"""Marshal one ordered encounter block into the standalone C++ engine.

Only built-in, non-overlapping entities with the original handlers and RNGs are
eligible. Another Python thread forces the original loop: snapshot/setstate of
the process-global RNGs must not discard another thread's draws. Native C++
workers do not count as Python threads. This module owns no numerical kernel,
selection rule or random-number generator.
"""
import ctypes as ct
from dataclasses import dataclass
import functools
import logging
import random
import threading

import numpy as np

from constants import PATH_NAMES, TITAN_NAMES
from entities import Pathstrider
from interaction_handler import InteractionHandler
from models import TitanToPathModel
from native_backend import get_native
from parliament_manager import ParliamentManager
from population_manager import PopulationManager


logger = logging.getLogger('OmphalosLogger')
_S, _L, _F = ct.c_size_t, ct.c_int64, ct.c_double
_D = ct.POINTER(ct.c_double)
_I = ct.POINTER(ct.c_int64)
_I32 = ct.POINTER(ct.c_int32)
_U = ct.POINTER(ct.c_uint32)
_B = ct.POINTER(ct.c_uint8)
_MT = ct.c_uint32 * 625
_PATH_NAMES, _TITAN_NAMES = tuple(PATH_NAMES), tuple(TITAN_NAMES)
_TRAITS = {'Mortal': 0, 'GoldenOne': 1, 'Reincarnator': 2}
_RELATIONS = {'CLASH': 1, 'SYNERGY': 2, 'REPULSION': 3, 'MENTORSHIP': 4}
_METHODS = {
    InteractionHandler: {name: getattr(InteractionHandler, name) for name in (
        'update_bins', 'discard_opponent', 'select_opponents', 'entity_interaction',
        '_handle_clash', '_handle_synergy', '_handle_repulsion',
        '_handle_mentorship_or_assimilation', '_handle_titan_battle')},
    PopulationManager: {name: getattr(PopulationManager, name) for name in (
        'recalculate_and_normalize_entity', 'normalize_affinities', 'track_distribution',
        'mark_paths_dirty', 'get_global_path_distribution')},
    ParliamentManager: {'get_zeitgeist_multiplier': ParliamentManager.get_zeitgeist_multiplier},
    TitanToPathModel: {'get_path_affinities': TitanToPathModel.get_path_affinities},
    Pathstrider: {name: getattr(Pathstrider, name) for name in (
        'path_affinities', 'dominant_path_idx', 'internal_purification')},
}
_RNG_FUNCTIONS = [(owner, name, getattr(owner, name)) for owner, names in (
    (random, ('random', 'getstate', 'setstate')),
    (np.random, ('random', 'rand', 'randint', 'get_state', 'set_state')))
    for name in names]


class _Config(ct.Structure):
    _fields_ = [('entities', _S), ('encounters', _S), ('opponent_index', ct.c_void_p),
                ('mapping', _D), ('mapping_count', _S), ('relations', _I32), ('relations_count', _S),
                ('interaction_modifiers', _D), ('modifiers_count', _S), ('tide', _D), ('tide_count', _S),
                ('purity_factor', _F), ('max_norm', _F), ('culling_strength', _F),
                ('reincarnator', _L), ('negative_world_titan', _L)]


class _State(ct.Structure):
    _fields_ = [('entity_capacity', _S), ('vector_capacity', _S), ('titans', _D), ('paths', _D),
                ('scores', _D), ('hit_points', _D), ('traits', _I32), ('aspects', _I32),
                ('bosses', _B), ('unlocked', _B), ('path_valid', _B), ('titan_replaced', _B),
                ('path_changed', _B), ('original_titan_mutated', _B), ('original_titans', _D),
                ('numpy_mt', _U), ('python_mt', _U), ('rng_words', _S)]


class _Event(ct.Structure):
    _fields_ = [('kind', ct.c_int32), ('path_or_aspect', ct.c_int32),
                ('first', _L), ('second', _L), ('damage', _F), ('remaining_hp', _F)]


class _Trace(ct.Structure):
    _fields_ = [('attempt', _L), ('first', _L), ('second', _L), ('culled', _L),
                ('applied', ct.c_int32), ('numpy_mt', _MT), ('python_mt', _MT)]


class _Result(ct.Structure):
    _fields_ = [('culled_ids', _I), ('culled_capacity', _S), ('culled_count', _S),
                ('updated_ids', _I), ('updated_capacity', _S), ('updated_count', _S),
                ('events', ct.POINTER(_Event)), ('event_capacity', _S), ('event_count', _S),
                ('trace', ct.POINTER(_Trace)), ('trace_capacity', _S), ('trace_count', _S),
                ('applied_encounters', _S), ('first_multiplier_paths', ct.c_double * 12)]


@dataclass
class EncounterResult:
    culled: set
    updated: set
    trace: list


def _ptr(array, pointer=_D):
    return array.ctypes.data_as(pointer)


@functools.lru_cache(maxsize=1)
def _entry_points(library):
    version = library.tiemu_encounters_abi_version
    version.argtypes, version.restype = [], ct.c_uint32
    if version() != 1:
        raise RuntimeError('Unsupported standalone encounter ABI version')
    run = library.tiemu_encounters_run
    run.argtypes, run.restype = [ct.POINTER(_Config), ct.POINTER(_State), ct.POINTER(_Result)], ct.c_int
    error = library.tiemu_encounters_last_error
    error.argtypes, error.restype = [], ct.c_char_p
    return run, error


def _original_methods(instance, cls):
    return (type(instance) is cls and all(
        name not in instance.__dict__ and getattr(cls, name) is original
        for name, original in _METHODS[cls].items()))


def _standard_rngs():
    if any(getattr(owner, name) is not original for owner, name, original in _RNG_FUNCTIONS):
        return False
    # A pre-import replacement must not become the accepted "original".
    python_owner = getattr(random, '_inst', None)
    numpy_owner = np.random.mtrand._rand
    if type(python_owner) is not random.Random or type(numpy_owner) is not np.random.RandomState:
        return False
    return all(getattr(original, '__self__', None) is (python_owner if owner is random else numpy_owner)
               for owner, _, original in _RNG_FUNCTIONS)


def _array(value, shape, *, owned=False, writable=False):
    return (type(value) is np.ndarray and value.dtype == np.dtype(np.float64)
            and value.shape == shape and value.flags.c_contiguous and value.flags.aligned
            and (not owned or value.flags.owndata) and (not writable or value.flags.writeable)
            and bool(np.all(np.isfinite(value))))


def _number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and bool(np.isfinite(value))


def _entity_number(value):
    # Fixed-width integers and lower-precision floating scalars have different
    # overflow/promotion rules. The newborn score exception is checked separately.
    return type(value) in (float, np.float64) and _number(value)


def _eligible(sim):
    if threading.active_count() != 1 or not _standard_rngs():
        return False
    if tuple(PATH_NAMES) != _PATH_NAMES or tuple(TITAN_NAMES) != _TITAN_NAMES:
        return False
    population = sim.population
    if type(population) is not list or not 2 <= len(population) <= (1 << 32)-1:
        return False
    if len({id(entity) for entity in population}) != len(population):
        return False
    manager, handler, parliament = sim.population_manager, sim.interaction_handler, sim.parliament_manager
    if not (_original_methods(manager, PopulationManager) and _original_methods(handler, InteractionHandler)
            and _original_methods(parliament, ParliamentManager)):
        return False
    if handler.population_manager is not manager or manager._tracked_population is not None:
        return False
    if not (_number(manager.purity_factor) and manager.purity_factor > 0
            and _number(manager.max_affinity_norm) and manager.max_affinity_norm > 0
            and _number(handler.culling_strength)
            and type(handler.culling_strength) in (int, float, np.float64)
            and handler._bin_size == 10):
        return False
    if not _array(manager.cosmic_tide_vector, (12,)):
        return False
    if type(handler.PATH_RELATIONSHIP_MATRIX) is not np.ndarray or handler.PATH_RELATIONSHIP_MATRIX.shape != (12, 12):
        return False
    if not all(type(value) in (str, np.str_) for value in handler.PATH_RELATIONSHIP_MATRIX.flat):
        return False
    if not _array(handler.PATH_INTERACTION_MATRIX, (12, 12)):
        return False
    if (type(parliament.seats) is not dict or set(parliament.seats) != set(_PATH_NAMES)
            or not all(_number(value) for value in parliament.seats.values())
            or not _number(parliament.total_seats) or parliament.total_seats <= 0
            or not _number(parliament.suppression_factor)):
        return False
    model = manager.titan_to_path_model
    if not _original_methods(model, TitanToPathModel) or handler.titan_to_path_model is not model:
        return False
    if not _array(model.titan_to_path_matrix, (12, 12)):
        return False
    reserved_ranges = [(array.ctypes.data, array.ctypes.data + array.nbytes) for array in (
        manager.cosmic_tide_vector, model.titan_to_path_matrix, handler.PATH_INTERACTION_MATRIX)]
    seen_arrays = set()
    for entity in population:
        if not _original_methods(entity, Pathstrider) or entity._titan_to_path_model is not model:
            return False
        if (type(entity.trait) is not str or entity.trait not in _TRAITS
                or entity.titan_aspect not in (None, *_TITAN_NAMES)
                or type(entity.name) is not str or type(entity.is_titan_boss) is not bool
                or type(entity.data_modification_unlocked) is not bool
                or not (_entity_number(entity.score) or (type(entity.score) is int and entity.score == 0))
                or not _entity_number(entity.hp) or entity.score == -1e-6
                or abs(entity.score) > 1e15):
            return False
        if not _array(entity.titan_affinities, (12,), owned=True, writable=True):
            return False
        cache = entity._path_affinities
        if cache is not None and not _array(cache, (12,), owned=True):
            return False
        for array in (entity.titan_affinities, cache):
            if array is not None:
                if id(array) in seen_arrays:
                    return False
                start, end = array.ctypes.data, array.ctypes.data + array.nbytes
                if any(start < right and left < end for left, right in reserved_ranges):
                    return False
                seen_arrays.add(id(array))
    return sim.reincarnator is None or any(entity is sim.reincarnator for entity in population)


def _emit_events(events, count, population):
    for index in range(count):
        event = events[index]
        first, second = population[event.first], population[event.second]
        if event.kind == 1:
            logger.info(f"\033[91m卡厄斯兰那在其命途 '{PATH_NAMES[event.path_or_aspect]}' 上对敌人 {second.name} 发动了无条件的毁灭！\033[0m")
        elif event.kind == 2:
            logger.info(f"\n\033[35m【逐火之旅】黄金裔 {first.name} ({TITAN_NAMES[event.path_or_aspect]}) 挑战泰坦 {second.name}！\033[0m")
        elif event.kind == 3:
            logger.info(f"{first.name} 对泰坦造成了 {event.damage:.2f} 点伤害！泰坦剩余HP: {event.remaining_hp:.2f}")
        elif event.kind == 4:
            logger.info(f"\033[92m泰坦 {second.name} 已被击败！黄金裔 {first.name} 继承了其火种，成为半神！\033[0m")
        elif event.kind == 5:
            logger.info(f"\033[91m泰坦发起了反击！{first.name} 灰飞烟灭！\033[0m")
        else:
            raise RuntimeError('Native encounter returned an unknown log event')


def try_run_encounters(sim, *, trace=False):
    """Return None without state/RNG changes when the original loop is required.

    After eligibility succeeds, ABI errors are fatal rather than falling back
    over a possibly advanced native index. A debug trace is opt-in and is never
    allocated or populated by normal production calls.
    """
    native = get_native()
    if native is None or getattr(native, '__binding__', None) != 'c-abi':
        return None
    try:
        eligible = _eligible(sim)
    except (AttributeError, TypeError, ValueError, OverflowError):
        eligible = False
    if not eligible:
        return None
    run, last_error = _entry_points(native._lib)
    population = sim.population
    count, encounters = len(population), min(2*len(population), 5000)
    manager, handler = sim.population_manager, sim.interaction_handler
    titans = np.stack([entity.titan_affinities for entity in population])
    original_titans = titans.copy()
    paths = np.zeros((count, 12), dtype=np.float64)
    valid = np.zeros(count, dtype=np.uint8)
    for i, entity in enumerate(population):
        if entity._path_affinities is not None:
            paths[i] = entity._path_affinities
            valid[i] = 1
    scores = np.array([entity.score for entity in population], dtype=np.float64)
    hp = np.array([entity.hp for entity in population], dtype=np.float64)
    traits = np.array([_TRAITS[entity.trait] for entity in population], dtype=np.int32)
    aspects = np.array([-1 if entity.titan_aspect is None else _TITAN_NAMES.index(entity.titan_aspect)
                        for entity in population], dtype=np.int32)
    bosses = np.array([entity.is_titan_boss for entity in population], dtype=np.uint8)
    unlocked = np.array([entity.data_modification_unlocked for entity in population], dtype=np.uint8)
    titan_replaced = np.zeros(count, dtype=np.uint8)
    path_changed = np.zeros(count, dtype=np.uint8)
    old_mutated = np.zeros(count, dtype=np.uint8)
    relations = np.array([_RELATIONS.get(str(value), 0)
                          for value in handler.PATH_RELATIONSHIP_MATRIX.flat], dtype=np.int32)
    culled_ids, updated_ids = np.empty(count, dtype=np.int64), np.empty(count, dtype=np.int64)
    events = (_Event * (encounters*3))()
    traces = (_Trace * encounters)() if trace else None
    # Recheck after data preparation, before touching the index or global RNGs.
    if threading.active_count() != 1 or not _standard_rngs():
        return None
    numpy_state, python_state = np.random.get_state(), random.getstate()
    if numpy_state[0] != 'MT19937' or python_state[0] != 3:
        return None
    numpy_mt = np.concatenate((numpy_state[1], np.array([numpy_state[2]], dtype=np.uint32)))
    python_mt = np.array(python_state[1], dtype=np.uint32)
    if numpy_mt.shape != (625,) or python_mt.shape != (625,):
        return None
    handler.update_bins(population)
    opponent_index = handler._native_opponents
    config = _Config(count, encounters, None, _ptr(manager.titan_to_path_model.titan_to_path_matrix), 144,
                     _ptr(relations, _I32), 144, _ptr(handler.PATH_INTERACTION_MATRIX), 144,
                     _ptr(manager.cosmic_tide_vector), 12, manager.purity_factor, manager.max_affinity_norm,
                     handler.culling_strength,
                     next((i for i, entity in enumerate(population) if entity is sim.reincarnator), -1),
                     _TITAN_NAMES.index('负世'))
    state = _State(count, count*12, _ptr(titans), _ptr(paths), _ptr(scores), _ptr(hp),
                   _ptr(traits, _I32), _ptr(aspects, _I32), _ptr(bosses, _B), _ptr(unlocked, _B),
                   _ptr(valid, _B), _ptr(titan_replaced, _B), _ptr(path_changed, _B), _ptr(old_mutated, _B),
                   _ptr(original_titans), _ptr(numpy_mt, _U), _ptr(python_mt, _U), 625)
    result = _Result(_ptr(culled_ids, _I), count, 0, _ptr(updated_ids, _I), count, 0,
                     events, encounters*3, 0, traces, encounters if trace else 0, 0, 0,
                     (ct.c_double * 12)())
    # Retain the original manager lifecycle, including replacement of stale
    # dictionaries/sets and finally cleanup. Numeric histogram work stays in C++.
    with manager.track_distribution(population):
        with opponent_index._lock:
            config.opponent_index = opponent_index._live()
            status = run(ct.byref(config), ct.byref(state), ct.byref(result))
            if status:
                raw = last_error()
                message = raw.decode('utf-8', errors='replace') if raw else 'Encounter engine failed'
                raise {1: ValueError, 2: IndexError, 3: RuntimeError}.get(status, RuntimeError)(message)
    np.random.set_state((numpy_state[0], numpy_mt[:624], int(numpy_mt[624]), *numpy_state[3:]))
    random.setstate((python_state[0], tuple(int(value) for value in python_mt), python_state[2]))
    # The unused multiplier arithmetic previously initialized this public seat
    # cache. Preserve that side effect once using the first actual old-path read.
    if result.applied_encounters:
        sim.parliament_manager.get_zeitgeist_multiplier(np.ctypeslib.as_array(result.first_multiplier_paths))
    # Even zero damage and a zero score promotion change integer scalar types.
    # Event-time writes identify these operations independently of value changes.
    score_written = {events[i].first for i in range(result.event_count) if events[i].kind == 4}
    hp_written = set()
    hp_numpy = [type(entity.hp) is np.float64 for entity in population]
    for i in range(result.event_count):
        event = events[i]
        if event.kind == 3:
            hp_written.add(event.second)
            # NumPy scalar damage promotes even an unchanged Python-float HP.
            # Score multiplication keeps NumPy-ness (int(0) only becomes float),
            # so its initial scalar kind is valid throughout this ordered replay.
            hp_numpy[event.second] |= type(population[event.first].score) is np.float64
    for i, entity in enumerate(population):
        if old_mutated[i]:
            entity.titan_affinities[:] = original_titans[i]
        if titan_replaced[i]:
            entity.titan_affinities = titans[i].copy()
        if path_changed[i]:
            entity._path_affinities = paths[i].copy() if valid[i] else None
        if i in score_written or scores[i] != entity.score:
            entity.score = np.float64(scores[i]) if isinstance(entity.score, np.generic) else float(scores[i])
        if i in hp_written or hp[i] != entity.hp:
            entity.hp = np.float64(hp[i]) if hp_numpy[i] else float(hp[i])
        if bool(unlocked[i]) != entity.data_modification_unlocked:
            entity.data_modification_unlocked = bool(unlocked[i])
    culled, updated = set(), set()
    for i in culled_ids[:result.culled_count]:
        culled.add(population[int(i)])
    for i in updated_ids[:result.updated_count]:
        updated.add(population[int(i)])
    _emit_events(events, result.event_count, population)
    debug_trace = []
    if trace:
        for entry in traces[:result.trace_count]:
            debug_trace.append({'attempt': int(entry.attempt), 'first': int(entry.first),
                                'second': int(entry.second), 'culled': int(entry.culled),
                                'applied': bool(entry.applied), 'numpy_state': list(entry.numpy_mt),
                                'python_state': list(entry.python_mt)})
    return EncounterResult(culled, updated, debug_trace)
