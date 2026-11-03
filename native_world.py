"""Lifetime and buffer adapter for the standalone resident C++ world.

Python entities remain the public mirrors while their remaining writers migrate.
Every native boundary imports their current contents; an unchanged ndarray
address is never treated as proof that its values are unchanged. The C++ handle
owns all numerical state and stable IDs between calls. Borrowed native views are
read only while the handle lock is held and never become public entity arrays.
"""
import ctypes as ct
import functools
import threading
from types import SimpleNamespace

import numpy as np

from constants import TITAN_NAMES
from entities import Pathstrider
from native_backend import get_native


_S, _D, _I = ct.c_size_t, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int64)
_I32, _U, _B = ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint8)
_H, _F = ct.c_void_p, ct.c_double
_ID = ct.POINTER(ct.c_uint64)
_TRAITS = {'Mortal': 0, 'GoldenOne': 1, 'Reincarnator': 2}
_ASPECTS = {name: index for index, name in enumerate(TITAN_NAMES)}


class _Source(ct.Structure):
    # Pointer-width integer fields avoid transient ctypes pointer/reference
    # graphs. The source table explicitly retains its ndarray owners until sync ends.
    _fields_ = [('stable_id', ct.c_uint64), ('titans', _H), ('paths', _H),
                ('score', _F), ('hit_points', _F), ('trait', ct.c_int32),
                ('aspect', ct.c_int32), ('boss', ct.c_uint8), ('unlocked', ct.c_uint8)]


class _State(ct.Structure):
    _fields_ = [('entity_capacity', _S), ('vector_capacity', _S), ('titans', _D), ('paths', _D),
                ('scores', _D), ('hit_points', _D), ('traits', _I32), ('aspects', _I32),
                ('bosses', _B), ('unlocked', _B), ('path_valid', _B), ('titan_replaced', _B),
                ('path_changed', _B), ('original_titan_mutated', _B), ('original_titans', _D),
                ('numpy_mt', _U), ('python_mt', _U), ('rng_words', _S)]


class _View(ct.Structure):
    _fields_ = [('entities', _S), ('stable_ids', _ID), ('state', _State)]


@functools.lru_cache(maxsize=1)
def _entry_points(library):
    def bind(name, arguments, result=ct.c_int):
        function = getattr(library, 'tiemu_world_' + name)
        function.argtypes, function.restype = arguments, result
        return function

    version = bind('abi_version', [], ct.c_uint32)
    if version() != 1:
        raise RuntimeError('Unsupported standalone resident world ABI version')
    # Struct/buffer pointers use void* here, so independent C ABI consumers can
    # use their own equivalent ctypes Structure classes on this same library.
    return SimpleNamespace(
        create=bind('create', [_H]), sync=bind('sync', [_H, _H, _S]),
        view=bind('get_view', [_H, _H]),
        run=bind('run_encounters', [_H, _H, _H, _H, _S, _H]),
        births=bind('births', [_H, _H, _S, _H, _S, _S, _H, _S, _H, _S,
                              _F, _F, _H, _H]),
        destroy=bind('destroy', [_H]))


def _sources(population):
    """Validate presentation/ownership once; finite values are checked in C++."""
    if type(population) is not list or len({id(p) for p in population}) != len(population):
        raise ValueError('Resident population must be an ordered list of distinct entities')
    if (Pathstrider.__getattribute__ is not object.__getattribute__
            or Pathstrider.__setattr__ is not object.__setattr__
            or any(name in Pathstrider.__dict__ for name in (
                'titan_affinities', '_path_affinities', 'score', 'hp', 'trait',
                'titan_aspect', 'is_titan_boss', 'data_modification_unlocked'))):
        raise ValueError('Custom entity attribute access requires the reference path')
    output = (_Source * len(population))()
    arrays = set()
    owners = []
    for index, entity in enumerate(population):
        if type(entity) is not Pathstrider:
            raise ValueError('Resident state supports built-in entities only')
        values, paths = entity.titan_affinities, entity._path_affinities
        addresses = []
        for array, writable in ((values, True), (paths, False)):
            if array is None and not writable:
                addresses.append(None)
                continue
            if (type(array) is not np.ndarray or array.dtype != np.float64
                    or array.shape != (12,) or not array.flags.c_contiguous
                    or not array.flags.aligned or not array.flags.owndata
                    or (writable and not array.flags.writeable) or id(array) in arrays):
                raise ValueError('Resident vectors must be distinct owned float64 rows')
            arrays.add(id(array))
            owners.append(array)
            addresses.append(array.ctypes.data)
        score, hp = entity.score, entity.hp
        trait, aspect = entity.trait, entity.titan_aspect
        boss, unlocked = entity.is_titan_boss, entity.data_modification_unlocked
        if (not (type(score) in (float, np.float64) or (type(score) is int and score == 0))
                or type(hp) not in (float, np.float64)
                or type(trait) is not str or trait not in _TRAITS
                or (aspect is not None and (type(aspect) is not str or aspect not in _ASPECTS))
                or type(boss) is not bool or type(unlocked) is not bool):
            raise ValueError('Unsupported resident entity scalar fields')
        output[index] = _Source(0, *addresses, score, hp, _TRAITS[trait],
                                -1 if aspect is None else _ASPECTS[aspect], boss, unlocked)
    output._owners = tuple(owners)
    return output


def _copy(pointer, shape, dtype=np.float64):
    count = int(np.prod(shape))
    if not count:
        return np.empty(shape, dtype=dtype)
    return np.ctypeslib.as_array(pointer, shape=(count,)).reshape(shape).copy()


class ResidentWorld:
    def __init__(self):
        self._lock = threading.RLock()
        self._handle = _H()
        self._population = ()
        self._by_identity = {}
        self._ids = ()
        self.sync_count = self.encounter_count = self.birth_count = 0
        self._native = get_native()
        if self._native is None or getattr(self._native, '__binding__', None) != 'c-abi':
            raise RuntimeError('Resident world requires the standalone C++ backend')
        self._api = _entry_points(self._native._lib)
        self._native._check(self._api.create(ct.byref(self._handle)))

    def _live(self):
        if not self._handle.value:
            raise RuntimeError('Resident world is closed')
        return self._handle

    def close(self):
        with self._lock:
            if self._handle.value:
                self._native._check(self._api.destroy(ct.byref(self._handle)))
            self._population = ()
            self._by_identity = {}
            self._ids = ()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _view(self):
        """Private borrowed view: caller must hold _lock until its last read."""
        output = _View()
        self._native._check(self._api.view(self._live(), ct.byref(output)))
        return output

    def sync(self, population):
        with self._lock:
            self._live()
            self._sync_prepared(population, _sources(population))

    def _sync_prepared(self, population, sources):
        """Use descriptors made by this call's full encounter eligibility scan."""
        with self._lock:
            owners = tuple(population)
            if len(owners) != len(sources):
                raise ValueError('Resident source count does not match population')
            for index, entity in enumerate(owners):
                sources[index].stable_id = self._by_identity.get(id(entity), 0)
            # Both prior and incoming populations remain strong references
            # across this transaction, preventing Python object-ID reuse.
            self._native._check(self._api.sync(self._live(), sources, len(owners)))
            view = self._view()
            ids = tuple(int(view.stable_ids[index]) for index in range(len(owners)))
            self._by_identity = {id(entity): stable_id for entity, stable_id in zip(owners, ids)}
            self._population, self._ids = owners, ids
            self.sync_count += 1

    def stable_ids(self):
        with self._lock:
            self._live()
            return self._ids

    def snapshot(self):
        with self._lock:
            view = self._view()
            count, state = view.entities, view.state
            return {'ids': _copy(view.stable_ids, (count,), np.uint64),
                    'titans': _copy(state.titans, (count, 12)),
                    'paths': _copy(state.paths, (count, 12)),
                    'scores': _copy(state.scores, (count,)),
                    'hp': _copy(state.hit_points, (count,)),
                    'traits': _copy(state.traits, (count,), np.int32),
                    'aspects': _copy(state.aspects, (count,), np.int32),
                    'bosses': _copy(state.bosses, (count,), np.uint8),
                    'unlocked': _copy(state.unlocked, (count,), np.uint8),
                    'path_valid': _copy(state.path_valid, (count,), np.uint8)}

    def run_encounters(self, config, result, numpy_mt, python_mt):
        for state in (numpy_mt, python_mt):
            if (type(state) is not np.ndarray or state.dtype != np.uint32
                    or state.shape != (625,) or not state.flags.c_contiguous
                    or not state.flags.aligned):
                raise ValueError('Encounter RNG state must be 625 aligned uint32 words')
        with self._lock:
            self._native._check(self._api.run(self._live(), ct.byref(config),
                                              numpy_mt.ctypes.data, python_mt.ctypes.data,
                                              625, ct.byref(result)))
            self.encounter_count += 1

    def births(self, blueprint, noise, modifier, tide, purity_factor, max_norm):
        blueprint = self._native._float_array(blueprint, 'blueprint', 1)
        noise = self._native._float_array(noise, 'noise', 2)
        tide = self._native._float_array(tide, 'tide', 1)
        if blueprint.shape != (12,) or noise.shape[1] != 12 or tide.shape != (12,):
            raise ValueError('Birth vectors must have width 12')
        if modifier is not None:
            modifier = self._native._float_array(modifier, 'modifier', 1)
            if modifier.shape != (12,):
                raise ValueError('Birth modifier must have width 12')
        output, count = _D(), _S()
        with self._lock:
            self._native._check(self._api.births(
                self._live(), blueprint.ctypes.data, blueprint.size,
                noise.ctypes.data, noise.size, len(noise),
                None if modifier is None else modifier.ctypes.data,
                0 if modifier is None else modifier.size, tide.ctypes.data, tide.size,
                purity_factor, max_norm, ct.byref(output), ct.byref(count)))
            if count.value != noise.size:
                raise RuntimeError('Native newborn output size does not match its input')
            values = _copy(output, noise.shape)
            self.birth_count += 1
            return values
