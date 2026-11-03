"""Resident metadata plans with the original ordered Python identity writes."""
import ctypes as ct
import math
import random
import threading
from types import MethodType

import numpy as np

from entities import Pathstrider
from native_backend import get_native
from native_world import ResidentWorld, _GoldenSource, _GoldenView
import population_manager as population_module


PopulationManager = population_module.PopulationManager
_GET_WORLD = PopulationManager.get_native_world
_WORLD_LIVE = ResidentWorld._live
_SHUFFLE = random.shuffle
_RANDOM_METHODS = {name: getattr(random.Random, name) for name in (
    '__getattribute__', 'shuffle', '_randbelow', 'getrandbits', 'random')}
_SOURCE_DTYPE = None
# Keep the original selection until cloud timings show a net benefit including
# metadata preparation, C ABI calls and the ordered Python identity writes.
NATIVE_GOLDEN_ENABLED = False


def _class_attribute(cls, name):
    # Class-level getattr can itself execute a descriptor installed by a hook.
    return next((base.__dict__[name] for base in cls.__mro__ if name in base.__dict__), None)


def _plain(cls, fields):
    return (_class_attribute(cls, '__getattribute__') is object.__getattribute__
            and _class_attribute(cls, '__setattr__') is object.__setattr__
            and not any(name in base.__dict__ for base in cls.__mro__
                        for name in (*fields, '__getattr__')))


def _bound(instance, name, original):
    method = getattr(instance, name, None)
    return type(method) is MethodType and method.__self__ is instance and method.__func__ is original


def _metadata(manager, population):
    # Check class hooks before touching instances: even a rejected getter may
    # draw RNG or mutate an earlier entity. Numeric vectors are never accessed.
    if (type(manager) is not PopulationManager
            or not _plain(PopulationManager, ('golden_one_cap', '_native_world'))
            or PopulationManager.__dict__.get('get_native_world') is not _GET_WORLD
            or not _bound(manager, 'get_native_world', _GET_WORLD)
            or not _plain(Pathstrider, ('trait', 'is_titan_boss', 'score',
                'titan_aspect', 'data_modification_unlocked', 'golden_one_tenure'))
            or type(_SHUFFLE) is not MethodType
            or _SHUFFLE.__func__ is not _RANDOM_METHODS['shuffle']
            or random.shuffle is not _SHUFFLE
            or any(_class_attribute(random.Random, name) is not original
                   for name, original in _RANDOM_METHODS.items())):
        return None
    owner = random._inst
    if (type(owner) is not random.Random or _SHUFFLE.__self__ is not owner
            or any(name in owner.__dict__ for name in ('shuffle', '_randbelow', 'getrandbits', 'random'))):
        return None
    aspects = population_module.TITAN_NAMES
    if type(aspects) is not list or len(aspects) != 12 or any(type(name) is not str for name in aspects):
        return None
    try:
        cap, world = manager.golden_one_cap, manager._native_world
    except AttributeError:
        return None
    if type(cap) is not int or not -(1 << 63) <= cap < (1 << 63):
        return None
    if world is not None and (
            type(world) is not ResidentWorld
            or not _plain(ResidentWorld, ('_handle', '_lock', '_api', 'golden_count'))
            or ResidentWorld.__dict__.get('_live') is not _WORLD_LIVE
            or not _bound(world, '_live', _WORLD_LIVE)
            or not world._handle.value):
        return None
    seen, rows = set(), []
    for entity in population:
        if type(entity) is not Pathstrider or id(entity) in seen:
            return None
        seen.add(id(entity))
        try:
            trait, boss, score = entity.trait, entity.is_titan_boss, entity.score
        except AttributeError:
            return None
        if (type(trait) is not str or type(boss) is not bool
                or type(score) not in (int, float, np.float64)
                or (type(score) is int and abs(score) > 2**53)
                or not math.isfinite(score)):
            return None
        # Unknown ordinary string traits behave like Mortal in this operation.
        role = 1 if trait == 'GoldenOne' else (2 if trait == 'Reincarnator' else 0)
        rows.append((score, role, boss))
    global _SOURCE_DTYPE
    if _SOURCE_DTYPE is None:
        dtype = np.dtype([('score', np.float64), ('trait', np.int32), ('boss', np.uint8)], align=True)
        if (dtype.itemsize != ct.sizeof(_GoldenSource) or dtype.alignment != ct.alignment(_GoldenSource)
                or any(dtype.fields[name][1] != getattr(_GoldenSource, name).offset
                       for name, _ in _GoldenSource._fields_)):
            raise RuntimeError('Resident golden source layout differs from the C ABI')
        _SOURCE_DTYPE = dtype
    sources = (_GoldenSource * len(rows))()
    np.ndarray((len(rows),), dtype=_SOURCE_DTYPE, buffer=sources)[:] = rows
    return sources, cap, aspects


def try_update_golden_ones(manager, population):
    """Apply a guarded C++ plan, or decline before any public state/RNG change."""
    if not NATIVE_GOLDEN_ENABLED:
        return False
    native = get_native()
    if (native is None or getattr(native, '__binding__', None) != 'c-abi'
            or type(population) is not list or threading.active_count() != 1):
        return False
    if not population:
        return True
    prepared = _metadata(manager, population)
    if prepared is None:
        return False
    sources, cap, aspects = prepared
    world = manager.get_native_world()
    with world._lock:
        if not world._handle.value:
            return False
        view = _GoldenView()
        status = world._api.golden_plan(world._live(), sources, len(population), cap, ct.byref(view))
        if status == 3:
            # An encounter-poisoned World cannot supply a plan. The C contract
            # rejects it before publication, so the untouched original loop can
            # still update Python identities and retain its exception prefix.
            return False
        native._check(status)
        for offset in range(view.reset_count):
            entity = population[view.reset_positions[offset]]
            entity.trait = 'Mortal'
            entity.titan_aspect = None
            entity.data_modification_unlocked = False
        titan_aspects = list(aspects)
        random.shuffle(titan_aspects)
        titan_count = min(view.selected_count, 12)
        for offset in range(titan_count):
            entity = population[view.selected_positions[offset]]
            entity.trait = 'GoldenOne'
            entity.golden_one_tenure = 0
            entity.titan_aspect = titan_aspects.pop()
        for offset in range(12, view.selected_count):
            entity = population[view.selected_positions[offset]]
            entity.trait = 'GoldenOne'
            entity.golden_one_tenure = 0
        world.golden_count += 1
    return True
