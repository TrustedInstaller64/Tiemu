"""Ordered C++ candidate pools; Python retains the original random draws.

An index covers one generation. Ordinary interactions only change the two
participants; every actual reincarnator interaction is a callback boundary and
requires a fresh, fully validated snapshot of the original population argument.
"""
import ctypes as ct
import functools
import math
import random
import threading
from types import MethodType, SimpleNamespace

import numpy as np

import aeonic_cycle_manager as cycle_module
from entities import Pathstrider
from native_backend import get_native


_MANAGER = cycle_module.AeonicCycleManager
_METHODS = cycle_module._NATIVE_AEONIC_METHODS
_CHOICE, _RANDOM, _ISFINITE = random.choice, random.random, np.isfinite
_RANDOM_METHODS = {name: getattr(random.Random, name) for name in (
    '__getattribute__', 'choice', '_randbelow', 'getrandbits', 'random')}
_H, _S = ct.c_void_p, ct.c_size_t


def _class_attribute(cls, name):
    # Class-level getattr can itself execute a descriptor installed by a hook.
    return next((base.__dict__[name] for base in cls.__mro__ if name in base.__dict__), None)


def _plain(cls, fields):
    return (_class_attribute(cls, '__getattribute__') is object.__getattribute__
            and _class_attribute(cls, '__setattr__') is object.__setattr__
            and not any(name in base.__dict__ for base in cls.__mro__
                        for name in (*fields, '__getattr__')))


def _bound(instance, name, function):
    value = getattr(instance, name, None)
    return type(value) is MethodType and value.__self__ is instance and value.__func__ is function


def _metadata(manager, population, reincarnator):
    if (type(manager) is not _MANAGER or type(population) is not list
            or cycle_module.__dict__.get('random') is not random
            or cycle_module.__dict__.get('np') is not np
            or (reincarnator is not None and type(reincarnator) is not Pathstrider)
            or threading.active_count() != 1 or np.isfinite is not _ISFINITE
            or not _plain(_MANAGER, ('defeated_titans_this_cycle',))
            or any(_MANAGER.__dict__.get(name) is not function for name, function in _METHODS.items())
            or not _plain(Pathstrider, ('trait', 'is_titan_form', 'held_fire_seeds', 'score',
                                       '__bool__', '__len__'))
            or _class_attribute(Pathstrider, '__eq__') is not object.__eq__
            or _class_attribute(Pathstrider, '__hash__') is not object.__hash__
            or random.choice is not _CHOICE or random.random is not _RANDOM
            or any(_class_attribute(random.Random, name) is not method for name, method in _RANDOM_METHODS.items())
            or any(not _bound(manager, name, function) for name, function in _METHODS.items())):
        return None
    owner = random._inst
    if (type(owner) is not random.Random or type(_CHOICE) is not MethodType
            or _CHOICE.__self__ is not owner or _CHOICE.__func__ is not _RANDOM_METHODS['choice']
            or getattr(_RANDOM, '__self__', None) is not owner
            or any(name in owner.__dict__ for name in ('choice', '_randbelow', 'getrandbits', 'random'))):
        return None
    try:
        defeated = manager.defeated_titans_this_cycle
    except AttributeError:
        return None
    if type(defeated) is not set or any(type(value) is not str for value in defeated):
        return None
    seen, seeds_seen, flags = set(), {id(defeated)}, []
    reincarnator_position = -1
    for position, entity in enumerate(population):
        if type(entity) is not Pathstrider or id(entity) in seen:
            return None
        seen.add(id(entity))
        try:
            trait, form, seeds, score = (entity.trait, entity.is_titan_form,
                                          entity.held_fire_seeds, entity.score)
        except AttributeError:
            return None
        if (type(trait) is not str or (form is not None and type(form) is not str)
                or type(seeds) is not set or id(seeds) in seeds_seen
                or any(type(value) is not str for value in seeds)
                or type(score) not in (int, float, np.float64)
                or (type(score) is int and abs(score) > 2**53) or not math.isfinite(score)):
            return None
        seeds_seen.add(id(seeds))
        flags.append((trait == 'GoldenOne', bool(form), bool(seeds)))
        if entity is reincarnator:
            reincarnator_position = position
    # A zero-row snapshot can occur after a callback and still represents the
    # original selector's empty result without consuming random numbers.
    arrays = np.asarray(flags, dtype=np.uint8).reshape(len(flags), 3).T.copy(order='C')
    return tuple(population), arrays, reincarnator_position


@functools.lru_cache(maxsize=1)
def _entry_points(library):
    def bind(name, args):
        function = getattr(library, 'tiemu_aeonic_' + name)
        function.argtypes, function.restype = args, ct.c_int
        return function
    return SimpleNamespace(
        create=bind('create', [_H, _H, _H, _S, ct.c_int64, _H]),
        size=bind('pool_size', [_H, ct.c_uint32, ct.c_int64, _H]),
        at=bind('pool_at', [_H, ct.c_uint32, ct.c_int64, _S, _H]),
        update=bind('update', [_H, _S, ct.c_uint8, ct.c_uint8]),
        destroy=bind('destroy', [_H]))


class _Pool:
    """random.choice sees the original ordered sequence, with native rank lookup."""
    def __init__(self, index, kind, excluded=-1):
        self.index, self.kind, self.excluded = index, kind, excluded
        count = _S()
        index._native._check(index._api.size(index._live(), kind, excluded, ct.byref(count)))
        self.count = count.value

    def __len__(self):
        return self.count

    def __getitem__(self, rank):
        position = _S()
        self.index._native._check(self.index._api.at(
            self.index._live(), self.kind, self.excluded, rank, ct.byref(position)))
        return position.value


class AeonicIndex:
    def __init__(self, native, manager, population, reincarnator, metadata):
        self._lock = threading.RLock()
        self._native, self._api, self._handle = native, _entry_points(native._lib), _H()
        self._manager, self._population, self._reincarnator = manager, population, reincarnator
        self._valid = False
        self._replace(metadata)

    def _replace(self, metadata):
        entities, flags, reincarnator_position = metadata
        handle = _H()
        self._native._check(self._api.create(*(flags[i].ctypes.data for i in range(3)),
            len(entities), reincarnator_position, ct.byref(handle)))
        if self._handle.value:
            self._native._check(self._api.destroy(ct.byref(self._handle)))
        self._handle, self._entities = handle, entities
        self._positions = {id(entity): i for i, entity in enumerate(entities)}
        self._golden, self._reincarnator_position = flags[0], reincarnator_position
        self._valid = True

    def _live(self):
        if not self._handle.value:
            raise RuntimeError('Aeonic index is closed')
        if not self._valid:
            raise RuntimeError('Aeonic index requires a successful refresh')
        return self._handle

    def select(self):
        with self._lock:
            self._live()
            attackers = _Pool(self, 0)
            if not attackers:
                return None, None
            attacker = random.choice(attackers)
            targets = _Pool(self, 1, attacker)
            if not targets:
                return self._entities[attacker], None
            if attacker == self._reincarnator_position:
                holders = _Pool(self, 2, attacker)
                defenders = holders if holders and random.random() < .75 else targets
            elif self._golden[attacker]:
                titans = _Pool(self, 3, attacker)
                defenders = titans if titans else targets
            else:
                defenders = targets
            defender = random.choice(defenders)
            return self._entities[attacker], self._entities[defender]

    def update_pair(self, attacker, defender):
        with self._lock:
            for entity in (attacker, defender):
                self._native._check(self._api.update(self._live(), self._positions[id(entity)],
                    bool(entity.is_titan_form), bool(entity.held_fire_seeds)))

    def refresh(self):
        with self._lock:
            if not self._handle.value:
                raise RuntimeError('Aeonic index is closed')
            metadata = _metadata(self._manager, self._population, self._reincarnator)
            if metadata is None:
                self._valid = False
                return False
            self._replace(metadata)
            return True

    def close(self):
        with self._lock:
            if self._handle.value:
                self._native._check(self._api.destroy(ct.byref(self._handle)))
            self._valid = False
            self._entities, self._positions = (), {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def try_create_aeonic_index(manager, population, reincarnator):
    native = get_native()
    if (native is None or getattr(native, '__binding__', None) != 'c-abi'
            or type(population) is not list or len(population) < 2):
        return None
    metadata = _metadata(manager, population, reincarnator)
    return None if metadata is None else AeonicIndex(native, manager, population, reincarnator, metadata)
