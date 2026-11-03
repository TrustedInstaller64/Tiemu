"""Checked public mirrors around the resident C++ scoring workspace.

The C++ World owns preparation, old/new paths and score buffers. NumPy retains
the original dot/tanh microkernels until their exact replacement is established.
Unsupported inputs return to the unchanged A3 batch path before any mutation.
"""
import ctypes as ct
import threading
from types import MethodType

import numpy as np

from constants import PATH_NAMES
from entities import Pathstrider
from models import TitanToPathModel
from native_backend import get_native
from native_paths import _PATH_PROPERTY, _MAP_PATHS
from native_world import ResidentWorld, _ScoringSource, _ScoringView
from parliament_manager import ParliamentManager


_MULTIPLIER = ParliamentManager.get_zeitgeist_multiplier
_SOURCE_DTYPE = None
_ERROR_MODE = {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}


def _plain(cls, fields):
    return (cls.__getattribute__ is object.__getattribute__
            and cls.__setattr__ is object.__setattr__
            and not any(name in base.__dict__ for base in cls.__mro__
                        for name in (*fields, '__getattr__')))


def _bound(instance, name, function):
    method = getattr(instance, name, None)
    return type(method) is MethodType and method.__self__ is instance and method.__func__ is function


def _array(value, shape, owned=False):
    return (type(value) is np.ndarray and value.dtype == np.float64
            and value.shape == shape and value.flags.c_contiguous
            and value.flags.aligned and (not owned or value.flags.owndata))


def _bounded(values):
    # One batch check, not a NumPy call for each small entity vector. Restrict
    # warning-producing cases so warning-as-error filters keep A3's prefix.
    return bool(np.all(np.isfinite(values)) and np.all(np.abs(values) <= 1e100))


def _seat_key(key):
    if key is None:
        return True
    return (type(key) is tuple and len(key) == 2 and type(key[0]) is tuple
            and type(key[1]) in (int, np.int64) and abs(int(key[1])) <= 2**30
            and all(type(pair) is tuple and len(pair) == 2 and type(pair[0]) is str
                    and type(pair[1]) in (int, np.int64) and abs(int(pair[1])) <= 2**30
                    for pair in key[0]))


def _prepare_sources(entities, parliament, distribution, evaluator, world, native):
    if (native is None or getattr(native, '__binding__', None) != 'c-abi'
            or len(entities) < 2 or type(world) is not ResidentWorld
            or not world._handle.value or type(evaluator) is not native.BatchEvaluator
            or type(parliament) is not ParliamentManager
            or threading.active_count() != 1 or np.geterr() != _ERROR_MODE
            or ParliamentManager.__dict__.get('get_zeitgeist_multiplier') is not _MULTIPLIER
            or native.BatchEvaluator.__dict__.get('concepts') is not native._ORIGINAL_BATCH_CONCEPTS
            or Pathstrider.path_affinities is not _PATH_PROPERTY
            or TitanToPathModel.get_path_affinities is not _MAP_PATHS
            or not _plain(Pathstrider, ('titan_affinities', '_path_affinities',
                '_titan_to_path_model', 'is_titan_boss', 'activity', 'stability', 'score', 'heroic_tendency'))
            or not _plain(TitanToPathModel, ('titan_to_path_matrix',))
            or not _plain(ParliamentManager, ('seats', 'total_seats', 'suppression_factor',
                                             '_seat_cache_key', '_normalized_seat_vector'))
            or not _plain(native.BatchEvaluator, ('_closed', '_handle', '_lock'))
            or evaluator._closed or not evaluator._handle.value
            or not _bound(evaluator, 'concepts', native._ORIGINAL_BATCH_CONCEPTS)
            or not _bound(parliament, 'get_zeitgeist_multiplier', _MULTIPLIER)):
        return None
    try:
        seats, total, suppression = parliament.seats, parliament.total_seats, parliament.suppression_factor
        key, normalized = parliament._seat_cache_key, parliament._normalized_seat_vector
    except AttributeError:
        return None
    if (type(seats) is not dict or any(type(name) is not str for name in seats)
            or any(type(value) not in (int, np.int64) or abs(int(value)) > 2**30
                   for value in seats.values())
            or any(name not in seats for name in PATH_NAMES)
            or type(total) not in (int, np.int64) or not 0 < int(total) <= 2**30
            or type(suppression) not in (int, float, np.float64)
            or abs(suppression) > 1e100 or not np.isfinite(suppression)
            or not _seat_key(key)
            or (normalized is not None and (not _array(normalized, (12,)) or not _bounded(normalized)))):
        return None
    if distribution is not None and (not _array(distribution, (12,)) or not _bounded(distribution)):
        return None
    matrix, seen, arrays, owners, rows = None, set(), set(), [], []
    for entity in entities:
        if type(entity) is not Pathstrider or id(entity) in seen:
            return None
        seen.add(id(entity))
        try:
            model, values, cached, boss = (entity._titan_to_path_model, entity.titan_affinities,
                                           entity._path_affinities, entity.is_titan_boss)
        except AttributeError:
            return None
        if type(model) is not TitanToPathModel or not _bound(model, 'get_path_affinities', _MAP_PATHS):
            return None
        try:
            current = model.titan_to_path_matrix
        except AttributeError:
            return None
        if matrix is None:
            if not _array(current, (12, 12)) or not _bounded(current):
                return None
            matrix = current
        elif current is not matrix:
            return None
        if type(boss) is not bool:
            return None
        addresses = []
        for array in (values, cached):
            if array is None and len(addresses) == 1:
                addresses.append(0)
                continue
            if not _array(array, (12,), owned=True) or id(array) in arrays:
                return None
            arrays.add(id(array))
            owners.append(array)
            addresses.append(array.ctypes.data)
        rows.append((*addresses, boss))
    if not _bounded(np.asarray(owners)):
        return None
    global _SOURCE_DTYPE
    if _SOURCE_DTYPE is None:
        dtype = np.dtype([('titans', np.uintp), ('cached_paths', np.uintp), ('boss', np.uint8)], align=True)
        if (dtype.itemsize != ct.sizeof(_ScoringSource) or dtype.alignment != ct.alignment(_ScoringSource)
                or any(dtype.fields[name][1] != getattr(_ScoringSource, name).offset
                       for name, _ in _ScoringSource._fields_)):
            raise RuntimeError('Resident scoring source layout differs from the C ABI')
        _SOURCE_DTYPE = dtype
    sources = (_ScoringSource * len(rows))()
    np.ndarray((len(rows),), dtype=_SOURCE_DTYPE, buffer=sources)[:] = rows
    sources._owners = tuple(owners)
    return sources, matrix


def try_evaluate_resident(entities, parliament, distribution, evaluator, world):
    native = get_native()
    prepared = _prepare_sources(entities, parliament, distribution, evaluator, world, native)
    if prepared is None:
        return False
    sources, matrix = prepared
    count = len(entities)
    # Fixed lock order. Never call evaluator.concepts while holding its Lock;
    # finish invokes the existing C evaluator directly, with no second pool.
    with world._lock, evaluator._lock:
        if evaluator._closed or not evaluator._handle.value or not world._handle.value:
            return False
        view = _ScoringView()
        native._check(world._api.score_prepare(world._live(), sources, count,
                                               matrix.ctypes.data, matrix.size, ct.byref(view)))
        old_paths = np.ctypeslib.as_array(view.old_paths, shape=(count * 12,)).reshape(count, 12)
        # Refresh the public seat cache using its original expression exactly
        # once. An unchanged key must retain user edits to the cached vector.
        parliament.get_zeitgeist_multiplier(old_paths[0])
        normalized = parliament._normalized_seat_vector
        if normalized is None:
            multipliers = np.ones(count)
        else:
            alignment = np.vecdot(old_paths, normalized) - np.vecdot(1 - old_paths, normalized)
            # Scalar tanh retains the original ufunc dispatch on every platform.
            wave = np.fromiter((np.tanh(value) for value in alignment * 3), dtype=np.float64, count=count)
            multipliers = np.fmax(0.9, np.fmin(1.2, 1.05 + 0.15 * wave * parliament.suppression_factor))
        native._check(world._api.score_finish(world._live(), view.token, evaluator._live(),
            multipliers.ctypes.data, count, None if distribution is None else distribution.ctypes.data,
            0 if distribution is None else distribution.size, ct.byref(view)))
        paths = np.ctypeslib.as_array(view.paths, shape=(count * 12,)).reshape(count, 12)
        stats = np.ctypeslib.as_array(view.stats, shape=(count * 4,)).reshape(count, 4)
        for index, entity in enumerate(entities):
            entity._path_affinities = paths[index].copy()
            entity.activity = float(stats[index, 0])
            entity.stability = float(stats[index, 1])
            entity.score = float(stats[index, 3])
            entity.heroic_tendency = entity.activity
        world.scoring_count += 1
    return True
