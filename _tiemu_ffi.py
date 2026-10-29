"""Thin NumPy adapter to the standalone Tiemu C ABI.

The shared library does not import Python or expose a Python extension entry
point. This module only owns buffers, validates presentation shapes, and manages
handle lifetimes. Numerical algorithms and worker threads belong to the core.
Inputs must remain unchanged in other threads until a call returns.
"""
import ctypes as _ct
import importlib.util as _import_util
import operator as _operator
import threading as _threading

import numpy as np


__binding__ = "c-abi"
_spec = _import_util.find_spec("_tiemu_core")
if _spec is None or not _spec.origin:
    raise ImportError("Build/install the standalone Tiemu C++ core (_tiemu_core) first")
try:
    # Do not import this library: it deliberately has no PyInit function.
    _lib = _ct.CDLL(_spec.origin)
except OSError as _exc:
    raise ImportError(f"Cannot load the standalone Tiemu C++ core: {_exc}") from _exc
__core_file__ = _spec.origin

_D = _ct.POINTER(_ct.c_double)
_I = _ct.POINTER(_ct.c_int64)
_U = _ct.POINTER(_ct.c_uint32)
_H = _ct.c_void_p
_HP = _ct.POINTER(_H)
_S = _ct.c_size_t
_L = _ct.c_int64
_F = _ct.c_double


def _bind(name, arguments, result=_ct.c_int):
    try:
        function = getattr(_lib, name)
    except AttributeError as exc:
        raise ImportError(f"Tiemu C ABI is missing required symbol {name}") from exc
    function.argtypes = arguments
    function.restype = result
    return function


_version = _bind("tiemu_abi_version", [], _ct.c_uint32)
_last_error = _bind("tiemu_last_error", [], _ct.c_char_p)
ABI_VERSION = int(_version())
if ABI_VERSION != 1:
    raise ImportError(f"Unsupported Tiemu C ABI version {ABI_VERSION}; expected 1")


def _check(status):
    """Read the current OS thread's error before any other status-returning call."""
    if status:
        raw = _last_error()
        message = raw.decode("utf-8", errors="replace") if raw else "Tiemu C ABI call failed"
        error = {1: ValueError, 2: IndexError, 3: RuntimeError}.get(status, RuntimeError)
        raise error(message)


_path_affinities = _bind("tiemu_path_affinities", [_D, _S, _S, _S, _D, _S, _S, _D, _S])
_concepts = _bind("tiemu_concepts", [_D, _S, _F, _D, _S, _ct.c_int, _D, _S])
_purify = _bind("tiemu_purify", [_D, _S, _F, _D, _S])
_normalize = _bind("tiemu_normalize", [_D, _S, _F, _D, _S])
_normalize_pipeline = _bind("tiemu_normalize_pipeline", [_D, _S, _F, _D, _S, _F, _D, _S])
_synergy = _bind("tiemu_synergy", [_D, _S, _D, _S, _D, _S, _D, _S])
_clash = _bind("tiemu_clash", [_D, _S, _D, _S, _F, _D, _S, _D, _S])
_mentorship = _bind("tiemu_mentorship", [_D, _S, _D, _S, _F, _D, _S, _D, _S])
_repulsion = _bind("tiemu_repulsion", [_D, _S, _D, _S, _D, _S])
_vote_totals = _bind("tiemu_vote_totals", [_I, _S, _D, _S, _D, _S, _S, _D, _S])
_inorganic_step = _bind("tiemu_inorganic_step", [_D, _S, _D, _S, _D, _S, _L, _D, _S, _D, _S])
_inorganic_run = _bind("tiemu_inorganic_run", [_D, _S, _D, _S, _U, _S, _L, _L,
                                             _D, _S, _D, _S, _U, _S])
_batch_create = _bind("tiemu_batch_create", [_L, _L, _HP])
_batch_close = _bind("tiemu_batch_close", [_H])
_batch_destroy = _bind("tiemu_batch_destroy", [_HP])
_batch_concepts = _bind("tiemu_batch_concepts", [_H, _D, _S, _S, _S, _D, _S, _S,
                                               _D, _S, _D, _S, _I, _S, _D, _S, _D, _S])
_hist_create = _bind("tiemu_histogram_create", [_I, _S, _L, _HP])
_hist_update = _bind("tiemu_histogram_update", [_H, _L, _L])
_hist_distribution = _bind("tiemu_histogram_distribution", [_H, _D, _S])
_hist_destroy = _bind("tiemu_histogram_destroy", [_HP])
_opponent_create = _bind("tiemu_opponent_create", [_D, _S, _F, _HP])
_opponent_alive = _bind("tiemu_opponent_alive_count", [_H, _I])
_opponent_entity = _bind("tiemu_opponent_entity_at", [_H, _L, _I])
_opponent_count = _bind("tiemu_opponent_count", [_H, _L, _F, _I])
_opponent_at = _bind("tiemu_opponent_at", [_H, _L, _F, _L, _I])
_opponent_remove = _bind("tiemu_opponent_remove", [_H, _L])
_opponent_destroy = _bind("tiemu_opponent_destroy", [_HP])


def _array(value, dtype, name, ndim=None, integer=False):
    if integer:
        source = np.asarray(value)
        if not np.can_cast(source.dtype, dtype, casting="safe"):
            raise TypeError(f"{name} requires safely convertible integer values")
        value = source
    result = np.asarray(value, dtype=dtype, order="C")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    # Contiguity alone does not ensure that a buffer is aligned for C++ loads.
    if not result.flags.aligned:
        result = result.copy(order="C")
    return result


def _float_array(value, name, ndim=None):
    return _array(value, np.float64, name, ndim)


def _int_array(value, name, ndim=1):
    return _array(value, np.int64, name, ndim, integer=True)


def _integer(value, name):
    result = _operator.index(value)
    if not -(1 << 63) <= result < (1 << 63):
        raise OverflowError(f"{name} is outside the signed 64-bit range")
    return result


def _ptr(array, pointer=_D):
    return array.ctypes.data_as(pointer)


def _distribution(value, paths):
    if value is None:
        return None
    result = _float_array(value, "distribution", 1)
    if result.size != paths:
        raise ValueError("distribution must match paths")
    return result


def _dist_args(distribution):
    return (None, 0) if distribution is None else (_ptr(distribution), distribution.size)


def path_affinities(values, matrix):
    values = _float_array(values, "values")
    matrix = _float_array(matrix, "matrix", 2)
    if values.ndim not in (1, 2):
        raise ValueError("values must be a vector or matrix")
    titans, paths = values.shape[-1], matrix.shape[1]
    if matrix.shape[0] != titans or paths == 0:
        raise ValueError("mapping matrix shape must match affinities and contain paths")
    rows = 1 if values.ndim == 1 else values.shape[0]
    output = np.empty((paths,) if values.ndim == 1 else (rows, paths), dtype=np.float64)
    _check(_path_affinities(_ptr(values), values.size, rows, titans,
                           _ptr(matrix), matrix.size, paths, _ptr(output), output.size))
    return output


def concepts(paths, multiplier, distribution=None, is_boss=False):
    paths = _float_array(paths, "paths", 1)
    if not paths.size:
        raise ValueError("paths must contain at least one path")
    distribution = _distribution(distribution, paths.size)
    output = np.empty(4, dtype=np.float64)
    _check(_concepts(_ptr(paths), paths.size, multiplier, *_dist_args(distribution),
                     bool(is_boss), _ptr(output), output.size))
    return tuple(map(float, output))


def purify(values, factor):
    values = _float_array(values, "values", 1)
    output = np.empty(values.shape, dtype=np.float64)
    _check(_purify(_ptr(values), values.size, factor, _ptr(output), output.size))
    return output


def normalize(values, max_norm):
    values = _float_array(values, "values", 1)
    output = np.empty(values.shape, dtype=np.float64)
    _check(_normalize(_ptr(values), values.size, max_norm, _ptr(output), output.size))
    return output


def normalize_pipeline(values, factor, tide, max_norm):
    values = _float_array(values, "values", 1)
    tide = _float_array(tide, "tide", 1)
    if tide.size != values.size:
        raise ValueError("tide must match values")
    output = np.empty(values.shape, dtype=np.float64)
    _check(_normalize_pipeline(_ptr(values), values.size, factor, _ptr(tide), tide.size,
                              max_norm, _ptr(output), output.size))
    return output


def _pair(first, second):
    first = _float_array(first, "first vector", 1)
    second = _float_array(second, "second vector", 1)
    if first.size != second.size:
        raise ValueError("interaction vectors must have the same length")
    return first, second


def _pair_call(function, first, second, *scalar):
    first, second = _pair(first, second)
    out_first = np.empty(first.shape, dtype=np.float64)
    out_second = np.empty(second.shape, dtype=np.float64)
    _check(function(_ptr(first), first.size, _ptr(second), second.size, *scalar,
                    _ptr(out_first), out_first.size, _ptr(out_second), out_second.size))
    return out_first, out_second


def synergy(a, b):
    return _pair_call(_synergy, a, b)


def clash(winner, loser, modifier):
    return _pair_call(_clash, winner, loser, modifier)


def mentorship(stronger, weaker, score_ratio):
    return _pair_call(_mentorship, stronger, weaker, score_ratio)


def repulsion(values, noise):
    values, noise = _pair(values, noise)
    if not values.size:
        raise ValueError("repulsion requires a nonempty vector")
    output = np.empty(values.shape, dtype=np.float64)
    _check(_repulsion(_ptr(values), values.size, _ptr(noise), noise.size,
                      _ptr(output), output.size))
    return output


def vote_totals(dom_indices, weighted_scores, vote_logits):
    dominant = _int_array(dom_indices, "dominant indices")
    weights = _float_array(weighted_scores, "weighted scores", 1)
    logits = _float_array(vote_logits, "vote logits", 2)
    if dominant.size != weights.size:
        raise ValueError("indices and weights must have the same length")
    if logits.shape[0] == 0 or logits.shape[0] != logits.shape[1]:
        raise ValueError("vote logits must be a nonempty square matrix")
    output = np.empty(logits.shape[0], dtype=np.float64)
    _check(_vote_totals(_ptr(dominant, _I), dominant.size, _ptr(weights), weights.size,
                        _ptr(logits), logits.size, logits.shape[0], _ptr(output), output.size))
    return output


def _grid_inputs(activity, stability, grid):
    activity = _float_array(activity, "activity", 1)
    stability = _float_array(stability, "stability", 1)
    grid = _integer(grid, "grid size")
    if grid <= 0 or activity.size != grid * grid or stability.size != activity.size:
        raise ValueError("activity/stability must match a positive grid squared")
    return activity, stability, grid


def inorganic_step(activity, stability, noise, grid_size):
    activity, stability, grid = _grid_inputs(activity, stability, grid_size)
    noise = _float_array(noise, "noise", 2)
    if noise.shape != (activity.size, 2):
        raise ValueError("noise must have shape (grid squared, 2)")
    out_activity = np.empty(activity.shape, dtype=np.float64)
    out_stability = np.empty(stability.shape, dtype=np.float64)
    _check(_inorganic_step(_ptr(activity), activity.size, _ptr(stability), stability.size,
                           _ptr(noise), noise.size, grid, _ptr(out_activity), out_activity.size,
                           _ptr(out_stability), out_stability.size))
    return out_activity, out_stability


def inorganic_run(activity, stability, mt, steps, grid):
    """Run a whole ordered grid segment in C++, including the CPython RNG stream."""
    activity, stability, grid = _grid_inputs(activity, stability, grid)
    mt = _array(mt, np.uint32, "MT state", 1, integer=True)
    steps = _integer(steps, "steps")
    if mt.size != 625 or steps < 0:
        raise ValueError("MT state must contain 625 words and steps must be nonnegative")
    out_activity = np.empty(activity.shape, dtype=np.float64)
    out_stability = np.empty(stability.shape, dtype=np.float64)
    out_mt = np.empty(mt.shape, dtype=np.uint32)
    _check(_inorganic_run(_ptr(activity), activity.size, _ptr(stability), stability.size,
                          _ptr(mt, _U), mt.size, steps, grid,
                          _ptr(out_activity), out_activity.size,
                          _ptr(out_stability), out_stability.size, _ptr(out_mt, _U), out_mt.size))
    return out_activity, out_stability, out_mt


class _Handle:
    def _initialize(self, destroy):
        self._lock = _threading.Lock()
        self._handle = _H()
        # Keep the function pointer alive even during interpreter shutdown.
        self._destroy = destroy

    def _live(self):
        if not self._handle.value:
            raise RuntimeError(f"{type(self).__name__} is closed")
        return self._handle

    def close(self):
        with self._lock:
            if self._handle.value:
                _check(self._destroy(_ct.byref(self._handle)))

    def __del__(self):
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        # A method retains self during its call. The same lock also protects
        # explicit close from racing a CDLL call that has released the GIL.
        try:
            with lock:
                if self._handle.value:
                    self._destroy(_ct.byref(self._handle))
        except Exception:
            pass


class BatchEvaluator(_Handle):
    def __init__(self, workers=1, min_rows=4096):
        self._initialize(_batch_destroy)
        workers = _integer(workers, "workers")
        min_rows = _integer(min_rows, "min_rows")
        _check(_batch_create(workers, min_rows, _ct.byref(self._handle)))

    def close(self):
        with self._lock:
            if self._handle.value:
                _check(_batch_close(self._handle))

    def concepts(self, values, matrix, multipliers, distribution, bosses):
        values = _float_array(values, "values", 2)
        matrix = _float_array(matrix, "matrix", 2)
        rows, titans = values.shape
        paths = matrix.shape[1]
        if matrix.shape[0] != titans or paths == 0:
            raise ValueError("mapping matrix must match values and contain paths")
        multipliers = _float_array(multipliers, "multipliers", 1)
        bosses = _int_array(bosses, "bosses")
        if multipliers.size != rows or bosses.size != rows:
            raise ValueError("multipliers and bosses must match the row count")
        distribution = _distribution(distribution, paths)
        out_paths = np.empty((rows, paths), dtype=np.float64)
        out_stats = np.empty((rows, 4), dtype=np.float64)
        with self._lock:
            _check(_batch_concepts(self._live(), _ptr(values), values.size, rows, titans,
                                   _ptr(matrix), matrix.size, paths,
                                   _ptr(multipliers), multipliers.size, *_dist_args(distribution),
                                   _ptr(bosses, _I), bosses.size,
                                   _ptr(out_paths), out_paths.size, _ptr(out_stats), out_stats.size))
        return out_paths, out_stats


class PathHistogram(_Handle):
    def __init__(self, indices, n_paths):
        self._initialize(_hist_destroy)
        indices = _int_array(indices, "indices")
        self._paths = _integer(n_paths, "number of paths")
        _check(_hist_create(_ptr(indices, _I), indices.size, self._paths, _ct.byref(self._handle)))

    def update(self, index, dom):
        index, dom = _integer(index, "index"), _integer(dom, "path")
        with self._lock:
            _check(_hist_update(self._live(), index, dom))

    def distribution(self):
        with self._lock:
            handle = self._live()
            output = np.empty(self._paths, dtype=np.float64)
            _check(_hist_distribution(handle, _ptr(output), output.size))
        return output


class OpponentIndex(_Handle):
    def __init__(self, scores, bin_size):
        self._initialize(_opponent_destroy)
        scores = _float_array(scores, "scores", 1)
        _check(_opponent_create(_ptr(scores), scores.size, bin_size, _ct.byref(self._handle)))

    def _result(self, function, *arguments):
        output = _L()
        with self._lock:
            _check(function(self._live(), *arguments, _ct.byref(output)))
        return int(output.value)

    @property
    def alive_count(self):
        return self._result(_opponent_alive)

    def entity_at(self, rank):
        return self._result(_opponent_entity, _integer(rank, "rank"))

    def opponent_count(self, entity_index, current_score):
        return self._result(_opponent_count, _integer(entity_index, "entity index"), current_score)

    def opponent_at(self, entity_index, current_score, rank):
        return self._result(_opponent_at, _integer(entity_index, "entity index"),
                            current_score, _integer(rank, "rank"))

    def remove(self, entity_index):
        entity_index = _integer(entity_index, "entity index")
        with self._lock:
            _check(_opponent_remove(self._live(), entity_index))
