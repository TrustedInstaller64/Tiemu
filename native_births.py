"""C2 batch newborn initialization with ordered Python naming and RNG calls.

Only independent numeric rows run on C++ workers. Publication of new entities
is deferred until the synchronous batch returns; callers must not concurrently
mutate/read birth-in-progress containers or input arrays. No global RNG state
is snapshotted or restored, and there is no second execution after RNG begins.
"""
import random
import sys

import numpy as np

from config import config
from constants import GREEK_ROOTS, TITAN_NAMES
from entities import Pathstrider
from models import TitanToPathModel
from native_backend import get_native


_ROOTS = tuple(GREEK_ROOTS)
_TITANS = tuple(TITAN_NAMES)
_ENTITY_METHODS = {name: Pathstrider.__dict__[name] for name in
                   ('__init__', '_initialize_lifecycle', '_from_native_affinities',
                    'internal_purification')}
_RNG = [(owner, name, getattr(owner, name)) for owner, names in
        ((random, ('sample', 'randint')), (np.random, ('normal',))) for name in names]
_NUMPY_CALLBACKS = {name: getattr(np, name) for name in ('array', 'dot')}
_MANAGER_CLASS = None
_MANAGER_METHODS = None
_MANAGER_MODULE = None
_FLOAT64 = np.dtype(np.float64)
_PLAIN_NUMBERS = (int, float, np.int64, np.float64)


def _register_population_manager(cls):
    global _MANAGER_CLASS, _MANAGER_METHODS, _MANAGER_MODULE
    if _MANAGER_CLASS is not None:
        raise RuntimeError('The native birth manager has already been registered')
    _MANAGER_CLASS = cls
    _MANAGER_MODULE = sys.modules[cls.__module__]
    _MANAGER_METHODS = {name: getattr(cls, name) for name in
                        ('generate_unique_name', 'recalculate_and_normalize_entity',
                         'normalize_affinities')}


def _array(value, shape):
    return (type(value) is np.ndarray and value.dtype == _FLOAT64 and value.shape == shape
            and bool(np.isfinite(value).all()))


def _supported(manager, population, count, blueprint, legacy):
    if (type(manager) is not _MANAGER_CLASS or type(population) is not list
            or type(count) not in (int, np.int64) or count <= 0
            or type(manager.existing_names) is not set
            or type(manager.name_to_entity_map) is not dict
            or not callable(manager._native_evaluator_factory)):
        return False
    if (_MANAGER_CLASS.__getattribute__ is not object.__getattribute__
            or _MANAGER_CLASS.__setattr__ is not object.__setattr__
            or Pathstrider.__getattribute__ is not object.__getattribute__
            or Pathstrider.__setattr__ is not object.__setattr__
            or Pathstrider.__new__ is not object.__new__
            or TitanToPathModel.__getattribute__ is not object.__getattribute__):
        return False
    if (getattr(_MANAGER_MODULE, 'Pathstrider') is not Pathstrider
            or getattr(_MANAGER_MODULE, 'np') is not np
            or getattr(_MANAGER_MODULE, 'random') is not random
            or getattr(_MANAGER_MODULE, 'GREEK_ROOTS') is not GREEK_ROOTS
            or tuple(GREEK_ROOTS) != _ROOTS or tuple(TITAN_NAMES) != _TITANS
            or any(type(root) is not str for root in GREEK_ROOTS)):
        return False
    if any(getattr(_MANAGER_CLASS, name) is not original or name in manager.__dict__
           for name, original in _MANAGER_METHODS.items()):
        return False
    if any(Pathstrider.__dict__.get(name) is not original
           for name, original in _ENTITY_METHODS.items()):
        return False
    if any(getattr(owner, name) is not original for owner, name, original in _RNG):
        return False
    if any(getattr(np, name) is not original for name, original in _NUMPY_CALLBACKS.items()):
        return False
    python_owner, numpy_owner = getattr(random, '_inst', None), np.random.mtrand._rand
    if type(python_owner) is not random.Random or type(numpy_owner) is not np.random.RandomState:
        return False
    if any(getattr(original, '__self__', None) is not
           (python_owner if owner is random else numpy_owner) for owner, _, original in _RNG):
        return False
    if (any(type(name) is not str for name in manager.existing_names)
            or any(type(name) is not str for name in manager.name_to_entity_map)):
        return False
    if (not _array(blueprint, (12,)) or not _array(manager.base_titan_affinities, (12,))
            or not _array(manager.cosmic_tide_vector, (12,))):
        return False
    if any(type(value) not in _PLAIN_NUMBERS or not np.isfinite(value) for value in
           (manager.mutation_rate, manager.purity_factor, manager.max_affinity_norm)):
        return False
    if manager.mutation_rate < 0 or manager.max_affinity_norm <= 0:
        return False
    if type(manager.titan_to_path_model) is not TitanToPathModel:
        return False
    if legacy is not None and (not _array(legacy, (12,)) or not _array(
            manager.titan_to_path_model.path_to_titan_feedback_matrix, (12, 12))):
        return False
    return True


def try_create_newborns(manager, population, count, blueprint, legacy_modifier=None):
    """Return False only before names/RNG change; committed batch failures raise."""
    if not config.get('runtime', {}).get('experimental_c_newborns', False):
        return False
    native = get_native()
    if native is None:
        return False
    try:
        if not _supported(manager, population, count, blueprint, legacy_modifier):
            return False
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    modifier = None
    if legacy_modifier is not None:
        # Same NumPy reduction as the reference, hoisted only for fixed built-in
        # arrays. Do not replace it with a differently rounded C++ dot product.
        try:
            modifier = np.dot(legacy_modifier,
                              manager.titan_to_path_model.path_to_titan_feedback_matrix)
        except (ValueError, TypeError, FloatingPointError):
            return False
        if not np.isfinite(modifier).all():
            return False
    evaluator = manager._native_evaluator_factory()
    if type(evaluator) is not native.BatchEvaluator:
        return False
    if evaluator._closed:
        raise RuntimeError('batch evaluator is closed')
    noise = np.empty((count, 12), dtype=np.float64)
    names = []
    for index in range(count):
        names.append(manager.generate_unique_name())
        noise[index] = np.random.normal(0, manager.mutation_rate, manager.base_titan_affinities.shape)
    values = evaluator.newborn_affinities(blueprint, noise, modifier, manager.purity_factor,
                                         manager.cosmic_tide_vector, manager.max_affinity_norm)
    for name, row in zip(names, values):
        entity = Pathstrider._from_native_affinities(name, row, manager.titan_to_path_model)
        population.append(entity)
        manager.name_to_entity_map[name] = entity
    return True
