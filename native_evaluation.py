"""Batch only independent entity scoring; world transitions remain ordered."""
import numpy as np

from config import config
from entities import Pathstrider
from models import TitanToPathModel
from native_backend import get_native
from parliament_manager import ParliamentManager


_C_METHODS = {
    Pathstrider: {name: getattr(Pathstrider, name) for name in
                 ('path_affinities', 'recalculate_concepts')},
    TitanToPathModel: {'get_path_affinities': TitanToPathModel.get_path_affinities},
    ParliamentManager: {'get_zeitgeist_multiplier': ParliamentManager.get_zeitgeist_multiplier},
}


def _c_scoring_supported(entities, parliament, model):
    """Restrict C1 to ordinary built-ins; custom callbacks retain ordered calls."""
    if type(parliament) is not ParliamentManager:
        return False
    for cls, methods in _C_METHODS.items():
        if (cls.__getattribute__ is not object.__getattribute__
                or cls.__setattr__ is not object.__setattr__
                or any(getattr(cls, name) is not original for name, original in methods.items())):
            return False
    for instance, cls in ((model, TitanToPathModel), (parliament, ParliamentManager)):
        if not instance.__dict__.keys().isdisjoint(_C_METHODS[cls]):
            return False
    matrix = model.titan_to_path_matrix
    if type(matrix) is not np.ndarray or matrix.dtype != np.float64:
        return False
    for entity in entities:
        if not entity.__dict__.keys().isdisjoint(_C_METHODS[Pathstrider]):
            return False
        current_model = entity._titan_to_path_model
        if not current_model.__dict__.keys().isdisjoint(_C_METHODS[TitanToPathModel]):
            return False
        values, cache = entity.titan_affinities, entity._path_affinities
        if type(values) is not np.ndarray or values.dtype != np.float64:
            return False
        if cache is not None and (type(cache) is not np.ndarray or cache.dtype != np.float64
                                  or cache.shape != (matrix.shape[1],)):
            return False
    return True


def _evaluate_c_scoring(entities, parliament, distribution, evaluator, matrix):
    # The old paths must be observed before any nonfinite titan repair. The
    # native formula uses these old caches, while mapping uses current titans.
    old_paths = np.stack([entity.path_affinities for entity in entities])
    parliament.get_zeitgeist_multiplier(old_paths[0])
    normalized_seats = parliament._normalized_seat_vector
    values = np.stack([entity.titan_affinities for entity in entities])
    # One bulk mask replaces thousands of tiny NumPy mask/reduction calls.
    for index in np.flatnonzero(~np.isfinite(values).all(axis=1)):
        entity = entities[int(index)]
        entity.titan_affinities = np.ones(len(entity.titan_affinities))
        values[index] = entity.titan_affinities
    bosses = np.array([bool(entity.is_titan_boss) for entity in entities], dtype=np.int64)
    return evaluator.concepts_zeitgeist(values, matrix, old_paths, normalized_seats,
                                        parliament.suppression_factor, distribution, bosses)


def _evaluate_scalar(entities, parliament, distribution):
    for entity in entities:
        multiplier = parliament.get_zeitgeist_multiplier(entity.path_affinities)
        entity.recalculate_concepts(multiplier, distribution)


def evaluate_entities(entities, parliament, distribution, evaluator=None):
    """Score a fixed set of entities without moving random or shared-state work.

    The multiplier observes each entity's existing lazy path cache, whereas
    scoring uses freshly mapped titan values. Keep those two reads distinct.
    An absent evaluator, custom entity/model, repeated entity, or heterogeneous
    mapping uses the original ordered scalar path.
    """
    entities = list(entities)
    if not entities:
        return
    if (evaluator is None or get_native() is None or len(entities) == 1
            or any(type(entity) is not Pathstrider for entity in entities)
            or len({id(entity) for entity in entities}) != len(entities)):
        _evaluate_scalar(entities, parliament, distribution)
        return

    model = entities[0]._titan_to_path_model
    if type(model) is not TitanToPathModel:
        _evaluate_scalar(entities, parliament, distribution)
        return
    matrix = model.titan_to_path_matrix
    if (matrix.ndim != 2 or any(
            type(entity._titan_to_path_model) is not TitanToPathModel
            or entity._titan_to_path_model.titan_to_path_matrix is not matrix
            or entity.titan_affinities.ndim != 1
            or entity.titan_affinities.size != matrix.shape[0]
            for entity in entities)):
        _evaluate_scalar(entities, parliament, distribution)
        return

    if (config.get('runtime', {}).get('experimental_c_scoring', False)
            and _c_scoring_supported(entities, parliament, model)):
        paths, stats = _evaluate_c_scoring(entities, parliament, distribution, evaluator, matrix)
    else:
        values = np.empty((len(entities), matrix.shape[0]), dtype=np.float64)
        multipliers = np.empty(len(entities), dtype=np.float64)
        bosses = np.empty(len(entities), dtype=np.int64)
        for index, entity in enumerate(entities):
            # This must precede nonfinite repair, just as in the scalar caller.
            multipliers[index] = parliament.get_zeitgeist_multiplier(entity.path_affinities)
            if not np.all(np.isfinite(entity.titan_affinities)):
                entity.titan_affinities = np.ones(len(entity.titan_affinities))
            values[index] = entity.titan_affinities
            bosses[index] = bool(entity.is_titan_boss)

        paths, stats = evaluator.concepts(values, matrix, multipliers, distribution, bosses)
    for index, entity in enumerate(entities):
        # Each entity owns its cache, as on the scalar path. A survivor must not
        # retain a whole obsolete population batch through a NumPy row view.
        entity._path_affinities = paths[index].copy()
        entity.activity = float(stats[index, 0])
        entity.stability = float(stats[index, 1])
        entity.score = float(stats[index, 3])
        entity.heroic_tendency = entity.activity
