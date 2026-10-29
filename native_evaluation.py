"""Batch only independent entity scoring; world transitions remain ordered."""
import numpy as np

from entities import Pathstrider
from models import TitanToPathModel
from native_backend import get_native


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
