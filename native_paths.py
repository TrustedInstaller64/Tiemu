"""Conservative batch access to the built-in entities' lazy path caches."""
import numpy as np

from constants import PATH_NAMES
from entities import Pathstrider
from models import TitanToPathModel

_PATH_PROPERTY = Pathstrider.path_affinities
_DOMINANT_PROPERTY = Pathstrider.dominant_path_idx
_MAP_PATHS = TitanToPathModel.get_path_affinities


def _native_dominant_paths(population, native):
    """Batch ordinary lazy caches; unsupported objects keep the original getters."""
    if (type(population) is not list or not population
            or Pathstrider.path_affinities is not _PATH_PROPERTY
            or Pathstrider.dominant_path_idx is not _DOMINANT_PROPERTY
            or TitanToPathModel.get_path_affinities is not _MAP_PATHS
            or Pathstrider.__getattribute__ is not object.__getattribute__
            or Pathstrider.__setattr__ is not object.__setattr__
            or TitanToPathModel.__getattribute__ is not object.__getattribute__):
        return None
    matrix, missing, seen = None, [], set()
    paths = len(PATH_NAMES)
    # Finish all validation before populating any cache, so fallback has no
    # extra side effects and custom getters retain their original call order.
    for entity in population:
        if type(entity) is not Pathstrider or id(entity) in seen:
            return None
        seen.add(id(entity))
        fields = entity.__dict__
        model = fields.get('_titan_to_path_model')
        if type(model) is not TitanToPathModel or 'get_path_affinities' in model.__dict__:
            return None
        current = model.__dict__.get('titan_to_path_matrix')
        if matrix is None:
            if (type(current) is not np.ndarray or current.dtype != np.float64
                    or current.ndim != 2 or current.shape[1] != paths):
                return None
            matrix = current
        elif current is not matrix:
            return None
        if '_path_affinities' not in fields:
            return None
        cached = fields['_path_affinities']
        if cached is None:
            titans = fields.get('titan_affinities')
            if (type(titans) is not np.ndarray or titans.dtype != np.float64
                    or titans.shape != (matrix.shape[0],)):
                return None
            missing.append(entity)
        elif (type(cached) is not np.ndarray or cached.dtype != np.float64
              or cached.shape != (paths,)):
            return None
    if missing:
        mapped = native.path_affinities(np.stack([p.titan_affinities for p in missing]), matrix)
        for entity, row in zip(missing, mapped):
            entity._path_affinities = row.copy()
    return np.argmax(np.stack([p._path_affinities for p in population]), axis=1).astype(np.int64)
