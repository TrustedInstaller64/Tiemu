# parliament_manager.py
import numpy as np
from native_backend import get_native
from constants import PATH_NAMES, PATH_RELATIONSHIP_MATRIX
from entities import Pathstrider
from models import TitanToPathModel

import logging 
logger = logging.getLogger("OmphalosLogger") 

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


class ParliamentManager:
    def __init__(self):
        self.seats = {name: 0 for name in PATH_NAMES}
        self.total_seats = 100
        self.suppression_factor = 1.0 # 默认为1.0，即无抑制
        self._seat_cache_key = None
        self._normalized_seat_vector = None

    def hold_election(self, population: list):
        """根据所有实体的投票，分配议会席位。"""
        if not population:
            return

        native = get_native()
        if native is None:
            total_votes = np.zeros(len(PATH_NAMES))
            for entity in population:
                vote_proposal = entity.generate_vote_proposal()
                vote_weight = entity.get_vote_weight() * entity.score
                total_votes += vote_proposal * vote_weight
        else:
            relationship_scores = {"SYNERGY": 0.5, "MENTORSHIP": 0.2,
                                   "REPULSION": -0.7, "CLASH": -1.0}
            logits = np.array([[relationship_scores.get(rel, 0.0) for rel in row]
                               for row in PATH_RELATIONSHIP_MATRIX], dtype=np.float64)
            np.fill_diagonal(logits, 1.0)
            dominant_paths = _native_dominant_paths(population, native)
            if dominant_paths is None:
                dominant_paths = np.array([p.dominant_path_idx for p in population], dtype=np.int64)
            total_votes = native.vote_totals(
                dominant_paths,
                np.array([p.get_vote_weight() * p.score for p in population], dtype=np.float64),
                logits)

        if np.sum(total_votes) == 0:
            # 如果没有有效投票，平均分配
            seats_per_path = self.total_seats // len(PATH_NAMES)
            for name in PATH_NAMES:
                self.seats[name] = seats_per_path
            return

        # 根据票数比例分配席位
        vote_distribution = total_votes / np.sum(total_votes)
        allocated_seats = (vote_distribution * self.total_seats).astype(int)

        # 处理取整问题
        remainder = self.total_seats - np.sum(allocated_seats)
        if remainder > 0:
            top_indices = np.argsort(vote_distribution)[-remainder:]
            allocated_seats[top_indices] += 1
        
        for i, name in enumerate(PATH_NAMES):
            self.seats[name] = allocated_seats[i]
        
        logger.info("\n\033[34m--- 公民议会换届 ---")
        dominant_party = max(self.seats, key=self.seats.get)
        logger.info(f"本世代议会已组成，命途主导 '{dominant_party}' (席位: {self.seats[dominant_party]}/{self.total_seats})")
        logger.info("--------------------------\033[0m")
        # 重置抑制因子，除非干预仍在持续
        if self.suppression_factor != 1.0:
             pass
    
    def get_zeitgeist_multiplier(self, entity_path_affinities: np.ndarray) -> float:
        """根据实体命途与当前议会格局，计算思潮乘数。"""
        # seats is a public mutable dict. Include both its complete contents
        # and total_seats so direct writes/replacement cannot stale this cache.
        key = (tuple(self.seats.items()), self.total_seats)
        if key != self._seat_cache_key:
            if np.sum(list(self.seats.values())) == 0:
                normalized = None
            else:
                seat_vector = np.array([self.seats[name] for name in PATH_NAMES])
                normalized = seat_vector / self.total_seats
            self._normalized_seat_vector = normalized
            self._seat_cache_key = key
        if self._normalized_seat_vector is None:
            return 1.0
        normalized_seat_vector = self._normalized_seat_vector
        alignment_score = np.dot(entity_path_affinities, normalized_seat_vector) - np.dot(1 - entity_path_affinities, normalized_seat_vector)
        
        # 将思潮带来的影响乘以抑制因子
        multiplier = 1.05 + (0.15 * np.tanh(alignment_score * 3) * self.suppression_factor)
        
        return max(0.9, min(1.2, multiplier))
