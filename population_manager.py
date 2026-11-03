import numpy as np
from contextlib import contextmanager
from native_backend import get_native
from native_paths import _native_dominant_paths
import random
import threading
from collections import Counter

from constants import GREEK_ROOTS, TITAN_NAMES, PATH_NAMES
from entities import Pathstrider
from models import TitanToPathModel
from native_world import ResidentWorld

import logging 
logger = logging.getLogger("OmphalosLogger") 


def _birth_array(value, shape):
    # This conservative range excludes overflow-sensitive exception/warning
    # paths before any names or random draws have advanced. It is many orders
    # above the normal configured world ranges, while keeping legacy dot,
    # purification and norm intermediates safely within float64.
    return (type(value) is np.ndarray and value.dtype == np.float64
            and value.shape == shape and bool(np.isfinite(value).all())
            and bool((np.abs(value) <= 1e30).all()))


def _native_births_supported(manager, population, count, blueprint, modifier):
    if (get_native() is None or type(manager) is not PopulationManager
            or type(population) is not list or type(count) is not int or count <= 0
            or threading.active_count() != 1
            or np.geterr() != {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}):
        return False
    # Reject class hooks before reading any instance fields. A getter may itself
    # consume randomness, so discovering it by calling it would already change
    # the reference loop's failure/callback prefix.
    if (Pathstrider.__new__ is not _BIRTH_ENTITY_NEW
            or Pathstrider.__init__ is not _BIRTH_ENTITY_INIT
            or Pathstrider.internal_purification is not _BIRTH_PURIFICATION
            or Pathstrider.__getattribute__ is not object.__getattribute__
            or Pathstrider.__setattr__ is not object.__setattr__
            or any(name in Pathstrider.__dict__ for name in (
                'name', '_titan_to_path_model', 'titan_affinities', 'trait',
                'golden_one_tenure', 'heroic_tendency', '_path_affinities', 'score',
                'hp', 'is_titan_boss', 'titan_aspect', 'data_modification_unlocked',
                'activity', 'stability', 'is_titan_form', 'held_fire_seeds'))
            or PopulationManager.__getattribute__ is not object.__getattribute__
            or PopulationManager.__setattr__ is not object.__setattr__
            or any(name in PopulationManager.__dict__ for name in (
                'existing_names', 'name_to_entity_map', 'base_titan_affinities',
                'mutation_rate', 'purity_factor', 'max_affinity_norm',
                'cosmic_tide_vector', 'titan_to_path_model', '_native_world'))
            or TitanToPathModel.__getattribute__ is not object.__getattribute__
            or TitanToPathModel.__setattr__ is not object.__setattr__
            or any(name in TitanToPathModel.__dict__ for name in (
                'titan_to_path_matrix', 'path_to_titan_feedback_matrix'))
            or any(getattr(random.Random, name) is not original
                   for name, original in _BIRTH_RANDOM_METHODS.items())):
        return False
    if any(name in manager.__dict__ or getattr(PopulationManager, name) is not original
           for name, original in _BIRTH_MANAGER_METHODS.items()):
        return False
    if any(getattr(owner, name) is not original for owner, name, original in _BIRTH_RNG):
        return False
    python_owner, numpy_owner = random._inst, np.random.mtrand._rand
    if (type(python_owner) is not random.Random or type(numpy_owner) is not np.random.RandomState
            or any(name in python_owner.__dict__ for name in ('sample', 'randint', 'random',
                                                              'getrandbits', '_randbelow', 'randrange'))):
        return False
    if any(getattr(original, '__self__', None) is not
           (python_owner if owner is random else numpy_owner)
           for owner, _, original in _BIRTH_RNG):
        return False
    if (type(manager.existing_names) is not set or type(manager.name_to_entity_map) is not dict
            or tuple(GREEK_ROOTS) != _BIRTH_ROOTS
            or type(manager.titan_to_path_model) is not TitanToPathModel):
        return False
    if not all(type(value) in (int, float, np.float64) and np.isfinite(value)
               for value in (manager.mutation_rate, manager.purity_factor, manager.max_affinity_norm)):
        return False
    if not (0 <= manager.mutation_rate <= 1e30 and 0 < manager.purity_factor <= 1e10
            and 0 < manager.max_affinity_norm <= 1e100):
        return False
    if not (_birth_array(blueprint, (12,))
            and _birth_array(manager.base_titan_affinities, (12,))
            and _birth_array(manager.cosmic_tide_vector, (12,))):
        return False
    if modifier is not None and not (
            _birth_array(modifier, (12,)) and _birth_array(
                manager.titan_to_path_model.path_to_titan_feedback_matrix, (12, 12))):
        return False
    return True

class PopulationManager:
    def __init__(self, existing_names: set, name_to_entity_map: dict, 
                 base_titan_affinities: np.ndarray, mutation_rate: float, 
                 purity_factor: float, golden_one_cap: int, 
                 golden_one_reversion_prob: float, titan_to_path_model: TitanToPathModel,
                 max_affinity_norm: float, min_norm: float, max_norm: float):
        
        self.existing_names = existing_names
        self.name_to_entity_map = name_to_entity_map
        self.base_titan_affinities = base_titan_affinities
        self.mutation_rate = mutation_rate
        self.purity_factor = purity_factor
        self.golden_one_cap = golden_one_cap
        self.golden_one_reversion_prob = golden_one_reversion_prob
        self.titan_to_path_model = titan_to_path_model
        self.max_affinity_norm = max_affinity_norm
        self.min_norm = min_norm
        self.max_norm = max_norm
        self._tracked_population = None
        self._path_histogram = None
        self._tracked_indices = {}
        self._dirty_paths = set()
        self.cosmic_tide_vector = np.zeros(len(TITAN_NAMES)) 
        self._native_world = None

    def get_native_world(self):
        if get_native() is None:
            return None
        if self._native_world is None:
            self._native_world = ResidentWorld()
        return self._native_world

    def close_native_world(self):
        world, self._native_world = self._native_world, None
        if world is not None:
            world.close()

    def generate_unique_name(self):
        while True:
            r1, r2 = random.sample(GREEK_ROOTS, 2)
            num = random.randint(1000, 9999)
            name = f"{r1}{r2}-{num}"
            if name not in self.existing_names:
                self.existing_names.add(name)
                return name

    def create_initial_population(self, num_entities: int, population: list, 
                                  cosmic_zeitgeist: np.ndarray):
        for _ in range(num_entities):
            name = self.generate_unique_name()
            entity = Pathstrider(name, self.base_titan_affinities + np.random.normal(0, self.mutation_rate, self.base_titan_affinities.shape), titan_to_path_model=self.titan_to_path_model)
            self.recalculate_and_normalize_entity(entity, None, None) 
            population.append(entity)
            self.name_to_entity_map[name] = entity
        
    def recalculate_and_normalize_entity(self, entity: Pathstrider, path_distribution: np.ndarray, cosmic_zeitgeist: np.ndarray):
        native = get_native()
        if native is not None and not self.purity_factor <= 0 and len(entity.titan_affinities) >= 2:
            entity.titan_affinities = native.normalize_pipeline(
                entity.titan_affinities, self.purity_factor,
                self.cosmic_tide_vector, self.max_affinity_norm)
            entity._path_affinities = None
            return
        # If purification is skipped, the original += changes any existing
        # aliases of the input array and leaves the path cache intact. Retain
        # that branch rather than replacing it with a new fused output array.
        entity.internal_purification(self.purity_factor)
        entity.titan_affinities += self.cosmic_tide_vector
        entity.titan_affinities = entity.titan_affinities.clip(min=0)
        self.normalize_affinities(entity)

    def normalize_affinities(self, entity_or_blueprint):
        target = entity_or_blueprint.titan_affinities if isinstance(entity_or_blueprint, Pathstrider) else entity_or_blueprint
        native = get_native()
        if native is not None:
            result = native.normalize(target, self.max_affinity_norm)
            if isinstance(entity_or_blueprint, Pathstrider):
                entity_or_blueprint.titan_affinities = result
            else:
                entity_or_blueprint[:] = result
            return None
        norm = np.linalg.norm(target)
        
        if not np.isfinite(norm) or norm == 0 or norm > self.max_affinity_norm:
            normalized_target = (target / (norm + 1e-9)) * self.max_affinity_norm
            if isinstance(entity_or_blueprint, Pathstrider):
                entity_or_blueprint.titan_affinities = normalized_target
            else: 
                entity_or_blueprint[:] = normalized_target
        return None 

    @contextmanager
    def track_distribution(self, population):
        """Cache only within one encounter loop; retain original lazy path evaluation."""
        if get_native() is None:
            yield
            return
        self._tracked_population = population
        self._path_histogram = None
        self._tracked_indices = {entity: i for i, entity in enumerate(population)}
        self._dirty_paths = set()
        try:
            yield
        finally:
            self._tracked_population = None
            self._path_histogram = None
            self._tracked_indices = {}
            self._dirty_paths.clear()

    def mark_paths_dirty(self, *entities):
        if self._tracked_population is not None:
            self._dirty_paths.update(p for p in entities if p in self._tracked_indices)

    def get_global_path_distribution(self, population: list):
        if not population:
            return np.ones(len(PATH_NAMES)) / len(PATH_NAMES)
        if population is self._tracked_population:
            native = get_native()
            if self._path_histogram is None:
                indices = np.array([p.dominant_path_idx for p in population], dtype=np.int64)
                self._path_histogram = native.PathHistogram(indices, len(PATH_NAMES))
            else:
                for p in sorted(self._dirty_paths, key=self._tracked_indices.__getitem__):
                    self._path_histogram.update(self._tracked_indices[p], int(p.dominant_path_idx))
            self._dirty_paths.clear()
            return self._path_histogram.distribution()
        native = get_native()
        indices = _native_dominant_paths(population, native) if native is not None else None
        if indices is not None:
            histogram = native.PathHistogram(indices, len(PATH_NAMES))
            try:
                return histogram.distribution()
            finally:
                histogram.close()
        path_counts = np.zeros(len(PATH_NAMES))
        for p in population:
            path_counts[p.dominant_path_idx] += 1
        distribution = path_counts / len(population)
        return distribution

    def update_golden_ones(self, population: list):
        """
        更新黄金裔：将上一代泰坦黄金裔清除，并选举12名新的天选黄金裔。
        """
        for p in population:
            if p.trait == "GoldenOne":
                p.trait = "Mortal"
                p.titan_aspect = None
                p.data_modification_unlocked = False

        # 选举新的黄金裔
        if not population: return
        candidates = [p for p in population if not p.is_titan_boss and p.trait != "Reincarnator"]
        candidates.sort(key=lambda p: p.score, reverse=True)

        num_titan_golden_ones = min(len(candidates), 12)
        titan_aspect_names = list(TITAN_NAMES) # 复制列表分配
        random.shuffle(titan_aspect_names)

        for i in range(num_titan_golden_ones):
            entity = candidates[i]
            entity.trait = "GoldenOne"
            entity.golden_one_tenure = 0
            entity.titan_aspect = titan_aspect_names.pop()

        # 分配普通黄金裔（如果上限更高）
        num_normal_golden_ones = min(len(candidates) - 12, self.golden_one_cap - 12)
        if num_normal_golden_ones > 0:
            for i in range(num_normal_golden_ones):
                entity = candidates[12 + i]
                entity.trait = "GoldenOne"
                entity.golden_one_tenure = 0

    def check_and_replenish_population(self, population: list, population_soft_cap: int, 
                                       aeonic_cycle_mode: bool, reincarnator: Pathstrider, 
                                       cosmic_zeitgeist: np.ndarray):
        if len(population) < population_soft_cap * 0.8:
            num_to_add = int(population_soft_cap - len(population))
            if num_to_add <= 0: return
            
            if aeonic_cycle_mode: 
                blueprint = self.base_titan_affinities * (1.5 * random.uniform(0.1, 1.5))
            else:
                golden_ones = [p for p in population if p.trait == "GoldenOne"]
                if not golden_ones: return
                template_entity = random.choice(golden_ones)
                blueprint = template_entity.titan_affinities
                
            self._add_new_entities(population, num_to_add, blueprint, cosmic_zeitgeist)

    def replenish_population_by_growth(self, population: list, num_to_add: int, cosmic_zeitgeist: np.ndarray, legacy_modifier:np.ndarray = None):
        """Replenish based on the current blueprint from the guide network."""
        blueprint = self.base_titan_affinities
        self._add_new_entities(population, num_to_add, blueprint, cosmic_zeitgeist, legacy_modifier)

    def _add_new_entities(self, population: list, num_to_add: int, blueprint: np.ndarray, 
                          cosmic_zeitgeist: np.ndarray, legacy_modifier: np.ndarray = None): # 新增参数
        dist_newborns = self.get_global_path_distribution(population)
        try:
            native_births = _native_births_supported(self, population, num_to_add, blueprint, legacy_modifier)
        except (AttributeError, TypeError, ValueError, OverflowError):
            native_births = False
        if native_births:
            # Keep the name/RandomState draw interleave. Only side-effect-free
            # numerical work moves past subsequent draws on this guarded path.
            names, noise = [], np.empty((num_to_add, 12), dtype=np.float64)
            modifier = (None if legacy_modifier is None else
                        np.dot(legacy_modifier, self.titan_to_path_model.path_to_titan_feedback_matrix) * 50.0)
            for index in range(num_to_add):
                names.append(self.generate_unique_name())
                noise[index] = np.random.normal(0, self.mutation_rate, self.base_titan_affinities.shape)
            world = self.get_native_world()
            values = world.births(blueprint, noise, modifier, self.cosmic_tide_vector,
                                  self.purity_factor, self.max_affinity_norm)
            for name, row in zip(names, values):
                entity = Pathstrider(name, row, titan_to_path_model=self.titan_to_path_model)
                population.append(entity)
                self.name_to_entity_map[name] = entity
            # The numerical birth buffer does not create live IDs until these
            # public objects exist. Reconcile removes culls and preserves order;
            # older numerical rows require a full sync before another encounter.
            # An unsupported pre-existing entity cannot invalidate completed
            # births; discard only the unavailable resident mirror in that case.
            try:
                world.reconcile(population)
            except (ValueError, AttributeError, TypeError):
                self.close_native_world()
            return
        for _ in range(num_to_add):
            name = self.generate_unique_name()
            new_affinities = blueprint + np.random.normal(0, self.mutation_rate, self.base_titan_affinities.shape)
            entity = Pathstrider(name, new_affinities, titan_to_path_model=self.titan_to_path_model)
            
            # 应用修正
            if legacy_modifier is not None:
                # 将命途修正值转换回对泰坦亲和度的影响
                # 这是通过反馈矩阵 (path_to_titan) 来实现的，虽然不完美...Sakaye不想改了
                titan_modifier = np.dot(legacy_modifier, self.titan_to_path_model.path_to_titan_feedback_matrix)
                entity.titan_affinities += titan_modifier * 50.0 # 50.0 是一个可调的魔法数字

            self.recalculate_and_normalize_entity(entity, dist_newborns, cosmic_zeitgeist)
            population.append(entity)
            self.name_to_entity_map[name] = entity


_BIRTH_ENTITY_INIT = Pathstrider.__init__
_BIRTH_ENTITY_NEW = Pathstrider.__new__
_BIRTH_PURIFICATION = Pathstrider.internal_purification
_BIRTH_ROOTS = tuple(GREEK_ROOTS)
_BIRTH_MANAGER_METHODS = {name: getattr(PopulationManager, name) for name in (
    'generate_unique_name', 'recalculate_and_normalize_entity', 'normalize_affinities',
    'get_global_path_distribution', 'get_native_world')}
_BIRTH_RNG = [(owner, name, getattr(owner, name)) for owner, names in (
    (random, ('sample', 'randint')), (np.random, ('normal',))) for name in names]
_BIRTH_RANDOM_METHODS = {name: getattr(random.Random, name) for name in (
    '__getattribute__', '_randbelow', 'getrandbits', 'random', 'randrange')}
