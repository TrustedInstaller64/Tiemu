"""Pure-data comparison of bounded C trajectories; never executes a world.

Exact drift, tolerance exceedance, decision checkpoints and distribution drift
are separate measures. Entity names can be reused after death: pointwise entity
errors are intentionally unavailable after the first discrete/RNG divergence.
"""
from bisect import bisect_right
import math


NUMERIC_FIELDS = ('score', 'hp', 'titan_affinities', 'cached_paths', 'activity',
                  'stability', 'heroic_tendency')
DISCRETE_FIELDS = ('trait', 'golden_one_tenure', 'is_titan_boss', 'titan_aspect',
                   'is_titan_form', 'held_fire_seeds', 'data_modification_unlocked')
NONFINITE = {'NaN': float('nan'), '+Infinity': float('inf'), '-Infinity': float('-inf')}


def _flatten(value):
    if isinstance(value, (tuple, list)):
        return [item for part in value for item in _flatten(part)]
    if isinstance(value, dict):
        return [item for key in sorted(value) for item in _flatten(value[key])]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, str) and value in NONFINITE:
        return [NONFINITE[value]]
    return []


def numeric_error(reference, candidate, rtol=1e-10, atol=1e-10):
    left, right = _flatten(reference), _flatten(candidate)
    finite_pairs, nonfinite_pairs, differing = [], 0, 0
    outside = 0
    for a, b in zip(left, right):
        if not math.isfinite(a) or not math.isfinite(b):
            nonfinite_pairs += 1
            # An equal signed infinity is identical; NaNs never assert identity.
            differing += a != b
            outside += a != b
        else:
            error = abs(a-b)
            finite_pairs.append((a, error))
            differing += error != 0
            outside += error > atol+rtol*abs(a)
    maximum = max((error for _, error in finite_pairs), default=None)
    squares = math.fsum(error*error for _, error in finite_pairs)
    reference_squares = math.fsum(a*a for a, _ in finite_pairs)
    return {'reference_count': len(left), 'candidate_count': len(right),
            'length_equal': len(left) == len(right),
            'finite_pair_count': len(finite_pairs), 'nonfinite_pair_count': nonfinite_pairs,
            'different_element_count': differing+abs(len(left)-len(right)),
            'outside_reference_tolerance': outside+abs(len(left)-len(right)),
            'max_absolute_error': maximum,
            'mean_absolute_error': (math.fsum(e for _, e in finite_pairs)/len(finite_pairs)
                                    if finite_pairs else None),
            'rmse': math.sqrt(squares/len(finite_pairs)) if finite_pairs else None,
            'relative_l2_error': (math.sqrt(squares/reference_squares) if reference_squares
                                  else 0.0 if finite_pairs and not squares else None),
            'max_scaled_relative_error': max((e/max(abs(a), 1e-12) for a, e in finite_pairs),
                                             default=None),
            'rtol': rtol, 'atol': atol, 'relative_denominator_floor': 1e-12}


def _quantile(values, fraction):
    if not values:
        return None
    point = (len(values)-1)*fraction
    low, high = math.floor(point), math.ceil(point)
    return values[low]+(values[high]-values[low])*(point-low)


def _stats(values):
    finite = sorted(value for value in values if math.isfinite(value))
    mean = math.fsum(finite)/len(finite) if finite else None
    return {'count': len(values), 'finite_count': len(finite),
            'nonfinite_count': len(values)-len(finite), 'mean': mean,
            'std': math.sqrt(math.fsum((x-mean)**2 for x in finite)/len(finite)) if finite else None,
            'q05': _quantile(finite, 0.05), 'q50': _quantile(finite, 0.5),
            'q95': _quantile(finite, 0.95)}, finite


def score_distribution(reference, candidate):
    a_stats, a = _stats(_flatten(reference))
    b_stats, b = _stats(_flatten(candidate))
    ks = wasserstein = None
    if a and b:
        points = sorted(set(a+b))
        ks, wasserstein = 0.0, 0.0
        previous, previous_difference = points[0], 0.0
        for point in points:
            wasserstein += abs(previous_difference)*(point-previous)
            difference = bisect_right(a, point)/len(a)-bisect_right(b, point)/len(b)
            ks = max(ks, abs(difference))
            previous, previous_difference = point, difference
    return {'reference': a_stats, 'candidate': b_stats, 'ks_distance': ks,
            'wasserstein_1': wasserstein,
            'finite_values_only': True, 'interpretation': 'Descriptive distances, no significance test.'}


def _population(state):
    rows = state['population']
    result = {row['name']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate entity name in snapshot')
    return result


def _cache_signature(population):
    return {name: None if row.get('cached_paths') is None else len(row['cached_paths'])
            for name, row in population.items()}


def _flags(left, right, a, b):
    def reincarnator_identity(state):
        entity = state.get('reincarnator')
        return None if entity is None else [state.get('reincarnator_population_index'),
                                           entity.get('name'),
                                           *[entity.get(field) for field in DISCRETE_FIELDS]]
    return {
        'population_order': [p['name'] for p in left['population']] !=
                            [p['name'] for p in right['population']],
        'membership': set(a) != set(b),
        'seats': left['seats'] != right['seats'],
        'culls': left.get('culled_names', []) != right.get('culled_names', []),
        'traits': {name: [row.get(field) for field in DISCRETE_FIELDS] for name, row in a.items()} !=
                  {name: [row.get(field) for field in DISCRETE_FIELDS] for name, row in b.items()},
        'golden': {name: row.get('titan_aspect') for name, row in a.items() if row.get('trait') == 'GoldenOne'} !=
                  {name: row.get('titan_aspect') for name, row in b.items() if row.get('trait') == 'GoldenOne'},
        'rng': left['rng'] != right['rng'],
        'networks': left['networks'] != right['networks'],
        'cache': _cache_signature(a) != _cache_signature(b),
        'reincarnator': reincarnator_identity(left) != reincarnator_identity(right),
        'reporting_path': any(a[name].get('reporting_dominant_idx') !=
                              b[name].get('reporting_dominant_idx') for name in a if name in b),
    }


def _checkpoint(left, right, individual_valid):
    a, b = _population(left), _population(right)
    common = [name for name in a if name in b]
    flags = _flags(left, right, a, b)
    discrete = any(flags[key] for key in ('population_order', 'membership', 'seats', 'culls',
                                          'traits', 'golden', 'reincarnator'))
    valid = individual_valid and not discrete and not flags['rng']
    counts_a, counts_b = left['dominant_path_counts'], right['dominant_path_counts']
    if len(counts_a) != len(counts_b):
        raise ValueError('snapshot path histogram dimensions differ')
    sum_a, sum_b = sum(counts_a), sum(counts_b)
    tv = (sum(abs(x/sum_a-y/sum_b) for x, y in zip(counts_a, counts_b))/2
          if sum_a and sum_b else 0.0 if not sum_a and not sum_b else None)
    numeric = ({field: numeric_error([a[name].get(field) for name in common],
                                    [b[name].get(field) for name in common])
                for field in NUMERIC_FIELDS} if valid else None)
    if valid and (left.get('reincarnator') is not None or right.get('reincarnator') is not None):
        numeric['reincarnator'] = numeric_error(left.get('reincarnator'), right.get('reincarnator'))
    world_numeric = {field: numeric_error(left.get(field), right.get(field))
                     for field in ('blueprint', 'zeitgeist', 'managers')}
    seats = set(left['seats']) | set(right['seats'])
    return {'population_reference': len(a), 'population_candidate': len(b),
            'population_count_delta': len(b)-len(a), 'common_name_count': len(common),
            'name_jaccard': len(common)/len(set(a)|set(b)) if a or b else 1.0,
            'common_name_reference_fraction': len(common)/len(a) if a else None,
            'common_name_candidate_fraction': len(common)/len(b) if b else None,
            'individual_comparison_valid': valid,
            'numeric': numeric, 'world_numeric': world_numeric,
            'path_total_variation': tv,
            'unknown_paths_reference': left.get('unknown_path_count', 0),
            'unknown_paths_candidate': right.get('unknown_path_count', 0),
            'seat_l1': sum(abs(left['seats'].get(key, 0)-right['seats'].get(key, 0)) for key in seats),
            'score_distribution': score_distribution([row['score'] for row in a.values()],
                                                      [row['score'] for row in b.values()]),
            'differences': flags, 'discrete_difference': discrete}


def compare_runs(reference, candidate):
    left, right = reference['snapshots'], candidate['snapshots']
    if not left or len(left) != len(right):
        raise ValueError('observed snapshot counts must be equal and nonzero')
    # Checkpoints are part of the contract, not a zip-to-shortest convenience.
    if [(s['step'], s['stage']) for s in left] != [(s['step'], s['stage']) for s in right]:
        raise ValueError('snapshot checkpoint locations differ')
    for key in ('seed', 'population', 'generations', 'reincarnator'):
        if reference['meta'][key] != candidate['meta'][key]:
            raise ValueError(f'incompatible experiment setting: {key}')
    first = {key: None for key in ('population_order', 'membership', 'seats', 'culls', 'traits',
                                   'golden', 'rng', 'networks', 'cache', 'reincarnator', 'reporting_path')}
    first_numeric = first_above_tolerance = first_discrete = None
    stopped_at = None
    individual_valid = True
    rows = []
    for before, after in zip(left, right):
        location = {'step': before['step'], 'stage': before['stage']}
        row = _checkpoint(before['state'], after['state'], individual_valid)
        individual_valid = row['individual_comparison_valid']
        if not individual_valid and stopped_at is None:
            stopped_at = location
        for key, changed in row['differences'].items():
            if changed and first[key] is None:
                first[key] = location
        if row['discrete_difference'] and first_discrete is None:
            first_discrete = location
        errors = [*(row['numeric'] or {}).values(), *row['world_numeric'].values()]
        if first_numeric is None and any(item['different_element_count'] for item in errors):
            first_numeric = location
        if first_above_tolerance is None and any(item['outside_reference_tolerance'] for item in errors):
            first_above_tolerance = location
        rows.append({**location, **row})
    final = _checkpoint(reference['final_state'], candidate['final_state'], individual_valid)
    a_networks, b_networks = reference.get('final_network_values', {}), candidate.get('final_network_values', {})
    network_errors = {name: numeric_error(a_networks.get(name, []), b_networks.get(name, []),
                                         rtol=1e-6, atol=1e-6)
                      for name in sorted(set(a_networks)|set(b_networks))}
    return {'first_numeric_difference': first_numeric,
            'first_numeric_above_reference_tolerance': first_above_tolerance,
            'first_discrete_difference': first_discrete,
            'individual_numeric_comparison_stopped_at': stopped_at,
            'individual_first_numeric_location_censored': stopped_at is not None,
            'first_differences': first, 'checkpoints': rows, 'final': final,
            'final_network_error': network_errors,
            'scope': 'First differences are first observed checkpoints, not first causal interactions. '
                     'Population order, membership, seats, culls and entity flags are exact decisions; '
                     'RNG, network hashes and observer-derived dominant paths are reported separately. '
                     'Pointwise entity numbers cease '
                     'after discrete/RNG divergence because names may be reused after death. Common '
                     'names after that point measure overlap, not lineage. World/network numerical '
                     'distances and empirical population distributions remain descriptive. Unknown/nonfinite '
                     'values are counted. A null first numeric checkpoint with censoring does not mean '
                     'no numerical drift; exact first entity drift is unavailable from the censoring point. '
                     'Flattened numerical comparisons assume the fixed schema/network architecture of this '
                     'experiment. No equivalence or distribution-generalization claim beyond samples.'}
