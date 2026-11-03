"""Cloud-only characterization of NumPy batch primitives for A scoring.

This reports exactness, not a performance promise or permission to widen test
tolerances. No project model is imported. Production still needs differential
tests against the frozen scalar expression.
"""
import json
import platform

import numpy as np


def original(paths, seats, suppression):
    return np.array([max(0.9, min(1.2, 1.05 + 0.15 * np.tanh(
        (np.dot(row, seats) - np.dot(1 - row, seats)) * 3) * suppression))
        for row in paths], dtype=np.float64)


def candidate(paths, seats, suppression, scalar_tanh):
    alignment = np.vecdot(paths, seats) - np.vecdot(1 - paths, seats)
    scaled = alignment * 3
    wave = (np.fromiter((np.tanh(value) for value in scaled), dtype=np.float64,
                        count=len(scaled)) if scalar_tanh else np.tanh(scaled))
    # Python min/max choose the first bound when the computed value is NaN.
    return np.fmax(0.9, np.fmin(1.2, 1.05 + (0.15 * wave * suppression)))


def main():
    rng = np.random.RandomState(731)
    ordinary = rng.uniform(0, 3, (2048, 12))
    ordinary /= ordinary.sum(axis=1, keepdims=True)
    signed = rng.uniform(-3, 3, (1024, 12))
    edges = np.vstack([np.zeros(12), np.eye(12), np.full(12, 1/12),
                       np.full(12, np.nan), np.full(12, np.inf), np.full(12, -np.inf)])
    paths = np.vstack([ordinary, signed, edges]).astype(np.float64)
    summary = {'python': platform.python_version(), 'platform': platform.platform(),
               'machine': platform.machine(), 'numpy': np.__version__,
               'rows': len(paths), 'cases': [],
               'scope': 'Raw float64 bit comparisons of the proposed batch expression. '
                        'No timing and no project execution; not a proof for all inputs.'}
    with np.errstate(all='ignore'):
        for denominator in (17, 100):
            seats = np.arange(2, 14, dtype=np.int64) / denominator
            for suppression in (0.0, 0.7, 1.0, -0.4, 10.0):
                expected = original(paths, seats, suppression)
                row = {'denominator': denominator, 'suppression': suppression}
                for name, scalar in (('vecdot_scalar_tanh', True), ('vecdot_vector_tanh', False)):
                    actual = candidate(paths, seats, suppression, scalar)
                    mismatch = actual.view(np.uint64) != expected.view(np.uint64)
                    row[name] = {'mismatches': int(mismatch.sum()),
                                 'max_absolute_difference': float(np.max(np.abs(actual - expected)))}
                summary['cases'].append(row)
    print(json.dumps({'scoring_math_probe': summary}), flush=True)


if __name__ == '__main__':
    main()
