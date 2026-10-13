import numpy as np
import pytest
from sort.core import initializers


def test_seeded_svd_uses_explicit_start_without_dense_fallback(monkeypatch):
    original = initializers.svds
    starts = []

    def checked_svd(matrix, **kwargs):
        assert 'random_state' not in kwargs and 'rng' not in kwargs
        starts.append(kwargs['v0'].copy())
        return original(matrix, **kwargs)

    def reject_dense_fallback(*args, **kwargs):
        pytest.fail('Seeded initialization unexpectedly fell back to dense SVD')

    monkeypatch.setattr(initializers, 'svds', checked_svd)
    monkeypatch.setattr(np.linalg, 'svd', reject_dense_fallback)
    matrix = np.random.RandomState(8).uniform(0.1, 2.0, (24, 9)).astype(np.float32)
    fits = [initializers.NNSVDInitializer(random_state=42, backend='numpy').initialize(matrix, 3)
            for _ in range(2)]
    assert len(starts) == 2
    expected = np.random.RandomState(42).standard_normal(9)
    for start in starts:
        np.testing.assert_array_equal(start, expected)
    for first, second in zip(fits[0], fits[1]):
        np.testing.assert_array_equal(first, second)


def test_dense_fallback_is_reported(monkeypatch):
    def rejected_svd(*args, **kwargs):
        raise ValueError('test solver failure')

    monkeypatch.setattr(initializers, 'svds', rejected_svd)
    matrix = np.random.RandomState(8).uniform(0.1, 2.0, (24, 9)).astype(np.float32)
    with pytest.warns(RuntimeWarning, match='using dense SVD'):
        w, h = initializers.NNSVDInitializer(random_state=42, backend='numpy').initialize(matrix, 3)
    assert np.isfinite(w).all() and np.isfinite(h).all()
