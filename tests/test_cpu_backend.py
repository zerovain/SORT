import numpy as np
from scipy import sparse

from sort.core import optimization


def test_numpy_w_update_when_cupy_is_unavailable(monkeypatch):
    monkeypatch.setattr(optimization, "CUPY_AVAILABLE", False)
    monkeypatch.setattr(optimization, "cp", None)
    monkeypatch.setattr(optimization, "cupy_sparse", None)

    W = np.full((4, 2), 0.5, dtype=np.float32)
    X = sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(4, 3) + 1)
    Q = np.array([[1.0, -0.2], [0.4, 0.7], [0.2, 0.5]], dtype=np.float32)
    L = sparse.eye(4, dtype=np.float32, format="csr")

    updated = optimization.update_W_multiplicative(
        W, X, Q, L=L, alpha=0.3, backend="numpy"
    )
    assert updated.shape == W.shape
    assert np.isfinite(updated).all()
    assert np.all(updated >= 0)
