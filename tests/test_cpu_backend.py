import numpy as np
import pytest
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


def test_deprecated_beta_does_not_change_the_q_update():
    torch = pytest.importorskip("torch")

    Q = torch.tensor(
        [[0.2, 1.0, 0.0], [0.1, 0.0, 1.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
        dtype=torch.float32,
    )
    W = np.array(
        [[1.0, 0.2, 0.4], [0.8, 0.5, 0.1], [0.7, 0.3, 0.6], [0.9, 0.4, 0.2]],
        dtype=np.float32,
    )
    X = sparse.csr_matrix(
        np.array(
            [[1.0, 0.2, 0.3, 0.4], [0.8, 0.5, 0.1, 0.2],
             [0.7, 0.3, 0.6, 0.1], [0.9, 0.4, 0.2, 0.5]],
            dtype=np.float32,
        )
    )

    outputs = []
    for beta in (0.0, 100.0):
        updated, _ = optimization.update_Q_riemannian(
            Q.clone(), W, X, beta=beta, lr=1e-3, n_steps=2, lambda_neg=1.0
        )
        outputs.append(updated.detach().cpu().numpy())

    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_allclose(
        outputs[0][:, 1:].T @ outputs[0][:, 1:], np.eye(2), atol=1e-6
    )
