"""  
Core optimization algorithms for spatial Semi-NMF.  
"""  

import numpy as np  
from scipy.sparse import issparse, spmatrix  
from typing import Optional, Union, Tuple  

try:  
    import cupy as cp  
    import cupyx.scipy.sparse as cupy_sparse  
    CUPY_AVAILABLE = True  
except ImportError:  
    cp = None
    cupy_sparse = None
    CUPY_AVAILABLE = False  

try:  
    import torch  
    import torch.nn.functional as F  
    TORCH_AVAILABLE = True  
except ImportError:  
    TORCH_AVAILABLE = False  


# ============================================================================  
# Multiplicative W update with numerical guards.
# ============================================================================  

def compute_positive_negative_parts(M):  
    """Decompose matrix M into positive and negative parts."""  
    if CUPY_AVAILABLE and isinstance(M, cp.ndarray):  
        xp = cp  
    else:  
        xp = np  
    M_pos = xp.maximum(M, 0)  
    M_neg = xp.maximum(-M, 0)  
    return M_pos, M_neg  


def update_W_multiplicative(  
    W: np.ndarray,  
    X: Union[np.ndarray, spmatrix],  
    Q: np.ndarray,  
    L: Optional[spmatrix] = None,  
    alpha: float = 0.0,  
    tv_manager: Optional[object] = None,  
    lambda_l1: Optional[np.ndarray] = None,  
    backend: str = 'auto',  
    data_manager: Optional[object] = None,  
) -> np.ndarray:  
    """  
    Multiplicative update for W with spatial regularization.  

    Update rule (Semi-NMF):  
        W <- W * sqrt( [XQ^+ + W(QQ^T)^-  + alpha*A*W]_ij  
                     / [XQ^- + W(QQ^T)^+  + alpha*D*W + lambda_l1]_ij )  

    NaN guard: denominator clamped to ≥ 1e-10;  
               result clamped to [0, 1e6] then NaN→0.  
    """  
    # ── backend setup ────────────────────────────────────────────────────────  
    if backend == 'auto':  
        xp = cp if (CUPY_AVAILABLE and isinstance(W, cp.ndarray)) else np  
        sparse_module = cupy_sparse if xp is cp else None  
    elif backend == 'cupy':  
        if not CUPY_AVAILABLE:  
            raise RuntimeError("CuPy not available")  
        xp = cp  
        sparse_module = cupy_sparse  
        if not isinstance(W, cp.ndarray):  
            W = cp.asarray(W, dtype=cp.float32)  
        if not isinstance(Q, cp.ndarray):  
            Q = cp.asarray(Q, dtype=cp.float32)  
    else:  
        xp = np  
        sparse_module = None  
        if CUPY_AVAILABLE and isinstance(W, cp.ndarray):  
            W = cp.asnumpy(W)  
            Q = cp.asnumpy(Q)  

    # ── float32 ──────────────────────────────────────────────────────────────  
    W = W.astype(xp.float32) if W.dtype != xp.float32 else W  
    Q = Q.astype(xp.float32) if Q.dtype != xp.float32 else Q  

    # ── Q^T, QQ^T ────────────────────────────────────────────────────────────  
    Qt        = Q.T  
    QQt       = Qt @ Q  
    Qt_pos,  Qt_neg  = compute_positive_negative_parts(Qt)  
    QQt_pos, QQt_neg = compute_positive_negative_parts(QQt)  

    # ── X @ Q ────────────────────────────────────────────────────────────────  
    from scipy.sparse import spmatrix as scipy_spmatrix  
    X_is_scipy = issparse(X) or isinstance(X, scipy_spmatrix)  
    X_is_cupy  = (CUPY_AVAILABLE and sparse_module is not None  
                  and isinstance(X, sparse_module.spmatrix))  

    if data_manager is not None and not data_manager.is_on_gpu() and xp is cp:  
        from .sparse_utils import sparse_dense_matmul_chunked_gpu  
        Q_np = cp.asnumpy(Q) if isinstance(Q, cp.ndarray) else Q  
        XQ   = sparse_dense_matmul_chunked_gpu(  
            data_manager.X_cpu,  
            cp.asarray(Q_np, dtype=cp.float32),  
            chunk_size='auto', transpose_X=False,  
            verbose=False, compute_dtype=np.float32,  
        )  
    elif X_is_cupy:  
        XQ = X @ Q  
    elif X_is_scipy:  
        XQ = X @ Q if xp is np else cupy_sparse.csr_matrix(X) @ Q  
    else:  
        if xp is cp:  
            X_gpu = X if isinstance(X, cp.ndarray) else cp.asarray(X, dtype=cp.float32)  
            XQ = X_gpu @ Q  
        else:  
            XQ = (np.asarray(X, dtype=np.float32) if not isinstance(X, np.ndarray)  
                  else X) @ Q  

    XQ_pos, XQ_neg = compute_positive_negative_parts(XQ)  

    # ── numerator / denominator ───────────────────────────────────────────────  
    numerator   = XQ_pos + W @ QQt_neg  
    denominator = XQ_neg + W @ QQt_pos + 1e-10  

    # ── spatial regularisation ────────────────────────────────────────────────  
    if alpha > 0:  
        if tv_manager is not None:  
            _, LW, D = tv_manager.get_regularization_terms(W)  
            DW = D * W  
            numerator   += alpha * (DW - LW)  
            denominator += alpha * DW  
        elif L is not None:  
            if sparse_module is not None and isinstance(L, sparse_module.spmatrix):  
                LW = L @ W  
                D  = cp.asarray(L.diagonal(), dtype=cp.float32)  
            elif issparse(L):  
                LW = L @ W  
                D  = np.asarray(L.diagonal(), dtype=np.float32)  
                if xp is cp:  
                    D = cp.asarray(D, dtype=cp.float32)  
            else:  
                LW = L @ W  
                D  = (cp.diag(L) if xp is cp else np.diag(L)).astype(xp.float32)  
            DW = D[:, None] * W  
            numerator   += alpha * (DW - LW)  
            denominator += alpha * DW  

    # ── L1 regularisation ─────────────────────────────────────────────────────  
    if lambda_l1 is not None:  
        if not isinstance(lambda_l1, xp.ndarray):  
            lambda_l1 = xp.asarray(lambda_l1, dtype=xp.float32)  
        denominator += lambda_l1[None, :]  

    # ── multiplicative step + NaN guard ──────────────────────────────────────  
    ratio   = numerator / denominator           # always ≥ 0  
    ratio   = xp.clip(ratio, 0.0, 1e6)         # prevent sqrt(Inf)  
    W_new   = W * xp.sqrt(ratio)  

    # replace NaN / Inf with the previous W value (safe fallback)  
    if xp is cp:  
        bad = cp.isnan(W_new) | cp.isinf(W_new)  
        if cp.any(bad):  
            import warnings  
            warnings.warn(  
                f"W update: {int(cp.sum(bad))} NaN/Inf entries replaced with W.",  
                RuntimeWarning, stacklevel=2,  
            )  
            W_new = cp.where(bad, W, W_new)  
    else:  
        bad = ~np.isfinite(W_new)  
        if np.any(bad):  
            import warnings  
            warnings.warn(  
                f"W update: {int(np.sum(bad))} NaN/Inf entries replaced with W.",  
                RuntimeWarning, stacklevel=2,  
            )  
            W_new = np.where(bad, W, W_new)  

    return W_new  


def update_W_hybrid_manual(XQ, Q, W, L, alpha, lambda_l1=None):
    """Manual W update for hybrid mode (precomputed XQ, Laplacian)."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("Hybrid manual update requires CuPy")

    XQ = XQ.astype(cp.float32) if XQ.dtype != cp.float32 else XQ
    Q  = Q.astype(cp.float32)  if Q.dtype  != cp.float32 else Q
    W  = W.astype(cp.float32)  if W.dtype  != cp.float32 else W

    # Q: (p, r)
    # W: (n, r)
    # XQ: (n, r)
    Qt        = Q.T
    QQt       = Qt @ Q              # (r, r)
    QQt_pos, QQt_neg = compute_positive_negative_parts(QQt)
    XQ_pos,  XQ_neg  = compute_positive_negative_parts(XQ)

    numerator   = XQ_pos + W @ QQt_neg
    denominator = XQ_neg + W @ QQt_pos + 1e-10

    if alpha > 0 and L is not None:
        LW = L @ W
        D  = cp.asarray(L.diagonal(), dtype=cp.float32)
        DW = D[:, None] * W
        numerator   += alpha * (DW - LW)
        denominator += alpha * DW

    if lambda_l1 is not None:
        lambda_l1 = lambda_l1.astype(cp.float32) if lambda_l1.dtype != cp.float32 else lambda_l1
        denominator += lambda_l1[None, :]

    ratio = cp.clip(numerator / denominator, 0.0, 1e6)
    W_new = W * cp.sqrt(ratio)
    bad   = cp.isnan(W_new) | cp.isinf(W_new)
    if cp.any(bad):
        W_new = cp.where(bad, W, W_new)
    return W_new


def update_W_hybrid_manual_tv(XQ, Q, W, tv_manager, alpha, lambda_l1=None):
    """Manual W update for hybrid mode (precomputed XQ, TV)."""
    if not CUPY_AVAILABLE:
        raise RuntimeError("Hybrid manual TV requires CuPy")

    XQ = XQ.astype(cp.float32) if XQ.dtype != cp.float32 else XQ
    Q  = Q.astype(cp.float32)  if Q.dtype  != cp.float32 else Q
    W  = W.astype(cp.float32)  if W.dtype  != cp.float32 else W

    # Q: (p, r)
    # W: (n, r)
    # XQ: (n, r)
    Qt        = Q.T
    QQt       = Qt @ Q              # (r, r)
    QQt_pos, QQt_neg = compute_positive_negative_parts(QQt)
    XQ_pos,  XQ_neg  = compute_positive_negative_parts(XQ)

    numerator   = XQ_pos + W @ QQt_neg
    denominator = XQ_neg + W @ QQt_pos + 1e-10

    if alpha > 0:
        _, LW, D = tv_manager.get_regularization_terms(W)
        DW = D * W
        numerator   += alpha * (DW - LW)
        denominator += alpha * DW

    if lambda_l1 is not None:
        lambda_l1 = lambda_l1.astype(cp.float32) if lambda_l1.dtype != cp.float32 else lambda_l1
        denominator += lambda_l1[None, :]

    ratio = cp.clip(numerator / denominator, 0.0, 1e6)
    W_new = W * cp.sqrt(ratio)
    bad   = cp.isnan(W_new) | cp.isinf(W_new)
    if cp.any(bad):
        W_new = cp.where(bad, W, W_new)
    return W_new
# ============================================================================
# Helper functions
# ============================================================================

def compute_positive_negative_parts(M):
    xp = cp if (CUPY_AVAILABLE and isinstance(M, cp.ndarray)) else np
    return xp.maximum(M, 0), xp.maximum(-M, 0)


def compute_loss_scale(
    X: Union[np.ndarray, 'spmatrix'],
    n_components: int,
) -> float:
    """
    Legacy scale estimator — kept for backwards compatibility.
    The Riemannian optimizer uses _compute_normalisation_scales instead.
    """
    n, p = X.shape
    r    = n_components
    if issparse(X):
        mu = float(X.data.mean()) if X.nnz > 0 else 1.0
    else:
        n_sample = min(1000, n)
        mu = float(np.mean(X[:n_sample]))
    mu = float(np.clip(mu, 0.1, 5.0))
    return (n * mu) / r


def _get_adaptive_lr(n: int) -> float:
    """
    Return the fixed Stiefel-manifold step size.

    The normalization scales handle dataset-size effects, leaving ``lr`` to
    control the approximate geodesic step length. The manuscript
    implementation uses 1e-3.
    """
    return 1e-3


def _compute_WtW_WtX_sparse_cpu(
    W: np.ndarray,
    X: Union[np.ndarray, 'spmatrix'],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute W^T W  (r×r)  and  W^T X  (r×p)  on CPU via scipy sparse SpMM.
    Results are small and uploaded to GPU by the caller.

    Returns
    -------
        WtW : (r, r)  float32
        WtX : (r, p)  float32; this is W^T X.
    """
    from scipy.sparse import issparse, csr_matrix

    if W.dtype != np.float32:
        W = W.astype(np.float32)

    WtW = (W.T @ W).astype(np.float32)          # (r, r)

    if issparse(X):
        if not isinstance(X, csr_matrix):
            X = X.tocsr()
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        # X.T @ W = (p×n)(n×r) = (p×r)  =>  .T = (r×p) = W^T X  ✓
        WtX = np.ascontiguousarray((X.T @ W).T, dtype=np.float32)

    elif CUPY_AVAILABLE and isinstance(X, cp.ndarray):
        WtX = (W.T @ cp.asnumpy(X).astype(np.float32)).astype(np.float32)

    elif CUPY_AVAILABLE and hasattr(X, 'get'):
        X_scipy = X.get()
        if not isinstance(X_scipy, csr_matrix):
            X_scipy = X_scipy.tocsr().astype(np.float32)
        WtX = np.ascontiguousarray((X_scipy.T @ W).T, dtype=np.float32)

    else:
        X_np = (X if (isinstance(X, np.ndarray) and X.dtype == np.float32)
                else np.asarray(X, dtype=np.float32))
        WtX  = (W.T @ X_np).astype(np.float32)  # (r, p)

    return WtW, WtX


def _compute_normalisation_scales(
    n: int,
    p: int,
    r: int,
    smooth_l1_delta: float = 0.1,
) -> dict:
    """
    Compute gradient-alignment scales from dimensions ``n``, ``p``, and ``r``.

    This assumes log-normalized X and signal columns on the Stiefel manifold,
    ``Q_s.T @ Q_s = I``. The scales keep regularization gradients at stable
    ratios to the reconstruction gradient across dataset sizes.

    Derivation:
      Reconstruction gradient with respect to Q_s:
        ||G_recon(Q_s)||_F ~ 2 * sqrt(n * p * r_s)

      Negative-loading gradient:
        G_neg = -2 * lambda_neg_eff * relu(-Q_s)
        Under the Stiefel constraint, entries are approximately N(0, 1/p):
          E[max(0, -Q_{jk})^2] = 1/(2p), over p*r_s entries
          => ||relu(-Q_s)||_F^2 ~ r_s/2
          => ||G_neg||_F ~ lambda_neg_eff * sqrt(2 * r_s)
        Matching condition:
          lambda_neg_eff * sqrt(2*r_s) ~ 2*sqrt(n*p*r_s)
          => S_neg = sqrt(2 * n * p)

      Smooth-L1 gradient in its approximately linear region:
        G_l1Q = lambda_l1_eff * sign(Q_s), over p*r_s entries
          => ||G_l1Q||_F = lambda_l1_eff * sqrt(p * r_s)
        Matching condition:
          lambda_l1_eff * sqrt(p*r_s) ~ 2*sqrt(n*p*r_s)
          => S_l1Q = 2 * sqrt(n)

      TV and W-L1 terms are already on the W reconstruction-gradient scale,
      so their scale is one.

    The fixed gradient clip protects retraction stability during Adam warm-up;
    it is tied to manifold geometry rather than dataset size.
    """
    r_s = max(r - 1, 1)

    S_neg  = float(np.sqrt(2.0 * n * p))    # ~ sqrt(n*p)
    S_l1Q  = float(2.0 * np.sqrt(float(n))) # ~ sqrt(n)
    S_l1W  = 1.0
    S_recon_ref = float(2.0 * np.sqrt(float(n * p * r_s)))

    return {
        'neg'       : S_neg,
        'l1_Q'      : S_l1Q,
        'l1_W'      : S_l1W,
        'recon_ref' : S_recon_ref,
        'ortho'     : 1.0,
    }


def _print_Q_diagnostics(d: dict, prefix: str = '') -> None:
    """Print compact Q-update scale and clipping diagnostics."""
    def flag(v, lo, hi):
        if v < lo:  return f'\033[33m↑{v:.4f}\033[0m'
        if v > hi:  return f'\033[31m↓{v:.4f}\033[0m'
        return             f'\033[32m✓{v:.4f}\033[0m'

    clip_str = '\033[31mCLIPPED\033[0m' if d['clipped'] else 'ok'

    print(
        f"  [Q-diag {prefix}] "
        f"recon={d['recon']:>11.3e} | "
        f"neg/recon={flag(d['ratio_neg'], 0.01, 0.10)} | "
        f"l1Q/recon={flag(d['ratio_l1Q'], 0.01, 0.10)} | "
        f"|G_riem|={d['riem_norm']:.2f}({clip_str}) | "
        f"rho={d['rho_actual']:.3f}"
    )


# ============================================================================
# Riemannian Adam optimizer on the Stiefel manifold
# ============================================================================

class RiemannianAdam:
    """
    Adam optimizer on the Stiefel manifold St(p, r_s) = {Q : Q^T Q = I}.

    Each step projects the Euclidean gradient into the tangent space,
    transports the previous moment into the current tangent space, applies
    the Adam update, and retracts by polar decomposition or QR.

    Parameters
    ----------
    lr    : step size on the approximate geodesic scale (default 1e-3)
    beta1 : 1st moment decay  (default 0.9)
    beta2 : 2nd moment decay  (default 0.999)
    eps   : numerical floor   (default 1e-8)
    """

    def __init__(
        self,
        lr:    float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps:   float = 1e-8,
    ):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m: Optional['torch.Tensor'] = None
        self.v: Optional['torch.Tensor'] = None
        self.t: int = 0

    @staticmethod
    def _project_tangent(
        Q_s: 'torch.Tensor',
        G:   'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Project Euclidean gradient G into T_{Q_s} St(p, r_s).

        grad_Riem = G - Q_s sym(Q_s^T G)
        The result satisfies Q_s^T Delta + Delta^T Q_s = 0.
        """
        QtG = Q_s.T @ G
        sym = (QtG + QtG.T) * 0.5
        return G - Q_s @ sym

    @staticmethod
    def _retract(
        Q_s:   'torch.Tensor',
        Delta: 'torch.Tensor',
        eta:   float,
        mode:  str = 'polar',
    ) -> 'torch.Tensor':
        """Retract the tangent step eta*Delta onto St(p, r_s)."""
        Y = Q_s + eta * Delta
        if mode == 'polar':
            try:
                U, _, Vt = torch.linalg.svd(Y, full_matrices=False)
                return U @ Vt
            except Exception:
                pass
        Q_out, R = torch.linalg.qr(Y)
        signs    = torch.sign(torch.diag(R)).unsqueeze(0)
        return Q_out * signs

    def step(
        self,
        Q_s:    'torch.Tensor',
        G_eucl: 'torch.Tensor',
        retraction: str = 'polar',
    ) -> 'torch.Tensor':
        """Apply one Riemannian Adam step."""
        self.t += 1

        G_riem = self._project_tangent(Q_s, G_eucl)

        if self.m is None:
            self.m = torch.zeros_like(G_riem)
            self.v = torch.zeros_like(G_riem)
        else:
            self.m = self._project_tangent(Q_s, self.m)

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * G_riem
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * G_riem ** 2

        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        Delta = m_hat / (v_hat.sqrt() + self.eps)

        return self._retract(Q_s, -Delta, self.lr, mode=retraction)

    def reset(self):
        """Reset optimizer moments when changing stages."""
        self.m = None
        self.v = None
        self.t = 0


# ============================================================================
# Main Riemannian Q update
# ============================================================================

def update_Q_riemannian(
    Q_torch:         'torch.Tensor',
    W:               np.ndarray,
    X:               Union[np.ndarray, 'spmatrix'],
    beta:            float = 0.0,
    lambda_l1:       Optional[np.ndarray] = None,
    riem_optimizer:  Optional['RiemannianAdam'] = None,
    lr:              Optional[float] = None,
    n_steps:         int = 10,
    background_col:  int = 0,
    lambda_neg:      float = 0.02,
    smooth_l1_delta: float = 0.1,
    grad_clip_norm:  float = 5.0,
    retraction:      str = 'polar',
    verbose_diag:    bool = False,
    diag_prefix:     str  = '',
    # ── deprecated / ignored ────────────────────────────────────────
    ortho_mode:      str = 'huber',
    huber_delta:     float = 0.5,
    loss_scale:      Optional[float] = None,
    ortho_scale:     Optional[float] = None,
    neg_scale:       Optional[float] = None,
    subsample_size:  Optional[int]   = None,
    proj_freq:       int = 0,
    optimizer:       Optional[object] = None,
) -> Tuple['torch.Tensor', 'RiemannianAdam']:
    """
    Update signal columns Q_s with Riemannian Adam and the background column
    q0 with unconstrained gradient descent. Gradient alignment is handled by
    S_neg and S_l1Q; clipping protects retraction during warm-up.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")

    device = Q_torch.device
    n, r   = W.shape
    p      = Q_torch.shape[0]
    r_s    = r - 1
    assert Q_torch.shape == (p, r), f"Q_torch shape {Q_torch.shape} != ({p}, {r})"

    # 1. Learning rate and optimizer.
    if lr is None:
        lr = _get_adaptive_lr(n)

    if riem_optimizer is None:
        riem_optimizer = RiemannianAdam(lr=lr)
    elif riem_optimizer.lr != lr:
        riem_optimizer.lr = lr

    # ── 2. WtW / WtX ─────────────────────────────────────────────────────
    WtW_np, WtX_np = _compute_WtW_WtX_sparse_cpu(W, X)

    with torch.no_grad():
        WtW  = torch.as_tensor(WtW_np,          dtype=torch.float32, device=device)
        WtXT = torch.as_tensor(WtX_np.T.copy(), dtype=torch.float32, device=device)
    del WtW_np, WtX_np

    if torch.isnan(WtW).any() or torch.isnan(WtXT).any():
        import warnings
        warnings.warn("NaN in WtW/WtXT — skipping Q update this epoch.",
                      RuntimeWarning, stacklevel=2)
        return Q_torch, riem_optimizer

    # 3. Gradient-alignment scales based only on n, p, and r.
    scales         = _compute_normalisation_scales(n, p, r, smooth_l1_delta)
    lambda_neg_eff = lambda_neg * scales['neg']   # lambda_neg * sqrt(2*n*p)

    if lambda_l1 is not None:
        lam_np = np.asarray(lambda_l1, dtype=np.float32) * scales['l1_Q']
                                                          # lambda_l1 * 2*sqrt(n)
        lam_t  = torch.as_tensor(lam_np, dtype=torch.float32, device=device)
        assert lam_t.shape[0] == r_s
    else:
        lam_t = None

    # 4. Optional negative-energy diagnostic.
    rho_actual = 0.0
    if verbose_diag:
        with torch.no_grad():
            Q_s_now = (Q_torch.detach()[:, 1:]
                       if background_col == 0
                       else Q_torch.detach()[
                           :, [c for c in range(r) if c != background_col]
                       ])
            neg_energy = float(torch.sum(F.relu(-Q_s_now) ** 2).item())
            rho_actual = neg_energy / max(float(r_s), 1e-12)

    # 5. Signal-column indices.
    sig_cols = (list(range(1, r)) if background_col == 0
                else [c for c in range(r) if c != background_col])

    # 6. Riemannian Adam loop.
    _diag = {}

    for step_i in range(n_steps):

        Q_val    = Q_torch.detach()
        q0       = Q_val[:, background_col]
        Q_s      = Q_val[:, sig_cols]

        Q_s_leaf = Q_s.clone().requires_grad_(True)
        q0_leaf  = q0.clone().requires_grad_(True)

        # Reconstruct full Q in original column order.
        cols  = []
        sig_i = 0
        for col in range(r):
            if col == background_col:
                cols.append(q0_leaf.unsqueeze(1))
            else:
                cols.append(Q_s_leaf[:, sig_i:sig_i + 1])
                sig_i += 1
        Q_full = torch.cat(cols, dim=1)

        # (i) Reconstruction loss.
        recon_loss = (
              torch.sum((Q_full @ WtW) * Q_full)
            - 2.0 * torch.sum(WtXT * Q_full)
        )

        # (ii) Negative-loading penalty on Q_s only.
        neg_loss = lambda_neg_eff * torch.sum(F.relu(-Q_s_leaf) ** 2)

        # (iii) Smooth-L1 sparsity penalty on Q_s only.
        if lam_t is not None:
            col_l1  = F.huber_loss(
                Q_s_leaf,
                torch.zeros_like(Q_s_leaf),
                delta=smooth_l1_delta,
                reduction='none',
            ).sum(dim=0)
            l1_loss = torch.dot(lam_t, col_l1)
        else:
            l1_loss = torch.zeros(1, device=device)

        loss = recon_loss + neg_loss + l1_loss

        if not torch.isfinite(loss):
            import warnings
            warnings.warn(
                f"[RiemAdam step {step_i}] non-finite loss "
                f"(recon={recon_loss.item():.3e}, neg={neg_loss.item():.3e})",
                RuntimeWarning, stacklevel=2,
            )
            break

        loss.backward()

        G_s  = Q_s_leaf.grad.detach()
        G_q0 = q0_leaf.grad.detach()

        # Clip the Riemannian gradient at the fixed geometric threshold.
        G_s_riem  = RiemannianAdam._project_tangent(Q_s, G_s)
        riem_norm = G_s_riem.norm()
        clipped   = bool(riem_norm > grad_clip_norm)
        if clipped:
            G_s = G_s * (grad_clip_norm / riem_norm)

        g_q0_norm = G_q0.norm()
        if g_q0_norm > grad_clip_norm:
            G_q0 = G_q0 * (grad_clip_norm / g_q0_norm)

        # Collect diagnostics at the last inner step.
        if verbose_diag and step_i == n_steps - 1:
            with torch.no_grad():
                r_val = float(recon_loss.item())
                n_val = float(neg_loss.item())
                l_val = float(l1_loss.item()) if lam_t is not None else 0.0
                _diag = {
                    'recon'      : r_val,
                    'neg'        : n_val,
                    'l1Q'        : l_val,
                    'ratio_neg'  : n_val / max(abs(r_val), 1e-12),
                    'ratio_l1Q'  : l_val / max(abs(r_val), 1e-12),
                    'riem_norm'  : float(riem_norm.item()),
                    'clipped'    : clipped,
                    'rho_actual' : rho_actual,
                    'S_neg'      : scales['neg'],
                    'lam_neg_eff': lambda_neg_eff,
                }

        # Riemannian Adam step for Q_s.
        Q_s_new = riem_optimizer.step(Q_s, G_s, retraction=retraction)

        # Unconstrained gradient step for q0.
        q0_new = q0 - lr * G_q0

        with torch.no_grad():
            Q_torch[:, background_col].copy_(q0_new)
            for i, col in enumerate(sig_cols):
                Q_torch[:, col].copy_(Q_s_new[:, i])

    # 7. Diagnostics.
    if verbose_diag and _diag:
        _print_Q_diagnostics(_diag, diag_prefix)

    return Q_torch, riem_optimizer

# ============================================================================
# Backward-compatible Q-update interface
# ============================================================================

def update_Q_adam(
    Q_torch:         'torch.Tensor',
    W:               np.ndarray,
    X:               Union[np.ndarray, 'spmatrix'],
    beta:            float = 0.05,
    lambda_l1:       Optional[np.ndarray] = None,
    optimizer:       Optional[object] = None,
    lr:              Optional[float] = None,
    n_steps:         int = 40,
    background_col:  int = 0,
    ortho_mode:      str = 'huber',
    huber_delta:     float = 0.5,
    lambda_neg:      float = 0.02,
    smooth_l1_delta: float = 0.1,
    grad_clip_norm:  float = 5.0,
    retraction:      str = 'polar',
    verbose_diag:    bool = False,
    diag_prefix:     str  = '',
    # ── deprecated ───────────────────────────────────────────────────
    loss_scale:      Optional[float] = None,
    ortho_scale:     Optional[float] = None,
    neg_scale:       Optional[float] = None,
    subsample_size:  Optional[int]   = None,
    proj_freq:       int = 0,
    riem_optimizer:  Optional[RiemannianAdam] = None,
) -> Tuple['torch.Tensor', RiemannianAdam]:
    """Preserve the external signature while using Riemannian Adam."""
    return update_Q_riemannian(
        Q_torch         = Q_torch,
        W               = W,
        X               = X,
        beta            = beta,
        lambda_l1       = lambda_l1,
        riem_optimizer  = riem_optimizer,
        lr              = lr,
        n_steps         = n_steps,
        background_col  = background_col,
        lambda_neg      = lambda_neg,
        smooth_l1_delta = smooth_l1_delta,
        grad_clip_norm  = grad_clip_norm,
        retraction      = retraction,
        verbose_diag    = verbose_diag,
        diag_prefix     = diag_prefix,
    )
