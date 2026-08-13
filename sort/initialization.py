"""
SymNMF initialization for SORT (Sparse-Aware, Memory-Efficient).
"""

import gc
import numpy as np
from scipy.sparse import issparse

from .core.initializers import NNSVDInitializer
from .core.updaters import HALSUpdater
from .core.procrustes import initialize_Q_from_W

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def _convert_to_dense(X, backend='numpy'):
    """
    Convert matrix to dense float32.

    Parameters
    ----------
    X : array-like
        Input matrix (sparse or dense)
    backend : str
        'numpy' or 'cupy'

    Returns
    -------
    X_dense : ndarray
        Dense float32 array
    """
    if backend == 'cupy' and CUPY_AVAILABLE:
        if issparse(X):
            X_dense = cp.asarray(X.toarray(), dtype=cp.float32)
        elif isinstance(X, cp.ndarray):
            X_dense = X.astype(cp.float32) if X.dtype != cp.float32 else X
        else:
            X_dense = cp.asarray(X, dtype=cp.float32)
    else:
        if issparse(X):
            X_dense = X.toarray().astype(np.float32)
        elif CUPY_AVAILABLE and isinstance(X, cp.ndarray):
            X_dense = cp.asnumpy(X).astype(np.float32)
        elif isinstance(X, np.ndarray):
            X_dense = X.astype(np.float32) if X.dtype != np.float32 else X
        else:
            X_dense = np.asarray(X, dtype=np.float32)

    return X_dense


def _compute_frobenius_norm_safe(Y, xp, verbose=False):
    """
    Compute Frobenius norm safely (avoids GPU OOM for large matrices).

    For large GPU arrays, transfers to CPU in chunks.

    Parameters
    ----------
    Y : ndarray
        Input matrix (can be on CPU or GPU)
    xp : module
        numpy or cupy
    verbose : bool
        Print progress

    Returns
    -------
    norm : float
        Frobenius norm of Y
    """
    if xp == np:
        return float(np.linalg.norm(Y, 'fro'))

    n, p = Y.shape
    element_size = Y.dtype.itemsize
    intermediate_size = n * p * element_size

    try:
        free_mem = cp.cuda.Device(0).mem_info[0]
        if intermediate_size * 2 < free_mem:
            if verbose:
                print(f"  Computing norm on GPU (safe: {intermediate_size/1e9:.1f} GB "
                      f"< {free_mem/1e9:.1f} GB)")
            return float(cp.linalg.norm(Y, 'fro'))
    except Exception:
        pass

    if verbose:
        print("  GPU memory insufficient for norm, computing on CPU...")

    Y_cpu = cp.asnumpy(Y)
    norm = float(np.linalg.norm(Y_cpu, 'fro'))
    del Y_cpu

    return norm


def _check_gpu_memory_sufficient(dense_bytes, multiplier=3, verbose=True):
    """
    Check whether GPU has enough free memory for a dense allocation.

    Parameters
    ----------
    dense_bytes : int
        Size (in bytes) of the dense matrix to allocate.
    multiplier : float
        Safety multiplier (accounts for intermediate buffers during HALS).
    verbose : bool
        Print diagnostic info.

    Returns
    -------
    sufficient : bool
        True if GPU free memory >= dense_bytes * multiplier.
    free_mem : int
        Current free GPU memory in bytes (0 if query failed).
    """
    if not CUPY_AVAILABLE:
        return False, 0

    try:
        free_mem, total_mem = cp.cuda.Device(0).mem_info
        needed = dense_bytes * multiplier
        if verbose:
            print(f"  GPU memory — free: {free_mem/1e9:.2f} GB, "
                  f"needed (×{multiplier}): {needed/1e9:.2f} GB / "
                  f"total: {total_mem/1e9:.2f} GB")
        return free_mem >= needed, free_mem
    except Exception:
        return False, 0


def initialize_W_symnmf(
    X, L, n_components,
    alpha=1.0,
    lambda1=0.1,
    lambda2=0.0,
    smooth_center_weight=0.5,
    max_iter=100,
    tol=1e-6,
    random_state=None,
    backend='auto',
    block_size=50000,
    verbose=True
):
    """
    Initialize W for SORT using SymNMF on smoothed expression.

    **UPDATED**: Uses sparse smoothing + safe norm computation + GPU memory
    guard to avoid OOM.

    Algorithm:
    1. Smooth X using Laplacian L → Y (sparse if X is sparse)
    2. Convert Y to dense for SymNMF (fallback to CPU if GPU memory is tight)
    3. Run SymNMF on Y: Y@Y^T ≈ W@H^T
    4. Select better factor (W or H)
    5. Initialize Q from W using Y

    Parameters
    ----------
    X : array-like, shape (n, p)
        Expression matrix (can be sparse or dense)
    L : sparse matrix, shape (n, n)
        Graph Laplacian matrix
    n_components : int
        Number of components
    alpha : float, default=1.0
        Symmetry regularization weight
    lambda1 : float, default=0.1
        L1 regularization weight
    lambda2 : float, default=0.0
        L2 regularization weight
    smooth_center_weight : float, default=0.5
        Center weight for spatial smoothing
    max_iter : int, default=100
        Maximum HALS iterations
    tol : float, default=1e-6
        Convergence tolerance
    random_state : int, optional
        Random seed
    backend : {'auto', 'cupy', 'numpy'}
        Computing backend (for HALS; smooth uses sparse CPU if X is sparse)
    block_size : int, default=50000
        Block size for sparse smoothing (ignored if X is dense)
    verbose : bool, default=True
        Print progress

    Returns
    -------
    W_init : ndarray, shape (n, n_components)
        Initial loading matrix (float32)
    Q_init : ndarray, shape (p, n_components)
        Initial signature matrix (float32)
    """

    # ── Determine backend for HALS ────────────────────────────────────────
    if backend == 'auto':
        if CUPY_AVAILABLE:
            if isinstance(X, cp.ndarray):
                backend = 'cupy'
                xp = cp
            else:
                try:
                    _ = cp.cuda.Device(0)
                    backend = 'cupy'
                    xp = cp
                except Exception:
                    backend = 'numpy'
                    xp = np
        else:
            backend = 'numpy'
            xp = np
    elif backend == 'cupy':
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        xp = cp
    else:
        xp = np

    X_is_sparse = issparse(X)

    if verbose:
        print("=" * 70)
        print("SymNMF Initialization (Sparse-Aware)")
        print("=" * 70)
        print(f"Data: {X.shape[0]:,} × {X.shape[1]:,}")
        print(f"Components: {n_components}")
        if X_is_sparse:
            print(f"Input format: sparse (nnz={X.nnz:,}, "
                  f"density={X.nnz/(X.shape[0]*X.shape[1]):.2%})")
        else:
            print("Input format: dense")
        print(f"HALS backend: {backend.upper()}")
        if X_is_sparse:
            print(f"Smooth backend: CPU sparse (block_size={block_size:,})")
        else:
            print(f"Smooth backend: {backend.upper()} (dense)")

    # ========================================================================
    # Step 1: Spatial smoothing
    # ========================================================================
    if verbose:
        print(f"\nStep 1: Spatial smoothing (center_weight={smooth_center_weight})...")

    if X_is_sparse:
        from .preprocessing import smooth_expression_from_laplacian_sparse

        Y_sparse = smooth_expression_from_laplacian_sparse(
            X, L, center_weight=smooth_center_weight,
            block_size=block_size, verbose=verbose
        )

        # ── Step 2: sparse → dense, with GPU memory guard ────────────────
        if verbose:
            print("\nStep 2: Converting smoothed result to dense for HALS...")

        n_rows, n_cols = Y_sparse.shape
        dense_bytes = int(n_rows) * int(n_cols) * 4  # float32 = 4 bytes

        if verbose:
            print(f"  Dense matrix size: {dense_bytes/1e9:.2f} GB "
                  f"({n_rows:,} × {n_cols:,} × float32)")

        # ── GPU memory guard: fallback to CPU if too tight ────────────────
        if backend == 'cupy':
            sufficient, free_mem = _check_gpu_memory_sufficient(
                dense_bytes, multiplier=3, verbose=verbose
            )
            if not sufficient:
                print(f"  ⚠️  Insufficient GPU memory "
                      f"(free={free_mem/1e9:.2f} GB, need≈{dense_bytes*3/1e9:.2f} GB). "
                      f"Falling back to CPU numpy for HALS.")
                backend = 'numpy'
                xp = np
                # Free any cached GPU blocks before proceeding on CPU
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

        # ── Perform the actual dense conversion ───────────────────────────
        if backend == 'cupy':
            Y = cp.asarray(Y_sparse.toarray(), dtype=cp.float32)
        else:
            Y = Y_sparse.toarray().astype(np.float32)

        # ── verbose stats: use CPU sparse data, NOT a new GPU bool array ──
        #    (the original `xp.sum(Y > 1e-10)` was the direct OOM cause)
        if verbose:
            Y_nnz_ratio = (Y_sparse.nnz / (n_rows * n_cols)) if n_rows * n_cols > 0 else 0.0
            if Y_sparse.nnz > 0:
                Y_min_v = float(Y_sparse.data.min())
                Y_max_v = float(Y_sparse.data.max())
            else:
                Y_min_v = Y_max_v = 0.0
            print(f"  Dense Y: shape={Y.shape}, non-zero (sparse nnz)={Y_nnz_ratio:.2%}")
            print(f"  Value range: [{Y_min_v:.4f}, {Y_max_v:.4f}]")

        # ── Release sparse matrix immediately to reclaim RAM ──────────────
        del Y_sparse
        gc.collect()
        if backend == 'cupy' and CUPY_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()

    else:
        # X is already dense — use existing smooth function
        from .preprocessing import smooth_expression_from_laplacian

        if backend == 'cupy' and not isinstance(X, cp.ndarray):
            X_compute = cp.asarray(X, dtype=cp.float32)
        elif backend == 'numpy' and CUPY_AVAILABLE and isinstance(X, cp.ndarray):
            X_compute = cp.asnumpy(X).astype(np.float32)
        else:
            X_compute = X.astype(np.float32) if X.dtype != np.float32 else X

        Y = smooth_expression_from_laplacian(
            X_compute, L, center_weight=smooth_center_weight,
            backend=backend, verbose=verbose
        )

        if verbose:
            Y_min, Y_max = float(xp.min(Y)), float(xp.max(Y))
            print(f"  Smoothed range: [{Y_min:.4f}, {Y_max:.4f}]")

    # ========================================================================
    # Step 3: NNDSVD initialization
    # ========================================================================
    step_num = 3 if X_is_sparse else 2
    if verbose:
        print(f"\nStep {step_num}: NNDSVD initialization...")

    initializer = NNSVDInitializer(random_state=random_state, backend=backend)
    W, H = initializer.initialize(Y, n_components)

    if verbose:
        W_nnz = float(xp.sum(W > 1e-10)) / W.size
        H_nnz = float(xp.sum(H > 1e-10)) / H.size
        print(f"  W non-zeros: {W_nnz:.1%}, H non-zeros: {H_nnz:.1%}")

    # ========================================================================
    # Step 4: HALS optimization
    # ========================================================================
    step_num += 1
    if verbose:
        print(f"\nStep {step_num}: HALS optimization (max_iter={max_iter})...")

    updater = HALSUpdater(
        alpha=alpha,
        lambda1=lambda1,
        lambda2=lambda2,
        backend=backend
    )

    objective_values = []

    Y_norm = _compute_frobenius_norm_safe(Y, xp, verbose=verbose)

    for iter_num in range(max_iter):
        W, H = updater.update(Y, W, H)

        if iter_num % 10 == 0 or iter_num == max_iter - 1:
            obj = _compute_objective(Y, W, H, alpha, lambda1, lambda2,
                                     Y_norm, xp, verbose=verbose)
            objective_values.append(obj)

            if verbose and (iter_num % 50 == 0 or iter_num < 10):
                residual = _compute_residual(Y, W, H, Y_norm, xp,
                                             verbose=verbose)
                print(f"  Iter {iter_num:4d}: obj={obj:.4e}, "
                      f"residual={residual:.4e}")

            if len(objective_values) >= 2:
                rel_change = (
                    abs(objective_values[-1] - objective_values[-2])
                    / (abs(objective_values[-2]) + 1e-16)
                )
                if rel_change < tol:
                    if verbose:
                        print(f"  ✓ Converged at iter {iter_num} "
                              f"(rel_change={rel_change:.2e})")
                    break

    # ========================================================================
    # Step 5: Select better factor
    # ========================================================================
    step_num += 1
    if verbose:
        print(f"\nStep {step_num}: Selecting better factor...")

    res_W = _compute_residual(Y, W, W, Y_norm, xp, verbose=False)
    res_H = _compute_residual(Y, H, H, Y_norm, xp, verbose=False)

    if res_H < res_W:
        W_init = H.copy()
        if verbose:
            print(f"  ✓ Selected H (residual: {res_H:.4e} < {res_W:.4e})")
    else:
        W_init = W.copy()
        if verbose:
            print(f"  ✓ Selected W (residual: {res_W:.4e} ≤ {res_H:.4e})")

    # Convert back to numpy float32
    if backend == 'cupy':
        W_init = cp.asnumpy(W_init).astype(np.float32)
    else:
        W_init = W_init.astype(np.float32)

    W_init = np.maximum(W_init, 0)

    # ========================================================================
    # Step 6: Initialize Q using Procrustes on SMOOTHED Y
    # ========================================================================
    step_num += 1
    if verbose:
        print(f"\nStep {step_num}: Initialize Q from smoothed Y (Procrustes)...")

    if backend == 'cupy':
        Y_np = cp.asnumpy(Y).astype(np.float32)
    else:
        Y_np = (Y.astype(np.float32)
                if isinstance(Y, np.ndarray)
                else np.asarray(Y, dtype=np.float32))

    Q_init = initialize_Q_from_W(W_init, Y_np, method='svd')
    Q_init = Q_init.astype(np.float32)

    if verbose:
        Q_nnz = np.sum(np.abs(Q_init) > 1e-10) / Q_init.size
        print(f"  Q non-zeros: {Q_nnz:.1%}")
        print("  (Q initialized to match W @ Q^T ≈ Y)")
        print("=" * 70)
        print("✓ Initialization complete")
        print("=" * 70 + "\n")

    return W_init, Q_init


def _compute_objective(Y, W, H, alpha, lambda1, lambda2, Y_norm, xp,
                        verbose=False):
    """
    Compute SymNMF objective (memory-efficient, avoids intermediate arrays).

    Parameters
    ----------
    Y : ndarray
        Data matrix
    W, H : ndarray
        Factor matrices
    alpha, lambda1, lambda2 : float
        Regularization weights
    Y_norm : float
        Pre-computed Frobenius norm of Y
    xp : module
        numpy or cupy
    verbose : bool
        Print debug info

    Returns
    -------
    objective : float
        Total objective value
    """
    Y_norm_4 = Y_norm ** 4

    WtW = W.T @ W
    HtH = H.T @ H
    WH_norm_sq = float(xp.sum(WtW * HtH))

    YtW = Y.T @ W
    YtH = Y.T @ H
    trace_term = float(xp.sum(YtW * YtH))

    recon_loss = Y_norm_4 + WH_norm_sq - 2 * trace_term
    recon_loss = max(0, recon_loss)

    sym_loss = alpha * float(xp.linalg.norm(W - H, 'fro') ** 2)
    l1_loss = lambda1 * float(xp.sum(xp.abs(W)) + xp.sum(xp.abs(H)))
    l2_loss = lambda2 * float(
        xp.linalg.norm(W, 'fro') ** 2 + xp.linalg.norm(H, 'fro') ** 2
    )

    if verbose:
        print(f"    Objective breakdown: recon={recon_loss:.2e}, "
              f"sym={sym_loss:.2e}, l1={l1_loss:.2e}, l2={l2_loss:.2e}")

    return recon_loss + sym_loss + l1_loss + l2_loss


def _compute_residual(Y, W, H, Y_norm, xp, verbose=False):
    """
    Compute relative residual (memory-efficient).

    Parameters
    ----------
    Y : ndarray
        Data matrix
    W, H : ndarray
        Factor matrices
    Y_norm : float
        Pre-computed Frobenius norm of Y
    xp : module
        numpy or cupy
    verbose : bool
        Print debug info

    Returns
    -------
    residual : float
        Relative residual ||Y^T Y - WH^T||_F / ||Y||_F^2
    """
    Y_norm_sq = Y_norm ** 2
    Y_norm_4 = Y_norm ** 4

    WtW = W.T @ W
    HtH = H.T @ H
    WH_norm_sq = float(xp.sum(WtW * HtH))

    YtW = Y.T @ W
    YtH = Y.T @ H
    trace_term = float(xp.sum(YtW * YtH))

    residual_sq = Y_norm_4 + WH_norm_sq - 2 * trace_term
    residual_sq = max(0, residual_sq)

    residual = float(xp.sqrt(residual_sq)) / Y_norm_sq

    if verbose:
        print(f"    Residual: {residual:.4e} "
              f"(||Y^T Y - WH^T||_F / ||Y||_F^2)")

    return residual