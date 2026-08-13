"""
Procrustes problem solvers for orthogonal matrix updates.
"""

import numpy as np
from scipy.linalg import svd
from scipy.sparse import issparse

try:
    import cupy as cp
    from cupyx.scipy.linalg import svd as cp_svd
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cp_svd = None
    CUPY_AVAILABLE = False


def procrustes(A, B, return_transformation=True):
    """
    Solve orthogonal Procrustes: min ||A - BQ^T||_F s.t. Q^T Q = I
    
    Parameters
    ----------
    A : array-like, shape (n, p)
        Target matrix (float32)
    B : array-like, shape (n, r)
        Source matrix (float32)
    return_transformation : bool, default=True
        Return transformation matrix Q
    
    Returns
    -------
    Q : array, shape (p, r)
        Orthogonal transformation matrix (float32)
    
    Notes
    -----
    Solution: Q = V @ U^T where B^T A = U S V^T (SVD)
    
    Reference: Schönemann (1966). Psychometrika 31(1), 1-10.
    """
    # Detect backend
    if CUPY_AVAILABLE and isinstance(B, cp.ndarray):
        xp = cp
        svd_fn = cp_svd
    else:
        xp = np
        svd_fn = svd
        if CUPY_AVAILABLE and isinstance(B, cp.ndarray):
            B = cp.asnumpy(B)
            A = cp.asnumpy(A)
    
    # Ensure float32
    if B.dtype != np.float32 and (xp == np or B.dtype != cp.float32):
        if xp == cp:
            B = B.astype(cp.float32)
            A = A.astype(cp.float32)
        else:
            B = B.astype(np.float32)
            A = A.astype(np.float32)
    
    # Compute B^T A
    BtA = B.T @ A  # (r, n) @ (n, p) = (r, p)
    
    # SVD
    U, S, Vt = svd_fn(BtA, full_matrices=False)
    
    # Optimal Q
    Q = Vt.T @ U.T  # (p, r)
    
    # Ensure float32 output
    if xp == cp:
        Q = Q.astype(cp.float32)
        if return_transformation:
            return cp.asnumpy(Q).astype(np.float32)
    else:
        Q = Q.astype(np.float32)
    
    if return_transformation:
        return Q
    else:
        return B @ Q.T


def update_q0_closed_form(w_0, W_s, Q_s, X):
    """
    Closed-form solution for background signature q_0.
    
    Given: X ≈ w_0 q_0^T + W_s Q_s^T
    Solve: min_{q_0} ||X - w_0 q_0^T - W_s Q_s^T||_F^2
    
    Solution: q_0 = (X^T w_0 - Q_s W_s^T w_0) / ||w_0||^2
    
    Parameters
    ----------
    w_0 : array-like, shape (n,)
        Background loading vector (float32)
    W_s : array-like, shape (n, r-1)
        Signal loading matrix (float32)
    Q_s : array-like, shape (p, r-1)
        Signal signature matrix (float32)
    X : array-like, shape (n, p)
        Data matrix (can be sparse, float32)
    
    Returns
    -------
    q_0 : array, shape (p,)
        Background signature vector (float32)
    """
    # Detect backend
    if CUPY_AVAILABLE and isinstance(w_0, cp.ndarray):
        xp = cp
    else:
        xp = np
        if CUPY_AVAILABLE and isinstance(w_0, cp.ndarray):
            w_0 = cp.asnumpy(w_0)
            W_s = cp.asnumpy(W_s)
            Q_s = cp.asnumpy(Q_s)
    
    # Ensure float32
    w_0 = w_0.astype(np.float32 if xp == np else cp.float32)
    W_s = W_s.astype(np.float32 if xp == np else cp.float32)
    Q_s = Q_s.astype(np.float32 if xp == np else cp.float32)
    
    # ||w_0||^2
    w0_norm_sq = xp.dot(w_0, w_0)
    
    # X^T w_0 (handle sparse)
    if issparse(X):
        Xt_w0 = X.T @ w_0
        # Convert to float32 if sparse computation returned different dtype
        if hasattr(Xt_w0, 'dtype') and Xt_w0.dtype != np.float32:
            Xt_w0 = Xt_w0.astype(np.float32)
    else:
        # Ensure X is float32
        if xp == cp:
            X_f32 = X.astype(cp.float32) if X.dtype != cp.float32 else X
        else:
            X_f32 = X.astype(np.float32) if X.dtype != np.float32 else X
        Xt_w0 = X_f32.T @ w_0
    
    # Q_s W_s^T w_0
    Qs_Wst_w0 = Q_s @ (W_s.T @ w_0)
    
    # Closed-form solution
    q_0 = (Xt_w0 - Qs_Wst_w0) / (w0_norm_sq + 1e-10)
    
    # Ensure float32 output
    if xp == cp:
        q_0 = cp.asnumpy(q_0).astype(np.float32)
    else:
        q_0 = q_0.astype(np.float32)
    
    return q_0


def update_Qs_procrustes(W_s, Q_s_prev, X, q_0, w_0):
    """
    Update signal signatures Q_s using Procrustes with orthogonality.
    
    Given: X ≈ w_0 q_0^T + W_s Q_s^T
    Solve: min_{Q_s} ||R - W_s Q_s^T||_F^2  s.t. Q_s^T Q_s = I
           where R = X - w_0 q_0^T
    
    Parameters
    ----------
    W_s : array-like, shape (n, r-1)
        Signal loading matrix (float32)
    Q_s_prev : array-like, shape (p, r-1)
        Previous signal signatures (unused, for API consistency)
    X : array-like, shape (n, p)
        Data matrix (can be sparse, float32)
    q_0 : array-like, shape (p,)
        Background signature (float32)
    w_0 : array-like, shape (n,)
        Background loading (float32)
    
    Returns
    -------
    Q_s : array, shape (p, r-1)
        Updated signal signatures with orthogonality (float32)
    """
    # Convert to numpy for stability
    if CUPY_AVAILABLE and isinstance(W_s, cp.ndarray):
        W_s = cp.asnumpy(W_s)
        w_0 = cp.asnumpy(w_0)
        q_0 = cp.asnumpy(q_0)
    
    # Ensure float32
    W_s = W_s.astype(np.float32)
    w_0 = w_0.astype(np.float32)
    q_0 = q_0.astype(np.float32)
    
    # Compute residual R = X - w_0 q_0^T
    if issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.array(X) if not isinstance(X, np.ndarray) else X
    
    # Ensure float32
    X_dense = X_dense.astype(np.float32)
    
    R = X_dense - np.outer(w_0, q_0)  # (n, p)
    
    # Solve Procrustes: min ||R - W_s Q_s^T||_F s.t. Q_s^T Q_s = I
    Q_s = procrustes(R, W_s, return_transformation=True)
    
    return Q_s.astype(np.float32)


def initialize_Q_from_W(W, X, method='svd'):
    """
    Initialize Q from W using SVD or least squares.
    
    Parameters
    ----------
    W : array-like, shape (n, r)
        Loading matrix (float32)
    X : array-like, shape (n, p)
        Data matrix (can be sparse, float32)
    method : str, default='svd'
        Initialization method: 'svd' or 'lstsq'
    
    Returns
    -------
    Q : array, shape (p, r)
        Initialized signature matrix (float32)
    """
    if CUPY_AVAILABLE and isinstance(W, cp.ndarray):
        xp = cp
        svd_fn = cp_svd
    else:
        xp = np
        svd_fn = svd
    
    # Ensure float32
    W = W.astype(np.float32 if xp == np else cp.float32)
    
    if method == 'svd':
        # W^T X = U S V^T, then Q = V
        if issparse(X):
            WtX = W.T @ X
            # Ensure float32
            if hasattr(WtX, 'dtype') and WtX.dtype != (np.float32 if xp == np else cp.float32):
                WtX = WtX.astype(np.float32 if xp == np else cp.float32)
        else:
            X_f32 = X.astype(np.float32 if xp == np else cp.float32)
            WtX = W.T @ X_f32
        
        U, S, Vt = svd_fn(WtX, full_matrices=False)
        Q = Vt.T  # (p, r)
    
    elif method == 'lstsq':
        # Q^T = (W^T W)^{-1} W^T X
        WtW = W.T @ W
        
        if issparse(X):
            WtX = W.T @ X
            if hasattr(WtX, 'dtype') and WtX.dtype != (np.float32 if xp == np else cp.float32):
                WtX = WtX.astype(np.float32 if xp == np else cp.float32)
        else:
            X_f32 = X.astype(np.float32 if xp == np else cp.float32)
            WtX = W.T @ X_f32
        
        Q = xp.linalg.solve(WtW, WtX).T
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure float32 output
    if xp == cp:
        Q = cp.asnumpy(Q).astype(np.float32)
    else:
        Q = Q.astype(np.float32)
    
    return Q
