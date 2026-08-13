"""
Initialization methods for SymNMF.
"""

import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import issparse

try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import svds as cp_svds
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class NNSVDInitializer:
    """
    Non-Negative SVD initialization with Low-Rank Correction (LRC).
    
    References:
    - Boutsidis & Gallopoulos (2008) - NNSVD
    - Atif, Qazi, Gillis (2019) - Low-Rank Correction
    """
    
    def __init__(self, random_state=None, backend='auto', lrc_maxiter=10, lrc_delta=0.05):
        """
        Parameters
        ----------
        random_state : int, optional
            Random seed
        backend : str, default='auto'
            Computing backend: 'auto', 'numpy', 'cupy'
        lrc_maxiter : int, default=10
            Maximum LRC iterations (0 to disable)
        lrc_delta : float, default=0.05
            LRC convergence threshold
        """
        self.random_state = random_state
        self.backend = backend
        self.lrc_maxiter = lrc_maxiter
        self.lrc_delta = lrc_delta
    
    def initialize(self, Y, n_components):
        """
        Initialize W and H from Y using NNSVD+LRC.
        
        Parameters
        ----------
        Y : array, shape (n, p)
            Input matrix (float32, can be sparse or dense)
        n_components : int
            Number of components
        
        Returns
        -------
        W : array, shape (n, r)
            Left factor (float32)
        H : array, shape (n, r)
            Right factor (float32)
        """
        # Determine backend
        if self.backend == 'auto':
            if CUPY_AVAILABLE and isinstance(Y, cp.ndarray):
                xp = cp
                svds_fn = cp_svds
                use_cupy = True
            else:
                xp = np
                svds_fn = svds
                use_cupy = False
        elif self.backend == 'cupy':
            if not CUPY_AVAILABLE:
                raise ValueError("CuPy not available")
            xp = cp
            svds_fn = cp_svds
            use_cupy = True
            if not isinstance(Y, cp.ndarray):
                Y = cp.asarray(Y, dtype=cp.float32)
        else:
            xp = np
            svds_fn = svds
            use_cupy = False
            if CUPY_AVAILABLE and isinstance(Y, cp.ndarray):
                Y = cp.asnumpy(Y)
        
        # Ensure float32
        if issparse(Y):
            # For sparse matrices, keep sparse format
            if Y.dtype != np.float32:
                Y = Y.astype(np.float32)
        else:
            # For dense matrices, convert dtype
            if use_cupy:
                if Y.dtype != cp.float32:
                    Y = Y.astype(cp.float32)
            else:
                if Y.dtype != np.float32:
                    Y = Y.astype(np.float32)
        
        n, p = Y.shape
        r = n_components
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
            if use_cupy:
                cp.random.seed(self.random_state)
        
        # Compute truncated SVD (use sparse SVD if Y is sparse)
        k_svd = min(int(np.ceil(r / 2.0 + 1)), min(n, p) - 1)
        
        try:
            # Use sparse SVD directly
            U, s, Vt = svds_fn(Y, k=k_svd)
            idx = xp.argsort(s)[::-1]
            U = U[:, idx]
            s = s[idx]
        except Exception as e:
            # Only fallback to dense if absolutely necessary
            if issparse(Y):
                raise RuntimeError(
                    f"Sparse SVD failed: {e}. "
                    "Try reducing n_components or use dense matrix."
                )
            
            # Dense SVD fallback
            if use_cupy:
                U, s, Vt = cp.linalg.svd(Y, full_matrices=False)
            else:
                U, s, Vt = np.linalg.svd(Y, full_matrices=False)
            
            U = U[:, :k_svd]
            s = s[:k_svd]
        
        # Eigendecomposition of Y@Y^T
        eigenvalues = s ** 2
        sqrt_eig = xp.sqrt(xp.maximum(eigenvalues, 0))
        
        self.Y_svd = U * sqrt_eig  # (n, k_svd)
        self.Z_svd = sqrt_eig[:, xp.newaxis] * U.T  # (k_svd, n)
        self.xp = xp
        
        # Initialize W and H using NNSVD
        W = xp.zeros((n, r), dtype=xp.float32)
        H = xp.zeros((n, r), dtype=xp.float32)
        
        # First component
        W[:, 0] = xp.abs(self.Y_svd[:, 0])
        H[:, 0] = xp.abs(self.Y_svd[:, 0])
        
        # Remaining components
        i = 1
        j = 1
        
        while i < r:
            if (i + 1) % 2 == 0:
                if j < k_svd:
                    W[:, i] = xp.maximum(self.Y_svd[:, j], 0)
                    H[:, i] = xp.maximum(self.Y_svd[:, j], 0)
                else:
                    if use_cupy:
                        W[:, i] = cp.random.rand(n).astype(cp.float32) * 0.001
                        H[:, i] = cp.random.rand(n).astype(cp.float32) * 0.001
                    else:
                        W[:, i] = np.random.rand(n).astype(np.float32) * 0.001
                        H[:, i] = np.random.rand(n).astype(np.float32) * 0.001
            else:
                j += 1
                if j < k_svd:
                    W[:, i] = xp.maximum(-self.Y_svd[:, j], 0)
                    H[:, i] = xp.maximum(-self.Y_svd[:, j], 0)
                else:
                    if use_cupy:
                        W[:, i] = cp.random.rand(n).astype(cp.float32) * 0.001
                        H[:, i] = cp.random.rand(n).astype(cp.float32) * 0.001
                    else:
                        W[:, i] = np.random.rand(n).astype(np.float32) * 0.001
                        H[:, i] = np.random.rand(n).astype(np.float32) * 0.001
            i += 1
        
        # Scale factors
        W, H = self._scale_factors(W, H)
        
        # LRC refinement
        if self.lrc_maxiter > 0:
            W, H = self._improve_with_lrc(W, H)
        
        return W, H
    
    def _scale_factors(self, W, H):
        """Scale W and H to minimize ||X - WH^T||_F."""
        xp = self.xp
        
        WtY = W.T @ self.Y_svd
        WtYZ = WtY @ self.Z_svd
        
        trace_num = xp.sum(WtYZ * H.T)
        
        WtW = W.T @ W
        HtH = H.T @ H
        trace_den = xp.sum(WtW * HtH)
        
        trace_den_scalar = float(trace_den)
        trace_num_scalar = float(trace_num)
        
        if trace_den_scalar > 1e-10 and trace_num_scalar > 0:
            scale = xp.sqrt(trace_num / trace_den)
            W = W * scale
            H = H * scale
        
        return W, H
    
    def _improve_with_lrc(self, W, H):
        """Improve W and H using LRA-based HALS."""
        errors = [self._compute_error(W, H)]
        
        for k in range(self.lrc_maxiter):
            W = self._lra_hals_update(self.Z_svd.T, self.Y_svd.T, H, W)
            H = self._lra_hals_update(self.Y_svd, self.Z_svd, W, H)
            
            error = self._compute_error(W, H)
            errors.append(error)
            
            if k > 0:
                improvement = float(errors[-2] - errors[-1])
                if improvement <= self.lrc_delta * float(errors[0]):
                    break
        
        return W, H
    
    def _lra_hals_update(self, Y, Z, W, H, eps=1e-16):
        """HALS update for min_{H>=0} ||YZ - WH^T||_F."""
        xp = self.xp
        n, r = H.shape
        
        WtW = W.T @ W
        WtY = W.T @ Y
        WtYZ = WtY @ Z
        
        H_new = H.copy()
        
        for j in range(r):
            residual = WtYZ[j, :] - H_new @ WtW[:, j]
            denom = xp.maximum(WtW[j, j], eps)
            
            H_new[:, j] = H_new[:, j] + residual / denom
            H_new[:, j] = xp.maximum(H_new[:, j], 0)
        
        return H_new
    
    def _compute_error(self, W, H):
        """Compute relative error ||X - WH^T||_F / ||X||_F."""
        xp = self.xp
        
        YtY = self.Y_svd.T @ self.Y_svd
        ZZt = self.Z_svd @ self.Z_svd.T
        norm_X_sq = xp.sum(YtY * ZZt)
        
        WtY = W.T @ self.Y_svd
        WtYZ = WtY @ self.Z_svd
        trace_WXH = xp.sum(WtYZ * H.T)
        
        WtW = W.T @ W
        HtH = H.T @ H
        trace_WWHH = xp.sum(WtW * HtH)
        
        error_sq = norm_X_sq - 2 * trace_WXH + trace_WWHH
        error_sq = xp.maximum(error_sq, 0)
        
        return float(xp.sqrt(error_sq / norm_X_sq))