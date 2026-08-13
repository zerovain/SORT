"""
Update rules for SymNMF (Memory-efficient version).

Key optimization: Avoid computing Y@Y^T explicitly
Instead use: (Y@Y^T)@G = Y@(Y^T@G)
"""

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class HALSUpdater:
    """
    Hierarchical Alternating Least Squares updater for SymNMF.
    
    Solves: min ||Y@Y^T - W@H^T||_F + alpha*||W - H||_F
    
    Memory-efficient implementation:
    - Never computes Y@Y^T explicitly (would be n×n)
    - Uses chain rule: (Y@Y^T)@G = Y@(Y^T@G)
    """
    
    def __init__(self, alpha=1.0, lambda1=0.0, lambda2=1e-8, 
                 eps=1e-16, backend='auto'):
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eps = eps
        self.backend = backend
    
    def update(self, Y, W, H):
        """
        Update W and H for one iteration.
        
        Parameters:
        -----------
        Y : array, shape (n, d)
            Input matrix (n samples, d features)
        W, H : array, shape (n, r)
            Factor matrices (n samples, r components)
        
        Returns:
        --------
        W_new, H_new : arrays
        
        Note:
        -----
        X = Y @ Y.T is never computed explicitly!
        Instead we use: X @ G = Y @ (Y.T @ G)
        """
        # Determine backend
        if self.backend == 'auto':
            if CUPY_AVAILABLE and isinstance(Y, cp.ndarray):
                xp = cp
            else:
                xp = np
        elif self.backend == 'cupy':
            xp = cp
        else:
            xp = np
        
        # Update W (X @ H where X = Y @ Y.T)
        W = self._update_factor(Y, W, H, H, xp)
        
        # Update H (X @ W where X = Y @ Y.T)
        H = self._update_factor(Y, H, W, W, xp)
        
        return W, H
    
    def _update_factor(self, Y, F, G1, G2, xp):
        """
        Update factor F using HALS with memory-efficient computation.
        
        Parameters:
        -----------
        Y : array, shape (n, d)
            Input matrix
        F : array, shape (n, r)
            Factor to update
        G1 : array, shape (n, r)
            Factor for Gram matrix (G1.T @ G1)
        G2 : array, shape (n, r)
            Factor for symmetry term (alpha * G2)
        xp : module
            numpy or cupy
        
        Mathematical formulation:
        -------------------------
        We want to compute: J = (Y @ Y.T) @ G1
        But Y @ Y.T is (n, n) which is too large!
        
        Solution: Use chain rule
        J = (Y @ Y.T) @ G1 = Y @ (Y.T @ G1)
        
        Steps:
        1. temp = Y.T @ G1  # (d, n) @ (n, r) = (d, r) ✓ small
        2. J = Y @ temp     # (n, d) @ (d, r) = (n, r) ✓ feasible
        """
        n, r = F.shape
        
        # Gram matrix: G1.T @ G1 (r, r) - small, no problem
        G = G1.T @ G1
        
        # Add L2 regularization to diagonal
        if self.lambda2 > 0:
            G.flat[::r+1] += self.lambda2
        
        # ===================================================================
        # KEY: Memory-efficient computation of J = (Y @ Y.T) @ G1
        # ===================================================================
        # Step 1: Compute Y.T @ G1 (d, r) - small intermediate matrix
        YtG1 = Y.T @ G1  # (d, n) @ (n, r) = (d, r)
        
        # Step 2: Compute Y @ (Y.T @ G1) (n, r) - final result
        J = Y @ YtG1  # (n, d) @ (d, r) = (n, r)
        # ===================================================================
        
        # Apply L1 regularization
        if self.lambda1 > 0:
            J -= self.lambda1 / 2
        
        # Column-wise HALS update
        F_new = F.copy()
        
        for j in range(r):
            if self.alpha > 0:  # SymNMF with symmetry constraint
                # numerator = G[j,j] * F[:,j] + J[:,j] - F @ G[:,j] + alpha * G2[:,j]
                # denominator = G[j,j] + alpha
                
                numerator = (G[j, j] * F_new[:, j] + 
                           J[:, j] - 
                           F_new @ G[:, j] + 
                           self.alpha * G2[:, j])
                
                denominator = G[j, j] + self.alpha
                
                F_new[:, j] = xp.maximum(
                    numerator / max(denominator, self.eps), 
                    self.eps
                )
                
            else:  # Standard NMF (no symmetry)
                # update = (J[:,j] - F @ G[:,j]) / G[j,j]
                # F[:,j] = F[:,j] + update
                
                update = (J[:, j] - F_new @ G[:, j]) / max(G[j, j], self.eps)
                F_new[:, j] = xp.maximum(F_new[:, j] + update, self.eps)
        
        return F_new


class MultiplicativeUpdater:
    """
    Multiplicative update rules for SymNMF (memory-efficient).
    
    Update rules:
    W_{ik} ← W_{ik} * sqrt((YY^T W + alpha*H)_{ik} / (WHH^T + alpha*W)_{ik})
    H_{ik} ← H_{ik} * sqrt((YY^T H + alpha*W)_{ik} / (HWW^T + alpha*H)_{ik})
    """
    
    def __init__(self, alpha=1.0, eps=1e-16, backend='auto'):
        self.alpha = alpha
        self.eps = eps
        self.backend = backend
    
    def update(self, Y, W, H):
        """Update W and H using multiplicative rules."""
        # Determine backend
        if self.backend == 'auto':
            if CUPY_AVAILABLE and isinstance(Y, cp.ndarray):
                xp = cp
            else:
                xp = np
        elif self.backend == 'cupy':
            xp = cp
        else:
            xp = np
        
        # Update W
        W = self._update_W(Y, W, H, xp)
        
        # Update H
        H = self._update_H(Y, W, H, xp)
        
        return W, H
    
    def _update_W(self, Y, W, H, xp):
        """
        Update W: W ← W * sqrt(numerator / denominator)
        
        numerator = (Y @ Y.T) @ W + alpha * H
        denominator = W @ H.T @ H + alpha * W
        """
        # Numerator: (Y @ Y.T) @ W = Y @ (Y.T @ W)
        YtW = Y.T @ W  # (d, r)
        numerator = Y @ YtW  # (n, r)
        
        if self.alpha > 0:
            numerator += self.alpha * H
        
        # Denominator: W @ (H.T @ H) + alpha * W
        HtH = H.T @ H  # (r, r)
        denominator = W @ HtH  # (n, r)
        
        if self.alpha > 0:
            denominator += self.alpha * W
        
        # Avoid division by zero
        denominator = xp.maximum(denominator, self.eps)
        
        # Multiplicative update
        W_new = W * xp.sqrt(numerator / denominator)
        
        return W_new
    
    def _update_H(self, Y, W, H, xp):
        """
        Update H: H ← H * sqrt(numerator / denominator)
        
        numerator = (Y @ Y.T) @ H + alpha * W
        denominator = H @ W.T @ W + alpha * H
        """
        # Numerator: (Y @ Y.T) @ H = Y @ (Y.T @ H)
        YtH = Y.T @ H  # (d, r)
        numerator = Y @ YtH  # (n, r)
        
        if self.alpha > 0:
            numerator += self.alpha * W
        
        # Denominator: H @ (W.T @ W) + alpha * H
        WtW = W.T @ W  # (r, r)
        denominator = H @ WtW  # (n, r)
        
        if self.alpha > 0:
            denominator += self.alpha * H
        
        # Avoid division by zero
        denominator = xp.maximum(denominator, self.eps)
        
        # Multiplicative update
        H_new = H * xp.sqrt(numerator / denominator)
        
        return H_new


# Registry of available updaters
UPDATERS = {
    'hals': HALSUpdater,
    'hals_gpu': HALSUpdater,  # Use backend='cupy'
    'multiplicative': MultiplicativeUpdater,
    'mult': MultiplicativeUpdater
}


def get_updater(name, **kwargs):
    """
    Get an updater instance by name.
    
    Parameters:
    -----------
    name : str
        Updater name ('hals', 'hals_gpu', 'multiplicative')
    **kwargs : dict
        Additional arguments passed to updater
    
    Returns:
    --------
    updater : Updater instance
    
    Examples:
    ---------
    >>> updater = get_updater('hals', alpha=1.0, lambda1=1000)
    >>> updater = get_updater('hals_gpu', backend='cupy')
    """
    if name not in UPDATERS:
        raise ValueError(f"Unknown updater: {name}. "
                        f"Available: {list(UPDATERS.keys())}")
    
    updater_class = UPDATERS[name]
    
    # Set backend for GPU version
    if name == 'hals_gpu':
        kwargs.setdefault('backend', 'cupy')
    
    return updater_class(**kwargs)