"""
High-level integration helpers for Total Variation regularization.

Provides TVWeightManager for managing TV weight lifecycle.
"""

from .tv_utils import compute_tv_weight_matrix


class TVWeightManager:
    """
    Manages TV weight computation and updates (IRLS outer loop).
    
    This class encapsulates the state and logic for TV regularization,
    keeping the main model code clean.
    
    Parameters
    ----------
    edges : ndarray, shape (|E|, 2)
        Edge list.
    w_static : ndarray, shape (|E|,)
        Static edge weights.
    epsilon : float, default=1e-6
        TV smoothing parameter.
    backend : {'numpy', 'cupy'}, default='numpy'
        Computing backend.
    
    Attributes
    ----------
    w_tilde : ndarray, shape (|E|, r)
        Current TV dynamic weights.
    
    Examples
    --------
    >>> manager = TVWeightManager(edges, w_static, epsilon=1e-6, backend='cupy')
    >>> manager.update(W)  # Compute initial weights
    >>> # ... optimization loop ...
    >>> manager.update(W)  # Recompute weights based on updated W
    """
    
    def __init__(self, edges, w_static, epsilon=1e-6, backend='numpy'):
        self.edges = edges
        self.w_static = w_static
        self.epsilon = epsilon
        self.backend = backend
        self.w_tilde = None
    
    def update(self, W):
        """
        Update TV weights based on current W.
        
        Parameters
        ----------
        W : ndarray, shape (n, r)
            Current loading matrix.
        """
        self.w_tilde = compute_tv_weight_matrix(
            W, self.edges, self.w_static, self.epsilon, self.backend
        )
    
    def is_initialized(self):
        """
        Check if TV weights have been computed.
        
        Returns
        -------
        bool
            True if w_tilde has been computed.
        """
        return self.w_tilde is not None