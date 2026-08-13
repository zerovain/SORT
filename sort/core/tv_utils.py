"""
Total Variation (TV) regularization utilities for spatial constraints.

Implements channel-independent TV with IRLS (Iteratively Reweighted Least Squares).
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from typing import Optional, Tuple

try:
    import cupy as cp
    import cupyx.scipy.sparse as cupy_sparse
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# ============================================================================
# Core TV computation functions
# ============================================================================

def compute_tv_weight_matrix(
    W: np.ndarray,
    edges: np.ndarray,
    w_static: np.ndarray,
    epsilon: float = 1e-6,
    backend: str = 'numpy'
) -> np.ndarray:
    """
    Compute TV dynamic weights for IRLS.
    
    For each edge (i,j) and channel k:
        w_tilde[e, k] = w_static[e] / sqrt((W[i,k] - W[j,k])^2 + epsilon^2)
    
    Parameters
    ----------
    W : ndarray, shape (n, r)
        Current loading matrix.
    edges : ndarray, shape (|E|, 2)
        Edge list where edges[e] = (i, j).
    w_static : ndarray, shape (|E|,)
        Static edge weights (from spatial graph).
    epsilon : float, default=1e-6
        Smoothing parameter to avoid division by zero.
    backend : {'numpy', 'cupy'}, default='numpy'
        Computing backend.
    
    Returns
    -------
    w_tilde : ndarray, shape (|E|, r)
        Dynamic TV weights for each edge and channel.
    
    Examples
    --------
    >>> edges = np.array([[0, 1], [1, 2], [2, 3]])
    >>> w_static = np.ones(3)
    >>> W = np.random.rand(4, 5)
    >>> w_tilde = compute_tv_weight_matrix(W, edges, w_static)
    >>> print(w_tilde.shape)
    (3, 5)
    """
    if backend == 'cupy':
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        xp = cp
        if not isinstance(W, cp.ndarray):
            W = cp.asarray(W)
        if not isinstance(edges, cp.ndarray):
            edges = cp.asarray(edges)
        if not isinstance(w_static, cp.ndarray):
            w_static = cp.asarray(w_static)
    else:
        xp = np
    
    # Extract node indices
    i_idx = edges[:, 0]
    j_idx = edges[:, 1]
    
    # Compute gradient for all edges and channels (vectorized)
    grad = W[i_idx, :] - W[j_idx, :]  # shape: (|E|, r)
    
    # Compute TV weights
    denominator = xp.sqrt(grad**2 + epsilon**2)  # (|E|, r)
    w_tilde = w_static[:, None] / denominator  # Broadcasting: (|E|, 1) / (|E|, r)
    
    return w_tilde


def laplacian_multiply_vectorized(
    W: np.ndarray,
    edges: np.ndarray,
    w_tilde: np.ndarray,
    backend: str = 'numpy'
) -> np.ndarray:
    """
    Compute L_k @ W[:, k] for all k using edge representation (no explicit matrix).
    
    Equivalent to: [L_1 @ W[:,1], L_2 @ W[:,2], ..., L_r @ W[:,r]]
    where L_k is the weighted Laplacian for channel k.
    
    Parameters
    ----------
    W : ndarray, shape (n, r)
        Loading matrix.
    edges : ndarray, shape (|E|, 2)
        Edge list.
    w_tilde : ndarray, shape (|E|, r)
        TV weight matrix.
    backend : {'numpy', 'cupy'}, default='numpy'
        Computing backend.
    
    Returns
    -------
    LW : ndarray, shape (n, r)
        Laplacian multiplication result for all channels.
    
    Notes
    -----
    Memory efficient: O(|E| * r) instead of O(n^2 * r).
    
    Examples
    --------
    >>> LW = laplacian_multiply_vectorized(W, edges, w_tilde)
    """
    if backend == 'cupy':
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        xp = cp
        if not isinstance(W, cp.ndarray):
            W = cp.asarray(W)
        if not isinstance(edges, cp.ndarray):
            edges = cp.asarray(edges)
        if not isinstance(w_tilde, cp.ndarray):
            w_tilde = cp.asarray(w_tilde)
    else:
        xp = np
    
    n, r = W.shape
    i_idx = edges[:, 0]
    j_idx = edges[:, 1]
    
    # Compute edge gradients (vectorized)
    grad = W[i_idx, :] - W[j_idx, :]  # (|E|, r)
    
    # Weight gradients
    weighted_grad = w_tilde * grad  # (|E|, r) element-wise
    
    # Aggregate to nodes using scatter_add
    LW = xp.zeros_like(W)
    
    if backend == 'cupy':
        # CuPy version using cupyx.scatter_add
        import cupyx
        cupyx.scatter_add(LW, i_idx, weighted_grad)
        cupyx.scatter_add(LW, j_idx, -weighted_grad)
    else:
        # NumPy version using add.at
        np.add.at(LW, i_idx, weighted_grad)
        np.add.at(LW, j_idx, -weighted_grad)
    
    return LW


def compute_degree_matrix(
    edges: np.ndarray,
    w_tilde: np.ndarray,
    num_nodes: int,
    backend: str = 'numpy'
) -> np.ndarray:
    """
    Compute degree matrix for each channel.
    
    D[i, k] = sum_{j in N(i)} w_tilde[edge(i,j), k]
    
    Parameters
    ----------
    edges : ndarray, shape (|E|, 2)
        Edge list (undirected graph, each edge appears once).
    w_tilde : ndarray, shape (|E|, r)
        TV weight matrix.
    num_nodes : int
        Number of nodes.
    backend : {'numpy', 'cupy'}, default='numpy'
        Computing backend.
    
    Returns
    -------
    D : ndarray, shape (n, r)
        Degree matrix (diagonal values for each channel).
    
    Examples
    --------
    >>> D = compute_degree_matrix(edges, w_tilde, num_nodes=100)
    """
    if backend == 'cupy':
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        xp = cp
        if not isinstance(edges, cp.ndarray):
            edges = cp.asarray(edges)
        if not isinstance(w_tilde, cp.ndarray):
            w_tilde = cp.asarray(w_tilde)
    else:
        xp = np
    
    r = w_tilde.shape[1]
    D = xp.zeros((num_nodes, r))
    
    i_idx = edges[:, 0]
    j_idx = edges[:, 1]
    
    # For undirected graph, each edge contributes to both endpoints
    if backend == 'cupy':
        import cupyx
        cupyx.scatter_add(D, i_idx, w_tilde)
        cupyx.scatter_add(D, j_idx, w_tilde)
    else:
        np.add.at(D, i_idx, w_tilde)
        np.add.at(D, j_idx, w_tilde)
    
    return D


def clip_tv_weights(
    w_tilde: np.ndarray,
    w_static: np.ndarray,
    max_ratio: float = 1000.0,
    backend: str = 'numpy'
) -> np.ndarray:
    """
    Clip TV weights to avoid numerical instability.
    
    Ensures: w_tilde[e, k] <= max_ratio * w_static[e]
    
    Parameters
    ----------
    w_tilde : ndarray, shape (|E|, r)
        TV weights.
    w_static : ndarray, shape (|E|,)
        Static edge weights.
    max_ratio : float, default=1000.0
        Maximum amplification factor.
    backend : {'numpy', 'cupy'}, default='numpy'
        Computing backend.
    
    Returns
    -------
    w_tilde_clipped : ndarray, shape (|E|, r)
        Clipped weights.
    
    Examples
    --------
    >>> w_tilde_safe = clip_tv_weights(w_tilde, w_static, max_ratio=1000)
    """
    if backend == 'cupy':
        xp = cp
    else:
        xp = np
    
    max_weights = max_ratio * w_static[:, None]  # (|E|, 1)
    w_tilde_clipped = xp.minimum(w_tilde, max_weights)
    
    return w_tilde_clipped


def validate_tv_weights(
    w_tilde: np.ndarray,
    w_static: np.ndarray,
    verbose: bool = True
) -> dict:
    """
    Validate and analyze TV weight statistics.
    
    Parameters
    ----------
    w_tilde : ndarray, shape (|E|, r)
        TV weights.
    w_static : ndarray, shape (|E|,)
        Static edge weights.
    verbose : bool, default=True
        Print statistics.
    
    Returns
    -------
    stats : dict
        Statistics dictionary with keys:
        - 'min', 'max', 'mean': Weight statistics
        - 'amplification_ratio': Max amplification over static weights
        - 'n_large_weights': Number of weights > 100 * w_static
    """
    # Convert to numpy for analysis
    if CUPY_AVAILABLE and isinstance(w_tilde, cp.ndarray):
        w_tilde = cp.asnumpy(w_tilde)
        w_static = cp.asnumpy(w_static)
    
    stats = {
        'min': float(w_tilde.min()),
        'max': float(w_tilde.max()),
        'mean': float(w_tilde.mean()),
        'amplification_ratio': float((w_tilde / w_static[:, None]).max()),
        'n_large_weights': int(np.sum(w_tilde > 100 * w_static[:, None])),
    }
    
    if verbose:
        print("\nTV Weight Statistics:")
        print(f"  Range: [{stats['min']:.2e}, {stats['max']:.2e}]")
        print(f"  Mean: {stats['mean']:.2e}")
        print(f"  Max amplification: {stats['amplification_ratio']:.1f}×")
        print(f"  Large weights (>100× static): {stats['n_large_weights']} / {w_tilde.size}")
    
    return stats


# ============================================================================
# Helper functions for integration with SORT
# ============================================================================

def extract_edge_list_from_laplacian(L: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract edge list and static weights from Laplacian matrix.
    
    Converts Laplacian L = D - A to adjacency A, then extracts edges.
    
    Parameters
    ----------
    L : sparse matrix, shape (n, n)
        Graph Laplacian matrix.
    
    Returns
    -------
    edges : ndarray, shape (|E|, 2)
        Edge list where edges[e] = (i, j) with i < j (undirected).
    w_static : ndarray, shape (|E|,)
        Static edge weights (normalized to [0, 1]).
    
    Examples
    --------
    >>> edges, w_static = extract_edge_list_from_laplacian(L)
    >>> print(f"Graph has {len(edges)} edges")
    """
    # Convert to CSR if needed
    if not isinstance(L, csr_matrix):
        L = csr_matrix(L)
    
    # Extract adjacency: A = D - L
    D = L.diagonal()
    A = -L.copy()
    A.setdiag(A.diagonal() + D)
    
    # Ensure non-negative (numerical errors can cause small negatives)
    A.data = np.maximum(A.data, 0)
    
    # Convert to COO for edge extraction
    A_coo = A.tocoo()
    
    # Extract edges (only upper triangle to avoid duplicates)
    mask = A_coo.row < A_coo.col
    edges = np.column_stack([A_coo.row[mask], A_coo.col[mask]])
    w_static = A_coo.data[mask].astype(np.float32)
    
    # Normalize static weights to [0, 1]
    if w_static.sum() > 0:
        w_static = w_static / w_static.max()
    else:
        w_static = np.ones(len(edges), dtype=np.float32)
    
    return edges, w_static


def prepare_tv_data_structures(L, strategy, verbose=False):
    """
    Prepare TV data structures (edges and static weights) from Laplacian.
    
    Parameters
    ----------
    L : sparse matrix
        Graph Laplacian.
    strategy : str
        Computing strategy ('full_gpu', 'hybrid', 'cpu_only').
    verbose : bool, default=False
        Print information.
    
    Returns
    -------
    edges_backend : ndarray
        Edge list (on appropriate device).
    w_static_backend : ndarray
        Static weights (on appropriate device).
    """
    # Extract edge list
    edges, w_static = extract_edge_list_from_laplacian(L)
    
    if verbose:
        print(f"  Extracted edge list: {len(edges)} edges")
        print(f"  Edge weight range: [{w_static.min():.4f}, {w_static.max():.4f}]")
        avg_degree = 2 * len(edges) / L.shape[0]
        print(f"  Average degree: {avg_degree:.1f}")
    
    # Transfer to appropriate backend
    if strategy in ['full_gpu', 'hybrid']:
        if not CUPY_AVAILABLE:
            raise RuntimeError("GPU strategy requested but CuPy not available")
        edges_backend = cp.asarray(edges)
        w_static_backend = cp.asarray(w_static)
    else:
        edges_backend = edges
        w_static_backend = w_static
    
    return edges_backend, w_static_backend