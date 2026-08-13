"""
Preprocessing functions for spatial transcriptomics data.

GPU-accelerated with intelligent chunking for large datasets.
"""

import numpy as np
from scipy.spatial import KDTree, Delaunay
from scipy.sparse import csr_matrix, issparse, diags, eye
from anndata import AnnData
from typing import Optional, Literal, Union, Tuple
import warnings

try:
    import cupy as cp
    import cupyx.scipy.sparse as cupy_sparse
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# ============================================================================
# GPU Memory Management Utilities
# ============================================================================

def _get_gpu_memory_info():
    """Get current GPU memory usage in bytes."""
    if not CUPY_AVAILABLE:
        return 0, 0
    
    try:
        mempool = cp.get_default_memory_pool()
        used = mempool.used_bytes()
        total = cp.cuda.Device().mem_info[1]
        return used, total
    except:
        return 0, 0


def _clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if CUPY_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()


# ============================================================================
# NEW: Sparse-Only Smoothing (CPU, Memory-Efficient)
# ============================================================================

def smooth_expression_sparse_cpu(X, A, center_weight=0.5, block_size=50000, verbose=True):
    """
    Spatial smoothing using sparse matrix operations (CPU, blocked).
    
    Y = (1 - center_weight) * A @ X + center_weight * X
    
    **Keeps everything sparse** and processes in blocks to save memory.
    
    Parameters
    ----------
    X : sparse matrix (n, p)
        Expression matrix (must be sparse, CSR format recommended)
    A : sparse matrix (n, n)
        Normalized adjacency matrix (row-stochastic)
    center_weight : float
        Weight for center spot (default 0.5)
        0 = full neighbor smoothing, 1 = no smoothing
    block_size : int
        Block size for row-wise processing (default 50000)
        Set to 0 or negative to process in single block
    verbose : bool
        Print progress
    
    Returns
    -------
    Y : sparse matrix (n, p)
        Smoothed expression (CSR format)
    """
    if not issparse(X):
        raise ValueError("X must be sparse. Use csr_matrix(X) to convert.")
    
    if not issparse(A):
        raise ValueError("A must be sparse. Use csr_matrix(A) to convert.")
    
    n, p = X.shape
    
    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError(f"Shape mismatch: A is {A.shape}, X is {X.shape}")
    
    if verbose:
        print(f"  Sparse smoothing (CPU): {n:,} spots × {p:,} genes")
        print(f"    X density: {X.nnz/X.size:.2%} ({X.nnz:,} non-zeros)")
        print(f"    A density: {A.nnz/A.size:.2%} ({A.nnz:,} non-zeros)")
    
    # Ensure CSR format for efficient row slicing
    X_csr = X.tocsr() if not isinstance(X, csr_matrix) else X
    A_csr = A.tocsr() if not isinstance(A, csr_matrix) else A
    
    # Compute smoothed expression
    neighbor_weight = 1.0 - center_weight
    
    if block_size <= 0 or n <= block_size:
        # Single block (no chunking)
        if verbose:
            print(f"    Processing in single block...")
        
        # Sparse @ Sparse (efficient!)
        neighbor_contrib = neighbor_weight * (A_csr @ X_csr)
        center_contrib = center_weight * X_csr
        Y = neighbor_contrib + center_contrib
        
    else:
        # Multiple blocks (row-wise chunking of A)
        n_blocks = int(np.ceil(n / block_size))
        if verbose:
            print(f"    Processing in {n_blocks} blocks (block_size={block_size:,})...")
        
        from scipy.sparse import vstack
        Y_blocks = []
        
        for block_idx in range(n_blocks):
            start_row = block_idx * block_size
            end_row = min(start_row + block_size, n)
            
            # Extract block rows from A and X
            A_block = A_csr[start_row:end_row, :]  # (block_size, n) sparse
            X_block = X_csr[start_row:end_row, :]  # (block_size, p) sparse
            
            # Compute for this block (sparse operations)
            neighbor_contrib_block = neighbor_weight * (A_block @ X_csr)
            center_contrib_block = center_weight * X_block
            Y_block = neighbor_contrib_block + center_contrib_block
            
            Y_blocks.append(Y_block)
            
            if verbose and (block_idx + 1) % max(1, n_blocks // 10) == 0:
                progress = (block_idx + 1) / n_blocks * 100
                print(f"      Progress: {progress:.1f}% ({block_idx+1}/{n_blocks})")
        
        # Combine blocks vertically
        Y = vstack(Y_blocks, format='csr')
    
    if verbose:
        print(f"    ✓ Smoothed: {Y.nnz:,} non-zeros ({Y.nnz/Y.size:.2%} density)")
    
    return Y


def smooth_expression_from_laplacian_sparse(
    X, L, center_weight=0.5, block_size=50000, verbose=True
):
    """
    Sparse version of smooth_expression_from_laplacian.
    
    Converts Laplacian L to normalized adjacency A, then applies sparse smoothing.
    
    **No conversion to dense** - keeps everything sparse throughout.
    
    Parameters
    ----------
    X : sparse matrix (n, p)
        Expression matrix (must be sparse)
    L : sparse matrix (n, n)
        Graph Laplacian (L = D - A)
    center_weight : float
        Weight for center spot (0=full smoothing, 1=no smoothing)
    block_size : int
        Block size for processing (default 50000)
    verbose : bool
        Print progress
    
    Returns
    -------
    Y : sparse matrix (n, p)
        Smoothed expression (CSR format)
    """
    if not issparse(X):
        raise ValueError("X must be sparse for sparse smoothing")
    
    if not issparse(L):
        raise ValueError("Laplacian L must be sparse")
    
    if verbose:
        print("  Converting Laplacian to normalized adjacency...")
    
    # L = D - A  =>  A = D - L
    # Extract diagonal (degree matrix)
    L_csr = csr_matrix(L)
    degrees = np.array(L_csr.diagonal())
    
    # Create degree matrix (sparse diagonal)
    D = diags(degrees, format='csr')
    
    # Compute adjacency: A = D - L
    A = D - L_csr
    
    # Ensure non-negative (due to numerical precision)
    A.data = np.maximum(A.data, 0)
    A.eliminate_zeros()
    
    # Normalize rows: A_norm[i,j] = A[i,j] / sum(A[i,:])
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    
    D_inv = diags(1.0 / row_sums, format='csr')
    A_normalized = D_inv @ A
    
    if verbose:
        print(f"    ✓ Normalized adjacency: {A_normalized.nnz:,} non-zeros")
    
    # Apply sparse smoothing
    Y = smooth_expression_sparse_cpu(
        X, A_normalized, center_weight, block_size, verbose
    )
    
    return Y


# ============================================================================
# GPU-Accelerated Sparse Matrix Operations
# ============================================================================

def _sparse_matmul_gpu_chunked(
    A_sparse,
    X,
    chunk_size: int = 50000,
    verbose: bool = True
):
    """
    GPU-accelerated sparse @ dense multiplication with chunking.
    
    Strategy:
    - Process A in row chunks to avoid GPU memory overflow
    - Each chunk: A_chunk @ X (sparse @ dense on GPU)
    - Accumulate results
    
    Parameters
    ----------
    A_sparse : sparse matrix
        Shape (n, m), can be on CPU or GPU
    X : ndarray
        Shape (m, p), can be on CPU or GPU (float32)
    chunk_size : int
        Number of rows to process at once
    verbose : bool
        Print progress
    
    Returns
    -------
    Y : cp.ndarray
        Result on GPU, shape (n, p), dtype=float32
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    n, m = A_sparse.shape
    m2, p = X.shape
    
    if m != m2:
        raise ValueError(f"Dimension mismatch: A is {A_sparse.shape}, X is {X.shape}")
    
    # Transfer to GPU if needed
    if not isinstance(A_sparse, cupy_sparse.csr_matrix):
        if verbose:
            print(f"  Transferring A ({A_sparse.shape}, nnz={A_sparse.nnz:,}) to GPU...")
        A_gpu = cupy_sparse.csr_matrix(A_sparse)
    else:
        A_gpu = A_sparse
    
    if not isinstance(X, cp.ndarray):
        if verbose:
            print(f"  Transferring X ({X.shape}) to GPU...")
        # Handle sparse X
        if issparse(X):
            X_gpu = cp.asarray(X.toarray(), dtype=cp.float32)
        else:
            X_gpu = cp.asarray(X, dtype=cp.float32)
    else:
        X_gpu = X.astype(cp.float32) if X.dtype != cp.float32 else X
    
    # Allocate output on GPU
    Y_gpu = cp.zeros((n, p), dtype=cp.float32)
    
    n_chunks = int(np.ceil(n / chunk_size))
    
    if verbose:
        print(f"  GPU chunked matmul: {n:,} rows in {n_chunks} chunks...")
    
    # Process in chunks
    for chunk_idx in range(n_chunks):
        start_row = chunk_idx * chunk_size
        end_row = min(start_row + chunk_size, n)
        
        # Get chunk (efficient on GPU sparse)
        A_chunk = A_gpu[start_row:end_row]
        
        # Sparse @ dense on GPU (fast!)
        Y_chunk = A_chunk @ X_gpu
        
        # Store result
        Y_gpu[start_row:end_row] = Y_chunk
        
        if verbose and (chunk_idx + 1) % max(1, n_chunks // 10) == 0:
            progress = (chunk_idx + 1) / n_chunks * 100
            print(f"    GPU progress: {progress:.1f}% ({chunk_idx+1}/{n_chunks})")
    
    if verbose:
        print(f"  ✓ GPU matmul complete")
    
    return Y_gpu


def _decide_backend_for_smoothing(n_spots, backend='auto', verbose=True):
    """
    Decide which backend to use for smoothing.
    
    Strategy:
    - n < 100k: GPU direct (no chunking)
    - 100k ≤ n < 400k: GPU chunked
    - n ≥ 400k: CPU chunked (safer for very large data)
    
    Returns
    -------
    backend : str
        'gpu', 'gpu_chunked', or 'cpu'
    chunk_size : int or None
    """
    if backend == 'numpy':
        return 'cpu', None
    
    if not CUPY_AVAILABLE:
        if verbose:
            print(f"  CuPy unavailable → Using CPU")
        return 'cpu', None
    
    # Check GPU availability
    try:
        mem_free = cp.cuda.Device(0).mem_info[0] / (1024**3)  # GB
    except:
        if verbose:
            print(f"  GPU unavailable → Using CPU")
        return 'cpu', None
    
    # Decide strategy based on data size
    if n_spots < 100000:
        if verbose:
            print(f"  Small data ({n_spots:,} spots) → Using GPU (direct)")
        return 'gpu', None
    elif n_spots < 400000:
        chunk_size = 50000
        if verbose:
            print(f"  Medium data ({n_spots:,} spots) → Using GPU (chunked {chunk_size:,})")
        return 'gpu_chunked', chunk_size
    else:
        # Very large: use CPU to be safe
        chunk_size = 20000
        if verbose:
            print(f"  Large data ({n_spots:,} spots) → Using CPU (chunked {chunk_size:,})")
        return 'cpu', chunk_size


# ============================================================================
# CPU Blocked Sparse Matrix Operations
# ============================================================================

def _sparse_matmul_cpu_blocked(
    A: csr_matrix,
    X: Union[np.ndarray, csr_matrix],
    block_size: int = 20000,
    verbose: bool = True,
) -> np.ndarray:
    """
    Memory-efficient sparse matrix multiplication on CPU: Y = A @ X
    
    Processes in blocks to handle large matrices.
    
    Parameters
    ----------
    A : csr_matrix
        Sparse matrix (n, m)
    X : ndarray or csr_matrix
        Dense or sparse matrix (m, p), dtype=float32
    block_size : int
        Rows per block
    verbose : bool
        Print progress
    
    Returns
    -------
    Y : np.ndarray
        Result (n, p), dtype=float32
    """
    if block_size is None:
        block_size = 5000
        if verbose:
            print(f"  block_size not specified, using default: {block_size}")

    n, m = A.shape
    m2, p = X.shape if hasattr(X, 'shape') else (m, 1)
    
    if m != m2:
        raise ValueError(f"Matrix dimensions don't match: A is {A.shape}, X is {X.shape}")
    
    n_blocks = int(np.ceil(n / block_size))
    
    if verbose:
        print(f"  CPU blocked matmul: {n:,} rows in {n_blocks} blocks...")
    
    # Allocate output
    Y = np.zeros((n, p), dtype=np.float32)
    
    # Block processing
    for block_idx in range(n_blocks):
        start_row = block_idx * block_size
        end_row = min(start_row + block_size, n)
        
        A_block = A[start_row:end_row]
        Y_block = A_block @ X
        
        if issparse(Y_block):
            Y[start_row:end_row] = Y_block.toarray()
        else:
            Y[start_row:end_row] = Y_block
        
        if verbose and (block_idx + 1) % max(1, n_blocks // 10) == 0:
            progress = (block_idx + 1) / n_blocks * 100
            print(f"    CPU progress: {progress:.1f}% ({block_idx+1}/{n_blocks})")
    
    if verbose:
        print(f"  ✓ CPU matmul complete")
    
    return Y


# Alias for backward compatibility
sparse_matmul_blocked_cpu = _sparse_matmul_cpu_blocked


# ============================================================================
# Updated Smoothing Functions with GPU Support
# ============================================================================

def smooth_expression_from_laplacian(
    X: Union[np.ndarray, csr_matrix, 'cp.ndarray'],
    L: csr_matrix,
    center_weight: float = 0.5,
    backend: str = 'auto',
    verbose: bool = False
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Smooth expression using Laplacian matrix.
    
    **GPU-accelerated with automatic chunking for large datasets.**
    
    **Note**: This function converts sparse X to dense for GPU/general use.
    For pure sparse operations, use `smooth_expression_from_laplacian_sparse()`.
    
    Parameters
    ----------
    X : array-like
        Expression matrix (n, p), dtype=float32
    L : sparse matrix
        Laplacian matrix (n, n)
    center_weight : float
        Weight for center (0=full smoothing, 1=no smoothing)
    backend : str
        'auto', 'cupy', or 'numpy'
    verbose : bool
        Print progress
    
    Returns
    -------
    X_smoothed : array
        Smoothed expression (same type as input), dtype=float32
    """
    # Extract adjacency from Laplacian: A = D - L
    if not issparse(L):
        D = np.diag(L)
        A = np.diag(D) - L
        A = np.maximum(A, 0)
        A = csr_matrix(A)
    else:
        D_diag = L.diagonal()
        D_sparse = diags(D_diag, format='csr')
        A = D_sparse - L
        A.data = np.maximum(A.data, 0)
        A.eliminate_zeros()
    
    return smooth_expression_with_graph(X, A, center_weight, backend, verbose)


def smooth_expression_with_graph(
    X: Union[np.ndarray, csr_matrix, 'cp.ndarray'],
    adjacency: csr_matrix,
    center_weight: float = 0.5,
    backend: str = 'auto',
    verbose: bool = False
) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Smooth expression matrix using graph adjacency.
    
    **GPU-accelerated with automatic strategy selection.**
    
    Formula: X_smooth = α * X + (1-α) * (D^{-1} A) X
    
    Strategy:
    - Small data (n < 100k): GPU direct
    - Medium (100k-400k): GPU chunked
    - Large (≥ 400k): CPU chunked
    
    **Note**: This converts sparse X to dense. For pure sparse, use 
    `smooth_expression_sparse_cpu()` directly.
    
    Parameters
    ----------
    X : array-like
        Expression matrix (n, p), dtype=float32
    adjacency : sparse matrix
        Adjacency matrix (n, n)
    center_weight : float
        Weight for center node
    backend : str
        'auto', 'cupy', or 'numpy'
    verbose : bool
        Print progress
    
    Returns
    -------
    X_smoothed : array
        Smoothed expression (same device as input), dtype=float32
    """
    if not 0 <= center_weight <= 1:
        raise ValueError(f"center_weight must be in [0, 1], got {center_weight}")
    
    n_spots = X.shape[0]
    input_is_gpu = CUPY_AVAILABLE and isinstance(X, cp.ndarray)
    X_is_sparse = issparse(X)
    
    # Decide backend strategy
    strategy, chunk_size = _decide_backend_for_smoothing(n_spots, backend, verbose)
    
    # Compute normalized adjacency: (1-α) * D^{-1} A
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    neighbor_weight = 1.0 - center_weight
    norm_factors = neighbor_weight / degrees
    D_inv = diags(norm_factors, format='csr')
    adjacency_norm = D_inv @ adjacency
    
    # ===== Execute based on strategy =====
    
    if strategy == 'gpu':
        # Direct GPU computation (small data)
        if not input_is_gpu:
            if X_is_sparse:
                X_gpu = cp.asarray(X.toarray(), dtype=cp.float32)
            else:
                X_gpu = cp.asarray(X, dtype=cp.float32)
        else:
            X_gpu = X.astype(cp.float32) if X.dtype != cp.float32 else X
        
        # Transfer adjacency to GPU
        A_gpu = cupy_sparse.csr_matrix(adjacency_norm)
        
        # Compute neighbor average on GPU
        X_neighbor_avg = A_gpu @ X_gpu
        
        # Add center contribution
        X_center = center_weight * X_gpu
        X_smoothed = X_center + X_neighbor_avg
        
        # Convert back if input was CPU
        if not input_is_gpu:
            X_smoothed = cp.asnumpy(X_smoothed)
    
    elif strategy == 'gpu_chunked':
        # GPU with chunking (medium data)
        if not input_is_gpu:
            X_compute = X.toarray().astype(np.float32) if X_is_sparse else X.astype(np.float32)
        else:
            X_compute = X.astype(cp.float32) if X.dtype != cp.float32 else X
        
        # Chunked GPU matmul
        X_neighbor_avg = _sparse_matmul_gpu_chunked(
            adjacency_norm, X_compute, chunk_size=chunk_size, verbose=verbose
        )
        
        # Center contribution
        if not input_is_gpu:
            X_center_cpu = center_weight * X_compute
            X_center = cp.asarray(X_center_cpu, dtype=cp.float32)
        else:
            X_center = center_weight * X_compute
        
        X_smoothed = X_center + X_neighbor_avg
        
        # Convert back if input was CPU
        if not input_is_gpu:
            X_smoothed = cp.asnumpy(X_smoothed)
    
    else:  # 'cpu'
        # CPU with blocking (large data)
        X_cpu = cp.asnumpy(X) if input_is_gpu else X
        
        # Ensure float32
        if X_is_sparse:
            X_cpu = X_cpu.astype(np.float32) if X_cpu.dtype != np.float32 else X_cpu
        else:
            if not isinstance(X_cpu, np.ndarray):
                X_cpu = np.asarray(X_cpu, dtype=np.float32)
            elif X_cpu.dtype != np.float32:
                X_cpu = X_cpu.astype(np.float32)
        
        # Blocked CPU matmul
        X_neighbor_avg = _sparse_matmul_cpu_blocked(
            adjacency_norm, X_cpu, block_size=chunk_size, verbose=verbose
        )
        
        # Add center contribution
        if X_is_sparse:
            X_center = center_weight * X_cpu.toarray()
        else:
            X_center = center_weight * X_cpu
        
        X_smoothed = X_center + X_neighbor_avg
        
        # Convert to GPU if input was GPU
        if input_is_gpu:
            if verbose:
                print(f"  Converting result back to GPU...")
            X_smoothed = cp.asarray(X_smoothed, dtype=cp.float32)
    
    return X_smoothed


# ============================================================================
# High-level AnnData interfaces
# ============================================================================

def build_spatial_graph(
    adata: AnnData,
    spatial_key: str = 'spatial',
    n_neighbors: int = 6,
    radius: Optional[float] = None,
    method: Literal['knn', 'radius', 'delaunay'] = 'knn',
    set_diag: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Build spatial neighborhood graph from coordinates.
    
    Stores adjacency matrix in `adata.obsp['spatial_connectivities']`.
    """
    if copy:
        adata = adata.copy()
    
    if spatial_key not in adata.obsm:
        raise ValueError(
            f"Spatial coordinates not found in .obsm['{spatial_key}']. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    
    coords = np.array(adata.obsm[spatial_key])
    n_obs = coords.shape[0]
    
    print(f"Building spatial graph using method='{method}'...")
    
    # Build graph based on method
    if method == 'knn':
        adjacency, distances = _build_knn_graph(coords, n_neighbors)
    elif method == 'radius':
        if radius is None:
            area = np.prod(coords.max(axis=0) - coords.min(axis=0))
            radius = np.sqrt(area / n_obs) * 1.5
            print(f"  Auto-calculated radius: {radius:.2f}")
        adjacency, distances = _build_radius_graph(coords, radius)
    elif method == 'delaunay':
        if coords.shape[1] != 2:
            raise ValueError("Delaunay triangulation requires 2D coordinates")
        adjacency, distances = _build_delaunay_graph(coords)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'knn', 'radius', 'delaunay'")
    
    if set_diag:
        adjacency.setdiag(1)
    
    # Store results
    adata.obsp['spatial_connectivities'] = adjacency
    adata.obsp['spatial_distances'] = distances
    adata.uns['spatial_graph'] = {
        'method': method,
        'n_neighbors': n_neighbors if method == 'knn' else None,
        'radius': radius if method == 'radius' else None,
        'spatial_key': spatial_key,
        'set_diag': set_diag,
    }
    
    avg_neighbors = adjacency.sum(axis=1).mean()
    print(f"  Graph built: {n_obs} nodes, avg {avg_neighbors:.1f} neighbors/node")
    
    return adata if copy else None


def compute_laplacian(
    adata: AnnData,
    normalized: bool = True,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Compute graph Laplacian from spatial adjacency.
    """
    if copy:
        adata = adata.copy()
    
    if 'spatial_connectivities' not in adata.obsp:
        raise ValueError("Spatial graph not found. Run build_spatial_graph() first.")
    
    A = adata.obsp['spatial_connectivities']
    n = A.shape[0]
    degrees = np.array(A.sum(axis=1)).flatten()
    
    if normalized:
        degrees_inv_sqrt = np.zeros_like(degrees)
        np.power(
            degrees, -0.5, out=degrees_inv_sqrt, where=degrees > 0
        )
        D_inv_sqrt = diags(degrees_inv_sqrt, format='csr')
        L = eye(n, format='csr') - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        D = diags(degrees, format='csr')
        L = D - A
    
    adata.uns['spatial_laplacian'] = L
    adata.uns['laplacian_params'] = {'normalized': normalized}
    
    laplacian_type = "normalized" if normalized else "unnormalized"
    print(f"Computed {laplacian_type} Laplacian")
    
    return adata if copy else None


def smooth_expression(
    adata: AnnData,
    center_weight: float = 0.5,
    layer: Optional[str] = None,
    target_layer: str = 'smoothed',
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Smooth gene expression using spatial graph.
    """
    if copy:
        adata = adata.copy()
    
    if 'spatial_laplacian' not in adata.uns:
        raise ValueError("Spatial Laplacian not found. Run compute_laplacian() first.")
    
    L = adata.uns['spatial_laplacian']
    X = adata.X if layer is None else adata.layers.get(layer)
    
    if X is None:
        raise ValueError(f"Layer '{layer}' not found in adata.layers")
    
    X_smoothed = smooth_expression_from_laplacian(
        X, L, center_weight=center_weight, backend='auto', verbose=True
    )
    
    adata.layers[target_layer] = X_smoothed
    adata.uns['smoothing_params'] = {
        'center_weight': center_weight,
        'source_layer': layer,
    }
    
    print(f"Expression smoothed (center_weight={center_weight}) → .layers['{target_layer}']")
    
    return adata if copy else None


# ============================================================================
# Low-level matrix operations
# ============================================================================

def build_spatial_graph_from_coords(
    coords: np.ndarray,
    radius: Optional[float] = None,
    k_neighbors: Optional[int] = None,
) -> csr_matrix:
    """Build spatial adjacency graph from coordinate array."""
    n = coords.shape[0]
    kdtree = KDTree(coords)
    
    if radius is not None:
        pairs = kdtree.query_pairs(radius, output_type='ndarray')
        if len(pairs) == 0:
            warnings.warn("No edges found with given radius. Try increasing radius.")
            return csr_matrix((n, n), dtype=np.float32)
        
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(rows), dtype=np.float32)
        
    elif k_neighbors is not None:
        distances, indices = kdtree.query(coords, k=k_neighbors+1)
        rows = np.repeat(np.arange(n), k_neighbors)
        cols = indices[:, 1:].flatten()
        data = np.ones(len(rows), dtype=np.float32)
    else:
        raise ValueError("Must specify either radius or k_neighbors")
    
    return csr_matrix((data, (rows, cols)), shape=(n, n))


# ============================================================================
# Private helper functions for graph construction
# ============================================================================

def _build_knn_graph(coords: np.ndarray, k: int) -> tuple:
    """Build k-nearest neighbors graph."""
    n = coords.shape[0]
    kdtree = KDTree(coords)
    distances, indices = kdtree.query(coords, k=k+1)
    
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    
    rows = np.repeat(np.arange(n), k)
    cols = indices.flatten()
    data = np.ones(len(rows), dtype=np.float32)
    
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    adjacency = adjacency + adjacency.T
    adjacency.data = np.ones(len(adjacency.data), dtype=np.float32)
    
    data_dist = distances.flatten().astype(np.float32)
    dist_matrix = csr_matrix((data_dist, (rows, cols)), shape=(n, n))
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    
    return adjacency, dist_matrix


def _build_radius_graph(coords: np.ndarray, radius: float) -> tuple:
    """Build radius-based graph."""
    kdtree = KDTree(coords)
    pairs = kdtree.query_pairs(radius, output_type='ndarray')
    
    if len(pairs) == 0:
        n = coords.shape[0]
        warnings.warn("No edges found with given radius")
        return csr_matrix((n, n), dtype=np.float32), csr_matrix((n, n), dtype=np.float32)
    
    n = coords.shape[0]
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    
    dists = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    data_dist = np.concatenate([dists, dists]).astype(np.float32)
    data = np.ones(len(rows), dtype=np.float32)
    
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    dist_matrix = csr_matrix((data_dist, (rows, cols)), shape=(n, n))
    
    return adjacency, dist_matrix


def _build_delaunay_graph(coords: np.ndarray) -> tuple:
    """Build Delaunay triangulation graph."""
    tri = Delaunay(coords)
    n = coords.shape[0]
    
    edges = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i+1, len(simplex)):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)
    
    if len(edges) == 0:
        warnings.warn("Delaunay triangulation produced no edges")
        return csr_matrix((n, n), dtype=np.float32), csr_matrix((n, n), dtype=np.float32)
    
    edges = np.array(list(edges))
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    
    dists = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    data_dist = np.concatenate([dists, dists]).astype(np.float32)
    data = np.ones(len(rows), dtype=np.float32)
    
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    dist_matrix = csr_matrix((data_dist, (rows, cols)), shape=(n, n))
    
    return adjacency, dist_matrix


# ============================================================================
# TV regularization utilities
# ============================================================================

def extract_edge_list(
    adata: AnnData,
    connectivity_key: str = 'spatial_connectivities',
    distance_key: str = 'spatial_distances',
    use_distances_as_weights: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract edge list and weights from spatial graph."""
    if connectivity_key not in adata.obsp:
        raise ValueError(
            f"Connectivity matrix '{connectivity_key}' not found in .obsp. "
            f"Run build_spatial_graph() first."
        )
    
    adjacency = adata.obsp[connectivity_key]
    
    if not issparse(adjacency):
        adjacency = csr_matrix(adjacency)
    
    adjacency_coo = adjacency.tocoo()
    mask = adjacency_coo.row < adjacency_coo.col
    edges = np.column_stack([adjacency_coo.row[mask], adjacency_coo.col[mask]])
    
    if use_distances_as_weights and distance_key in adata.obsp:
        distances = adata.obsp[distance_key]
        if not issparse(distances):
            distances = csr_matrix(distances)
        distances_coo = distances.tocoo()
        
        edge_distances = distances_coo.data[mask]
        w_static = 1.0 / (edge_distances + 1e-6)
        w_static = w_static / w_static.max()
    else:
        w_static = np.ones(len(edges), dtype=np.float32)
    
    print(f"Extracted edge list: {len(edges)} edges from graph")
    
    return edges, w_static


def store_edge_list(
    adata: AnnData,
    use_distances_as_weights: bool = True,
    copy: bool = False,
) -> Optional[AnnData]:
    """Extract and store edge list in AnnData for TV regularization."""
    if copy:
        adata = adata.copy()
    
    edges, w_static = extract_edge_list(adata, use_distances_as_weights=use_distances_as_weights)
    
    adata.uns['tv_edges'] = edges
    adata.uns['tv_weights'] = w_static
    adata.uns['tv_params'] = {'use_distances_as_weights': use_distances_as_weights}
    
    print(f"Stored edge list in .uns['tv_edges'] and .uns['tv_weights']")
    
    return adata if copy else None
