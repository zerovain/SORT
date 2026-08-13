"""
Sparse matrix utilities and conversions with GPU optimization.

Key features:
- Chunked GPU computation for large datasets
- Automatic data placement management

Thresholds:
- n < 200k: Direct computation
- 200k ≤ n < 400k: Chunked computation, X on GPU
- n ≥ 400k: Chunked computation, X on CPU
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix, csc_matrix, spmatrix
from typing import Union, Literal, Optional

try:
    import cupy as cp
    import cupyx.scipy.sparse as cupy_sparse
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cupy_sparse = None


# ============================================================================
# GPU Data Manager
# ============================================================================

class GPUDataManager:
    """Manage data placement and transfer for GPU computation."""
    
    def __init__(
        self, 
        X_sparse: spmatrix,
        backend: Literal['gpu', 'cpu'] = 'gpu',
        n_threshold_gpu: int = 400000,
        verbose: bool = False
    ):
        """
        Parameters
        ----------
        X_sparse : scipy.sparse matrix
            Sparse data matrix on CPU (float32)
        backend : {'gpu', 'cpu'}
            Computing backend
        n_threshold_gpu : int
            Threshold for keeping X on GPU (default: 400k)
        verbose : bool
            Print placement information
        """
        if not issparse(X_sparse):
            raise ValueError("X_sparse must be a scipy sparse matrix")
        
        self.X_cpu = X_sparse.tocsr()
        
        # Ensure float32
        if self.X_cpu.dtype != np.float32:
            self.X_cpu = self.X_cpu.astype(np.float32)
        
        self.n, self.p = self.X_cpu.shape
        self.backend = backend
        self.n_threshold_gpu = n_threshold_gpu
        self.verbose = verbose
        
        if backend == 'gpu' and CUPY_AVAILABLE and self.n < n_threshold_gpu:
            self.X_gpu = cupy_sparse.csr_matrix(self.X_cpu)
            self.X_on_gpu = True
            if verbose:
                mem_gb = (self.X_cpu.nnz * 4 + self.X_cpu.nnz * 4 + (self.n + 1) * 4) / 1e9
                print(f"[GPU Manager] X on GPU (float32): n={self.n:,}, memory={mem_gb:.2f} GB")
        else:
            self.X_gpu = None
            self.X_on_gpu = False
            if backend == 'gpu' and verbose:
                print(f"[GPU Manager] X on CPU: n={self.n:,} ≥ {n_threshold_gpu:,}")
    
    def get_X(self) -> Union[spmatrix, 'cupy_sparse.csr_matrix']:
        """Get X in appropriate format."""
        return self.X_gpu if self.X_on_gpu else self.X_cpu
    
    def is_on_gpu(self) -> bool:
        """Check if X is stored on GPU."""
        return self.X_on_gpu
    
    def get_memory_info(self) -> dict:
        """Get memory usage information."""
        nnz = self.X_cpu.nnz
        dtype_size = 4  # float32
        mem_bytes = nnz * dtype_size + nnz * 4 + (self.n + 1) * 4
        
        return {
            'shape': (self.n, self.p),
            'nnz': nnz,
            'sparsity': 1 - nnz / (self.n * self.p),
            'dtype': np.float32,
            'memory_gb': mem_bytes / 1e9,
            'location': 'GPU' if self.X_on_gpu else 'CPU'
        }
    
    def free_gpu_memory(self):
        """Free GPU memory if X was stored there."""
        if self.X_on_gpu and self.X_gpu is not None:
            del self.X_gpu
            self.X_gpu = None
            self.X_on_gpu = False
            if CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()


# ============================================================================
# Chunked GPU Matrix Multiplication
# ============================================================================

def sparse_dense_matmul_chunked_gpu(
    X_cpu_sparse: spmatrix,
    Y_gpu_dense: 'cp.ndarray',
    chunk_size: Union[int, Literal['auto']] = 'auto',
    transpose_X: bool = False,
    verbose: bool = False
) -> 'cp.ndarray':
    """
    GPU-optimized sparse-dense matmul with chunked CPU→GPU transfer.
    
    X stays on CPU (float32), transfer chunks to GPU for computation.
    
    Parameters
    ----------
    X_cpu_sparse : scipy.sparse.csr_matrix
        Sparse matrix on CPU (float32), shape (n, p)
    Y_gpu_dense : cupy.ndarray
        Dense matrix on GPU, shape (n, r) or (p, r), dtype=float32
    chunk_size : int or 'auto'
        Rows per batch. 'auto': n<200k→no chunk, 200k≤n<400k→100k, n≥400k→150k
    transpose_X : bool
        If True, compute X.T @ Y
    verbose : bool
        Print progress
    
    Returns
    -------
    result : cupy.ndarray
        Result on GPU, dtype=float32
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    
    if not issparse(X_cpu_sparse):
        raise ValueError("X_cpu_sparse must be scipy sparse matrix")
    
    n, p = X_cpu_sparse.shape
    
    if not isinstance(X_cpu_sparse, csr_matrix):
        X_cpu_sparse = X_cpu_sparse.tocsr()
    
    # Ensure float32
    if X_cpu_sparse.dtype != np.float32:
        X_cpu_sparse = X_cpu_sparse.astype(np.float32)
    
    if not isinstance(Y_gpu_dense, cp.ndarray):
        Y_gpu_dense = cp.asarray(Y_gpu_dense, dtype=cp.float32)
    elif Y_gpu_dense.dtype != cp.float32:
        Y_gpu_dense = Y_gpu_dense.astype(cp.float32)
    
    # Determine chunk size
    if chunk_size == 'auto':
        if n < 200000:
            chunk_size = n
        elif n < 400000:
            chunk_size = 100000
        else:
            chunk_size = 150000
    
    # Fast path: no chunking
    if chunk_size >= n:
        X_gpu = cupy_sparse.csr_matrix(X_cpu_sparse)
        result = X_gpu.T @ Y_gpu_dense if transpose_X else X_gpu @ Y_gpu_dense
        del X_gpu
        return result
    
    # Chunked computation
    if verbose:
        print(f"[Chunked MatMul] n={n:,}, chunk={chunk_size:,}")
    
    n_chunks = int(np.ceil(n / chunk_size))
    
    if transpose_X:
        result = cp.zeros((p, Y_gpu_dense.shape[1]), dtype=cp.float32)
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n)
            
            X_chunk_cpu = X_cpu_sparse[start_idx:end_idx, :]
            Y_chunk_gpu = Y_gpu_dense[start_idx:end_idx, :]
            
            X_chunk_gpu = cupy_sparse.csr_matrix(X_chunk_cpu)
            result += X_chunk_gpu.T @ Y_chunk_gpu
            del X_chunk_gpu
            
            if verbose and (i + 1) % max(1, n_chunks // 10) == 0:
                print(f"  {i+1}/{n_chunks} ({100*(i+1)/n_chunks:.0f}%)")
    
    else:
        result = cp.zeros((n, Y_gpu_dense.shape[1]), dtype=cp.float32)
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n)
            
            X_chunk_cpu = X_cpu_sparse[start_idx:end_idx, :]
            X_chunk_gpu = cupy_sparse.csr_matrix(X_chunk_cpu)
            result[start_idx:end_idx, :] = X_chunk_gpu @ Y_gpu_dense
            del X_chunk_gpu
            
            if verbose and (i + 1) % max(1, n_chunks // 10) == 0:
                print(f"  {i+1}/{n_chunks} ({100*(i+1)/n_chunks:.0f}%)")
    
    return result


# ============================================================================
# Memory Estimation
# ============================================================================

def estimate_gpu_memory_usage(
    X_sparse: spmatrix,
    n_components: int,
    include_X_on_gpu: bool = True
) -> dict:
    """
    Estimate GPU memory usage.
    
    Parameters
    ----------
    X_sparse : sparse matrix
        Data matrix
    n_components : int
        Number of components
    include_X_on_gpu : bool
        Whether to include X in GPU memory estimate
    
    Returns
    -------
    dict
        Memory estimates with keys: X_gpu_gb, W_Q_L_gb, total_with_X_gb,
        total_without_X_gb, recommend_X_on_cpu
    """
    n, p = X_sparse.shape
    r = n_components
    nnz = X_sparse.nnz
    
    # X on GPU (sparse CSR, float32)
    X_gpu_mem = nnz * 4 + nnz * 4 + (n + 1) * 4
    
    # W, Q, L on GPU (float32)
    W_mem = n * r * 4
    Q_mem = p * r * 4
    L_mem = nnz * 12
    WQL_mem = W_mem + Q_mem + L_mem
    
    X_gpu_gb = X_gpu_mem / 1e9
    WQL_gb = WQL_mem / 1e9
    total_with_X_gb = (X_gpu_mem + WQL_mem) / 1e9
    
    return {
        'X_gpu_gb': X_gpu_gb,
        'W_Q_L_gb': WQL_gb,
        'total_with_X_gb': total_with_X_gb,
        'total_without_X_gb': WQL_gb,
        'recommend_X_on_cpu': total_with_X_gb > 6.0
    }


# ============================================================================
# Backend Conversion
# ============================================================================

def convert_to_backend(
    X: Union[np.ndarray, spmatrix],
    backend: Literal['numpy', 'cupy', 'torch'] = 'numpy',
    sparse_format: Literal['csr', 'csc'] = 'csr',
    dtype: Optional[np.dtype] = None
) -> Union[np.ndarray, spmatrix, 'cp.ndarray']:
    """
    Convert matrix to specified backend.
    
    Parameters
    ----------
    X : array-like
        Input matrix
    backend : {'numpy', 'cupy', 'torch'}
        Target backend
    sparse_format : {'csr', 'csc'}
        Sparse format
    dtype : dtype, optional
        Target dtype (default: float32)
    """
    if backend == 'numpy':
        if issparse(X):
            X_out = X.tocsr() if sparse_format == 'csr' else X.tocsc()
        else:
            X_out = np.asarray(X)
        
        if dtype is not None and X_out.dtype != dtype:
            X_out = X_out.astype(dtype)
        
        return X_out
    
    elif backend == 'cupy':
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        
        if issparse(X):
            if dtype is not None and X.dtype != dtype:
                X = X.astype(dtype)
            
            return cupy_sparse.csr_matrix(X) if sparse_format == 'csr' else cupy_sparse.csc_matrix(X)
        else:
            X_out = cp.asarray(X)
            if dtype is not None:
                cp_dtype = getattr(cp, str(np.dtype(dtype).name))
                X_out = X_out.astype(cp_dtype)
            return X_out
    
    elif backend == 'torch':
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch not available")
        
        if issparse(X):
            X_coo = X.tocoo()
            if dtype is not None and X_coo.dtype != dtype:
                X_coo = X_coo.astype(dtype)
            
            indices = torch.LongTensor(np.vstack([X_coo.row, X_coo.col]))
            values = torch.FloatTensor(X_coo.data)
            return torch.sparse.FloatTensor(indices, values, X_coo.shape)
        else:
            torch_dtype = torch.float32 if dtype is None else getattr(torch, str(np.dtype(dtype).name))
            return torch.tensor(X, dtype=torch_dtype)
    
    else:
        raise ValueError(f"Unknown backend: {backend}")


def sparse_matmul(
    A: Union[np.ndarray, spmatrix],
    B: Union[np.ndarray, spmatrix],
    backend: str = 'auto'
) -> Union[np.ndarray, 'cp.ndarray']:
    """Unified sparse matrix multiplication across backends."""
    if backend == 'auto':
        if CUPY_AVAILABLE and (isinstance(A, cp.ndarray) or isinstance(A, cupy_sparse.spmatrix)):
            backend = 'cupy'
        else:
            backend = 'numpy'
    
    if backend == 'cupy':
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        
        if cupy_sparse.issparse(A):
            B_cp = cp.asarray(B) if not isinstance(B, cp.ndarray) else B
            return A @ B_cp
        else:
            A_cp = cp.asarray(A) if not isinstance(A, cp.ndarray) else A
            B_cp = cp.asarray(B) if not isinstance(B, cp.ndarray) else B
            return A_cp @ B_cp
    
    else:
        if issparse(A):
            B_np = np.asarray(B) if not isinstance(B, np.ndarray) else B
            return A @ B_np
        else:
            A_np = np.asarray(A) if not isinstance(A, np.ndarray) else A
            B_np = np.asarray(B) if not isinstance(B, np.ndarray) else B
            return A_np @ B_np


def optimize_sparse_format(
    X: spmatrix,
    operation: Literal['row_slicing', 'col_slicing', 'matmul_left', 'matmul_right'] = 'row_slicing'
) -> spmatrix:
    """Choose optimal sparse matrix format for operation."""
    if not issparse(X):
        return X
    
    if operation in ['row_slicing', 'matmul_left']:
        return X.tocsr()
    elif operation in ['col_slicing', 'matmul_right']:
        return X.tocsc()
    else:
        return X