"""
Memory management and strategy selection utilities.

Strategies:
- n < 200k: Direct computation, X on GPU
- 200k ≤ n < 400k: Chunked (100k), X on GPU
- n ≥ 400k: Chunked (150k), X on CPU
"""

import numpy as np
from scipy.sparse import issparse
from typing import Optional, Tuple, Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def check_gpu_memory() -> Optional[dict]:
    """
    Query GPU memory status.
    
    Returns
    -------
    dict or None
        GPU memory info or None if unavailable
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    
    try:
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        
        return {
            'total_GB': total_memory,
            'allocated_GB': allocated,
            'cached_GB': cached,
            'free_GB': total_memory - cached,
            'device_name': gpu_props.name
        }
    except Exception:
        return None


def estimate_memory_usage(
    X, 
    W=None, 
    Q=None, 
    L=None
) -> dict:
    """
    Estimate memory requirements for matrices.
    
    Parameters
    ----------
    X : array-like, shape (n, p)
        Data matrix
    W : array-like, shape (n, r), optional
        Loading matrix (float32)
    Q : array-like, shape (p, r), optional
        Signature matrix (float32)
    L : array-like, shape (n, n), optional
        Laplacian matrix (float32)
    
    Returns
    -------
    dict
        Memory estimates
    """
    memory_dict = {}
    
    # X (float32 for sparse)
    if issparse(X):
        nnz = X.nnz
        dtype_size = 4  # float32
        X_memory_gb = (nnz * dtype_size + nnz * 4 + (X.shape[0] + 1) * 4) / 1e9
        sparsity_pct = (1 - nnz / (X.shape[0] * X.shape[1])) * 100
        
        memory_dict['X'] = {
            'memory_gb': X_memory_gb,
            'description': f"{X_memory_gb:.3f} GB (sparse float32, {sparsity_pct:.1f}% zeros)"
        }
    else:
        dtype_size = 4  # float32
        X_memory_gb = X.size * dtype_size / 1e9
        memory_dict['X'] = {
            'memory_gb': X_memory_gb,
            'description': f"{X_memory_gb:.3f} GB (dense)"
        }
    
    # W, Q (float32)
    if W is not None:
        W_memory_gb = W.nbytes / 1e9
        memory_dict['W'] = {
            'memory_gb': W_memory_gb,
            'description': f"{W_memory_gb*1000:.1f} MB"
        }
    
    if Q is not None:
        Q_memory_gb = Q.nbytes / 1e9
        memory_dict['Q'] = {
            'memory_gb': Q_memory_gb,
            'description': f"{Q_memory_gb*1000:.1f} MB"
        }
    
    # L (sparse float32)
    if L is not None:
        if issparse(L):
            L_memory_gb = (L.nnz * 12) / 1e9
            memory_dict['L'] = {
                'memory_gb': L_memory_gb,
                'description': f"{L_memory_gb*1000:.1f} MB (sparse)"
            }
        else:
            L_memory_gb = L.nbytes / 1e9
            memory_dict['L'] = {
                'memory_gb': L_memory_gb,
                'description': f"{L_memory_gb:.2f} GB"
            }
    
    # Temporary matrices
    if W is not None and Q is not None:
        n, p = X.shape
        r = Q.shape[1]
        temp_memory_gb = (n * r * 4) / 1e9
        memory_dict['temp'] = {
            'memory_gb': temp_memory_gb,
            'description': f"{temp_memory_gb*1000:.1f} MB (temporary)"
        }
    
    # Total with safety margin
    total_gb = sum(v['memory_gb'] for v in memory_dict.values() if 'memory_gb' in v)
    memory_dict['total_gb'] = total_gb * 2
    
    return memory_dict


def select_backend_strategy(
    n_spots: int,
    n_features: int,
    n_components: int,
    backend: str = 'auto',
    available_memory_gb: Optional[float] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Select optimal backend and computation strategy.
    
    This is the main entry point for strategy selection.
    
    Parameters
    ----------
    n_spots : int
        Number of spots/cells
    n_features : int
        Number of features/genes
    n_components : int
        Number of components
    backend : str
        'auto', 'cupy', 'numpy'
    available_memory_gb : float, optional
        Available GPU memory in GB
    verbose : bool
        Print strategy information
    
    Returns
    -------
    dict
        Strategy configuration with keys:
        - backend: 'cupy' or 'numpy'
        - use_chunking: bool
        - chunk_size: int or None
        - X_location: 'gpu' or 'cpu'
        - W_location: 'gpu' or 'cpu'
        - Q_location: 'gpu' or 'cpu'
        - reason: str
    """
    # Force CPU backend
    if backend == 'numpy':
        strategy = {
            'backend': 'numpy',
            'use_chunking': False,
            'chunk_size': None,
            'X_location': 'cpu',
            'W_location': 'cpu',
            'Q_location': 'cpu',
            'L_location': 'cpu',
            'reason': 'User requested numpy backend'
        }
        if verbose:
            print(f"Backend strategy: {strategy['reason']}")
        return strategy
    
    # Check GPU availability
    if not CUPY_AVAILABLE or not TORCH_AVAILABLE:
        strategy = {
            'backend': 'numpy',
            'use_chunking': False,
            'chunk_size': None,
            'X_location': 'cpu',
            'W_location': 'cpu',
            'Q_location': 'cpu',
            'L_location': 'cpu',
            'reason': 'GPU not available'
        }
        if verbose:
            print(f"Backend strategy: {strategy['reason']}")
        return strategy
    
    # Check GPU memory
    if available_memory_gb is None:
        gpu_info = check_gpu_memory()
        if gpu_info is not None:
            available_memory_gb = gpu_info['free_GB']
        else:
            # GPU not available, use CPU
            strategy = {
                'backend': 'numpy',
                'use_chunking': False,
                'chunk_size': None,
                'X_location': 'cpu',
                'W_location': 'cpu',
                'Q_location': 'cpu',
                'L_location': 'cpu',
                'reason': 'GPU memory unavailable'
            }
            if verbose:
                print(f"Backend strategy: {strategy['reason']}")
            return strategy
    
    # Estimate memory requirements (float32)
    # Rough estimates: sparse X (1% density), W (n × r), Q (p × r)
    sparsity = 0.01  # Assume 1% density for sparse X
    X_memory_gb = (n_spots * n_features * sparsity * 4) / 1e9  # float32
    W_memory_gb = (n_spots * n_components * 4) / 1e9  # float32
    Q_memory_gb = (n_features * n_components * 4) / 1e9  # float32
    L_memory_gb = (n_spots * 8 * 12) / 1e9  # Sparse (8 neighbors avg)
    
    total_memory_gb = X_memory_gb + W_memory_gb + Q_memory_gb + L_memory_gb
    
    # Strategy selection based on data size
    if n_spots < 200000:
        # Small data: everything on GPU
        if total_memory_gb * 1.5 < available_memory_gb:
            strategy = {
                'backend': 'cupy',
                'use_chunking': False,
                'chunk_size': None,
                'X_location': 'gpu',
                'W_location': 'gpu',
                'Q_location': 'gpu',
                'L_location': 'gpu',
                'reason': f'Small data (n={n_spots:,}), all on GPU'
            }
        else:
            # Not enough GPU memory, use CPU
            strategy = {
                'backend': 'numpy',
                'use_chunking': False,
                'chunk_size': None,
                'X_location': 'cpu',
                'W_location': 'cpu',
                'Q_location': 'cpu',
                'L_location': 'cpu',
                'reason': f'Insufficient GPU memory ({available_memory_gb:.1f} GB available)'
            }
    
    elif 200000 <= n_spots < 400000:
        # Medium data: chunked computation, X on GPU
        chunk_size = 100000
        chunk_memory_gb = (chunk_size * n_features * sparsity * 4) / 1e9
        
        if (chunk_memory_gb + W_memory_gb + Q_memory_gb + L_memory_gb) * 1.5 < available_memory_gb:
            strategy = {
                'backend': 'cupy',
                'use_chunking': True,
                'chunk_size': chunk_size,
                'X_location': 'gpu',
                'W_location': 'gpu',
                'Q_location': 'gpu',
                'L_location': 'gpu',
                'reason': f'Medium data (n={n_spots:,}), chunked GPU computation'
            }
        else:
            # Use CPU
            strategy = {
                'backend': 'numpy',
                'use_chunking': False,
                'chunk_size': None,
                'X_location': 'cpu',
                'W_location': 'cpu',
                'Q_location': 'cpu',
                'L_location': 'cpu',
                'reason': f'Insufficient GPU memory for chunked computation'
            }
    
    else:
        # Large data: X on CPU, W/Q/L on GPU
        chunk_size = 150000
        
        if (W_memory_gb + Q_memory_gb + L_memory_gb) * 1.5 < available_memory_gb:
            strategy = {
                'backend': 'cupy',
                'use_chunking': True,
                'chunk_size': chunk_size,
                'X_location': 'cpu',
                'W_location': 'gpu',
                'Q_location': 'gpu',
                'L_location': 'gpu',
                'reason': f'Large data (n={n_spots:,}), X on CPU, chunked GPU updates'
            }
        else:
            # Use CPU
            strategy = {
                'backend': 'numpy',
                'use_chunking': False,
                'chunk_size': None,
                'X_location': 'cpu',
                'W_location': 'cpu',
                'Q_location': 'cpu',
                'L_location': 'cpu',
                'reason': f'Insufficient GPU memory, using CPU'
            }
    
    if verbose:
        print(f"\nBackend Strategy:")
        print(f"  Backend: {strategy['backend']}")
        print(f"  Chunking: {strategy['use_chunking']}")
        if strategy['use_chunking']:
            print(f"  Chunk size: {strategy['chunk_size']:,}")
        print(f"  X location: {strategy['X_location']}")
        print(f"  W location: {strategy['W_location']}")
        print(f"  Q location: {strategy['Q_location']}")
        print(f"  Reason: {strategy['reason']}")
        if available_memory_gb is not None:
            print(f"  GPU memory available: {available_memory_gb:.1f} GB")
            print(f"  Estimated total memory: {total_memory_gb:.1f} GB")
    
    return strategy


def get_data_placement_strategy(
    X_sparse,
    n_components: int,
    backend: str = 'auto',
    verbose: bool = False
) -> dict:
    """
    Determine optimal data placement strategy (legacy interface).
    
    This function provides backward compatibility.
    New code should use select_backend_strategy().
    
    Parameters
    ----------
    X_sparse : sparse matrix
        Data matrix
    n_components : int
        Number of components
    backend : str
        'auto', 'gpu', or 'cpu'
    verbose : bool
        Print information
    
    Returns
    -------
    dict
        Strategy configuration
    """
    n, p = X_sparse.shape
    
    # Map old backend names
    if backend == 'gpu':
        backend = 'cupy'
    elif backend == 'cpu':
        backend = 'numpy'
    
    strategy = select_backend_strategy(
        n_spots=n,
        n_features=p,
        n_components=n_components,
        backend=backend,
        verbose=verbose
    )
    
    # Add legacy keys for backward compatibility
    strategy['use_chunked_matmul'] = strategy['use_chunking']
    
    return strategy


def should_use_chunked_matmul(n_spots: int, backend: str = 'auto') -> Tuple[bool, Optional[int]]:
    """
    Determine if chunked matmul should be used.
    
    Parameters
    ----------
    n_spots : int
        Number of spots
    backend : str
        'cpu', 'gpu', or 'auto'
    
    Returns
    -------
    use_chunked : bool
    chunk_size : int or None
    """
    if backend == 'cpu' or backend == 'numpy':
        return False, None
    
    if n_spots < 200000:
        return False, None
    elif n_spots < 400000:
        return True, 100000
    else:
        return True, 150000


def get_optimal_chunk_size(n_spots: int, available_memory_gb: Optional[float] = None) -> int:
    """
    Calculate optimal chunk size.
    
    Parameters
    ----------
    n_spots : int
        Number of spots
    available_memory_gb : float, optional
        Available GPU memory in GB
    
    Returns
    -------
    chunk_size : int
    """
    if available_memory_gb is None:
        if n_spots < 200000:
            return n_spots
        elif n_spots < 400000:
            return 100000
        else:
            return 150000
    else:
        # Adaptive: chunk should use < 20% of available memory
        # Heuristic: chunk_size × 3000 genes × 4 bytes (float32)
        target_mem_gb = available_memory_gb * 0.2
        chunk_size = int(target_mem_gb * 1e9 / (3000 * 4))
        return max(50000, min(chunk_size, 200000))


def print_memory_strategy(strategy: Dict[str, Any], n_spots: int, n_features: int, n_components: int):
    """
    Print memory strategy summary.
    
    Parameters
    ----------
    strategy : dict
        Strategy from select_backend_strategy()
    n_spots : int
        Number of spots
    n_features : int
        Number of features
    n_components : int
        Number of components
    """
    print("\n" + "="*70)
    print("Memory Management Strategy")
    print("="*70)
    print(f"Data size: {n_spots:,} spots × {n_features:,} features")
    print(f"Components: {n_components}")
    print(f"\nStrategy:")
    print(f"  Backend: {strategy['backend'].upper()}")
    print(f"  Chunking: {'Yes' if strategy['use_chunking'] else 'No'}")
    if strategy['use_chunking']:
        print(f"  Chunk size: {strategy['chunk_size']:,} spots")
    print(f"\nData placement:")
    print(f"  X: {strategy['X_location'].upper()}")
    print(f"  W: {strategy['W_location'].upper()}")
    print(f"  Q: {strategy['Q_location'].upper()}")
    print(f"  L: {strategy['L_location'].upper()}")
    print(f"\nReason: {strategy['reason']}")
    print("="*70 + "\n")