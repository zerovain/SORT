"""
General utility functions for SORT.

Provides helper functions for data manipulation, logging, and model I/O.
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import issparse, spmatrix
from typing import Optional, Dict, Any, Union


def subsample_indices(n: int, n_sub: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random subsample indices.
    
    Parameters
    ----------
    n : int
        Total number of samples.
    n_sub : int
        Desired subsample size.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    indices : ndarray
        Random subsample indices (sorted).
    
    Examples
    --------
    >>> idx = subsample_indices(1000, 100, seed=42)
    >>> print(len(idx))  # 100
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    n_sub = min(n_sub, n)
    indices = rng.choice(n, n_sub, replace=False)
    indices.sort()  # Sort for better memory access patterns
    return indices


def compute_relative_error(
    X: Union[np.ndarray, spmatrix],
    W: np.ndarray,
    Q: np.ndarray,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> float:
    """
    Compute relative reconstruction error: ||X - WQ^T||_F / ||X||_F.
    
    Parameters
    ----------
    X : array-like, shape (n, p)
        Data matrix (can be sparse).
    W : ndarray, shape (n, r)
        Loading matrix.
    Q : ndarray, shape (p, r)
        Signature matrix.
    sample_size : int, optional
        If provided, compute on a random subsample (for large X).
    seed : int, optional
        Random seed for subsampling.
    
    Returns
    -------
    relative_error : float
        Relative Frobenius norm error.
    
    Examples
    --------
    >>> error = compute_relative_error(X, W, Q)
    >>> print(f"Reconstruction error: {error:.2%}")
    """
    # Subsample if requested
    if sample_size and X.shape[0] > sample_size:
        idx = subsample_indices(X.shape[0], sample_size, seed=seed)
        X_sample = X[idx, :].toarray() if issparse(X) else X[idx, :]
        W_sample = W[idx, :]
        
        residual = X_sample - W_sample @ Q.T
        err_sample = np.linalg.norm(residual, 'fro')
        norm_sample = np.linalg.norm(X_sample, 'fro')
        
        return err_sample / (norm_sample + 1e-10)
    else:
        X_dense = X.toarray() if issparse(X) else X
        residual = X_dense - W @ Q.T
        err = np.linalg.norm(residual, 'fro')
        norm = np.linalg.norm(X_dense, 'fro')
        
        return err / (norm + 1e-10)


def log_progress(
    epoch: int,
    metrics: Dict[str, Any],
    stage: Optional[int] = None,
    interval: int = 10,
):
    """
    Print training progress in a formatted way.
    
    Parameters
    ----------
    epoch : int
        Current epoch number.
    metrics : dict
        Dictionary of metrics to log (e.g., {'recon_err': 0.123, 'ortho': 0.045}).
    stage : int, optional
        Training stage (1 or 2). If provided, adds stage prefix.
    interval : int, default=10
        Only print every `interval` epochs.
    
    Examples
    --------
    >>> log_progress(10, {'recon_err': 0.123, 'ortho': 0.045}, stage=1)
    [Stage 1] Epoch   10 | recon_err: 0.1230 | ortho: 0.0450
    """
    if epoch % interval != 0 and epoch != 0:
        return
    
    # Build message
    if stage is not None:
        msg = f"[Stage {stage}] Epoch {epoch:4d}"
    else:
        msg = f"Epoch {epoch:4d}"
    
    # Add metrics
    for key, val in metrics.items():
        if isinstance(val, (int, np.integer)):
            msg += f" | {key}: {val:d}"
        elif isinstance(val, (float, np.floating)):
            msg += f" | {key}: {val:.4f}"
        else:
            msg += f" | {key}: {val}"
    
    print(msg)


def save_model(model_dict: Dict[str, Any], filepath: Union[str, Path]):
    """
    Save SORT model to disk.
    
    Parameters
    ----------
    model_dict : dict
        Dictionary containing model parameters. Should include:
        - 'W': Loading matrix
        - 'Q': Signature matrix
        - 'params': Model hyperparameters
        - Any other metadata
    filepath : str or Path
        Output file path (.pkl).
    
    Examples
    --------
    >>> model_dict = {
    ...     'W': model.W,
    ...     'Q': model.Q,
    ...     'params': {'alpha': 0.1, 'beta': 0.01},
    ... }
    >>> save_model(model_dict, 'sort_model.pkl')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Model saved to {filepath}")


def load_model(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load SORT model from disk.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path (.pkl).
    
    Returns
    -------
    model_dict : dict
        Dictionary containing model parameters.
    
    Examples
    --------
    >>> model_dict = load_model('sort_model.pkl')
    >>> W = model_dict['W']
    >>> Q = model_dict['Q']
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model_dict = pickle.load(f)
    
    print(f"✓ Model loaded from {filepath}")
    return model_dict


def analyze_sparsity(
    W: np.ndarray,
    Q: np.ndarray,
    threshold: float = 1e-3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyze sparsity of W and Q matrices.
    
    Parameters
    ----------
    W : ndarray, shape (n, r)
        Loading matrix.
    Q : ndarray, shape (p, r)
        Signature matrix.
    threshold : float, default=1e-3
        Values below this (in absolute value) are considered zero.
    verbose : bool, default=True
        Whether to print results.
    
    Returns
    -------
    stats : dict
        Dictionary with sparsity statistics:
        - 'W_overall_sparsity': Overall W sparsity (fraction of zeros)
        - 'Q_overall_sparsity': Overall Q sparsity
        - 'W_per_column': Per-column W sparsity
        - 'Q_per_column': Per-column Q sparsity
    
    Examples
    --------
    >>> stats = analyze_sparsity(model.W, model.Q, threshold=0.01)
    Sparsity Analysis (threshold=0.010)
    ==============================
    W overall: 45.2%
    Q overall: 62.1%
    ...
    """
    W_sparsity = np.mean(np.abs(W) < threshold)
    Q_sparsity = np.mean(np.abs(Q) < threshold)
    
    W_col_sparsity = np.mean(np.abs(W) < threshold, axis=0)
    Q_col_sparsity = np.mean(np.abs(Q) < threshold, axis=0)
    
    stats = {
        'W_overall_sparsity': float(W_sparsity),
        'Q_overall_sparsity': float(Q_sparsity),
        'W_per_column': W_col_sparsity,
        'Q_per_column': Q_col_sparsity,
    }
    
    if verbose:
        print("="*70)
        print(f"Sparsity Analysis (threshold={threshold:.3f})")
        print("="*70)
        print(f"W overall sparsity: {W_sparsity:.1%}")
        print(f"Q overall sparsity: {Q_sparsity:.1%}")
        print(f"\nW per-column sparsity:")
        print(f"  Min:  {W_col_sparsity.min():.1%}")
        print(f"  Max:  {W_col_sparsity.max():.1%}")
        print(f"  Mean: {W_col_sparsity.mean():.1%}")
        print(f"\nQ per-column sparsity:")
        print(f"  Min:  {Q_col_sparsity.min():.1%}")
        print(f"  Max:  {Q_col_sparsity.max():.1%}")
        print(f"  Mean: {Q_col_sparsity.mean():.1%}")
        print("="*70)
    
    return stats


def normalize_components(
    W: np.ndarray,
    Q: np.ndarray,
    mode: str = 'Q',
) -> tuple:
    """
    Normalize W and Q so that one has unit norm per column.
    
    Parameters
    ----------
    W : ndarray, shape (n, r)
        Loading matrix.
    Q : ndarray, shape (p, r)
        Signature matrix.
    mode : {'W', 'Q', 'both'}, default='Q'
        Which matrix to normalize:
        - 'W': Normalize W to unit norm, scale Q accordingly
        - 'Q': Normalize Q to unit norm, scale W accordingly
        - 'both': Normalize both to have same scale
    
    Returns
    -------
    W_norm : ndarray
        Normalized W.
    Q_norm : ndarray
        Normalized Q.
    
    Examples
    --------
    >>> W_norm, Q_norm = normalize_components(W, Q, mode='Q')
    >>> np.testing.assert_allclose(np.linalg.norm(Q_norm, axis=0), 1.0)
    """
    if mode == 'Q':
        # Normalize Q to unit norm
        Q_norms = np.linalg.norm(Q, axis=0, keepdims=True)
        Q_norm = Q / (Q_norms + 1e-10)
        W_norm = W * Q_norms.T
    
    elif mode == 'W':
        # Normalize W to unit norm
        W_norms = np.linalg.norm(W, axis=0, keepdims=True)
        W_norm = W / (W_norms + 1e-10)
        Q_norm = Q * W_norms.T
    
    elif mode == 'both':
        # Balance scales
        W_norms = np.linalg.norm(W, axis=0)
        Q_norms = np.linalg.norm(Q, axis=0)
        scales = np.sqrt(W_norms * Q_norms)
        
        W_norm = W * (scales / (W_norms + 1e-10))
        Q_norm = Q * (scales / (Q_norms + 1e-10))
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return W_norm, Q_norm
    

def detect_device(
    prefer: str = 'auto',
    verbose: bool = True,
    test_memory: bool = True
) -> str:
    """
    Detect and select computing device.
    
    Parameters
    ----------
    prefer : str, default 'auto'
        Preferred device: 'auto', 'cuda', or 'cpu'
        - 'auto': Use GPU if available
        - 'cuda': Force GPU (raise error if unavailable)
        - 'cpu': Force CPU
    verbose : bool, default True
        Print detection info
    test_memory : bool, default True
        Test GPU memory accessibility
        
    Returns
    -------
    device : str
        Selected device: 'cuda' or 'cpu'
        
    Raises
    ------
    RuntimeError
        If prefer='cuda' but GPU unavailable
        
    Examples
    --------
    >>> device = detect_device()
    Auto-detected device: cuda (NVIDIA A100, 40GB available)
    
    >>> device = detect_device(prefer='cpu')
    Using device: cpu (forced by user)
    """
    # Force CPU
    if prefer == 'cpu':
        if verbose:
            print("Using device: cpu (forced by user)")
        return 'cpu'
    
    # Try GPU
    try:
        import cupy as cp
        
        # Test basic GPU access
        _ = cp.zeros(1)
        
        # Get GPU info
        gpu_id = cp.cuda.runtime.getDeviceCount()
        if gpu_id == 0:
            raise RuntimeError("No GPU devices found")
        
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props['name'].decode()
        
        # Test memory if requested
        if test_memory:
            mem_info = cp.cuda.runtime.memGetInfo()
            free_mem_gb = mem_info[0] / 1e9
            total_mem_gb = mem_info[1] / 1e9
            
            if verbose:
                print(f"Auto-detected device: cuda ({gpu_name})")
                print(f"  GPU memory: {free_mem_gb:.1f}GB / {total_mem_gb:.1f}GB available")
            
            # Warn if low memory
            if free_mem_gb < 2.0:
                warnings.warn(
                    f"Low GPU memory: {free_mem_gb:.1f}GB available. "
                    "Consider using device='cpu' for large datasets.",
                    UserWarning
                )
        else:
            if verbose:
                print(f"Auto-detected device: cuda ({gpu_name})")
        
        return 'cuda'
        
    except (ImportError, Exception) as e:
        # GPU not available
        if prefer == 'cuda':
            raise RuntimeError(
                f"GPU requested but unavailable: {e}\n"
                "Install cupy: pip install cupy-cuda11x (or cupy-cuda12x)"
            )
        
        if verbose:
            if isinstance(e, ImportError):
                print("Using device: cpu (cupy not installed)")
            else:
                print(f"Using device: cpu (GPU error: {e})")
        
        return 'cpu'
