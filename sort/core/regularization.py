import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def compute_l1_weights_fixed(n_cols, lambda_base=0.01):
    return np.full(n_cols, lambda_base)

def compute_l1_weights_adaptive(matrix, lambda_base=0.01, method='l2_norm'):
    if CUPY_AVAILABLE and isinstance(matrix, cp.ndarray):
        xp = cp
    else:
        xp = np
    
    if method == 'l2_norm':
        scale = xp.sqrt(xp.sum(matrix**2, axis=0))
    elif method == 'mean':
        scale = xp.mean(xp.abs(matrix), axis=0)
    elif method == 'median':
        scale = xp.median(xp.abs(matrix), axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    mean_scale = xp.mean(scale)
    weights = lambda_base * scale / (mean_scale + 1e-10)
    
    if CUPY_AVAILABLE and isinstance(weights, cp.ndarray):
        weights = cp.asnumpy(weights)
    
    return weights

def analyze_scale_variation(W, Q, verbose=True):
    w_l2_norms = np.linalg.norm(W, axis=0)
    w_means = np.mean(np.abs(W), axis=0)
    
    Q_s = Q[:, 1:]
    q_l2_norms = np.linalg.norm(Q_s, axis=0)
    q_l1_norms = np.sum(np.abs(Q_s), axis=0)
    
    stats = {
        'W_l2_ratio': w_l2_norms.max() / (w_l2_norms.min() + 1e-10),
        'W_mean_ratio': w_means.max() / (w_means.min() + 1e-10),
        'Q_l2_ratio': q_l2_norms.max() / (q_l2_norms.min() + 1e-10),
        'Q_l1_ratio': q_l1_norms.max() / (q_l1_norms.min() + 1e-10),
    }
    
    if verbose:
        print("="*70)
        print("Scale Variation Analysis")
        print("="*70)
        print(f"W L2 ratio: {stats['W_l2_ratio']:.1f}×")
        print(f"Q_s L2 ratio: {stats['Q_l2_ratio']:.2f}×")
        
        if stats['W_l2_ratio'] > 5:
            print("✅ W: Use ADAPTIVE (ratio > 5×)")
        else:
            print("⚠️  W: FIXED sufficient (ratio < 5×)")
        
        if stats['Q_l2_ratio'] > 1.5:
            print("⚠️  Q_s: Consider ADAPTIVE (ratio > 1.5×)")
        else:
            print("✅ Q_s: Use FIXED (consistent scale)")
        print("="*70)
    
    return stats