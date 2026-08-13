"""
Analysis functions for SORT results.

Provides tools to analyze and interpret SORT decomposition results,
including gene ranking, component correlation, and quality metrics.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from typing import Optional, Union, List, Literal
import warnings


def get_top_genes(
    adata: AnnData,
    component: Union[int, str],
    n_genes: int = 50,
    criterion: Literal['absolute', 'positive', 'negative'] = 'absolute',
    return_scores: bool = False,
) -> Union[List[str], pd.DataFrame]:
    """
    Get top genes for a specific component.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with SORT results.
    component : int or str
        Component index (0, 1, ...) or name ('Background', 'Signal_1', ...).
    n_genes : int, default=50
        Number of top genes to return.
    criterion : {'absolute', 'positive', 'negative'}, default='absolute'
        Ranking criterion:
        - 'absolute': By absolute signature value
        - 'positive': By positive signature value (most upregulated)
        - 'negative': By negative signature value (most downregulated)
    return_scores : bool, default=False
        If True, return DataFrame with scores. Otherwise return list of gene names.
    
    Returns
    -------
    list of str or DataFrame
        If return_scores=False: List of gene names
        If return_scores=True: DataFrame with columns ['gene', 'score', 'rank']
    
    Examples
    --------
    >>> # Get top 50 genes by absolute value
    >>> genes = sort.get_top_genes(adata, component='Background', n_genes=50)
    >>> 
    >>> # Get top upregulated genes with scores
    >>> df = sort.get_top_genes(adata, component=1, criterion='positive', 
    ...                          n_genes=30, return_scores=True)
    >>> print(df.head())
    """
    _check_sort_results(adata)
    
    Q = adata.varm['sort_signatures']
    
    # Parse component
    if isinstance(component, str):
        comp_names = adata.uns['sort']['component_names']
        try:
            component_idx = comp_names.index(component)
        except ValueError:
            raise ValueError(
                f"Component '{component}' not found. "
                f"Available components: {comp_names}"
            )
    else:
        component_idx = component
        if component_idx < 0 or component_idx >= Q.shape[1]:
            raise ValueError(
                f"Component index {component_idx} out of range [0, {Q.shape[1]-1}]"
            )
    
    # Get signature
    signature = Q[:, component_idx]
    
    # Rank genes based on criterion
    if criterion == 'absolute':
        ranking_scores = np.abs(signature)
    elif criterion == 'positive':
        ranking_scores = signature
    elif criterion == 'negative':
        ranking_scores = -signature
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    # Get top indices
    top_idx = np.argsort(ranking_scores)[::-1][:n_genes]
    
    if return_scores:
        df = pd.DataFrame({
            'gene': adata.var_names[top_idx].tolist(),
            'score': signature[top_idx],
            'abs_score': np.abs(signature[top_idx]),
            'rank': np.arange(1, len(top_idx) + 1),
        })
        return df
    else:
        return adata.var_names[top_idx].tolist()


def rank_genes_components(
    adata: AnnData,
    n_genes: int = 100,
    criterion: Literal['absolute', 'positive', 'negative'] = 'absolute',
) -> pd.DataFrame:
    """
    Rank genes for all components.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with SORT results.
    n_genes : int, default=100
        Number of top genes per component.
    criterion : {'absolute', 'positive', 'negative'}, default='absolute'
        Ranking criterion.
    
    Returns
    -------
    DataFrame
        Ranked genes with columns ['component', 'gene', 'score', 'abs_score', 'rank'].
    
    Examples
    --------
    >>> df = sort.rank_genes_components(adata, n_genes=50)
    >>> # Save to file
    >>> df.to_csv('sort_ranked_genes.csv', index=False)
    >>> 
    >>> # Get genes for specific component
    >>> signal1_genes = df[df['component'] == 'Signal_1']
    """
    _check_sort_results(adata)
    
    Q = adata.varm['sort_signatures']
    n_components = Q.shape[1]
    comp_names = adata.uns['sort']['component_names']
    
    df_list = []
    
    for i in range(n_components):
        signature = Q[:, i]
        
        # Rank based on criterion
        if criterion == 'absolute':
            ranking_scores = np.abs(signature)
        elif criterion == 'positive':
            ranking_scores = signature
        elif criterion == 'negative':
            ranking_scores = -signature
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        top_idx = np.argsort(ranking_scores)[::-1][:n_genes]
        
        df_comp = pd.DataFrame({
            'component': comp_names[i],
            'component_idx': i,
            'gene': adata.var_names[top_idx].tolist(),
            'score': signature[top_idx],
            'abs_score': np.abs(signature[top_idx]),
            'rank': np.arange(1, len(top_idx) + 1),
        })
        df_list.append(df_comp)
    
    df = pd.concat(df_list, ignore_index=True)
    return df


def get_background(adata: AnnData, copy: bool = False) -> tuple:
    """
    Extract background component.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with SORT results.
    copy : bool, default=False
        Whether to return copies (to prevent modification).
    
    Returns
    -------
    w_0 : ndarray, shape (n_obs,)
        Background loadings (per spot).
    q_0 : ndarray, shape (n_vars,)
        Background signature (per gene).
    
    Examples
    --------
    >>> w_bg, q_bg = sort.get_background(adata)
    >>> 
    >>> # Add to adata for visualization
    >>> adata.obs['background_loading'] = w_bg
    """
    _check_sort_results(adata)
    
    W = adata.obsm['X_sort']
    Q = adata.varm['sort_signatures']
    
    w_0 = W[:, 0]
    q_0 = Q[:, 0]
    
    if copy:
        return w_0.copy(), q_0.copy()
    else:
        return w_0, q_0


def get_signals(adata: AnnData, copy: bool = False) -> tuple:
    """
    Extract signal components.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with SORT results.
    copy : bool, default=False
        Whether to return copies.
    
    Returns
    -------
    W_s : ndarray, shape (n_obs, n_components-1)
        Signal loadings.
    Q_s : ndarray, shape (n_vars, n_components-1)
        Signal signatures.
    
    Examples
    --------
    >>> W_signals, Q_signals = sort.get_signals(adata)
    >>> print(f"Number of signal components: {W_signals.shape[1]}")
    """
    _check_sort_results(adata)
    
    W = adata.obsm['X_sort']
    Q = adata.varm['sort_signatures']
    
    W_s = W[:, 1:]
    Q_s = Q[:, 1:]
    
    if copy:
        return W_s.copy(), Q_s.copy()
    else:
        return W_s, Q_s


def compute_reconstruction_error(
    adata: AnnData,
    layer: Optional[str] = None,
    per_gene: bool = False,
    per_obs: bool = False,
) -> Union[float, np.ndarray]:
    """
    Compute reconstruction error.
    
    Computes relative reconstruction error: ||X - WQ^T||_F / ||X||_F
    
    Parameters
    ----------
    adata : AnnData
        AnnData with SORT results.
    layer : str, optional
        Layer to compare against. If None, uses `.X`.
    per_gene : bool, default=False
        If True, return per-gene errors (shape: n_vars).
    per_obs : bool, default=False
        If True, return per-observation errors (shape: n_obs).
    
    Returns
    -------
    float or ndarray
        Relative reconstruction error(s).
        - If per_gene=False and per_obs=False: single scalar
        - If per_gene=True: array of shape (n_vars,)
        - If per_obs=True: array of shape (n_obs,)
    
    Examples
    --------
    >>> # Overall error
    >>> error = sort.compute_reconstruction_error(adata)
    >>> print(f"Reconstruction error: {error:.2%}")
    >>> 
    >>> # Per-gene errors
    >>> gene_errors = sort.compute_reconstruction_error(adata, per_gene=True)
    >>> adata.var['reconstruction_error'] = gene_errors
    >>> 
    >>> # Per-spot errors
    >>> spot_errors = sort.compute_reconstruction_error(adata, per_obs=True)
    >>> adata.obs['reconstruction_error'] = spot_errors
    """
    _check_sort_results(adata)
    
    W = adata.obsm['X_sort']
    Q = adata.varm['sort_signatures']
    
    # Get data matrix
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        X = adata.layers[layer]
    
    # Convert to dense if sparse
    if issparse(X):
        X = X.toarray()
    
    # Compute reconstruction
    X_recon = W @ Q.T
    
    # Compute errors
    residual = X - X_recon
    
    if per_gene:
        # Per-gene relative error
        gene_residual_norm = np.linalg.norm(residual, axis=0)
        gene_norm = np.linalg.norm(X, axis=0)
        errors = gene_residual_norm / (gene_norm + 1e-10)
        return errors
    
    elif per_obs:
        # Per-observation relative error
        obs_residual_norm = np.linalg.norm(residual, axis=1)
        obs_norm = np.linalg.norm(X, axis=1)
        errors = obs_residual_norm / (obs_norm + 1e-10)
        return errors
    
    else:
        # Overall relative error
        error = np.linalg.norm(residual, 'fro') / (np.linalg.norm(X, 'fro') + 1e-10)
        return error


def correlate_components(
    adata: AnnData,
    obs_keys: Optional[List[str]] = None,
    method: Literal['pearson', 'spearman'] = 'pearson',
) -> pd.DataFrame:
    """
    Correlate SORT loadings with metadata.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with SORT results.
    obs_keys : list of str, optional
        Metadata keys in `.obs` to correlate. If None, uses all numeric columns.
    method : {'pearson', 'spearman'}, default='pearson'
        Correlation method.
    
    Returns
    -------
    DataFrame
        Correlation coefficients with columns:
        ['component', 'variable', 'correlation', 'abs_correlation'].
    
    Examples
    --------
    >>> # Correlate with QC metrics
    >>> df = sort.correlate_components(adata, 
    ...                                 obs_keys=['total_counts', 'n_genes_by_counts'])
    >>> print(df.sort_values('abs_correlation', ascending=False))
    >>> 
    >>> # Find components correlated with specific metadata
    >>> batch_corr = df[df['variable'] == 'batch']
    """
    _check_sort_results(adata)
    
    W = adata.obsm['X_sort']
    n_components = W.shape[1]
    comp_names = adata.uns['sort']['component_names']
    
    # Determine which obs keys to use
    if obs_keys is None:
        obs_keys = adata.obs.select_dtypes(include=[np.number]).columns.tolist()
        if len(obs_keys) == 0:
            raise ValueError("No numeric columns found in .obs")
    
    results = []
    
    for key in obs_keys:
        if key not in adata.obs:
            warnings.warn(f"Key '{key}' not found in .obs, skipping", UserWarning)
            continue
        
        values = adata.obs[key].values
        
        # Skip if constant
        if np.std(values) < 1e-10:
            warnings.warn(f"Key '{key}' has zero variance, skipping", UserWarning)
            continue
        
        for i in range(n_components):
            loadings = W[:, i]
            
            # Compute correlation
            if method == 'pearson':
                # Pearson correlation
                corr = np.corrcoef(loadings, values)[0, 1]
            elif method == 'spearman':
                # Spearman correlation (rank-based)
                from scipy.stats import spearmanr
                corr, _ = spearmanr(loadings, values)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results.append({
                'component': comp_names[i],
                'component_idx': i,
                'variable': key,
                'correlation': corr,
                'abs_correlation': abs(corr),
            })
    
    df = pd.DataFrame(results)
    return df


def compute_component_orthogonality(adata: AnnData) -> dict:
    """
    Compute orthogonality metrics for signal components.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with SORT results.
    
    Returns
    -------
    dict
        Dictionary with orthogonality metrics:
        - 'Q_orthogonality': ||Q_s^T Q_s - I||_F (lower is better)
        - 'W_orthogonality': ||W_s^T W_s - I||_F (informative only)
        - 'Q_gram_matrix': Q_s^T Q_s (for visualization)
        - 'W_gram_matrix': W_s^T W_s (for visualization)
    
    Examples
    --------
    >>> metrics = sort.compute_component_orthogonality(adata)
    >>> print(f"Q orthogonality violation: {metrics['Q_orthogonality']:.4f}")
    """
    _check_sort_results(adata)
    
    W_s, Q_s = get_signals(adata)
    
    # Normalize to unit vectors for fair comparison
    W_s_norm = W_s / (np.linalg.norm(W_s, axis=0, keepdims=True) + 1e-10)
    Q_s_norm = Q_s / (np.linalg.norm(Q_s, axis=0, keepdims=True) + 1e-10)
    
    # Compute Gram matrices
    Q_gram = Q_s_norm.T @ Q_s_norm
    W_gram = W_s_norm.T @ W_s_norm
    
    # Orthogonality violations
    n_signals = Q_s.shape[1]
    I = np.eye(n_signals)
    
    Q_ortho_violation = np.linalg.norm(Q_gram - I, 'fro')
    W_ortho_violation = np.linalg.norm(W_gram - I, 'fro')
    
    return {
        'Q_orthogonality': Q_ortho_violation,
        'W_orthogonality': W_ortho_violation,
        'Q_gram_matrix': Q_gram,
        'W_gram_matrix': W_gram,
    }


def compute_component_sparsity(adata: AnnData, threshold: float = 1e-3) -> pd.DataFrame:
    """
    Compute sparsity of components.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with SORT results.
    threshold : float, default=1e-3
        Values below this are considered zero.
    
    Returns
    -------
    DataFrame
        Sparsity metrics per component.
    
    Examples
    --------
    >>> df = sort.compute_component_sparsity(adata)
    >>> print(df)
    """
    _check_sort_results(adata)
    
    W = adata.obsm['X_sort']
    Q = adata.varm['sort_signatures']
    comp_names = adata.uns['sort']['component_names']
    
    results = []
    
    for i in range(W.shape[1]):
        W_sparsity = np.mean(np.abs(W[:, i]) < threshold)
        Q_sparsity = np.mean(np.abs(Q[:, i]) < threshold)
        
        results.append({
            'component': comp_names[i],
            'W_sparsity': W_sparsity,
            'Q_sparsity': Q_sparsity,
            'W_l0_norm': np.sum(np.abs(W[:, i]) >= threshold),
            'Q_l0_norm': np.sum(np.abs(Q[:, i]) >= threshold),
        })
    
    df = pd.DataFrame(results)
    return df


def _check_sort_results(adata: AnnData):
    """Check if SORT has been run on adata."""
    if 'sort' not in adata.uns:
        raise ValueError(
            "SORT results not found. Run sort.decompose() first:\n"
            "  >>> sort.decompose(adata, n_components=10)"
        )
    if 'X_sort' not in adata.obsm:
        raise ValueError("SORT loadings not found in .obsm['X_sort']")
    if 'sort_signatures' not in adata.varm:
        raise ValueError("SORT signatures not found in .varm['sort_signatures']")