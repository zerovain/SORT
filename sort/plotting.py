"""
Visualization functions for SORT results.

Provides spatial and matrix visualizations for SORT decomposition results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import gridspec
from anndata import AnnData
from scipy.sparse import issparse
from typing import Optional, Union, List, Literal
import warnings


def spatial_loadings(
    adata: AnnData,
    component: Union[int, str, List[Union[int, str]]],
    spatial_key: str = 'spatial',
    color_map: str = 'viridis',
    spot_size: Optional[float] = None,
    ncols: int = 3,
    figsize: Optional[tuple] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    save: Optional[str] = None,
    **kwargs,
):
    """
    Plot spatial distribution of component loadings.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with SORT results.
    component : int, str, or list
        Component(s) to plot. Can be:
        - int: Component index (e.g., 0, 1, 2)
        - str: Component name (e.g., 'Background', 'Signal_1')
        - list: Multiple components
    spatial_key : str, default='spatial'
        Key in `.obsm` for spatial coordinates.
    color_map : str, default='viridis'
        Matplotlib colormap name.
    spot_size : float, optional
        Scatter plot point size. If None, auto-calculated.
    ncols : int, default=3
        Number of columns in subplot layout.
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated.
    vmin, vmax : float, optional
        Color scale limits. If None, uses data range.
    title : str, optional
        Overall figure title.
    save : str, optional
        Path to save figure.
    **kwargs
        Additional arguments passed to `scatter()`.
    
    Examples
    --------
    >>> # Plot background
    >>> sort.spatial_loadings(adata, component='Background')
    >>> 
    >>> # Plot multiple signals
    >>> sort.spatial_loadings(adata, component=[1, 2, 3, 4], ncols=2)
    >>> 
    >>> # Custom styling
    >>> sort.spatial_loadings(adata, component='Signal_1', 
    ...                        color_map='RdBu_r', spot_size=50)
    """
    from .analysis import _check_sort_results
    
    _check_sort_results(adata)
    
    # Validate spatial coordinates
    if spatial_key not in adata.obsm:
        raise ValueError(
            f"Spatial coordinates not found in .obsm['{spatial_key}']. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    
    coords = adata.obsm[spatial_key]
    W = adata.obsm['X_sort']
    comp_names = adata.uns['sort']['component_names']
    
    # Parse components
    if isinstance(component, (int, str)):
        components = [component]
    else:
        components = list(component)
    
    # Resolve component indices and labels
    comp_indices = []
    comp_labels = []
    for comp in components:
        if isinstance(comp, str):
            try:
                idx = comp_names.index(comp)
            except ValueError:
                warnings.warn(
                    f"Component '{comp}' not found. Available: {comp_names}",
                    UserWarning
                )
                continue
        else:
            idx = comp
            if idx < 0 or idx >= W.shape[1]:
                warnings.warn(
                    f"Component index {idx} out of range [0, {W.shape[1]-1}]",
                    UserWarning
                )
                continue
        comp_indices.append(idx)
        comp_labels.append(comp_names[idx])
    
    n_plots = len(comp_indices)
    if n_plots == 0:
        raise ValueError("No valid components to plot")
    
    # Setup figure layout
    nrows = int(np.ceil(n_plots / ncols))
    if figsize is None:
        figsize = (5 * ncols, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    # Auto calculate spot size
    if spot_size is None:
        area = np.prod(coords.max(axis=0) - coords.min(axis=0))
        spot_size = max(1, area / coords.shape[0] * 3)
    
    # Plot each component
    for i, (idx, label) in enumerate(zip(comp_indices, comp_labels)):
        ax = axes[i]
        
        values = W[:, idx]
        
        # Create scatter plot
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            s=spot_size,
            cmap=color_map,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
        
        # Styling
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xlabel('Spatial X', fontsize=10)
        ax.set_ylabel('Spatial Y', fontsize=10)
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Loading', fontsize=10)
    
    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save}")
    
    plt.show()


def signature_heatmap(
    adata: AnnData,
    components: Optional[List[Union[int, str]]] = None,
    n_genes: int = 50,
    figsize: Optional[tuple] = None,
    cmap: str = 'RdBu_r',
    center: Optional[float] = 0,
    show_gene_names: bool = True,
    gene_fontsize: int = 8,
    save: Optional[str] = None,
    **kwargs,
):
    """
    Plot heatmap of gene signatures for components.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with SORT results.
    components : list, optional
        Components to plot. If None, plots all.
    n_genes : int, default=50
        Number of top genes per component to show.
    figsize : tuple, optional
        Figure size (width, height).
    cmap : str, default='RdBu_r'
        Colormap. Diverging colormaps work well.
    center : float, optional
        Value to center colormap. If None, uses data center.
    show_gene_names : bool, default=True
        Whether to show gene names on y-axis.
    gene_fontsize : int, default=8
        Font size for gene names.
    save : str, optional
        Path to save figure.
    **kwargs
        Additional arguments for `plt.imshow()`.
    
    Examples
    --------
    >>> # Plot all components
    >>> sort.signature_heatmap(adata, n_genes=30)
    >>> 
    >>> # Plot specific components
    >>> sort.signature_heatmap(adata, components=['Signal_1', 'Signal_2'], 
    ...                        n_genes=50, cmap='coolwarm')
    """
    from .analysis import _check_sort_results, get_top_genes
    
    _check_sort_results(adata)
    
    Q = adata.varm['sort_signatures']
    comp_names = adata.uns['sort']['component_names']
    
    # Determine components to plot
    if components is None:
        components = list(range(Q.shape[1]))
    
    # Collect top genes for each component
    all_genes = []
    for comp in components:
        genes = get_top_genes(adata, comp, n_genes=n_genes)
        all_genes.extend(genes)
    
    # Get unique genes (preserving order)
    unique_genes = list(dict.fromkeys(all_genes))
    gene_indices = [adata.var_names.get_loc(g) for g in unique_genes]
    
    # Resolve component indices
    comp_indices = []
    comp_labels = []
    for comp in components:
        if isinstance(comp, str):
            try:
                idx = comp_names.index(comp)
            except ValueError:
                warnings.warn(f"Component '{comp}' not found, skipping", UserWarning)
                continue
        else:
            idx = comp
        comp_indices.append(idx)
        comp_labels.append(comp_names[idx])
    
    # Extract signature submatrix
    signature_matrix = Q[gene_indices, :][:, comp_indices]
    
    # Auto figure size
    if figsize is None:
        width = max(8, len(comp_indices) * 1.5)
        height = max(6, len(unique_genes) * 0.12)
        figsize = (width, height)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color normalization
    if center is not None:
        vmax = max(abs(signature_matrix.min()), abs(signature_matrix.max()))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=center, vmax=vmax)
    else:
        norm = None
    
    # Plot heatmap
    im = ax.imshow(signature_matrix, aspect='auto', cmap=cmap, norm=norm, **kwargs)
    
    # X-axis: Components
    ax.set_xticks(np.arange(len(comp_labels)))
    ax.set_xticklabels(comp_labels, rotation=45, ha='right', fontsize=11)
    ax.set_xlabel('Components', fontweight='bold', fontsize=12)
    
    # Y-axis: Genes
    if show_gene_names and len(unique_genes) <= 100:
        ax.set_yticks(np.arange(len(unique_genes)))
        ax.set_yticklabels(unique_genes, fontsize=gene_fontsize)
        ax.set_ylabel('Genes', fontweight='bold', fontsize=12)
    else:
        ax.set_yticks([])
        ax.set_ylabel(f'Top {len(unique_genes)} genes', fontweight='bold', fontsize=12)
    
    ax.set_title('Gene Signatures', fontsize=14, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Signature score', fontsize=11)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save}")
    
    plt.show()


def spatial_reconstruction(
    adata: AnnData,
    genes: Union[str, List[str]],
    spatial_key: str = 'spatial',
    layer: str = 'sort_reconstructed',
    original_layer: Optional[str] = None,
    spot_size: Optional[float] = None,
    figsize: Optional[tuple] = None,
    cmap: str = 'viridis',
    save: Optional[str] = None,
):
    """
    Compare original vs reconstructed gene expression spatially.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with SORT results.
    genes : str or list of str
        Gene name(s) to visualize.
    spatial_key : str, default='spatial'
        Key in `.obsm` for spatial coordinates.
    layer : str, default='sort_reconstructed'
        Layer with reconstructed expression.
    original_layer : str, optional
        Layer with original expression. If None, uses `.X`.
    spot_size : float, optional
        Scatter point size.
    figsize : tuple, optional
        Figure size (width, height).
    cmap : str, default='viridis'
        Colormap.
    save : str, optional
        Path to save figure.
    
    Examples
    --------
    >>> # Compare single gene
    >>> sort.spatial_reconstruction(adata, genes='CD3D')
    >>> 
    >>> # Compare multiple genes
    >>> sort.spatial_reconstruction(adata, genes=['CD3D', 'CD8A', 'FOXP3'])
    """
    from .analysis import _check_sort_results
    
    _check_sort_results(adata)
    
    if isinstance(genes, str):
        genes = [genes]
    
    # Validate inputs
    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial coordinates not found in .obsm['{spatial_key}']")
    
    if layer not in adata.layers:
        raise ValueError(
            f"Reconstructed layer '{layer}' not found. Run decompose() first."
        )
    
    coords = adata.obsm[spatial_key]
    n_genes = len(genes)
    
    # Auto figure size
    if figsize is None:
        figsize = (10, 4 * n_genes)
    
    fig, axes = plt.subplots(n_genes, 2, figsize=figsize)
    if n_genes == 1:
        axes = axes.reshape(1, -1)
    
    # Auto spot size
    if spot_size is None:
        area = np.prod(coords.max(axis=0) - coords.min(axis=0))
        spot_size = max(1, area / coords.shape[0] * 3)
    
    for i, gene in enumerate(genes):
        if gene not in adata.var_names:
            warnings.warn(f"Gene '{gene}' not found, skipping", UserWarning)
            axes[i, 0].text(0.5, 0.5, f"Gene '{gene}' not found",
                          ha='center', va='center', transform=axes[i, 0].transAxes)
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
            continue
        
        gene_idx = adata.var_names.get_loc(gene)
        
        # Get original expression
        if original_layer is None:
            original = adata.X[:, gene_idx]
        else:
            original = adata.layers[original_layer][:, gene_idx]
        
        if issparse(original):
            original = original.toarray().flatten()
        else:
            original = np.asarray(original).flatten()
        
        # Get reconstructed expression
        reconstructed = adata.layers[layer][:, gene_idx]
        if issparse(reconstructed):
            reconstructed = reconstructed.toarray().flatten()
        else:
            reconstructed = np.asarray(reconstructed).flatten()
        
        # Shared color scale
        vmin = min(original.min(), reconstructed.min())
        vmax = max(original.max(), reconstructed.max())
        
        # Plot original
        sc1 = axes[i, 0].scatter(coords[:, 0], coords[:, 1], c=original,
                                s=spot_size, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'{gene} - Original', fontweight='bold', fontsize=12)
        axes[i, 0].set_aspect('equal')
        axes[i, 0].axis('off')
        plt.colorbar(sc1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Plot reconstructed
        sc2 = axes[i, 1].scatter(coords[:, 0], coords[:, 1], c=reconstructed,
                                s=spot_size, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'{gene} - Reconstructed', fontweight='bold', fontsize=12)
        axes[i, 1].set_aspect('equal')
        axes[i, 1].axis('off')
        plt.colorbar(sc2, ax=axes[i, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save}")
    
    plt.show()


def plot_component_correlation(
    adata: AnnData,
    figsize: tuple = (10, 8),
    cmap: str = 'coolwarm',
    save: Optional[str] = None,
):
    """
    Plot correlation matrix between components.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with SORT results.
    figsize : tuple, default=(10, 8)
        Figure size.
    cmap : str, default='coolwarm'
        Colormap.
    save : str, optional
        Path to save figure.
    
    Examples
    --------
    >>> sort.plot_component_correlation(adata)
    """
    from .analysis import _check_sort_results
    
    _check_sort_results(adata)
    
    W = adata.obsm['X_sort']
    comp_names = adata.uns['sort']['component_names']
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(W.T)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    # Labels
    ax.set_xticks(np.arange(len(comp_names)))
    ax.set_yticks(np.arange(len(comp_names)))
    ax.set_xticklabels(comp_names, rotation=45, ha='right')
    ax.set_yticklabels(comp_names)
    
    ax.set_title('Component Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Add correlation values
    for i in range(len(comp_names)):
        for j in range(len(comp_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                         ha='center', va='center', color='black', fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save}")
    
    plt.show()