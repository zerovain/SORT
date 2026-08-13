"""
SORT: Spatial Orthogonal Regularized Transcriptomic decomposition

A spatial transcriptomics analysis tool that decomposes expression into
background and spatially coherent signal components using semi-NMF.

Main Features
-------------
- Background-signal decomposition
- Spatial regularization via graph Laplacian
- Orthogonal signal signatures
- GPU acceleration support
- AnnData integration

Basic Usage
-----------
>>> import scanpy as sc
>>> import sort
>>>
>>> # Load and preprocess
>>> adata = sc.read_h5ad('spatial_data.h5ad')
>>> sc.pp.highly_variable_genes(adata, n_top_genes=3000)
>>>
>>> # Build spatial graph
>>> sort.build_spatial_graph(adata, n_neighbors=6)
>>> sort.compute_laplacian(adata)
>>>
>>> # Run SORT decomposition
>>> result = sort.fit(adata, n_components=10, random_state=42)
>>>
>>> # Downstream analysis
>>> Q_norm = sort.compute_qscore_matrix(adata.varm['sort_signatures'])
>>> gsp    = sort.compute_gene_specificity(adata.obsm['X_sort'],
...                                        adata.varm['sort_signatures'])
>>> ratios, dominant = sort.compute_dominance_ratios(gsp['GSp'])
>>> genelist, qdf    = sort.extract_enrichment_genelist(
...                        Q_norm, dominant, ratios, gene_names)
"""

__version__ = '1.0.0'

# ==============================================================================
# Core API
# ==============================================================================

from .model import decompose, SpatialSemiNMF
from .api import SORT, fit
from .config import SORTConfig
from .result import SORTResult
from .posthoc import (
    PosthocBResult,
    compute_posthoc_b,
    rank_genes_for_gep,
    score_signatures_from_b,
)

# ==============================================================================
# Preprocessing
# ==============================================================================

from .preprocessing import (
    build_spatial_graph,
    compute_laplacian,
    smooth_expression,
    build_spatial_graph_from_coords,
    smooth_expression_with_graph,
    smooth_expression_from_laplacian,
)

# ==============================================================================
# Analysis Tools
# ==============================================================================

from .analysis import (
    get_top_genes,
    rank_genes_components,
    get_background,
    get_signals,
    compute_reconstruction_error,
    correlate_components,
)

# ==============================================================================
# Visualization
# ==============================================================================

from .plotting import (
    spatial_loadings,
    signature_heatmap,
    spatial_reconstruction,
)

# ==============================================================================
# Configuration
# ==============================================================================

from .settings import settings

# ==============================================================================
# Initialization (Advanced)
# ==============================================================================

from .initialization import initialize_W_symnmf

# ==============================================================================
# Pipelines
# ==============================================================================

from .pipelines import (
    merge_samples,
    build_per_sample_graph,
    align_samples_grid,
    export_sort_results,
    normalize_Q_scale_W,
)

# ==============================================================================
# Downstream Analysis
# ==============================================================================

from .downstream import (
    compute_qscore_matrix,
    compute_gene_specificity,
    compute_dominance_ratios,
    extract_enrichment_genelist,
    classify_geps,
)

# ==============================================================================
# Public API
# ==============================================================================

__all__ = [
    # Minimal public API
    'SORT',
    'fit',
    'SORTConfig',
    'SORTResult',
    'PosthocBResult',
    'compute_posthoc_b',
    'rank_genes_for_gep',
    'score_signatures_from_b',
    # Core
    'decompose',
    'SpatialSemiNMF',

    # Preprocessing
    'build_spatial_graph',
    'compute_laplacian',
    'smooth_expression',
    'build_spatial_graph_from_coords',
    'smooth_expression_with_graph',
    'smooth_expression_from_laplacian',

    # Analysis
    'get_top_genes',
    'rank_genes_components',
    'get_background',
    'get_signals',
    'compute_reconstruction_error',
    'correlate_components',

    # Visualization
    'spatial_loadings',
    'signature_heatmap',
    'spatial_reconstruction',

    # Configuration
    'settings',

    # Advanced
    'initialize_W_symnmf',

    # Pipelines
    'merge_samples',
    'build_per_sample_graph',
    'align_samples_grid',
    'export_sort_results',
    'normalize_Q_scale_W',

    # Downstream
    'compute_qscore_matrix',
    'compute_gene_specificity',
    'compute_dominance_ratios',
    'extract_enrichment_genelist',
    'classify_geps',
]
