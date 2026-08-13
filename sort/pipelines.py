"""Graph, assembly and compatibility utilities.

Biological preprocessing is intentionally not part of the public SORT API;
release tutorials show the corresponding Scanpy calls explicitly. The legacy
preprocessing wrappers in this module remain only for internal compatibility
with earlier scripts and are excluded from ``__all__``.
"""

import os  
import hashlib
import anndata as ad
import numpy as np  
import pandas as pd  
import scanpy as sc  
from scipy.sparse import issparse, csr_matrix  
from typing import Optional, List, Dict  
import pickle  

from .preprocessing import compute_laplacian  
from .model import decompose  
from .utils import detect_device  

__all__ = [  
    'merge_samples',  
    'build_per_sample_graph',  
    'align_samples_grid',  
    'export_sort_results',  
    'normalize_Q_scale_W',  
]  


# ==============================================================================  
# Helper Functions  
# ==============================================================================  

def _get_array_stats(X):  
    """Get statistics from sparse or dense array."""  
    if issparse(X):  
        if X.nnz == 0:  
            return np.array([]), np.array([])  
        gene_cells = np.array((X > 0).sum(axis=0)).flatten()  
        gene_means = np.array(X.mean(axis=0)).flatten()  
    else:  
        gene_cells = (X > 0).sum(axis=0)  
        gene_means = X.mean(axis=0)  
    return gene_cells, gene_means  


def _check_abnormal_values(X):  
    """Check for NaN, Inf, and negative values."""  
    if issparse(X):  
        if X.nnz == 0:  
            return False, None  
        data = X.data  
        has_abnormal = np.isnan(data).any() or np.isinf(data).any() or (data < 0).any()  
        
        if has_abnormal:  
            bad_genes = np.zeros(X.shape[1], dtype=bool)  
            for gene_idx in range(X.shape[1]):  
                gene_data = X[:, gene_idx].data  
                if len(gene_data) > 0:  
                    if np.isnan(gene_data).any() or np.isinf(gene_data).any() or (gene_data < 0).any():  
                        bad_genes[gene_idx] = True  
            return True, bad_genes  
    else:  
        bad_genes = np.isnan(X).any(axis=0) | np.isinf(X).any(axis=0) | (X < 0).any(axis=0)  
        return bad_genes.any(), bad_genes  
    
    return False, None  


def _print_gene_list(genes, max_display=10, prefix=""):  
    """Print gene list with truncation."""  
    if len(genes) <= max_display:  
        print(f"{prefix}{', '.join(genes)}")  
    else:  
        print(f"{prefix}{', '.join(genes[:max_display])}... (+{len(genes)-max_display} more)")  


# ==============================================================================  
# Dataset Merging  
# ==============================================================================  

def merge_samples(  
    adata_list: List,  
    cohort_labels: Optional[List[str]] = None,  
    join: str = 'inner',  
    batch_key: str = 'sample_id',  
):  
    """  
    Merge multiple AnnData objects with spatial coordinates.  
    
    Parameters  
    ----------  
    adata_list : list of AnnData  
        List of spatial AnnData objects to merge  
    cohort_labels : list of str, optional  
        Labels for each cohort (e.g., ['IDH_WT', 'UKF'])  
    join : str, default 'inner'  
        How to merge genes: 'inner' or 'outer'  
    batch_key : str, default 'sample_id'  
        Column added to identify the input sample after concatenation  
    """  
    if len(adata_list) < 2:  
        raise ValueError("Need at least 2 AnnData objects to merge")  
    if join not in {"inner", "outer"}:
        raise ValueError("join must be 'inner' or 'outer'")
    if not isinstance(batch_key, str) or not batch_key:
        raise ValueError("batch_key must be a non-empty string")

    # Avoid mutating caller-owned objects when adding cohort labels.
    adata_list = [adata.copy() for adata in adata_list]
    
    for i, adata in enumerate(adata_list):  
        if 'spatial' not in adata.obsm:  
            raise ValueError(f"AnnData {i} missing 'spatial' coordinates in obsm")  
    
    print(f"Merging {len(adata_list)} datasets...")  
    
    if cohort_labels is not None:  
        if len(cohort_labels) != len(adata_list):  
            raise ValueError("cohort_labels must match adata_list length")  
        for adata, label in zip(adata_list, cohort_labels):  
            adata.obs['cohort'] = label  
    
    if join == 'inner':  
        common_genes = adata_list[0].var_names  
        for adata in adata_list[1:]:  
            common_genes = common_genes.intersection(adata.var_names)  
        if len(common_genes) == 0:
            raise ValueError("No common genes remain for join='inner'")
        print(f"  Common genes: {len(common_genes):,}")  
        adata_list = [adata[:, common_genes].copy() for adata in adata_list]  
    
    spatial_coords = [adata.obsm['spatial'].copy() for adata in adata_list]  
    
    adata_combined = ad.concat(
        adata_list,
        join=join,
        label=batch_key,
        keys=[str(i) for i in range(len(adata_list))],
        index_unique=None,
        merge="same",
        uns_merge="same",
    )
    
    adata_combined.obsm['spatial'] = np.vstack(spatial_coords).astype(np.float32)  
    
    print(f"✓ Merged: {adata_combined.shape[0]:,} spots × {adata_combined.shape[1]:,} genes")  
    if batch_key in adata_combined.obs.columns:  
        print(f"  Samples: {adata_combined.obs[batch_key].nunique()}")  
    
    return adata_combined  


# ==============================================================================  
# Standard Preprocessing  
# ==============================================================================  

def standard_preprocess(
    adata,
    target_sum: float = 1e4,
    log_transform: bool = True,
    n_hvg: Optional[int] = 3000,
    batch_key: Optional[str] = 'sample_id',
    min_genes: int = 200,
    min_cells_pct: float = 0.01,
    min_mean_expression: float = 0.01,
    use_float32: bool = True,
    to_sparse: bool = True,
    copy: bool = False,
):
    """Compatibility wrapper; public workflows should use explicit Scanpy."""
    if copy:
        adata = adata.copy()
    
    print("="*80)
    print("Preprocessing Spatial Data")
    print("="*80)
    print(f"Original: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes\n")
    
    # Step 1: Cell/Spot QC Filtering
    # Modify the working AnnData object in place.
    if min_genes > 0:
        print(f"[1/7] Cell filtering (min_genes={min_genes})...")
        if 'N_genes' not in adata.obs.columns:
            adata.obs['N_genes'] = np.array((adata.X > 0).sum(axis=1)).flatten() if issparse(adata.X) else (adata.X > 0).sum(axis=1)
        
        n_before = adata.shape[0]
        mask = adata.obs['N_genes'] >= min_genes
        adata._inplace_subset_obs(mask)
        print(f"      Removed {n_before - adata.shape[0]:,}, remaining {adata.shape[0]:,}\n")
    else:
        print(f"[1/7] Cell filtering: skipped\n")
    
    # Step 2: Gene QC Filtering
    # Filter gene columns in place.
    min_cells_absolute = max(3, int(adata.shape[0] * min_cells_pct))
    print(f"[2/7] Gene filtering (min_cells={min_cells_absolute}, min_mean={min_mean_expression})...")
    
    gene_cells, gene_means = _get_array_stats(adata.X)
    keep_genes = (gene_cells >= min_cells_absolute) & (gene_means >= min_mean_expression if min_mean_expression > 0 else True)
    
    n_removed = (~keep_genes).sum()
    if n_removed > 0:
        print(f"      Removed {n_removed:,} genes")
        _print_gene_list(adata.var_names[~keep_genes].tolist(), prefix="      Examples: ")
        adata._inplace_subset_var(keep_genes)
    else:
        print(f"      ✓ All genes passed")
    print(f"      Remaining: {adata.shape[1]:,} genes\n")
    
    # Step 3: Check Abnormal Values
    # Remove invalid gene columns in place.
    print(f"[3/7] Checking abnormal values...")
    has_abnormal, bad_genes = _check_abnormal_values(adata.X)
    
    if has_abnormal and bad_genes is not None and bad_genes.sum() > 0:
        n_bad = bad_genes.sum()
        print(f"      ⚠ Removing {n_bad} genes with NaN/Inf/negative values")
        _print_gene_list(adata.var_names[bad_genes].tolist(), prefix="      Examples: ")
        adata._inplace_subset_var(~bad_genes)
    else:
        print(f"      ✓ No abnormal values\n")
    
    # Steps 4--6 preserve the paper workflow: normalize and log-transform the
    # complete QC-passing matrix before batch-aware Seurat-v3 feature selection.
    # Step 4: Library-size normalization
    if target_sum is not None:
        print(f"[4/7] Normalization (target_sum={target_sum})...")
        sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
        print(f"      ✓ Normalized\n")
    else:
        print(f"[4/7] Normalization: skipped\n")

    # Step 5: Log Transformation
    if log_transform:
        print(f"[5/7] Log transformation...")
        sc.pp.log1p(adata)
        print(f"      ✓ Applied log1p\n")
    else:
        print(f"[5/7] Log transformation: skipped\n")

    # Step 6: HVG Selection with Validation
    if n_hvg is not None:
        print(f"[6/7] HVG selection on normalized log-expression (n_hvg={n_hvg})...")
        
        effective_batch_key = batch_key if batch_key in adata.obs.columns else None
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_hvg, batch_key=effective_batch_key,
            flavor='seurat_v3', subset=False
        )
        
        n_hvg_initial = adata.var['highly_variable'].sum()
        print(f"      Initial: {n_hvg_initial}")
        
        # Validate HVGs
        hvg_mask = adata.var['highly_variable'].values
        hvg_indices = np.where(hvg_mask)[0]
        X_hvg = adata[:, hvg_mask].X
        
        hvg_cells, hvg_means = _get_array_stats(X_hvg)
        
        # Identify problematic HVGs
        problematic_mask = np.zeros(len(hvg_indices), dtype=bool)
        min_hvg_cells = max(5, int(adata.shape[0] * 0.01))
        
        problematic_mask |= (hvg_means < 0) | (hvg_means < -0.5) | (hvg_cells < min_hvg_cells)
        
        n_problematic = problematic_mask.sum()
        if n_problematic > 0:
            print(f"      ⚠ Removing {n_problematic} problematic HVGs")
            hvg_names = adata.var_names[hvg_mask]
            _print_gene_list(hvg_names[problematic_mask].tolist(), max_display=15, prefix="      Examples: ")
            
            problematic_indices = hvg_indices[problematic_mask]
            for idx in problematic_indices:
                adata.var.iloc[idx, adata.var.columns.get_loc('highly_variable')] = False
            
            print(f"      Final: {adata.var['highly_variable'].sum()}")
        else:
            print(f"      ✓ All HVGs validated")
        
        adata._inplace_subset_var(adata.var['highly_variable'].values)
        print(f"      Shape: {adata.shape[0]:,} × {adata.shape[1]:,}\n")
    else:
        print(f"[6/7] HVG selection: skipped\n")
    
    # Step 7: Data Type Conversion
    print(f"[7/7] Data type conversion...")
    
    # Final cleanup of abnormal values
    if issparse(adata.X) and adata.X.nnz > 0:
        data = adata.X.data
        if np.isnan(data).any() or np.isinf(data).any():
            adata.X.data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            adata.X.eliminate_zeros()
    elif not issparse(adata.X):
        if np.isnan(adata.X).any() or np.isinf(adata.X).any():
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Convert to sparse
    if to_sparse and not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    
    # Convert to float32
    if use_float32:
        if issparse(adata.X):
            if adata.X.dtype != np.float32:
                adata.X.data = adata.X.data.astype(np.float32)
        else:
            if adata.X.dtype != np.float32:
                adata.X = adata.X.astype(np.float32)
    
    print(f"      ✓ Converted to {'sparse ' if issparse(adata.X) else ''}float32\n")
    
    # Summary
    print("="*80)
    print(f"✓ Preprocessing Complete: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    
    if issparse(adata.X):
        sparsity = 1 - (adata.X.nnz / (adata.shape[0] * adata.shape[1]))
        print(f"  Sparsity: {sparsity*100:.2f}%")
        if adata.X.nnz > 0:
            print(f"  Range: [{adata.X.data.min():.4f}, {adata.X.data.max():.4f}]")
    else:
        print(f"  Range: [{adata.X.min():.4f}, {adata.X.max():.4f}]")
    
    print("="*80 + "\n")
    
    # Return the modified reference for pipeline composition.
    return adata


# ==============================================================================  
# Per-Sample Spatial Graph  
# ==============================================================================  

def build_per_sample_graph(  
    adata,  
    n_neighbors: int = 8,  
    sample_key: str = 'sample_id',  
    coord_type: str = 'generic',  
    spatial_key: str = 'spatial',  
):  
    """  
    Build spatial graph separately for each sample, then combine.  
    
    For multi-sample datasets, this avoids creating edges between samples.  
    Each sample gets an independent spatial graph, then Laplacians are  
    combined as a block diagonal matrix.  
    
    Parameters  
    ----------  
    adata : AnnData  
        Spatial data with multiple samples  
    n_neighbors : int, default 8  
        Number of nearest neighbors per spot/cell  
    sample_key : str, default 'sample_id'  
        Key in obs identifying samples  
    coord_type : str, default 'generic'  
        Coordinate type for squidpy ('generic', 'grid', or 'visium')  
    spatial_key : str, default 'spatial'  
        Key in obsm with spatial coordinates  
    """  
    import squidpy as sq  
    
    if sample_key not in adata.obs.columns:  
        raise ValueError(f"'{sample_key}' not found in adata.obs")  
    if spatial_key not in adata.obsm:
        raise ValueError(f"'{spatial_key}' not found in adata.obsm")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    if adata.obs[sample_key].isna().any():
        raise ValueError(f"'{sample_key}' contains missing sample identifiers")
    
    # Preserve first-occurrence order for reporting, but assemble every local
    # graph with its actual global row indices. This remains correct even when
    # sample rows are not contiguous or labels do not sort like row blocks.
    sample_ids = list(pd.unique(adata.obs[sample_key]))
    print(f"Building spatial graphs for {len(sample_ids)} samples (k={n_neighbors})...")  
    
    global_rows = []
    global_cols = []
    global_data = []
    
    for i, sample_id in enumerate(sample_ids, 1):  
        sample_mask = np.asarray(adata.obs[sample_key] == sample_id)
        sample_indices = np.flatnonzero(sample_mask)
        adata_sample = adata[sample_mask].copy()  
        n_spots = adata_sample.shape[0]  
        
        if n_spots < n_neighbors + 1:  
            print(f"  ⚠ [{i}/{len(sample_ids)}] {sample_id}: only {n_spots} spots, skipping")  
            continue  
        
        sq.gr.spatial_neighbors(  
            adata_sample, coord_type=coord_type,  
            n_neighs=n_neighbors, spatial_key=spatial_key  
        )  
        compute_laplacian(adata_sample)  
        local = adata_sample.uns['spatial_laplacian'].tocoo()
        global_rows.append(sample_indices[local.row])
        global_cols.append(sample_indices[local.col])
        global_data.append(local.data)
        
        if i % 10 == 0 or i == len(sample_ids):  
            print(f"  [{i}/{len(sample_ids)}] Processed")  
    
    if global_data:
        rows = np.concatenate(global_rows)
        cols = np.concatenate(global_cols)
        values = np.concatenate(global_data)
        L_combined = csr_matrix(
            (values, (rows, cols)), shape=(adata.n_obs, adata.n_obs)
        )
    else:
        L_combined = csr_matrix((adata.n_obs, adata.n_obs), dtype=np.float32)
    adata.uns['spatial_laplacian'] = L_combined  
    
    print(f"✓ Combined Laplacian: {L_combined.shape}, {L_combined.nnz:,} non-zeros\n")  
    return adata  


# ==============================================================================  
# Coordinate Alignment  
# ==============================================================================  

def _alignment_cache_signature(adata, sample_key, spatial_key, settings):
    """Hash row identity, input coordinates and layout settings."""
    digest = hashlib.sha256()
    for values in (adata.obs_names, adata.obs[sample_key]):
        for value in values:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    coords = np.ascontiguousarray(np.asarray(adata.obsm[spatial_key]))
    digest.update(str(coords.dtype).encode("ascii"))
    digest.update(coords.view(np.uint8))
    digest.update(repr(settings).encode("utf-8"))
    return digest.hexdigest()

def align_samples_grid(  
    adata,  
    samples_per_row: int = 5,  
    normalize_size: bool = True,  
    target_width: float = 100.0,  
    horizontal_gap: float = 20.0,  
    vertical_gap: float = 20.0,  
    sample_key: str = 'sample_id',  
    spatial_key: str = 'spatial',  
    output_key: str = 'spatial_aligned',  
    cache_file: Optional[str] = None,  
):  
    """  
    Align multiple samples in a grid layout for visualization.  
    
    Arranges samples in a grid while optionally normalizing their sizes  
    to make them comparable. Preserves aspect ratios.  
    
    Parameters  
    ----------  
    adata : AnnData  
        Data with multiple samples  
    samples_per_row : int, default 5  
        Number of samples per row in grid  
    normalize_size : bool, default True  
        Normalize each sample to similar size (preserves aspect ratio)  
    target_width : float, default 100  
        Target width after normalization  
    horizontal_gap : float, default 20  
        Gap between samples horizontally  
    vertical_gap : float, default 20  
        Gap between samples vertically  
    sample_key : str, default 'sample_id'  
        Key in obs identifying samples  
    spatial_key : str, default 'spatial'  
        Input spatial coordinates key in obsm  
    output_key : str, default 'spatial_aligned'  
        Output aligned coordinates key in obsm  
    cache_file : str, optional  
        Path to cache aligned coordinates  
    """  
    if sample_key not in adata.obs.columns:
        raise ValueError(f"'{sample_key}' not found in adata.obs")
    if spatial_key not in adata.obsm:
        raise ValueError(f"'{spatial_key}' not found in adata.obsm")
    if samples_per_row < 1:
        raise ValueError("samples_per_row must be at least 1")
    if adata.obs[sample_key].isna().any():
        raise ValueError(f"'{sample_key}' contains missing sample identifiers")
    coords_input = np.asarray(adata.obsm[spatial_key])
    if coords_input.ndim != 2 or coords_input.shape != (adata.n_obs, 2):
        raise ValueError(
            f"obsm['{spatial_key}'] must have shape ({adata.n_obs}, 2); "
            f"found {coords_input.shape}"
        )

    cache_settings = (
        samples_per_row, normalize_size, target_width,
        horizontal_gap, vertical_gap, sample_key, spatial_key,
    )
    cache_signature = _alignment_cache_signature(
        adata, sample_key, spatial_key, cache_settings
    )

    if cache_file and os.path.exists(cache_file):  
        print(f"✓ Loading from cache: {cache_file}")  
        with open(cache_file, 'rb') as f:  
            payload = pickle.load(f)
        if payload.get('signature') != cache_signature:
            raise ValueError(
                "Alignment cache does not match the current observations, "
                "coordinates or layout settings"
            )
        cached = np.asarray(payload['coords'])
        if cached.shape != (adata.n_obs, 2):
            raise ValueError(
                f"Cached coordinates have shape {cached.shape}; "
                f"expected ({adata.n_obs}, 2)"
            )
        adata.obsm[output_key] = cached
        return adata  

    sample_ids = list(pd.unique(adata.obs[sample_key]))
    print(f"Aligning {len(sample_ids)} samples in {samples_per_row}-column grid...")  
    
    aligned_coords = np.zeros((adata.shape[0], 2), dtype=np.float32)  
    sample_data = {}  
    
    # Normalize each sample  
    for sample_id in sample_ids:  
        mask = adata.obs[sample_key] == sample_id  
        coords = adata.obsm[spatial_key][mask].copy()  
        
        mins, maxs = coords.min(axis=0), coords.max(axis=0)  
        width, height = maxs[0] - mins[0], maxs[1] - mins[1]  
        coords_centered = coords - mins  
        
        if normalize_size and width > 0:  
            scale = target_width / width  
            coords_normalized = coords_centered * scale  
            normalized_width, normalized_height = target_width, height * scale  
        else:  
            coords_normalized = coords_centered  
            normalized_width, normalized_height = width, height  
        
        sample_data[sample_id] = {  
            'coords': coords_normalized,  
            'width': normalized_width,  
            'height': normalized_height,  
        }  
    
    # Arrange in grid  
    x_offset = y_offset = row_max_height = 0  
    
    for i, sample_id in enumerate(sample_ids):  
        data = sample_data[sample_id]  
        mask = adata.obs[sample_key] == sample_id  
        spot_indices = np.where(mask)[0]  
        
        coords_placed = data['coords'].copy()  
        coords_placed[:, 0] += x_offset  
        coords_placed[:, 1] += y_offset  
        aligned_coords[spot_indices] = coords_placed  
        
        row_max_height = max(row_max_height, data['height'])  
        col = i % samples_per_row  
        
        if col < samples_per_row - 1:  
            x_offset += data['width'] + horizontal_gap  
        else:  
            x_offset = 0  
            y_offset += row_max_height + vertical_gap  
            row_max_height = 0  
    
    adata.obsm[output_key] = aligned_coords  
    
    if cache_file:  
        cache_parent = os.path.dirname(cache_file)
        if cache_parent:
            os.makedirs(cache_parent, exist_ok=True)
        with open(cache_file, 'wb') as f:  
            pickle.dump({'coords': aligned_coords, 'signature': cache_signature}, f)
        print(f"  ✓ Cached to: {cache_file}")  
    
    print(f"✓ Aligned coordinates saved to obsm['{output_key}']\n")  
    return adata  


# ==============================================================================  
# Q Normalization and W Scaling  
# ==============================================================================  

def normalize_Q_scale_W(  
    adata,  
    W_key: str = 'X_sort',  
    Q_key: str = 'sort_signatures',  
    W_output_key: str = 'X_sort_scaled',  
    Q_output_key: str = 'sort_signatures_normalized',  
    scaling_key: str = 'sort_scaling_factors',  
):  
    """  
    L2-normalize Q signatures and scale W accordingly.  
    
    This preserves reconstruction: X ≈ W @ Q.T = W_scaled @ Q_normalized.T  
    
    Useful for:  
    - Comparing component magnitudes across datasets  
    - Interpreting gene signature weights independently of loading scales  
    - Better numerical stability  
    
    Parameters  
    ----------  
    adata : AnnData  
        Data with SORT results  
    W_key : str, default 'X_sort'  
        Key in obsm for loadings  
    Q_key : str, default 'sort_signatures'  
        Key in varm for signatures  
    W_output_key : str, default 'X_sort_scaled'  
        Output key for scaled loadings  
    Q_output_key : str, default 'sort_signatures_normalized'  
        Output key for normalized signatures  
    scaling_key : str, default 'sort_scaling_factors'  
        Key in uns to store scaling factors  
    """  
    if W_key not in adata.obsm or Q_key not in adata.varm:  
        raise ValueError(f"Missing '{W_key}' or '{Q_key}'")  
    
    print("Normalizing Q and scaling W...")  
    
    W, Q = adata.obsm[W_key].copy(), adata.varm[Q_key].copy()  
    q_norms = np.linalg.norm(Q, axis=0, keepdims=True)  
    q_norms = np.where(q_norms == 0, 1.0, q_norms)  
    
    Q_normalized = Q / q_norms  
    W_scaled = W * q_norms  
    
    # Check reconstruction equivalence in row chunks. Materializing both full
    # N-by-G reconstructions can require tens of gigabytes for atlas inputs.
    recon_error = 0.0
    for start in range(0, W.shape[0], 512):
        stop = min(start + 512, W.shape[0])
        original = W[start:stop] @ Q.T
        rescaled = W_scaled[start:stop] @ Q_normalized.T
        if original.size:
            recon_error = max(recon_error, float(np.max(np.abs(original - rescaled))))
    
    adata.obsm[W_output_key] = W_scaled.astype(np.float32)  
    adata.varm[Q_output_key] = Q_normalized.astype(np.float32)  
    adata.uns[scaling_key] = q_norms.flatten().astype(np.float32)  
    
    print(f"  Q norms - mean: {q_norms.mean():.4f}, range: [{q_norms.min():.4f}, {q_norms.max():.4f}]")  
    print(f"  Reconstruction error: {recon_error:.2e}")  
    print(f"✓ Stored: obsm['{W_output_key}'], varm['{Q_output_key}'], uns['{scaling_key}']\n")  
    
    return adata  


# ==============================================================================  
# Complete Pipeline  
# ==============================================================================  

def run_sort_pipeline(  
    adata,  
    n_components: int = 40,  
    n_hvg: Optional[int] = 3000,  
    n_neighbors: int = 8,  
    sample_key: str = 'sample_id',  
    # SORT model parameters  
    alpha: float = 0.6,  
    beta: float = 0.5,  # Deprecated; retained for archived callers.
    lambda_l1_W: float = 0.5,  
    lambda_l1_Q: float = 80.0,  
    use_tv: bool = True,  
    # Preprocessing parameters  
    preprocess: bool = True,  
    target_sum: Optional[float] = 1e4,  
    log_transform: bool = True,  
    min_genes: int = 200,  
    min_cells_pct: float = 0.01,  
    min_mean_expression: float = 0.01,  
    # Graph building  
    build_graph: bool = True,  
    # Post-processing  
    normalize_wq: bool = False,  
    # System  
    device: str = 'auto',  
    random_state: int = 42,  
    copy: bool = False,  
    lambda_neg: float = 1.0,
    **kwargs  
):  
    """Compatibility wrapper retained for archived internal callers.

    Complete SORT analysis pipeline with full parameter control.  
    
    Workflow:  
    1. Preprocessing (optional, with customizable parameters)  
    2. Spatial graph building (per-sample if multiple samples)  
    3. SORT decomposition  
    4. Q normalization and W scaling (optional)  
    
    Parameters  
    ----------  
    adata : AnnData  
        Spatial transcriptomics data  
    n_components : int, default 40  
        Number of SORT components (including background)  
    n_hvg : int, optional, default 3000  
        Number of highly variable genes (None = use all genes)  
    n_neighbors : int, default 8  
        Number of neighbors for spatial graph  
    sample_key : str, default 'sample_id'  
        Key identifying samples (for multi-sample data)  
        
    SORT Model Parameters  
    ---------------------  
    alpha : float, default 0.6  
        Weight for spatial regularization (0.0-1.0), higher = more smoothing  
    beta : float, default 0.5  
        Deprecated compatibility argument; ignored by the current optimizer.
    lambda_l1_W : float, default 0.5  
        L1 penalty for loadings W (sparsity in cells)  
    lambda_l1_Q : float, default 80.0  
        L1 penalty for signatures Q (sparsity in genes)  
    use_tv : bool, default True  
        Use total variation regularization  
        
    Preprocessing Parameters  
    ------------------------  
    preprocess : bool, default True  
        Run preprocessing pipeline  
    target_sum : float, optional, default 1e4  
        Target sum for normalization (None to skip)  
    log_transform : bool, default True  
        Apply log1p transformation  
    min_genes : int, default 200  
        Minimum genes per cell/spot for QC  
    min_cells_pct : float, default 0.01  
        Minimum percentage of cells expressing gene  
    min_mean_expression : float, default 0.01  
        Minimum mean expression per gene  
        
    Graph & Post-processing  
    -----------------------  
    build_graph : bool, default True  
        Build spatial graph  
    normalize_wq : bool, default False  
        L2-normalize Q and scale W after decomposition  
        
    System  
    ------  
    device : str, default 'auto'  
        'auto', 'cupy', or 'numpy'  
    random_state : int, default 42  
        Random seed  
    copy : bool, default False  
        Return a copy  
    lambda_neg : float, default 1.0
        Negative-loading penalty forwarded to ``decompose``. The default is
        the value reached by the manuscript pipeline.
        
    **kwargs : dict  
        Additional arguments for decompose():  
        stage1_epochs, stage2_epochs, subsample_size, W_init, Q_init,
        auto_init, init_kwargs,
        l1_weight_strategy, tv_epsilon, tv_update_freq, tv_stage, verbose  
    """  
    if copy:
        adata = adata.copy()
    
    print("="*80)
    print("SORT ANALYSIS PIPELINE")
    print("="*80)
    
    if device == 'auto':
        device = detect_device()
    print(f"Device: {device}\n")
    
    # Step 1: Preprocessing
    # Keep the returned reference for the following steps.
    if preprocess:
        print("="*80)
        print("Step 1: Preprocessing")
        print("="*80)
        adata = standard_preprocess(
            adata,
            target_sum=target_sum,
            log_transform=log_transform,
            n_hvg=n_hvg,
            batch_key=sample_key,
            min_genes=min_genes,
            min_cells_pct=min_cells_pct,
            min_mean_expression=min_mean_expression,
            copy=False
        )
    else:
        print("Step 1: Preprocessing - SKIPPED\n")
    
    # Step 2: Spatial graph
    if build_graph:
        print("="*80)
        print("Step 2: Building Spatial Graph")
        print("="*80)
        
        if sample_key in adata.obs.columns:
            n_samples = adata.obs[sample_key].nunique()
            if n_samples > 1:
                print(f"Multi-sample dataset detected ({n_samples} samples)")
                build_per_sample_graph(adata, n_neighbors=n_neighbors, sample_key=sample_key)
            else:
                print("Single sample detected")
                from .preprocessing import build_spatial_graph
                build_spatial_graph(adata, n_neighbors=n_neighbors)
                compute_laplacian(adata)
                print()
        else:
            print("No sample_key - treating as single sample")
            from .preprocessing import build_spatial_graph
            build_spatial_graph(adata, n_neighbors=n_neighbors)
            compute_laplacian(adata)
            print()
    else:
        print("Step 2: Spatial Graph - SKIPPED\n")
    
    # Step 3: SORT decomposition
    print("="*80)
    print("Step 3: SORT Decomposition")
    print("="*80)
    
    # Preprocessing above has already selected the modeled genes.
    valid_decompose_params = {
        'stage1_epochs', 'stage2_epochs',
        'subsample_size', 'W_init', 'Q_init', 'auto_init', 'init_kwargs',
        'l1_weight_strategy', 'tv_epsilon', 'tv_update_freq', 'tv_stage', 'verbose'
    }
    
    decompose_kwargs = {k: v for k, v in kwargs.items() if k in valid_decompose_params}
    
    # Decompose the current X without applying a second gene filter.
    decompose_kwargs['use_highly_variable'] = False
    decompose_kwargs['layer'] = None
    
    unused_params = set(kwargs.keys()) - valid_decompose_params
    if unused_params:
        names = ", ".join(sorted(unused_params))
        raise TypeError(
            f"Unsupported run_sort_pipeline parameter(s): {names}"
        )
    
    decompose(
        adata,
        n_components=n_components,
        alpha=alpha,
        beta=beta,
        lambda_l1_W=lambda_l1_W,
        lambda_l1_Q=lambda_l1_Q,
        lambda_neg=lambda_neg,
        use_tv=use_tv,
        device=device,
        random_state=random_state,
        copy=False,
        **decompose_kwargs
    )
    
    # Step 4: Q normalization (optional)
    if normalize_wq:
        print("\n" + "="*80)
        print("Step 4: Normalizing Q and Scaling W")
        print("="*80)
        normalize_Q_scale_W(adata)
    
    # Summary
    print("="*80)
    print("✓ SORT Pipeline Complete")
    print("="*80)
    print(f"Dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    print(f"Results: adata.obsm['X_sort'] ({adata.obsm['X_sort'].shape})")
    print(f"         adata.varm['sort_signatures'] ({adata.varm['sort_signatures'].shape})")
    
    if normalize_wq:
        print(f"Normalized: adata.obsm['X_sort_scaled']")
        print(f"            adata.varm['sort_signatures_normalized']")
    
    if 'sort_params' in adata.uns:
        print(f"Parameters: adata.uns['sort_params']")
    
    print("="*80 + "\n")
    
    if copy:
        return adata


# ==============================================================================
# Results Export
# ==============================================================================

def export_sort_results(
    adata,
    output_dir: str,
    prefix: str = 'sort',
    top_genes: int = 50,
    use_normalized: bool = False,
    save_adata: bool = False,
    metadata_cols: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Export SORT results to files.
    
    Saves:
    - W_loadings.npy: Cell/spot loadings
    - Q_signatures.npy: Gene signatures
    - gene_names.csv: Gene names corresponding to Q rows
    - spatial_coordinates.csv: Spatial coordinates for W rows
    - top_genes.csv: Top genes per component
    - metadata.csv: Cell/spot metadata (optional)
    - parameters.csv: Analysis parameters
    - (optional) results.h5ad: Full AnnData object
    
    Parameters
    ----------
    adata : AnnData
        Data with SORT results
    output_dir : str
        Output directory path
    prefix : str, default 'sort'
        Prefix for output files
    top_genes : int, default 50
        Number of top genes to export per component
    use_normalized : bool, default False
        Export normalized Q and scaled W (if available)
    save_adata : bool, default False
        Save full AnnData object
    metadata_cols : list of str, optional
        Specific columns to include in metadata.csv
        
    Returns
    -------
    file_paths : dict
        Dictionary mapping file types to paths
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting results to: {output_dir}")
    
    file_paths = {}
    
    # Select W and Q
    if use_normalized and 'X_sort_scaled' in adata.obsm:
        W = adata.obsm['X_sort_scaled']
        Q = adata.varm['sort_signatures_normalized']
        suffix = '_scaled'
    else:
        W = adata.obsm['X_sort']
        Q = adata.varm['sort_signatures']
        suffix = ''
    
    # 1. Loadings and Signatures
    w_path = os.path.join(output_dir, f'{prefix}_W_loadings{suffix}.npy')
    q_path = os.path.join(output_dir, f'{prefix}_Q_signatures{suffix}.npy')
    np.save(w_path, W)
    np.save(q_path, Q)
    file_paths['loadings'] = w_path
    file_paths['signatures'] = q_path
    print(f"  ✓ W: {W.shape}")
    print(f"  ✓ Q: {Q.shape}")
    
    # 2. Gene Names
    gene_names_path = os.path.join(output_dir, f'{prefix}_gene_names.csv')
    pd.DataFrame({
        'gene_index': np.arange(len(adata.var_names)),
        'gene_name': adata.var_names.tolist()
    }).to_csv(gene_names_path, index=False)
    file_paths['gene_names'] = gene_names_path
    print(f"  ✓ Genes: {len(adata.var_names)}")
    
    # 3. Spatial Coordinates
    if 'spatial' in adata.obsm:
        coords_df = pd.DataFrame({
            'spot_index': np.arange(adata.shape[0]),
            'spatial_x': adata.obsm['spatial'][:, 0],
            'spatial_y': adata.obsm['spatial'][:, 1]
        })
        
        if 'sample_id' in adata.obs.columns:
            coords_df['sample_id'] = adata.obs['sample_id'].values
        
        if 'spatial_aligned' in adata.obsm:
            coords_df['aligned_x'] = adata.obsm['spatial_aligned'][:, 0]
            coords_df['aligned_y'] = adata.obsm['spatial_aligned'][:, 1]
        
        coords_path = os.path.join(output_dir, f'{prefix}_spatial_coordinates.csv')
        coords_df.to_csv(coords_path, index=False)
        file_paths['spatial_coords'] = coords_path
        print(f"  ✓ Coordinates: {adata.shape[0]} spots")
    
    # 4. Scaling Factors (if normalized)
    if use_normalized and 'sort_scaling_factors' in adata.uns:
        scaling_path = os.path.join(output_dir, f'{prefix}_scaling_factors.npy')
        np.save(scaling_path, adata.uns['sort_scaling_factors'])
        file_paths['scaling'] = scaling_path
        print(f"  ✓ Scaling factors")
    
    # 5. Top Genes
    top_genes_list = []
    for comp in range(Q.shape[1]):
        comp_name = 'Background' if comp == 0 else f'Signal_{comp}'
        q_comp = Q[:, comp]
        top_idx = np.argsort(np.abs(q_comp))[-top_genes:][::-1]
        
        for rank, idx in enumerate(top_idx, 1):
            top_genes_list.append({
                'component': comp_name,
                'component_id': comp,
                'rank': rank,
                'gene': adata.var_names[idx],
                'score': q_comp[idx],
                'abs_score': np.abs(q_comp[idx])
            })
    
    top_genes_path = os.path.join(output_dir, f'{prefix}_top_genes.csv')
    pd.DataFrame(top_genes_list).to_csv(top_genes_path, index=False)
    file_paths['top_genes'] = top_genes_path
    print(f"  ✓ Top genes: {len(top_genes_list)} entries")
    
    # 6. Metadata
    if metadata_cols is None:
        metadata_cols = [col for col in adata.obs.columns 
                        if col in ['sample_id', 'cohort', 'patient', 'tumor', 
                                  'region', 'Mouse_ID', 'Slice_ID', 'Sample_type',
                                  'Tier1', 'Tier3', 'patient_id', 'ukf_id']]
    
    if metadata_cols:
        metadata = adata.obs[metadata_cols].copy()
        metadata.insert(0, 'spot_index', np.arange(len(metadata)))
        
        metadata_path = os.path.join(output_dir, f'{prefix}_metadata.csv')
        metadata.to_csv(metadata_path, index=False)
        file_paths['metadata'] = metadata_path
        print(f"  ✓ Metadata: {len(metadata_cols)} columns")
    
    # 7. Parameters
    if 'sort_params' in adata.uns:
        params_path = os.path.join(output_dir, f'{prefix}_parameters.csv')
        pd.DataFrame([adata.uns['sort_params']]).to_csv(params_path, index=False)
        file_paths['parameters'] = params_path
        print(f"  ✓ Parameters")
    
    # 8. Full AnnData (optional)
    if save_adata:
        adata_path = os.path.join(output_dir, f'{prefix}_results.h5ad')
        adata.write_h5ad(adata_path)
        file_paths['adata'] = adata_path
        print(f"  ✓ Full AnnData")
    
    print(f"\n✓ Exported {len(file_paths)} file(s)\n")
    
    return file_paths
