# GEP characterization utilities.

"""
sort_gep_characterize.py

GEP characterization pipeline for SORT decomposition results.

Implements model-aware gene-program association metrics derived from the
low-rank reconstruction X_hat = W @ Q.T, including reconstruction variance
contribution (R_var), reconstruction concordance (C_recon), and relative
exclusivity (E_gk). Also provides GEP quality assessment combining W-side
activity statistics with Q-side gene support metrics, and a model-aware
GEP annotation score against reference signatures.

Pipeline
--------
stats          = precompute_stats(W, Q)
R_var          = compute_variance_contribution(stats)
C_recon        = compute_reconstruction_concordance(stats)
E_gk           = compute_exclusivity(R_var)
gene_df        = rank_genes_per_gep(R_var, C_recon, E_gk, gene_names)
quality_df     = assess_gep_quality(R_var, C_recon, W, sample_ids)
annotation_df  = annotate_geps_from_signatures(C_recon, Q, signature_dict)

Dependencies
------------
sort_downstream.py  (classify_geps, used internally in assess_gep_quality)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

__all__ = [
    'precompute_stats',
    'compute_variance_contribution',
    'compute_reconstruction_concordance',
    'compute_exclusivity',
    'rank_genes_per_gep',
    'assess_gep_quality',
    'annotate_geps_from_signatures',
]

_EPS = 1e-10


# ==============================================================================
# 0. Precompute shared statistics
# ==============================================================================

def precompute_stats(
    W: np.ndarray,
    Q: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute and cache all statistics needed for downstream gene-program metrics.

    All metrics operate on the low-rank reconstruction X_hat = W @ Q.T.
    Rather than materialising X_hat explicitly (which may be large), all
    required quantities are derived from the program covariance matrix
    Sigma_W and the cross-product matrix M = Q @ Sigma_W.

    Parameters
    ----------
    W : ndarray (n_spots, K)
        Program activity matrix. Column 0 (mean offset) should be excluded
        before calling this function if not relevant to biological analysis.
    Q : ndarray (n_genes, K)
        Gene loading matrix, same column ordering as W.

    Returns
    -------
    dict with keys
        W          : (n_spots, K)   original W (stored for classify_geps)
        Q          : (n_genes, K)   original Q
        Sigma_W    : (K, K)         sample covariance of program activities
        sigma2_W   : (K,)           per-program marginal variance (diag of Sigma_W)
        M          : (n_genes, K)   Q @ Sigma_W
        v_xhat     : (n_genes,)     per-gene reconstructed variance = sum_k M_gk * q_gk
        active_k   : (K,) bool      programs with sigma2_W > eps (used to mask outputs)
    """
    n_spots, K = W.shape
    n_genes    = Q.shape[0]

    if Q.shape[1] != K:
        raise ValueError(f"W has {K} components but Q has {Q.shape[1]}")

    W_c     = W - W.mean(axis=0)
    Sigma_W = (W_c.T @ W_c) / (n_spots - 1)          # (K, K)
    sigma2_W = np.diag(Sigma_W)                        # (K,)
    active_k = sigma2_W > _EPS

    M       = Q @ Sigma_W                              # (n_genes, K)
    v_xhat  = (M * Q).sum(axis=1)                      # (n_genes,)

    return dict(
        W=W, Q=Q,
        Sigma_W=Sigma_W, sigma2_W=sigma2_W,
        M=M, v_xhat=v_xhat,
        active_k=active_k,
    )


# ==============================================================================
# 1. Reconstruction variance contribution  R_var[g, k]
# ==============================================================================

def compute_variance_contribution(
    stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Marginal variance contribution of each program to each gene's reconstructed
    variance.

    R_var[g, k] = q[g,k]^2 * sigma2_W[k] / (v_xhat[g] + eps)

    Note: sum_k R_var[g, k] != 1 in general because cross-program covariance
    terms are not included in the numerator. R_var is therefore a marginal
    contribution proxy, not a strict variance partition.

    Invariance: R_var is scale-invariant with respect to absolute gene
    expression level, correcting the systematic downweighting of lowly
    expressed genes in loading-only scores.

    Programs flagged inactive (sigma2_W < eps) receive R_var = 0.

    Parameters
    ----------
    stats : dict returned by precompute_stats

    Returns
    -------
    R_var : ndarray (n_genes, K)
    """
    Q        = stats['Q']
    sigma2_W = stats['sigma2_W']
    v_xhat   = stats['v_xhat']
    active_k = stats['active_k']

    numerator = Q ** 2 * sigma2_W[np.newaxis, :]      # (n_genes, K)
    denom     = v_xhat[:, np.newaxis] + _EPS           # (n_genes, 1)
    R_var     = numerator / denom

    # zero out inactive programs
    R_var[:, ~active_k] = 0.0

    return R_var


# ==============================================================================
# 2. Reconstruction concordance  C_recon[g, k]
# ==============================================================================

def compute_reconstruction_concordance(
    stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Signed Pearson correlation between gene g's reconstructed expression and
    program k's spatial activity profile.

    C_recon[g, k] = sign(q[g,k]) * M[g,k] / sqrt(v_xhat[g] * sigma2_W[k])

    where M = Q @ Sigma_W.

    Properties
    ----------
    - Values in [-1, 1].
    - Invariant to additive (spatially uniform) background expression,
      because Pearson correlation is shift-invariant.
    - Uses only precomputed quantities; X_hat is never materialised.
    - Programs flagged inactive receive C_recon = 0.

    Parameters
    ----------
    stats : dict returned by precompute_stats

    Returns
    -------
    C_recon : ndarray (n_genes, K)
    """
    Q        = stats['Q']
    M        = stats['M']
    sigma2_W = stats['sigma2_W']
    v_xhat   = stats['v_xhat']
    active_k = stats['active_k']

    denom   = (np.sqrt(v_xhat[:, np.newaxis])
               * np.sqrt(sigma2_W[np.newaxis, :])
               + _EPS)                                  # (n_genes, K)

    C_recon = np.sign(Q) * M / denom
    C_recon = np.clip(C_recon, -1.0, 1.0)

    C_recon[:, ~active_k] = 0.0

    return C_recon


# ==============================================================================
# 3. Relative exclusivity  E_gk
# ==============================================================================

def compute_exclusivity(
    R_var: np.ndarray,
) -> np.ndarray:
    """
    Relative exclusivity of program k for gene g.

    E_gk = R_var[g, k] - max_{l != k} R_var[g, l]

    Positive values indicate that program k contributes more reconstructed
    variance to gene g than any other single program. Treated as a graded
    confidence score; strong exclusivity is biologically uncommon due to
    background expression in multiple compartments.

    Parameters
    ----------
    R_var : ndarray (n_genes, K)

    Returns
    -------
    E_gk : ndarray (n_genes, K)
    """
    K     = R_var.shape[1]
    E_gk  = np.empty_like(R_var)

    for k in range(K):
        other_cols    = np.delete(R_var, k, axis=1)   # (n_genes, K-1)
        max_other     = other_cols.max(axis=1)         # (n_genes,)
        E_gk[:, k]   = R_var[:, k] - max_other

    return E_gk


# ==============================================================================
# 4. Per-GEP gene ranking table
# ==============================================================================

def rank_genes_per_gep(
    R_var: np.ndarray,
    C_recon: np.ndarray,
    E_gk: np.ndarray,
    gene_names: List[str],
    c_recon_threshold: float = 0.4,
    r_var_top_quantile: float = 0.80,
    min_exclusive_e: float = 0.0,
    n_anchor: int = 50,
    n_representative: int = 50,
    n_exclusive: int = 50,
    verbose: bool = True,
) -> Dict[int, pd.DataFrame]:
    """
    For each GEP, derive three gene lists and return as a ranked DataFrame.

    Gene categories
    ---------------
    Anchor marker
        q[g,k] > 0
        R_var[g,k] in top r_var_top_quantile of column k
        C_recon[g,k] >= c_recon_threshold
        argmax_l R_var[g,l] == k  (program k is the dominant contributor)
        Used for overrepresentation analysis and GEP annotation.

    Representative gene
        q[g,k] > 0
        C_recon[g,k] >= c_recon_threshold
        No exclusivity requirement.
        Used for spatial visualisation.

    Exclusive marker
        Anchor marker criteria AND E_gk[g,k] >= min_exclusive_e
        High-confidence program-specific genes for cross-dataset validation.

    Parameters
    ----------
    R_var              : ndarray (n_genes, K)
    C_recon            : ndarray (n_genes, K)
    E_gk               : ndarray (n_genes, K)
    gene_names         : list[str], length n_genes
    c_recon_threshold  : float, default 0.4
    r_var_top_quantile : float, default 0.80  (top 20% of column)
    min_exclusive_e    : float, default 0.0   (E_gk > 0 means dominant)
    n_anchor           : int, default 50
    n_representative   : int, default 50
    n_exclusive        : int, default 50
    verbose            : bool

    Returns
    -------
    dict[int, DataFrame]
        Keys are GEP indices. Each DataFrame has columns:
        gene, R_var, C_recon, E_gk, category
        sorted by C_recon descending within each category.
    """
    gene_names = np.asarray(gene_names)
    n_genes, K = R_var.shape
    Q          = np.sign(C_recon)  # sign proxy: sufficient for pos/neg test

    # actual sign from R_var numerator signs is ambiguous; use C_recon sign
    # (C_recon already encodes sign(q[g,k]))
    pos_loading = C_recon > 0      # approximation: C_recon > 0 iff q[g,k] > 0
                                   # when gene has any reconstructed variance

    dominant_gep = R_var.argmax(axis=1)   # (n_genes,)
    result = {}

    for k in range(K):
        r_thr  = np.quantile(R_var[:, k], r_var_top_quantile)
        r_col  = R_var[:, k]
        c_col  = C_recon[:, k]
        e_col  = E_gk[:, k]

        # masks
        pos_mask      = pos_loading[:, k]
        r_top_mask    = r_col >= r_thr
        c_thr_mask    = c_col >= c_recon_threshold
        dominant_mask = dominant_gep == k
        exclusive_mask = e_col >= min_exclusive_e

        anchor_mask       = pos_mask & r_top_mask & c_thr_mask & dominant_mask
        representative_mask = pos_mask & c_thr_mask
        exclusive_mask_full = anchor_mask & exclusive_mask

        def _build(mask, sort_by, n_top, label):
            idx  = np.where(mask)[0]
            if len(idx) == 0:
                return pd.DataFrame(columns=['gene', 'R_var', 'C_recon',
                                             'E_gk', 'category'])
            order = np.argsort(sort_by[idx])[::-1][:n_top]
            sel   = idx[order]
            return pd.DataFrame({
                'gene'    : gene_names[sel],
                'R_var'   : r_col[sel].round(6),
                'C_recon' : c_col[sel].round(6),
                'E_gk'    : e_col[sel].round(6),
                'category': label,
            })

        frames = [
            _build(anchor_mask,        c_col, n_anchor,        'anchor'),
            _build(representative_mask, c_col, n_representative, 'representative'),
            _build(exclusive_mask_full, c_col, n_exclusive,      'exclusive'),
        ]
        df = pd.concat(frames, ignore_index=True)
        result[k] = df

        if verbose:
            n_a = anchor_mask.sum()
            n_r = representative_mask.sum()
            n_e = exclusive_mask_full.sum()
            print(f"  GEP {k:>3d} | anchor={n_a:>4d}  "
                  f"representative={n_r:>4d}  exclusive={n_e:>4d}")

    return result


# ==============================================================================
# 5. GEP quality assessment  (W-side + Q-side)
# ==============================================================================

def assess_gep_quality(
    R_var: np.ndarray,
    C_recon: np.ndarray,
    W: np.ndarray,
    sample_ids: np.ndarray,
    c_recon_threshold: float = 0.4,
    n_eff_min: int = 5,
    n_high_concordance_min: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Assess per-GEP quality using Q-side gene support metrics and W-side
    activity classification.

    Q-side metrics
    --------------
    N_eff : effective gene support size (inverse-Simpson over R_var column)
        N_eff_k = (sum_g R_var[g,k])^2 / sum_g R_var[g,k]^2
        Reflects how many genes collectively account for the program variance.
    n_high_concordance : number of genes with C_recon[g,k] >= threshold
    top1_r_var : maximum single-gene variance contribution
        If top1_r_var > 0.5, the GEP may be driven by a single gene.

    W-side classification
    ---------------------
    Delegated to classify_geps from sort_downstream (imported lazily to
    avoid circular imports). Results are merged on GEP index.

    Quality flags
    -------------
    single_gene_driven : top1_r_var > 0.5
    low_gene_support   : N_eff < n_eff_min OR n_high_concordance < n_high_concordance_min
    interpretable      : not single_gene_driven AND not low_gene_support

    Parameters
    ----------
    R_var              : ndarray (n_genes, K)
    C_recon            : ndarray (n_genes, K)
    W                  : ndarray (n_spots, K)
    sample_ids         : ndarray (n_spots,)
    c_recon_threshold  : float, default 0.4
    n_eff_min          : int,   default 5
    n_high_concordance_min : int, default 5
    verbose            : bool

    Returns
    -------
    DataFrame with columns:
        GEP, N_eff, n_high_concordance, top1_r_var,
        single_gene_driven, low_gene_support, interpretable,
        Between, Within, Category  (from classify_geps)
    """
    from downstream import classify_geps   # lazy import

    K = R_var.shape[1]
    records = []

    for k in range(K):
        r_col  = R_var[:, k]
        c_col  = C_recon[:, k]

        r_sum  = r_col.sum()
        r_ss   = (r_col ** 2).sum()
        N_eff  = (r_sum ** 2 / r_ss) if r_ss > _EPS else 0.0

        n_hc   = int((c_col >= c_recon_threshold).sum())
        top1   = float(r_col.max())

        single_gene = top1 > 0.5
        low_support = (N_eff < n_eff_min) or (n_hc < n_high_concordance_min)
        interp      = not single_gene and not low_support

        records.append(dict(
            GEP=k,
            N_eff=round(N_eff, 2),
            n_high_concordance=n_hc,
            top1_r_var=round(top1, 4),
            single_gene_driven=single_gene,
            low_gene_support=low_support,
            interpretable=interp,
        ))

    q_df = pd.DataFrame(records)

    # W-side classification
    w_df = classify_geps(W, sample_ids, verbose=verbose)

    result = q_df.merge(w_df[['GEP', 'Between', 'Within', 'Category',
                               'frac_active', 'between_cv',
                               'within_cv', 'bimodal_frac']],
                        on='GEP', how='left')

    if verbose:
        print(f"\n[assess_gep_quality] {K} GEPs")
        print(f"  interpretable      : {result['interpretable'].sum()}")
        print(f"  single_gene_driven : {result['single_gene_driven'].sum()}")
        print(f"  low_gene_support   : {result['low_gene_support'].sum()}")

    return result


# ==============================================================================
# 6. Model-aware GEP annotation against reference signatures
# ==============================================================================

def annotate_geps_from_signatures(
    C_recon: np.ndarray,
    Q: np.ndarray,
    signature_dict: Dict[str, pd.DataFrame],
    gene_names: List[str],
    logfc_col: str = 'avg_logFC',
    logfc_min: float = 1.0,
    pct_diff_col: Optional[Tuple[str, str]] = ('pct_1', 'pct_2'),
    pct_diff_min: float = 20.0,
    min_genes: int = 5,
    score_threshold: float = 0.4,
    margin_threshold: float = 0.1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Score each GEP against a set of reference signatures using logFC-weighted
    mean reconstruction concordance.

    Score(k, S) = sum_{g in S+_k} w_g * C_recon[g,k]
                  / sum_{g in S+_k} w_g

    where S+_k is the subset of signature genes present in the panel with
    positive loading on program k, and w_g = logFC_g (from the reference table).

    Signature pre-filtering
    -----------------------
    Genes in each reference signature are filtered to those with
        logFC >= logfc_min
        pct_1 - pct_2 >= pct_diff_min  (if pct columns are available)
    to exclude broadly expressed genes unlikely to carry cluster-specific
    spatial information. GEP-signature pairs with fewer than min_genes
    qualifying genes are scored as NaN and excluded from annotation.

    Annotation assignment
    ---------------------
    A GEP is assigned the label of the highest-scoring signature provided:
        score  >= score_threshold
        margin >= margin_threshold  (gap to second-best signature)
    GEPs not meeting these criteria are labelled "unassigned".

    Parameters
    ----------
    C_recon        : ndarray (n_genes, K)
    Q              : ndarray (n_genes, K)
    signature_dict : dict[str, DataFrame]
        Keys are signature labels. Each DataFrame must contain at least
        a 'Symbol' (or index) column with gene names and logfc_col.
        If the DataFrame has no logfc_col, uniform weights are used.
    gene_names     : list[str], length n_genes
    logfc_col      : str, default 'avg_logFC'
    logfc_min      : float, default 1.0
    pct_diff_col   : tuple(str, str) or None
        Column names for pct_1 and pct_2. Set None to skip pct filtering.
    pct_diff_min   : float, default 20.0
    min_genes      : int, default 5
    score_threshold : float, default 0.4
    margin_threshold: float, default 0.1
    verbose        : bool

    Returns
    -------
    DataFrame with columns:
        GEP,
        [sig_name]_score  for each signature,
        [sig_name]_n_genes for each signature,
        best_signature, best_score, second_score, margin, annotation
    """
    gene_names  = np.asarray(gene_names)
    gene_index  = {g: i for i, g in enumerate(gene_names)}
    K           = C_recon.shape[1]
    sig_names   = list(signature_dict.keys())

    # pre-filter signatures and build (gene_idx, weight) lists
    filtered_sigs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sig_name, sig_df in signature_dict.items():
        # resolve gene column
        if 'Symbol' in sig_df.columns:
            genes = sig_df['Symbol'].values
        else:
            genes = sig_df.index.values

        # logFC filter
        if logfc_col in sig_df.columns:
            weights = sig_df[logfc_col].values.astype(float)
            keep    = weights >= logfc_min
        else:
            weights = np.ones(len(genes))
            keep    = np.ones(len(genes), dtype=bool)

        # pct difference filter
        if pct_diff_col is not None:
            p1_col, p2_col = pct_diff_col
            if p1_col in sig_df.columns and p2_col in sig_df.columns:
                pct_diff = sig_df[p1_col].values - sig_df[p2_col].values
                keep    &= pct_diff >= pct_diff_min

        genes   = genes[keep]
        weights = weights[keep]

        # intersect with panel
        idxs, ws = [], []
        for g, w in zip(genes, weights):
            if g in gene_index:
                idxs.append(gene_index[g])
                ws.append(w)

        if len(idxs) == 0:
            filtered_sigs[sig_name] = (np.array([], dtype=int),
                                       np.array([], dtype=float))
        else:
            filtered_sigs[sig_name] = (np.array(idxs), np.array(ws))

    # score matrix: (K, n_sigs)
    score_matrix = np.full((K, len(sig_names)), np.nan)
    ngene_matrix = np.zeros((K, len(sig_names)), dtype=int)

    for j, sig_name in enumerate(sig_names):
        idxs, ws = filtered_sigs[sig_name]
        if len(idxs) == 0:
            continue
        for k in range(K):
            # restrict to genes with positive loading on program k
            pos_mask = Q[idxs, k] > 0
            sel_idx  = idxs[pos_mask]
            sel_ws   = ws[pos_mask]
            if len(sel_idx) < min_genes:
                ngene_matrix[k, j] = len(sel_idx)
                continue
            c_vals             = C_recon[sel_idx, k]
            score_matrix[k, j] = np.average(c_vals, weights=sel_ws)
            ngene_matrix[k, j] = len(sel_idx)

    # build result DataFrame
    records = []
    for k in range(K):
        row = {'GEP': k}
        for j, sig_name in enumerate(sig_names):
            row[f'{sig_name}_score']   = (round(float(score_matrix[k, j]), 4)
                                          if not np.isnan(score_matrix[k, j])
                                          else np.nan)
            row[f'{sig_name}_n_genes'] = int(ngene_matrix[k, j])

        # find best and second-best (ignoring NaN)
        valid_mask = ~np.isnan(score_matrix[k])
        if valid_mask.sum() == 0:
            row.update(best_signature='unassigned', best_score=np.nan,
                       second_score=np.nan, margin=np.nan,
                       annotation='unassigned')
        else:
            scores_valid = score_matrix[k].copy()
            scores_valid[~valid_mask] = -np.inf
            order        = np.argsort(scores_valid)[::-1]
            best_j       = order[0]
            best_score   = float(score_matrix[k, best_j])
            second_score = (float(score_matrix[k, order[1]])
                            if valid_mask.sum() >= 2 else np.nan)
            margin       = (best_score - second_score
                            if not np.isnan(second_score) else best_score)

            if (best_score >= score_threshold and margin >= margin_threshold):
                annotation = sig_names[best_j]
            else:
                annotation = 'unassigned'

            row.update(
                best_signature=sig_names[best_j],
                best_score=round(best_score, 4),
                second_score=round(second_score, 4) if not np.isnan(second_score) else np.nan,
                margin=round(margin, 4),
                annotation=annotation,
            )
        records.append(row)

    result = pd.DataFrame(records)

    if verbose:
        n_assigned = (result['annotation'] != 'unassigned').sum()
        print(f"[annotate_geps] {K} GEPs | "
              f"{len(sig_names)} signatures | "
              f"annotated: {n_assigned}/{K}")
        for _, r in result[result['annotation'] != 'unassigned'].iterrows():
            print(f"  GEP {int(r.GEP):>3d} → {r.annotation:<25s} "
                  f"score={r.best_score:.3f}  margin={r.margin:.3f}")

    return result
