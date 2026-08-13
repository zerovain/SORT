"""
sort_downstream.py

Downstream analysis tools for SORT decomposition results.

Core pipeline
-------------
Q_norm               = compute_qscore_matrix(Q)
gsp_results          = compute_gene_specificity(W, Q)
dominance_ratios, dominant_geps = compute_dominance_ratios(gsp_results['GSp'])
genelist_dict, quality_df       = extract_enrichment_genelist(
                                      Q_norm, dominant_geps, dominance_ratios,
                                      gene_names)
gep_df               = classify_geps(W, sample_ids)
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from typing import Dict, List, Optional, Tuple

__all__ = [
    'compute_qscore_matrix',
    'compute_gene_specificity',
    'compute_dominance_ratios',
    'extract_enrichment_genelist',
    'classify_geps',
]


# ==============================================================================
# 1. Q-score matrix  (L2-normalized, signed)
# ==============================================================================

def compute_qscore_matrix(Q: np.ndarray) -> np.ndarray:
    """
    L2-normalize each column of Q to unit length.

    Signs are preserved: positive = up-regulation in that GEP.

    Parameters
    ----------
    Q : ndarray (n_genes, n_components)

    Returns
    -------
    Q_norm : ndarray (n_genes, n_components)
    """
    Q_norm = Q.copy()
    norms = np.linalg.norm(Q_norm, axis=0)
    norms[norms == 0] = 1.0
    Q_norm /= norms[np.newaxis, :]
    return Q_norm


# ==============================================================================
# 2. Gene Specificity (GSp) matrix
# ==============================================================================

def compute_gene_specificity(
    W: np.ndarray,
    Q: np.ndarray,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute Gene Specificity (GSp) matrix (positive-only, L1-normalized per gene).

    Algorithm
    ---------
    1. Lambda[k] = max(W[:, k])
    2. Q_tilde   = Q * Lambda
    3. Q_pos     = max(Q_tilde, 0)
    4. GSp[j,:]  = Q_pos[j,:] / sum(Q_pos[j,:])

    Genes with all-zero Q_pos rows have no positive contribution to any GEP.
    Their GSp row remains all-zero; they are naturally excluded by the
    positive-value filter (L1) in extract_enrichment_genelist.

    Parameters
    ----------
    W : ndarray (n_spots, n_components)
    Q : ndarray (n_genes, n_components)
    verbose : bool

    Returns
    -------
    dict with keys:
        'GSp'     : ndarray (n_genes, n_components)  row sums = 1 (or 0 for zero rows)
        'Q_tilde' : ndarray (n_genes, n_components)
        'Q_pos'   : ndarray (n_genes, n_components)
        'Lambda'  : ndarray (n_components,)
    """
    n_genes, n_components = Q.shape

    Lambda  = np.maximum(W.max(axis=0), 1e-10)
    Q_tilde = Q * Lambda[np.newaxis, :]
    Q_pos   = np.maximum(Q_tilde, 0.0)

    row_sums  = Q_pos.sum(axis=1)
    zero_mask = row_sums == 0

    if verbose:
        pct_neg = (Q_tilde < 0).sum() / Q_tilde.size * 100
        print(f"[GSp] negative entries zeroed : {pct_neg:.1f}%")
        if zero_mask.sum():
            print(f"[GSp] all-zero rows (excluded): {zero_mask.sum()}")

    # safe divide: zero rows stay zero in GSp
    row_sums_safe = np.where(zero_mask, 1.0, row_sums)
    GSp = Q_pos / row_sums_safe[:, np.newaxis]
    # zero rows: GSp remains 0 (not assigned uniform distribution)

    return {'GSp': GSp, 'Q_tilde': Q_tilde, 'Q_pos': Q_pos, 'Lambda': Lambda}


# ==============================================================================
# 3. Dominance analysis
# ==============================================================================

def compute_dominance_ratios(
    GSp: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-gene dominance ratio and dominant GEP index.

    dominance_ratio[j] = GSp[j, argmax] / (GSp[j, second_argmax] + 1e-8)

    Interpretation
    --------------
    Because gene expression is non-negative, the positive part of each gene's
    Q vector necessarily dominates, so max_GSp is always meaningful.
    The ratio measures how exclusively a gene belongs to its top GEP:

        ratio > 5  : highly specific  (top GEP > 5x the runner-up)
        ratio 2–5  : moderately specific
        ratio ≤ 2  : shared across GEPs

    Parameters
    ----------
    GSp : ndarray (n_genes, n_components)

    Returns
    -------
    dominance_ratios : ndarray (n_genes,)
    dominant_geps    : ndarray (n_genes,)
    """
    sorted_idx   = np.argsort(GSp, axis=1)[:, ::-1]
    n_genes      = GSp.shape[0]
    rows         = np.arange(n_genes)

    max_vals     = GSp[rows, sorted_idx[:, 0]]
    second_vals  = (GSp[rows, sorted_idx[:, 1]]
                    if GSp.shape[1] > 1 else np.zeros(n_genes))

    dominance_ratios = max_vals / (second_vals + 1e-8)
    dominant_geps    = sorted_idx[:, 0]

    if verbose:
        r = dominance_ratios
        print(f"[Dominance] mean={r.mean():.2f}  median={np.median(r):.2f}  "
              f"max={r.max():.2f}")
        print(f"  highly specific  (ratio > 5) : {(r > 5).sum():>6d} "
              f"({(r > 5).mean()*100:.1f}%)")
        print(f"  moderately spec  (2 < r ≤ 5) : {((r > 2) & (r <= 5)).sum():>6d}")
        print(f"  shared           (ratio ≤ 2)  : {(r <= 2).sum():>6d}")

    return dominance_ratios, dominant_geps


# ==============================================================================
# 4. List A — enrichment gene list
# ==============================================================================

def extract_enrichment_genelist(
    Q_norm: np.ndarray,
    dominant_geps: np.ndarray,
    dominance_ratios: np.ndarray,
    gene_names: List[str],
    n_genes_per_gep: int = 50,
    min_dominance_ratio: Optional[float] = 2.0,
    min_genes: int = 10,
    verbose: bool = True,
) -> Tuple[Dict[int, List[str]], pd.DataFrame]:
    """
    Extract per-GEP marker gene lists for hypergeometric enrichment analysis.

    Filtering layers applied per GEP k
    ------------------------------------
    L1 sign        : Q_norm[j, k] > 0
    L2 intensity   : Q_norm[j, k] > mean of positive values in column k
    L3 specificity : dominant_geps[j] == k
    L4 dominance   : dominance_ratios[j] >= min_dominance_ratio  (optional)

    L3 logically implies L1 when GSp is built positive-only, but L1 is kept
    explicit for clarity and robustness.
    L4 guards against genes whose argmax GEP wins only by a narrow margin.

    Genes passing all active layers are ranked by Q_norm descending;
    up to n_genes_per_gep are selected.

    Parameters
    ----------
    Q_norm              : ndarray (n_genes, n_components)
    dominant_geps       : ndarray (n_genes,)
    dominance_ratios    : ndarray (n_genes,)
    gene_names          : list[str]
    n_genes_per_gep     : int, default 50
    min_dominance_ratio : float or None
        Minimum dominance ratio for L4. Set None to disable.
        Default 2.0 — requires dominant GEP to be at least 2x the runner-up.
    min_genes           : int — GEPs below this are flagged low-quality
    verbose             : bool

    Returns
    -------
    genelist_dict : dict[int, list[str]]
        Genes ranked by descending Q-score, one list per GEP.
    quality_df    : DataFrame (n_components × 9)
        gep_idx, n_positive, q_mean_positive, q_max,
        n_above_mean, n_dominant, n_pass_l4, n_selected, is_low_quality
    """
    gene_names = np.asarray(gene_names)
    n_genes, n_components = Q_norm.shape

    assert len(gene_names)     == n_genes, "gene_names length mismatch"
    assert len(dominant_geps)  == n_genes, "dominant_geps length mismatch"
    assert len(dominance_ratios) == n_genes, "dominance_ratios length mismatch"

    use_l4 = min_dominance_ratio is not None
    genelist_dict: Dict[int, List[str]] = {}
    records = []

    if verbose:
        print(f"[EnrichmentGenelist] {n_components} GEPs | "
              f"top {n_genes_per_gep} | "
              f"L4 min_ratio={'off' if not use_l4 else min_dominance_ratio}")

    for k in range(n_components):
        q_col    = Q_norm[:, k]

        # L1
        pos_mask = q_col > 0
        n_pos    = int(pos_mask.sum())
        q_mean   = float(q_col[pos_mask].mean()) if n_pos > 0 else 0.0
        q_max    = float(q_col.max())

        # L2
        l2_mask  = q_col > q_mean

        # L3
        l3_mask  = dominant_geps == k

        # L4
        l4_mask  = (dominance_ratios >= min_dominance_ratio
                    if use_l4 else np.ones(n_genes, dtype=bool))

        combined  = l2_mask & l3_mask & l4_mask
        cand_idx  = np.where(combined)[0]
        n_dom     = int((l2_mask & l3_mask).sum())   # before L4, for reporting
        n_pass_l4 = len(cand_idx)

        if n_pass_l4 > 0:
            order    = np.argsort(q_col[cand_idx])[::-1]
            selected = cand_idx[order][:n_genes_per_gep]
        else:
            selected = np.array([], dtype=int)

        n_sel = len(selected)
        genelist_dict[k] = gene_names[selected].tolist()

        lq = n_sel < min_genes
        records.append(dict(
            gep_idx=k,
            n_positive=n_pos,
            q_mean_positive=round(q_mean, 6),
            q_max=round(q_max, 6),
            n_above_mean=int(l2_mask.sum()),
            n_dominant=n_dom,
            n_pass_l4=n_pass_l4,
            n_selected=n_sel,
            is_low_quality=lq,
        ))

        if verbose:
            flag = "  ← LOW QUALITY" if lq else ""
            print(f"  GEP {k:>3d} | "
                  f"L1={n_pos:>5d}  "
                  f"L2={int(l2_mask.sum()):>5d}  "
                  f"L3={n_dom:>5d}  "
                  f"L4={n_pass_l4:>5d}  "
                  f"sel={n_sel:>3d}{flag}")

    quality_df = pd.DataFrame(records)

    if verbose:
        n_lq = quality_df['is_low_quality'].sum()
        print(f"  → {n_lq}/{n_components} GEPs flagged low quality")

    return genelist_dict, quality_df


# ==============================================================================
# 5. GEP classification  (Between-sample × Within-sample)
# ==============================================================================

def classify_geps(
    W: np.ndarray,
    sample_ids: np.ndarray,
    activity_threshold: float = 0.3,
    min_spots: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Classify GEPs on a Between-sample × Within-sample grid.

    Between-sample
    --------------
    Global   : frac_active ≥ 75th-pct  AND  between_cv < 60th-pct
    Subgroup : frac_active ≥ 25th-pct  (including high-coverage high-CV)
    Rare     : frac_active < 25th-pct

    Within-sample
    -------------
    Uniform    : within_cv < 0.3  AND  bimodal_frac ≤ 10%
    Structured : bimodal_frac > 20%  OR  (bimodal_frac > 10% AND within_cv > 0.5)
    Diffuse    : otherwise

    Bimodality via Sarle's coefficient b = (skew²+1) / kurtosis; b > 0.55 → bimodal.
    A sample needs ≥ min_spots to contribute to within-sample statistics.

    Parameters
    ----------
    W                  : ndarray (n_spots, n_components)
    sample_ids         : ndarray (n_spots,)
    activity_threshold : float  — sample "active" if 95th-pct of normalised W > this
    min_spots          : int
    verbose            : bool

    Returns
    -------
    DataFrame columns:
        GEP, Between, Within, Category,
        n_active, frac_active, between_cv, within_cv, bimodal_frac
    """
    samples   = np.unique(sample_ids)
    n_samples = len(samples)
    n_geps    = W.shape[1]
    records   = []

    for k in range(n_geps):
        vals = W[:, k]
        p99  = np.percentile(vals, 99)
        norm = vals / p99 if p99 > 0 else vals

        active = [s for s in samples
                  if np.percentile(norm[sample_ids == s], 95) > activity_threshold]
        n_act  = len(active)
        f_act  = n_act / n_samples

        if n_act >= 2:
            means = [vals[sample_ids == s].mean() for s in active]
            b_cv  = np.std(means) / (np.mean(means) + 1e-10)
        else:
            b_cv  = 0.0

        w_cvs, bimodal, analyzed = [], 0, 0
        for s in active:
            sv = vals[sample_ids == s]
            if len(sv) < min_spots:
                continue
            analyzed += 1
            w_cvs.append(np.std(sv) / (np.mean(sv) + 1e-10))
            sv_z = (sv - sv.mean()) / (sv.std() + 1e-10)
            sk = skew(sv_z)
            kt = kurtosis(sv_z, fisher=False)
            if kt > 0 and (sk**2 + 1) / kt > 0.55:
                bimodal += 1

        w_cv   = float(np.mean(w_cvs)) if w_cvs else 0.0
        b_frac = bimodal / analyzed    if analyzed else 0.0

        records.append(dict(GEP=k, n_active=n_act, frac_active=f_act,
                            between_cv=b_cv, within_cv=w_cv,
                            bimodal_frac=b_frac))

    feat = pd.DataFrame(records)

    # adaptive thresholds
    frac_hi = np.percentile(feat['frac_active'], 75)
    frac_lo = np.percentile(feat['frac_active'], 25)
    bcv_thr = np.percentile(feat['between_cv'],  60)

    rows = []
    for _, r in feat.iterrows():
        fa, bcv = r['frac_active'], r['between_cv']
        wcv, bf = r['within_cv'],   r['bimodal_frac']

        # Between — explicit four-branch to avoid coverage ambiguity
        if   fa >= frac_hi and bcv < bcv_thr:  b_cat = 'Global'
        elif fa >= frac_hi and bcv >= bcv_thr: b_cat = 'Subgroup'
        elif fa >= frac_lo:                    b_cat = 'Subgroup'
        else:                                  b_cat = 'Rare'

        # Within
        if   wcv < 0.3 and bf <= 0.10:                    w_cat = 'Uniform'
        elif bf > 0.20 or (bf > 0.10 and wcv > 0.5):      w_cat = 'Structured'
        else:                                              w_cat = 'Diffuse'

        rows.append(dict(GEP=int(r['GEP']), Between=b_cat, Within=w_cat,
                         Category=f"{b_cat}-{w_cat}"))

    result = feat.merge(pd.DataFrame(rows), on='GEP')

    if verbose:
        print(f"[classify_geps] {n_geps} GEPs | "
              f"frac_hi={frac_hi:.2f} frac_lo={frac_lo:.2f} bcv={bcv_thr:.2f}")
        for cat, grp in result.groupby('Category'):
            gep_list = sorted(grp.GEP.tolist())
            preview  = str(gep_list[:8]) + ('...' if len(gep_list) > 8 else '')
            print(f"  {cat:<22s}: {len(grp):>2d}  {preview}")

    return result