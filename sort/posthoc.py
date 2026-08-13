"""Post hoc gene association analysis conditional on fitted SORT activities."""

from dataclasses import dataclass, field
import warnings
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np


@dataclass
class PosthocBResult:
    """Gene-by-component coefficients from expression ~ intercept + fixed W.

    The coefficients are conditional association estimates for annotation.
    They are not differential-expression statistics or causal effects.
    """

    B: np.ndarray
    intercept: Optional[np.ndarray]
    stderr: Optional[np.ndarray]
    t: Optional[np.ndarray]
    pvalue: Optional[np.ndarray]
    r2: Optional[np.ndarray]
    gene_names: np.ndarray
    component_names: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.B = np.asarray(self.B)
        self.gene_names = np.asarray(self.gene_names, dtype=str)
        self.component_names = np.asarray(self.component_names, dtype=str)
        if self.B.ndim != 2:
            raise ValueError("B must be a genes-by-components matrix")
        n_genes, n_components = self.B.shape
        if len(self.gene_names) != n_genes:
            raise ValueError("gene_names length must equal B.shape[0]")
        if len(self.component_names) != n_components:
            raise ValueError("component_names length must equal B.shape[1]")
        if self.intercept is not None:
            self.intercept = _validate_vector(
                self.intercept, n_genes, "intercept"
            )
        for name in ("stderr", "t", "pvalue"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value)
                if array.shape != self.B.shape:
                    raise ValueError(f"{name} shape must equal B.shape")
                setattr(self, name, array)
        if self.r2 is not None:
            self.r2 = _validate_vector(self.r2, n_genes, "r2")

    def component_index(self, gep: Union[int, str]) -> int:
        """Resolve a positional integer or component label such as GEP18."""

        if isinstance(gep, (int, np.integer)):
            index = int(gep)
        elif isinstance(gep, str):
            matches = np.flatnonzero(self.component_names == gep)
            if len(matches) == 1:
                return int(matches[0])
            if gep.upper().startswith("GEP") and gep[3:].isdigit():
                index = int(gep[3:])
            else:
                raise ValueError(f"Unknown component name: {gep!r}")
        else:
            raise TypeError("gep must be an integer index or component name")
        if index < 0 or index >= self.B.shape[1]:
            raise IndexError(
                f"Component index {index} is outside [0, {self.B.shape[1]})"
            )
        return index


def compute_posthoc_b(
    X,
    W,
    gene_names=None,
    component_names=None,
    add_intercept=True,
    standardize_X=False,
    standardize_W=False,
    chunk_size=5000,
    ridge=0.0,
    return_stats=True,
    dtype=None,
) -> PosthocBResult:
    """Regress each expression gene on all fixed SORT activity components.

    Parameters
    ----------
    X
        Spots-by-genes dense, sparse, or sliceable expression matrix.
    W
        Spots-by-components SORT activity matrix.
    add_intercept
        Add an unpenalized intercept to every gene regression.
    standardize_X, standardize_W
        Z-score columns across spots. Zero-variance columns are safely mapped
        to zero after centering.
    chunk_size
        Number of genes densified and solved together. ``None`` uses all genes.
    ridge
        Nonnegative L2 penalty. The intercept is never penalized.
    return_stats
        Compute standard errors, t statistics, two-sided p-values when SciPy is
        available, and per-gene R-squared.
    dtype
        Floating calculation/output dtype. The default is float64.

    Returns
    -------
    PosthocBResult
        B is genes-by-components. Coefficients describe conditional post hoc
        associations and must not be interpreted as differential expression or
        causal effects.
    """

    from scipy import sparse

    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("X must be a spots-by-genes matrix")
    work_dtype = np.dtype(np.float64 if dtype is None else dtype)
    if work_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("dtype must be float32 or float64")

    w = np.asarray(W, dtype=work_dtype)
    if w.ndim != 2:
        raise ValueError("W must be a spots-by-components matrix")
    n_spots, n_components = w.shape
    if X.shape[0] != n_spots:
        raise ValueError(
            f"X and W must have the same spot rows; got {X.shape[0]} and {n_spots}"
        )
    n_genes = int(X.shape[1])
    if n_spots == 0 or n_genes == 0 or n_components == 0:
        raise ValueError("X and W dimensions must all be nonzero")
    if not np.isfinite(w).all():
        raise ValueError("W contains non-finite values")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")

    genes = _names_or_default(gene_names, n_genes, "gene", width=0)
    components = _names_or_default(
        component_names, n_components, "GEP", width=max(2, len(str(n_components - 1)))
    )

    if standardize_W:
        w, w_means, w_scales = _zscore_columns(w)
    else:
        w_means = None
        w_scales = None

    if add_intercept:
        design = np.column_stack(
            [np.ones(n_spots, dtype=work_dtype), w]
        )
    else:
        design = w
    n_parameters = design.shape[1]
    if n_spots < 1:
        raise ValueError("At least one spot is required")

    if chunk_size is None:
        effective_chunk = n_genes
    else:
        effective_chunk = int(chunk_size)
        if effective_chunk < 1:
            raise ValueError("chunk_size must be positive or None")

    xtx = design.T @ design
    rank = int(np.linalg.matrix_rank(design))
    if rank < n_parameters:
        warnings.warn(
            f"Design matrix is rank deficient ({rank} < {n_parameters}); "
            "coefficients use the minimum-norm solution.",
            RuntimeWarning,
            stacklevel=2,
        )
    ridge_matrix = np.eye(n_parameters, dtype=work_dtype) * float(ridge)
    if add_intercept:
        ridge_matrix[0, 0] = 0.0
    system = xtx + ridge_matrix

    if ridge == 0:
        covariance_base = np.linalg.pinv(xtx, rcond=1e-12)
    else:
        try:
            system_inv = np.linalg.inv(system)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Ridge system is singular; using a pseudoinverse.",
                RuntimeWarning,
                stacklevel=2,
            )
            system_inv = np.linalg.pinv(system, rcond=1e-12)
        covariance_base = system_inv @ xtx @ system_inv.T

    b_matrix = np.empty((n_genes, n_components), dtype=work_dtype)
    intercept = (
        np.empty(n_genes, dtype=work_dtype) if add_intercept else None
    )
    stderr = (
        np.full((n_genes, n_components), np.nan, dtype=work_dtype)
        if return_stats
        else None
    )
    t_values = (
        np.full((n_genes, n_components), np.nan, dtype=work_dtype)
        if return_stats
        else None
    )
    p_values = (
        np.full((n_genes, n_components), np.nan, dtype=work_dtype)
        if return_stats
        else None
    )
    r2_values = (
        np.full(n_genes, np.nan, dtype=work_dtype) if return_stats else None
    )

    df = n_spots - n_parameters
    if return_stats and df <= 0:
        warnings.warn(
            f"Residual degrees of freedom are nonpositive (df={df}); "
            "stderr, t, and pvalue will be NaN.",
            RuntimeWarning,
            stacklevel=2,
        )

    pvalue_available = False
    if return_stats and df > 0:
        try:
            from scipy.stats import t as student_t
        except ImportError:
            student_t = None
        pvalue_available = student_t is not None
    else:
        student_t = None

    for start in range(0, n_genes, effective_chunk):
        end = min(start + effective_chunk, n_genes)
        x_chunk = X[:, start:end]
        if sparse.issparse(x_chunk):
            y = x_chunk.toarray()
        else:
            y = np.asarray(x_chunk)
        y = np.asarray(y, dtype=work_dtype)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape != (n_spots, end - start):
            raise ValueError(
                f"X slice has unexpected shape {y.shape}; "
                f"expected {(n_spots, end - start)}"
            )
        if not np.isfinite(y).all():
            raise ValueError(f"X contains non-finite values in genes {start}:{end}")

        if standardize_X:
            y, _, _ = _zscore_columns(y)

        xty = design.T @ y
        if ridge == 0:
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        else:
            try:
                coefficients = np.linalg.solve(system, xty)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Ridge solve failed; using a pseudoinverse.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                coefficients = np.linalg.pinv(system, rcond=1e-12) @ xty

        if add_intercept:
            assert intercept is not None
            intercept[start:end] = coefficients[0]
            component_coefficients = coefficients[1:]
            component_slice = slice(1, None)
        else:
            component_coefficients = coefficients
            component_slice = slice(None)
        b_matrix[start:end] = component_coefficients.T

        if not return_stats:
            continue

        yty = np.sum(y * y, axis=0)
        fitted_cross = np.sum(coefficients * xty, axis=0)
        fitted_quadratic = np.sum(
            coefficients * (xtx @ coefficients), axis=0
        )
        sse = np.maximum(
            yty - 2.0 * fitted_cross + fitted_quadratic, 0.0
        )
        sum_y = np.sum(y, axis=0)
        tss = np.maximum(yty - (sum_y * sum_y) / float(n_spots), 0.0)
        assert r2_values is not None
        r2_values[start:end] = np.divide(
            sse,
            tss,
            out=np.full_like(sse, np.nan),
            where=tss > np.finfo(work_dtype).eps,
        )
        r2_values[start:end] = 1.0 - r2_values[start:end]

        if df <= 0:
            continue
        sigma2 = sse / float(df)
        variance_diag = np.diag(covariance_base)[:, None] * sigma2[None, :]
        se_all = np.sqrt(np.maximum(variance_diag, 0.0))
        se_components = se_all[component_slice]
        assert stderr is not None and t_values is not None
        stderr[start:end] = se_components.T
        t_chunk = np.divide(
            component_coefficients,
            se_components,
            out=np.full_like(component_coefficients, np.nan),
            where=se_components > 0,
        )
        t_values[start:end] = t_chunk.T
        if student_t is not None:
            assert p_values is not None
            p_values[start:end] = (
                2.0 * student_t.sf(np.abs(t_chunk), df=df)
            ).T

    return PosthocBResult(
        B=b_matrix,
        intercept=intercept,
        stderr=stderr,
        t=t_values,
        pvalue=p_values if pvalue_available else None,
        r2=r2_values,
        gene_names=genes,
        component_names=components,
        params={
            "n_spots": n_spots,
            "n_genes": n_genes,
            "n_components": n_components,
            "add_intercept": bool(add_intercept),
            "standardize_X": bool(standardize_X),
            "standardize_W": bool(standardize_W),
            "W_means": None if w_means is None else w_means.tolist(),
            "W_scales": None if w_scales is None else w_scales.tolist(),
            "chunk_size": None if chunk_size is None else effective_chunk,
            "ridge": float(ridge),
            "ridge_penalizes_intercept": False,
            "return_stats": bool(return_stats),
            "df": int(df),
            "design_rank": rank,
            "dtype": work_dtype.name,
            "pvalue_distribution": (
                "student_t_two_sided" if pvalue_available else None
            ),
            "coefficient_interpretation": (
                "conditional_posthoc_association_not_DE_or_causal_effect"
            ),
            "ridge_inference_note": (
                "standard errors use the fixed-design ridge covariance; "
                "t and p values are approximate when ridge > 0"
                if ridge > 0
                else None
            ),
            "B_orientation": "genes_by_components",
            "X_orientation": "spots_by_genes",
            "W_orientation": "spots_by_components",
        },
    )


def rank_genes_for_gep(
    posthoc_result,
    gep,
    n_top=30,
    n_bottom=30,
    sort_by="coef",
    min_abs_coef=None,
):
    """Rank positive and negative gene associations for one component."""

    import pandas as pd

    if not isinstance(posthoc_result, PosthocBResult):
        raise TypeError("posthoc_result must be a PosthocBResult")
    if n_top < 0 or n_bottom < 0:
        raise ValueError("n_top and n_bottom must be nonnegative")
    allowed = {"coef", "abs_coef", "t", "abs_t"}
    if sort_by not in allowed:
        raise ValueError(f"sort_by must be one of {sorted(allowed)}")
    if sort_by in {"t", "abs_t"} and posthoc_result.t is None:
        raise ValueError("t statistics are unavailable; rerun with return_stats=True")
    if min_abs_coef is not None and min_abs_coef < 0:
        raise ValueError("min_abs_coef must be nonnegative")

    index = posthoc_result.component_index(gep)
    component = str(posthoc_result.component_names[index])
    coefficients = posthoc_result.B[:, index]
    t_values = (
        posthoc_result.t[:, index]
        if posthoc_result.t is not None
        else np.full(len(coefficients), np.nan)
    )
    keep = np.isfinite(coefficients)
    if sort_by in {"t", "abs_t"}:
        keep &= np.isfinite(t_values)
    if min_abs_coef is not None:
        keep &= np.abs(coefficients) >= min_abs_coef

    positive = np.flatnonzero(keep & (coefficients > 0))
    negative = np.flatnonzero(keep & (coefficients < 0))
    positive = _sort_gene_indices(
        positive, coefficients, t_values, sort_by, direction="positive"
    )[:n_top]
    negative = _sort_gene_indices(
        negative, coefficients, t_values, sort_by, direction="negative"
    )[:n_bottom]

    records = []
    for direction, indices in (("positive", positive), ("negative", negative)):
        for rank, gene_index in enumerate(indices, start=1):
            records.append(
                {
                    "gene": str(posthoc_result.gene_names[gene_index]),
                    "component": component,
                    "coef": float(coefficients[gene_index]),
                    "stderr": _optional_stat(
                        posthoc_result.stderr, gene_index, index
                    ),
                    "t": _optional_stat(posthoc_result.t, gene_index, index),
                    "pvalue": _optional_stat(
                        posthoc_result.pvalue, gene_index, index
                    ),
                    "direction": direction,
                    "rank": rank,
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "gene",
            "component",
            "coef",
            "stderr",
            "t",
            "pvalue",
            "direction",
            "rank",
        ],
    )


def score_signatures_from_b(
    posthoc_result,
    signatures,
    min_genes=3,
    zscore_across_components=True,
    case_insensitive=False,
):
    """Score annotation signatures by mean B coefficient across matched genes.

    Returns a wide score DataFrame and an overlap DataFrame. Row-wise
    z-scoring, when requested, is an annotation display transformation only.
    """

    import pandas as pd

    if not isinstance(posthoc_result, PosthocBResult):
        raise TypeError("posthoc_result must be a PosthocBResult")
    if not isinstance(signatures, Mapping):
        raise TypeError("signatures must map names to gene sequences")
    if min_genes < 1:
        raise ValueError("min_genes must be positive")

    lookup: Dict[str, int] = {}
    for index, gene in enumerate(posthoc_result.gene_names):
        key = str(gene).upper() if case_insensitive else str(gene)
        lookup.setdefault(key, index)

    score_rows = []
    overlap_records = []
    signature_names = []
    for signature_name, signature_genes in signatures.items():
        name = str(signature_name)
        if isinstance(signature_genes, str):
            requested = [signature_genes]
        else:
            requested = list(
                dict.fromkeys(str(gene) for gene in signature_genes)
            )
        matched_indices = []
        matched_genes = []
        missing_genes = []
        for gene in requested:
            key = gene.upper() if case_insensitive else gene
            if key in lookup:
                matched_indices.append(lookup[key])
                matched_genes.append(gene)
            else:
                missing_genes.append(gene)

        meets_minimum = len(matched_indices) >= min_genes
        if meets_minimum:
            score = np.mean(posthoc_result.B[matched_indices], axis=0)
        else:
            score = np.full(posthoc_result.B.shape[1], np.nan, dtype=float)
        score_rows.append(score)
        signature_names.append(name)
        overlap_records.append(
            {
                "signature": name,
                "n_requested": len(requested),
                "n_overlap": len(matched_indices),
                "meets_min_genes": meets_minimum,
                "matched_genes": ",".join(matched_genes),
                "missing_genes": ",".join(missing_genes),
            }
        )

    score_matrix = (
        np.asarray(score_rows, dtype=float)
        if score_rows
        else np.empty((0, posthoc_result.B.shape[1]), dtype=float)
    )
    if zscore_across_components and score_matrix.size:
        for row_index in range(score_matrix.shape[0]):
            row = score_matrix[row_index]
            if not np.isfinite(row).all():
                continue
            mean = float(np.mean(row))
            std = float(np.std(row))
            if std > 0:
                score_matrix[row_index] = (row - mean) / std
            else:
                score_matrix[row_index] = 0.0

    score_df = pd.DataFrame(
        score_matrix,
        index=pd.Index(signature_names, name="signature"),
        columns=posthoc_result.component_names,
    )
    overlap_df = pd.DataFrame(overlap_records)
    return score_df, overlap_df


def _validate_vector(value, length: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError(f"{name} shape must be ({length},)")
    return array


def _names_or_default(names, length: int, prefix: str, width: int) -> np.ndarray:
    if names is None:
        if width:
            values = [f"{prefix}{index:0{width}d}" for index in range(length)]
        else:
            values = [f"{prefix}_{index}" for index in range(length)]
    else:
        values = [str(value) for value in names]
    if len(values) != length:
        raise ValueError(f"{prefix} names length must be {length}")
    if len(set(values)) != len(values):
        raise ValueError(f"{prefix} names must be unique")
    return np.asarray(values, dtype=str)


def _zscore_columns(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    safe_scales = np.where(scales > 0, scales, 1.0)
    standardized = (matrix - means) / safe_scales
    return standardized, means, safe_scales


def _sort_gene_indices(
    indices: np.ndarray,
    coefficients: np.ndarray,
    t_values: np.ndarray,
    sort_by: str,
    direction: str,
) -> np.ndarray:
    if sort_by == "coef":
        values = coefficients[indices]
        order = np.argsort(-values if direction == "positive" else values)
    elif sort_by == "abs_coef":
        order = np.argsort(-np.abs(coefficients[indices]))
    elif sort_by == "t":
        values = t_values[indices]
        order = np.argsort(-values if direction == "positive" else values)
    else:
        order = np.argsort(-np.abs(t_values[indices]))
    return indices[order]


def _optional_stat(
    values: Optional[np.ndarray], gene_index: int, component_index: int
) -> float:
    if values is None:
        return float("nan")
    return float(values[gene_index, component_index])
