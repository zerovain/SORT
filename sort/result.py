"""Validated result container for the public SORT API."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .config import SORTConfig


@dataclass
class SORTResult:
    """Fitted matrices with explicit orientations and provenance metadata."""

    W: np.ndarray
    Q: np.ndarray
    gene_names: np.ndarray
    observation_names: np.ndarray
    config: SORTConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    reconstruction: Optional[np.ndarray] = None
    adata: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W)
        self.Q = _as_genes_by_components(self.Q, self.W.shape[1])
        self.gene_names = np.asarray(self.gene_names, dtype=str)
        self.observation_names = np.asarray(self.observation_names, dtype=str)
        if self.W.ndim != 2:
            raise ValueError("W must be spots-by-components")
        if self.W.shape[0] != len(self.observation_names):
            raise ValueError("observation_names must contain one value per W row")
        if self.Q.shape[0] != len(self.gene_names):
            raise ValueError("gene_names must contain one value per Q row")
        if np.any(self.W < 0):
            raise ValueError("W must be nonnegative")

    def __repr__(self) -> str:
        """Return a compact notebook-friendly result summary."""

        elapsed = self.metadata.get("fit_time_seconds")
        timing = "" if elapsed is None else f", fit_time={float(elapsed):.1f}s"
        return (
            "SORTResult("
            f"W={self.W.shape} spots_x_components, "
            f"Q={self.Q.shape} genes_x_components, "
            f"background_component=0{timing})"
        )

    @classmethod
    def from_adata(cls, adata, *, config: SORTConfig) -> "SORTResult":
        """Build a result from keys written by the retained implementation."""

        if "X_sort" not in adata.obsm or "sort_signatures" not in adata.varm:
            raise ValueError("AnnData does not contain fitted SORT W/Q matrices")
        reconstruction = adata.layers.get("sort_reconstructed")
        return cls(
            W=np.asarray(adata.obsm["X_sort"]),
            Q=np.asarray(adata.varm["sort_signatures"]),
            gene_names=np.asarray(adata.var_names, dtype=str),
            observation_names=np.asarray(adata.obs_names, dtype=str),
            config=config,
            metadata={
                "W_orientation": "spots_by_components",
                "Q_orientation": "genes_by_components",
                "implementation": "manuscript_equivalent",
                "sort_params": dict(adata.uns.get("sort", {}).get("params", {})),
            },
            reconstruction=(
                None if reconstruction is None else np.asarray(reconstruction)
            ),
            adata=adata,
        )

    def reconstruct(
        self,
        rows: Optional[Sequence[int]] = None,
        genes: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Materialize a requested reconstruction slice."""

        w = self.W if rows is None else self.W[np.asarray(rows)]
        q = self.Q if genes is None else self.Q[np.asarray(genes)]
        return np.asarray(w @ q.T, dtype=np.float32)

    def save(self, path: str, *, include_reconstruction: bool = False) -> Path:
        """Save W/Q and metadata; dense reconstruction is opt-in for export."""

        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "W.npy", self.W)
        np.save(output / "Q.npy", self.Q)
        np.save(output / "gene_names.npy", self.gene_names)
        np.save(output / "observation_names.npy", self.observation_names)
        if include_reconstruction and self.reconstruction is not None:
            np.save(output / "reconstruction.npy", self.reconstruction)
        payload = {
            "config": self.config.to_dict(),
            "metadata": self.metadata,
            "orientations": {
                "W": "spots_by_components",
                "Q": "genes_by_components",
            },
        }
        (output / "metadata.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        return output


def _as_genes_by_components(Q: np.ndarray, n_components: int) -> np.ndarray:
    """Validate Q orientation against K and transpose only if unambiguous."""

    q = np.asarray(Q)
    if q.ndim != 2:
        raise ValueError("Q must be two-dimensional")
    row_match = q.shape[0] == n_components
    column_match = q.shape[1] == n_components
    if column_match and not row_match:
        return q
    if row_match and not column_match:
        return q.T
    if row_match and column_match:
        raise ValueError(
            "Q orientation is ambiguous because both axes equal the number of components"
        )
    raise ValueError(
        f"Q shape {q.shape} is incompatible with W component count {n_components}"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
