"""Small public API that delegates to the manuscript implementation."""

from time import perf_counter
from typing import Optional

from .config import SORTConfig
from .result import SORTResult


def _resolve_auto_device() -> str:
    """Choose CUDA only when both required GPU runtimes are usable."""

    try:
        import torch

        if not torch.cuda.is_available():
            return "numpy"
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            return "numpy"
    except Exception:
        return "numpy"
    return "cuda"


def fit(adata, n_components: int, *, copy: bool = False, **kwargs) -> SORTResult:
    """Fit SORT through a concise functional interface.

    Results are stored in ``adata.obsm['X_sort']`` and
    ``adata.varm['sort_signatures']`` and are also returned as a validated
    :class:`SORTResult`. Stable biological-analysis settings are defaults;
    dataset-specific overrides remain available as keyword arguments.
    """

    return SORT(n_components=n_components, **kwargs).fit(adata, copy=copy)


class SORT:
    """Fit SORT on an AnnData object with an existing spatial Laplacian.

    The wrapper does not construct a graph or implement an optimizer. It calls
    :func:`sort.decompose` with the same arguments used by the retained
    manuscript implementation.
    """

    def __init__(self, config: Optional[SORTConfig] = None, **kwargs):
        if config is not None and kwargs:
            raise TypeError("pass either config or keyword parameters, not both")
        self.config = config if config is not None else SORTConfig(**kwargs)
        self.result_: Optional[SORTResult] = None

    def fit(self, adata, *, copy: bool = False) -> SORTResult:
        """Fit and return a validated spots-by-components result.

        ``adata.uns['spatial_laplacian']`` must already exist. Use the public
        graph helpers explicitly so the graph construction is visible and
        reproducible.
        """

        from .model import decompose

        target = adata.copy() if copy else adata
        kwargs = self.config.to_decompose_kwargs()
        resolved_device = (
            _resolve_auto_device() if kwargs["device"] == "auto" else kwargs["device"]
        )
        kwargs["device"] = resolved_device
        concise = kwargs["verbose"] is None
        if concise:
            kwargs["verbose"] = False
            print(
                f"SORT: {target.n_obs:,} locations x {target.n_vars:,} genes; "
                f"{self.config.n_components} components"
            )

        started = perf_counter()
        decompose(target, copy=False, **kwargs)
        elapsed = perf_counter() - started
        result = SORTResult.from_adata(target, config=self.config)
        result.metadata["fit_time_seconds"] = elapsed
        result.metadata["resolved_device"] = resolved_device
        self.result_ = result
        if concise:
            print(f"Finished in {elapsed:.1f} s")
            print(
                "Stored .obsm['X_sort'] "
                f"{result.W.shape} and .varm['sort_signatures'] {result.Q.shape}"
            )
        return result
