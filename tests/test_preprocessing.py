import anndata as ad
import numpy as np
import scanpy as sc

from sort.pipelines import standard_preprocess


def test_standard_preprocess_uses_manuscript_normalize_log_hvg_order(monkeypatch):
    events = []
    adata = ad.AnnData(
        np.array(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7],
                [5, 6, 7, 8],
            ],
            dtype=np.float32,
        )
    )

    def record_hvg(current, **kwargs):
        events.append("hvg")
        assert current.n_vars == 4
        assert np.any(current.X != np.rint(current.X))
        current.var["highly_variable"] = [True, True, False, False]

    def record_normalize(current, **kwargs):
        events.append("normalize")
        assert current.n_vars == 4
        current.X = current.X / 3.0

    def record_log1p(current, **kwargs):
        events.append("log1p")
        current.X = np.log1p(current.X)

    monkeypatch.setattr(sc.pp, "highly_variable_genes", record_hvg)
    monkeypatch.setattr(sc.pp, "normalize_total", record_normalize)
    monkeypatch.setattr(sc.pp, "log1p", record_log1p)

    result = standard_preprocess(
        adata,
        n_hvg=2,
        batch_key=None,
        min_genes=0,
        min_cells_pct=0,
        min_mean_expression=0,
    )

    assert events == ["normalize", "log1p", "hvg"]
    assert result.shape == (6, 2)


def test_standard_preprocess_allows_missing_default_batch_key(monkeypatch):
    adata = ad.AnnData(np.ones((6, 4), dtype=np.float32))
    observed = {}

    def record_hvg(current, **kwargs):
        observed["batch_key"] = kwargs["batch_key"]
        current.var["highly_variable"] = True

    monkeypatch.setattr(sc.pp, "highly_variable_genes", record_hvg)
    monkeypatch.setattr(sc.pp, "normalize_total", lambda *args, **kwargs: None)
    monkeypatch.setattr(sc.pp, "log1p", lambda *args, **kwargs: None)

    standard_preprocess(
        adata,
        n_hvg=4,
        min_genes=0,
        min_cells_pct=0,
        min_mean_expression=0,
    )

    assert observed["batch_key"] is None
