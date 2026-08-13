import inspect

import anndata as ad
import numpy as np
import pytest

import sort
from sort import SORT, SORTConfig, SORTResult, decompose, fit
from sort.model import SpatialSemiNMF


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="n_components"):
        SORTConfig(n_components=1)
    with pytest.raises(ValueError, match="regularization"):
        SORTConfig(lambda_neg=-1)


def test_high_level_defaults_match_common_biological_settings():
    config = SORTConfig()
    assert config.n_components == 25
    assert config.alpha == 0.3
    assert config.beta == 0.5
    assert config.lambda_l1_W == 0.3
    assert config.lambda_l1_Q == 300.0
    assert config.l1_weight_strategy == "adaptive"
    assert config.lambda_neg == 1.0
    assert config.use_tv is True
    assert config.tv_epsilon == 0.01
    assert config.tv_update_freq == 5
    assert config.tv_stage == "stage2"
    assert (config.stage1_epochs, config.stage2_epochs) == (50, 100)
    assert config.random_state == 42


def test_preprocessing_pipeline_is_not_part_of_the_public_api():
    assert not hasattr(sort, "standard_preprocess")
    assert not hasattr(sort, "run_sort_pipeline")


def test_adaptive_l1_is_the_default_at_every_public_model_entry_point():
    assert (
        inspect.signature(SpatialSemiNMF).parameters["l1_weight_strategy"].default
        == "adaptive"
    )
    assert (
        inspect.signature(decompose).parameters["l1_weight_strategy"].default
        == "adaptive"
    )


def test_sort_accepts_config_or_keywords_but_not_both():
    assert SORT(SORTConfig(n_components=3)).config.n_components == 3
    assert SORT(n_components=4).config.n_components == 4
    with pytest.raises(TypeError, match="either config"):
        SORT(SORTConfig(), n_components=3)


def test_result_orients_q_only_when_unambiguous():
    result = SORTResult(
        W=np.ones((5, 2)),
        Q=np.ones((2, 7)),
        gene_names=np.asarray([f"g{i}" for i in range(7)]),
        observation_names=np.asarray([f"s{i}" for i in range(5)]),
        config=SORTConfig(n_components=2),
    )
    assert result.Q.shape == (7, 2)
    with pytest.raises(ValueError, match="ambiguous"):
        SORTResult(
            W=np.ones((5, 2)),
            Q=np.ones((2, 2)),
            gene_names=np.asarray(["g0", "g1"]),
            observation_names=np.asarray([f"s{i}" for i in range(5)]),
            config=SORTConfig(n_components=2),
        )


def test_sort_default_output_is_concise(monkeypatch, capsys):
    adata = ad.AnnData(np.ones((5, 7), dtype=np.float32))
    adata.uns["spatial_laplacian"] = np.eye(5, dtype=np.float32)

    def fake_decompose(target, *, copy, **kwargs):
        assert copy is False
        assert kwargs["verbose"] is False
        assert kwargs["device"] == "numpy"
        target.obsm["X_sort"] = np.ones((5, 3), dtype=np.float32)
        target.varm["sort_signatures"] = np.ones((7, 3), dtype=np.float32)
        target.uns["sort"] = {"params": {"n_components": 3}}

    monkeypatch.setattr("sort.model.decompose", fake_decompose)
    monkeypatch.setattr("sort.api._resolve_auto_device", lambda: "numpy")
    result = SORT(SORTConfig(n_components=3, random_state=17)).fit(adata)

    output = capsys.readouterr().out
    assert "SORT: 5 locations x 7 genes; 3 components" in output
    assert ".obsm['X_sort'] (5, 3)" in output
    assert repr(result).startswith("SORTResult(W=(5, 3) spots_x_components")
    assert result.metadata["resolved_device"] == "numpy"


def test_functional_fit_forwards_to_sort(monkeypatch):
    sentinel = object()

    def fake_fit(self, adata, *, copy=False):
        assert self.config.n_components == 4
        assert self.config.random_state == 23
        assert copy is True
        return sentinel

    monkeypatch.setattr(SORT, "fit", fake_fit)
    assert fit(object(), n_components=4, random_state=23, copy=True) is sentinel
