import anndata as ad
import numpy as np
import pytest

from sort import align_samples_grid, merge_samples, normalize_Q_scale_W
from sort.pipelines import run_sort_pipeline


def _sample(name, genes):
    result = ad.AnnData(
        np.ones((2, len(genes)), dtype=np.float32),
        obs={"source": [name, name]},
        var={"gene": genes},
    )
    result.var_names = genes
    result.obs_names = [f"{name}_{index}" for index in range(2)]
    result.obsm["spatial"] = np.array([[0, 0], [1, 1]], dtype=np.float32)
    return result


def test_merge_samples_default_batch_key_and_no_input_mutation():
    first = _sample("first", ["A", "B"])
    second = _sample("second", ["A", "B"])

    merged = merge_samples([first, second], cohort_labels=["x", "y"])

    assert "sample_id" in merged.obs
    assert list(merged.obs["sample_id"].astype(str)) == ["0", "0", "1", "1"]
    assert "cohort" not in first.obs
    assert "cohort" not in second.obs


def test_merge_samples_rejects_invalid_join():
    first = _sample("first", ["A"])
    second = _sample("second", ["A"])
    with pytest.raises(ValueError, match="join must"):
        merge_samples([first, second], join="left")


def test_merge_samples_rejects_empty_inner_intersection():
    first = _sample("first", ["A"])
    second = _sample("second", ["B"])
    with pytest.raises(ValueError, match="No common genes"):
        merge_samples([first, second])


def test_normalize_q_scale_w_preserves_reconstruction():
    adata = ad.AnnData(np.ones((4, 3), dtype=np.float32))
    adata.obsm["X_sort"] = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32
    )
    adata.varm["sort_signatures"] = np.array(
        [[1, 0], [0, 2], [2, 1]], dtype=np.float32
    )
    expected = adata.obsm["X_sort"] @ adata.varm["sort_signatures"].T

    normalize_Q_scale_W(adata)

    observed = (
        adata.obsm["X_sort_scaled"]
        @ adata.varm["sort_signatures_normalized"].T
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_alignment_cache_rejects_different_row_identity(tmp_path):
    first = _sample("first", ["A"])
    second = _sample("second", ["A"])
    merged = merge_samples([first, second])
    cache = tmp_path / "alignment.pkl"
    align_samples_grid(merged, cache_file=str(cache))

    changed = merged.copy()
    changed.obs_names = [f"changed_{index}" for index in range(changed.n_obs)]
    with pytest.raises(ValueError, match="does not match"):
        align_samples_grid(changed, cache_file=str(cache))


def test_pipeline_rejects_unknown_keyword():
    adata = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="not_a_parameter"):
        run_sort_pipeline(
            adata,
            preprocess=False,
            build_graph=False,
            not_a_parameter=True,
        )
