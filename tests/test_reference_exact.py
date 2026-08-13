"""Exact regression against a tiny fixture from the manuscript package."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from sort import SORT, SORTConfig, build_spatial_graph, compute_laplacian


REFERENCE = Path(__file__).parent / "data" / "tiny_cpu_reference.npz"


def test_tiny_cpu_reference_is_exact():
    reference = np.load(REFERENCE)
    adata = ad.AnnData(sparse.csr_matrix(reference["X"]))
    adata.obs_names = reference["obs_names"].astype(str)
    adata.var_names = reference["var_names"].astype(str)
    adata.obsm["spatial"] = reference["coordinates"]

    build_spatial_graph(adata, n_neighbors=4)
    compute_laplacian(adata)
    np.testing.assert_array_equal(
        adata.uns["spatial_laplacian"].toarray(), reference["laplacian"]
    )

    config = SORTConfig(
        n_components=3,
        use_highly_variable=False,
        alpha=0.2,
        beta=0.05,
        lambda_l1_W=0.02,
        lambda_l1_Q=0.03,
        l1_weight_strategy="fixed",
        lambda_neg=1.0,
        use_tv=True,
        tv_epsilon=1e-2,
        tv_update_freq=1,
        tv_stage="stage2",
        stage1_epochs=2,
        stage2_epochs=2,
        adam_steps=2,
        grad_clip_norm=1.0,
        device="numpy",
        random_state=17,
        auto_init=True,
        init_kwargs={
            "max_iter": 3,
            "alpha": 0.2,
            "lambda1": 0.02,
            "lambda2": 0.0,
            "smooth_center_weight": 0.5,
            "tol": 1e-5,
            "backend": "numpy",
            "verbose": False,
        },
        verbose=False,
    )
    result = SORT(config).fit(adata)

    np.testing.assert_array_equal(result.W, reference["W"])
    np.testing.assert_array_equal(result.Q, reference["Q"])
    np.testing.assert_array_equal(result.reconstruction, reference["reconstruction"])
    assert result.metadata["sort_params"] == json.loads(str(reference["params_json"]))
    assert np.all(result.W >= 0)
