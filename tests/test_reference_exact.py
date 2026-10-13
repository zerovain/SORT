"""Float32 optimizer regression with frozen graph and manuscript initialization.

The original end-to-end fixture remains in data/tiny_cpu_reference.npz.
KNN tie-breaking and SVD initialization are tested separately from optimization.
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from sort import SORT, SORTConfig, build_spatial_graph, compute_laplacian


REFERENCE = Path(__file__).parent / "data" / "tiny_optimizer_reference_v2.npz"


def test_fixed_graph_and_initialization_match_manuscript_optimizer():
    reference = np.load(REFERENCE)
    adata = ad.AnnData(sparse.csr_matrix(reference["X"]))
    adata.obs_names = reference["obs_names"].astype(str)
    adata.var_names = reference["var_names"].astype(str)
    adata.obsm["spatial"] = reference["coordinates"]

    adata.uns["spatial_laplacian"] = sparse.csr_matrix(reference["laplacian"])
    np.random.seed(17)

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
        auto_init=False,
        store_reconstruction=True,
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
    # Warm-start matrices are forwarded by the retained low-level entry point.
    from sort import decompose
    decompose(adata, **config.to_decompose_kwargs(),
              W_init=reference["W_init"], Q_init=reference["Q_init"])

    # Cross-platform float32 arithmetic, not a permissive end-to-end tolerance.
    for actual, expected in [(adata.obsm["X_sort"], reference["W"]),
                             (adata.varm["sort_signatures"], reference["Q"]),
                             (adata.layers["sort_reconstructed"], reference["reconstruction"])]:
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert np.all(adata.obsm["X_sort"] >= 0)
    provenance = json.loads(str(reference["provenance_json"]))
    assert provenance["seed"] == 17
