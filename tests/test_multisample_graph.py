import anndata as ad
import numpy as np

import sort


def test_per_sample_graph_respects_interleaved_global_rows():
    adata = ad.AnnData(np.ones((6, 2), dtype=np.float32))
    adata.obs["sample"] = ["z", "a", "z", "a", "z", "a"]
    adata.obsm["spatial"] = np.array(
        [[0, 0], [100, 0], [1, 0], [101, 0], [2, 0], [102, 0]],
        dtype=np.float32,
    )

    sort.build_per_sample_graph(adata, n_neighbors=1, sample_key="sample")
    laplacian = adata.uns["spatial_laplacian"].tocsr()
    z = np.array([0, 2, 4])
    a = np.array([1, 3, 5])
    assert laplacian[z][:, a].nnz == 0
    assert laplacian[a][:, z].nnz == 0
