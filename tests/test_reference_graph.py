"""Freeze the directed kNN convention used by Squidpy 1.2.3."""
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from sklearn.neighbors import NearestNeighbors

import sort


def test_knn_laplacian_matches_historical_direction():
    # Uneven spacing makes the directed graph asymmetric; there are no ties.
    coords = np.array([[0., 0.], [0.8, 0.1], [2.7, 0.3],
                       [6.5, 1.1], [12.2, 0.4], [20.9, 2.3]])
    adata = ad.AnnData(np.ones((len(coords), 2), dtype=np.float32))
    adata.obs['sample_id'] = 'section'
    adata.obsm['spatial'] = coords
    k = 2
    neighbors = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors()[1]
    adjacency = csr_matrix((np.ones(neighbors.size),
                            (neighbors.ravel(), np.repeat(np.arange(len(coords)), k))),
                           shape=(len(coords), len(coords)))
    assert (adjacency != adjacency.T).nnz > 0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    np.power(degree, -0.5, out=inverse, where=degree > 0)
    d = diags(inverse)
    expected = eye(len(coords), format='csr') - d @ adjacency @ d
    sort.build_per_sample_graph(adata, n_neighbors=k)
    actual = adata.uns['spatial_laplacian']
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
