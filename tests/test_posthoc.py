import numpy as np
from scipy import sparse

from sort import compute_posthoc_b, rank_genes_for_gep


def test_posthoc_dense_sparse_and_chunk_equivalence():
    rng = np.random.default_rng(7)
    W = rng.random((30, 3))
    true_B = rng.normal(size=(8, 3))
    intercept = rng.normal(size=8)
    X = intercept[None, :] + W @ true_B.T
    genes = [f"g{i}" for i in range(8)]

    dense = compute_posthoc_b(X, W, gene_names=genes, chunk_size=3)
    sparse_result = compute_posthoc_b(
        sparse.csr_matrix(X), W, gene_names=genes, chunk_size=None
    )
    np.testing.assert_allclose(dense.B, true_B, atol=1e-10)
    np.testing.assert_allclose(sparse_result.B, dense.B, rtol=1e-12, atol=1e-12)
    assert dense.params["B_orientation"] == "genes_by_components"

    ranked = rank_genes_for_gep(dense, 1, n_top=2, n_bottom=2)
    assert set(ranked["direction"]) <= {"positive", "negative"}
