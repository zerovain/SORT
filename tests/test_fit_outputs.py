import anndata as ad
import numpy as np
import pytest
from scipy import sparse
import sort


def make_data():
    rng = np.random.default_rng(8)
    data = ad.AnnData(sparse.csr_matrix(rng.uniform(0.1, 2, (24, 9)).astype(np.float32)))
    data.var_names = [f"gene_{j}" for j in range(9)]
    data.var["highly_variable"] = np.arange(9) < 5
    data.uns["spatial_laplacian"] = sparse.eye(24, dtype=np.float32, format="csr")
    return data


def short_fit(data, **kwargs):
    return sort.fit(data, n_components=3, stage1_epochs=2, stage2_epochs=2,
                    adam_steps=2, device="numpy", random_state=17,
                    init_kwargs={"max_iter": 3}, **kwargs)


def test_unsliced_hvg_preserves_gene_axis_without_dense_reconstruction(tmp_path):
    data = make_data()
    result = short_fit(data, verbose=False)
    assert result.W.shape == (24, 3) and result.Q.shape == (9, 3)
    np.testing.assert_array_equal(result.gene_names, data.var_names)
    np.testing.assert_array_equal(result.Q[5:], 0)
    assert result.metadata["fitted_gene_mask"] == [True] * 5 + [False] * 4
    assert result.reconstruction is None and "sort_reconstructed" not in data.layers
    small = result.reconstruct(rows=[0, 2], genes=[0, 4])
    assert small.shape == (2, 2)
    result.save(str(tmp_path))
    assert not (tmp_path / "reconstruction.npy").exists()


def test_explicit_reconstruction_has_full_input_gene_axis(tmp_path):
    data = make_data()
    result = short_fit(data, verbose=False, store_reconstruction=True)
    assert result.reconstruction.shape == data.shape
    np.testing.assert_array_equal(result.reconstruction[:, 5:], 0)
    np.testing.assert_allclose(result.reconstruction, result.W @ result.Q.T, atol=1e-6)
    result.save(str(tmp_path), include_reconstruction=True)
    np.testing.assert_array_equal(np.load(tmp_path / "reconstruction.npy"), result.reconstruction)


def test_repeatability_and_progress_do_not_change_fit(capsys):
    fits = [short_fit(make_data(), verbose=mode) for mode in (False, False, True)]
    for other in fits[1:]:
        np.testing.assert_array_equal(other.W, fits[0].W)
        np.testing.assert_array_equal(other.Q, fits[0].Q)
    output = capsys.readouterr().out
    assert "device=numpy" in output and "Stage 1: 2/2" in output and "Stage 2: 2/2" in output
    assert "Soft Orthogonality" not in output and "Q-diag" not in output


def test_old_reconstruction_is_removed_on_refit():
    data = make_data()
    short_fit(data, verbose=False, store_reconstruction=True)
    short_fit(data, verbose=False)
    assert "sort_reconstructed" not in data.layers


def test_empty_hvg_selection_has_a_clear_error():
    data = make_data()
    data.var["highly_variable"] = False
    with pytest.raises(ValueError, match="No genes selected"):
        short_fit(data, verbose=False)
