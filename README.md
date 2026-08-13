# SORT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8–3.11](https://img.shields.io/badge/python-3.8--3.11-blue.svg)](pyproject.toml)

SORT (Spatial Orthogonal Regularized Transcriptomic decomposition) learns a
nonnegative spot-by-component activity matrix `W` and a shared
gene-by-component loading matrix `Q` from spatial transcriptomic data.

This release preserves the optimizer and initialization used for the
associated manuscript. Interface and packaging changes are checked against a
frozen deterministic reference. Filtering, normalization and feature selection
are ordinary Scanpy steps and remain explicit in each analysis workflow.

## Installation

Clone the repository and create the tested environment:

```bash
git clone https://github.com/zerovain/SORT.git
cd SORT
conda env create -f environment.yml
conda activate sort-paper-release
python -m pip install -e . --no-deps
```

GPU execution requires a CUDA-compatible PyTorch installation and CuPy. With
`device="auto"`, the high-level API uses CUDA only when both runtimes are
available and otherwise falls back to NumPy. The exact tested environment is
recorded in `environment.yml`.

## Basic use

Graph construction is explicit because it is part of the analysis definition.

```python
import scanpy as sc
import sort

adata = sc.read_h5ad("prepared_spatial_data.h5ad")

# Example preprocessing; adapt filtering and feature selection to the dataset.
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
adata = adata[:, adata.var["highly_variable"]].copy()

sort.build_spatial_graph(adata, n_neighbors=8)
sort.compute_laplacian(adata)

result = sort.fit(
    adata,
    n_components=25,
)

print(result.W.shape)  # spots x components
print(result.Q.shape)  # genes x components
```

The lower-level manuscript function remains available for exact archived
workflows:

```python
import sort

sort.decompose(adata, n_components=25, random_state=42)
```

## Result conventions

- `X`: spots by genes;
- `W`: spots by components and nonnegative;
- `Q`: genes by components;
- component 0 is the fitted background in the manuscript analyses.

Whenever an external `Q` is loaded, validate its shape against
`W.shape[1]`. Do not assume that every historical file uses the same
orientation.

`decompose` retains the manuscript output contract, including
`adata.layers['sort_reconstructed']`. `SORTResult.save()` omits that dense
array unless `include_reconstruction=True`; this changes export size only and
does not change the fitted AnnData object.

## Tutorials

The standalone tutorial repository is available at
[zerovain/SORT_tutorial](https://github.com/zerovain/SORT_tutorial).

The `tutorials/notebooks/` directory follows the short, task-oriented style of
common spatial-transcriptomics vignettes:

1. a complete simulated eight-section atlas from counts through fitted `W/Q`;
2. the complete PDAC atlas workflow;
3. the complete eight-stage embryogenesis workflow.

Each tutorial is limited to one or two focused figures. Paper-panel assembly,
survival analysis, full benchmark grids, and enrichment screens remain in the
separate archived manuscript-analysis code.

Only the simulated H5AD is redistributed. Biological tutorials start from the
original repositories and provide preparation scripts; processed third-party
H5AD files are not bundled.

The paired Python sources generate the three canonical Jupyter notebooks and
static Markdown or HTML without changing the paper environment. See
`docs/TUTORIALS.md` for the isolated build commands and `docs/index.rst` for
the Read the Docs homepage.

Direct notebook links:

- [Simulated spatial atlas](tutorials/notebooks/00_simulated_atlas.ipynb)
- [Complete PDAC atlas](tutorials/notebooks/01_pdac_atlas.ipynb)
- [Mouse embryogenesis atlas](tutorials/notebooks/02_embryogenesis.ipynb)

The repository includes a complete Read the Docs configuration. Until the
hosted site is activated, GitHub renders the executed notebooks directly.

## Reproducibility

Run the lightweight checks with:

```bash
python -m pytest
```

The deterministic reference test compares the release to a fixture generated
by the manuscript package using the tested CPU environment. GPU reruns are
assessed with numeric tolerances and matched-component correlations rather
than described as bitwise deterministic.

## Interpretation boundary

A learned component is a gene-expression program, not automatically a cell
type, pathway, differential-expression result, or causal effect. Post-hoc `B`
coefficients are conditional associations for interpretation and are distinct
from fitted `Q` loadings.

## Citation and license

See `CITATION.cff` for citation metadata. The software is released under the
[MIT License](LICENSE).

## Support

Please report bugs and usage questions through the
[GitHub issue tracker](https://github.com/zerovain/SORT/issues).
