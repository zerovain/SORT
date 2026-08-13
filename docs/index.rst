SORT tutorial
=============

SORT (Spatial Orthogonal Regularized Transcriptomic decomposition) learns
spatial gene-expression programs shared across sections, samples or stages.
The tutorials show the complete analysis path from a prepared count matrix to
the fitted spot-by-program matrix ``W`` and gene-by-program matrix ``Q``.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   Tutorial 1: Simulated spatial atlas <tutorials/00_simulated_atlas>
   Tutorial 2: Complete PDAC atlas <tutorials/01_pdac_atlas>
   Tutorial 3: Mouse embryogenesis atlas <tutorials/02_embryogenesis>

.. toctree::
   :hidden:

   TUTORIALS
   DATA_AVAILABILITY
   REPRODUCIBILITY
   PAPER_RELEASE_SCOPE

Introduction
------------

SORT is designed for joint analysis of spatial transcriptomic atlases. It
models nonnegative spatial activities while learning shared gene-expression
profiles and incorporating a within-section spatial graph.

Main features
-------------

* unsupervised discovery of spatial gene-expression programs;
* joint modeling of multiple sections, samples or developmental stages;
* explicit per-section spatial graph construction;
* spot-by-program activities and gene-by-program profiles with validated
  orientations;
* CPU and CUDA execution through the same analysis interface.

Installation
------------

Create the tested environment and install the package from the repository:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate sort-paper-release
   python -m pip install -e . --no-deps

CUDA execution additionally requires a compatible PyTorch and CuPy
installation. With ``device="auto"``, SORT uses CUDA only when both runtimes
are available and otherwise falls back to NumPy.

Quick start
-----------

Preprocessing remains visible because filtering and feature selection depend
on the dataset. SORT-specific steps begin with construction of the spatial
graph:

.. code-block:: python

   import scanpy as sc
   import sort

   adata = sc.read_h5ad("prepared_spatial_data.h5ad")
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)
   sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
   adata = adata[:, adata.var["highly_variable"]].copy()

   sort.build_spatial_graph(adata, n_neighbors=8)
   sort.compute_laplacian(adata)
   result = sort.fit(adata, n_components=25)

   result.W.shape  # locations x components
   result.Q.shape  # genes x components

Tutorial datasets
-----------------

The simulation input is included in the repository. The PDAC and
embryogenesis notebooks use complete public atlases assembled locally from
their original repositories; processed third-party H5AD files are not
redistributed. See :doc:`TUTORIALS` for preparation and build instructions.

Output interpretation
---------------------

``result.W`` contains nonnegative spatial activities and ``result.Q`` contains
the corresponding gene loadings. Component 0 is the fitted background in the
paper analyses. A learned component is a gene-expression program and should
not automatically be interpreted as a cell type, pathway or causal effect.

Reproducibility and paper analyses
----------------------------------

The tutorials teach the public interface; they are not an exhaustive
collection of every paper workflow. Exact figure and supplementary analyses,
including dataset-specific settings, are retained in the companion Zenodo
archive. See :doc:`REPRODUCIBILITY` and :doc:`PAPER_RELEASE_SCOPE`.

Citation and license
--------------------

Citation metadata are provided in ``CITATION.cff``. SORT is released under the
MIT License.

Support
-------

Please use the `SORT GitHub issue tracker
<https://github.com/zerovain/SORT/issues>`_ for bug reports and usage
questions.
