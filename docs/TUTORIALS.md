# Tutorial publishing

Reviewable Jupytext sources under `tutorials/notebooks/` generate ordinary
Jupyter notebooks. GitHub renders the `.ipynb` files directly, and the same
sources can generate static HTML or Markdown for a documentation website.

The complete seed-03 simulation is bundled at
`tutorials/data/simulation/seed_0003.h5ad`. Biological H5AD files are not
redistributed. After downloading the original data, create local inputs with:

```bash
python tutorials/prepare/prepare_pdac_atlas.py \
  --visium-root /path/to/GSE282302/visium_samples \
  --output tutorials/data/pdac/pdac_atlas_counts.h5ad

python tutorials/prepare/prepare_embryogenesis.py \
  --input-dir /path/to/MOSTA \
  --output tutorials/data/embryogenesis/embryogenesis_counts.h5ad

```

The assembly scripts only combine source files and attach identifiers. Scanpy
preprocessing and SORT graph construction remain visible in each notebook.

Install notebook tooling in an isolated environment when the analysis
environment must remain frozen:

```bash
python -m venv .venv-docs
.venv-docs/bin/python -m pip install -e '.[docs]'
.venv-docs/bin/python tutorials/build_docs.py
```

The build command synchronizes the three notebooks into `docs/tutorials/` for
Read the Docs. Use `--format html` or `--format markdown` for standalone pages.
The `--execute` option executes all three notebooks and therefore requires the
two local biological H5AD files and an appropriate compute node.

To preview the Read the Docs site locally:

```bash
.venv-docs/bin/sphinx-build -b html docs docs/_build/html
```

Read the Docs uses `.readthedocs.yaml`, installs `docs/requirements.txt`, and
renders the committed notebook outputs without executing the atlas fits.
