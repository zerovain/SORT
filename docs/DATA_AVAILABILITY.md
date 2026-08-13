# Data inputs

The repository distributes one author-generated simulation input:

- seed 0003; 10,368 locations, eight sections and the 5,000 genes supplied to
  SORT;
- raw integer counts in `.X`, spatial coordinates in `.obsm['spatial']`,
  section metadata, aligned spatial and gene-signature truth, and the exact
  block-diagonal spatial Laplacian;
- no fitted `W`, `Q` or reconstructed expression;
- SHA-256: `e52b52fe4bbfaea03e850083e11a44181a967b8e3510b0133007016e96d7413a`.

Processed third-party biological H5AD files are not redistributed. Tutorials
instead start from the original sources:

- PDAC Visium: GEO
  [GSE282302](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE282302);
- mouse embryogenesis: [MOSTA](https://db.cngb.org/stomics/mosta/);
- cross-species cerebellum:
  [CBMSTA STOmics](http://db.cngb.org/stomics/cbmsta).

Preparation scripts accept user-supplied source directories and write new
model-ready files. They never edit raw data in place. Private filesystem paths
and local raw filenames are not part of the public interface.
