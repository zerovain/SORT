# SORT tutorials

The notebooks follow a short spatial-transcriptomics vignette structure:
load data, show the Scanpy preprocessing, construct the section-wise spatial
graph, fit SORT and visualize one or two biologically interpretable results.

1. `00_simulated_atlas`: complete seed-03 simulation (10,368 locations,
   5,000 modeled genes and eight sections).
2. `01_pdac_atlas`: complete 108-section GSE282302 PDAC atlas.
3. `02_embryogenesis`: complete eight-stage MOSTA embryogenesis series.

Only the author-generated simulation H5AD is distributed. The biological
notebooks link to the original repositories and use the assembly scripts under
`tutorials/prepare/`; the resulting H5AD files stay local. Normalization,
`log1p`, highly variable gene selection and graph construction are shown in the
notebooks rather than hidden in a SORT convenience pipeline.

The notebooks use the documented preprocessing order and fixed analysis
settings. The PDAC and embryogenesis notebooks use the package-default
initializer. Full paper reproduction—including datasets not used as public
tutorials—remains in the companion Zenodo analysis archive.

See `docs/TUTORIALS.md` for notebook and webpage build instructions.
