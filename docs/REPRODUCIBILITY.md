# Reproducibility

The paper release keeps the manuscript optimizer and initialization path
unchanged. The companion Zenodo workflows, rather than software convenience
defaults, freeze the exact preprocessing used by each paper analysis. Exact
reproduction additionally depends on input ordering, graph, software
environment, backend, and random seed.

## Required records

For each fit, retain:

- input dataset/accession and preprocessing description;
- observation, sample, and gene order;
- spatial-graph method and parameters;
- the mapping from each sample-local graph row to the global observation row,
  plus a zero-cross-sample-edge check;
- all SORT and initializer parameters;
- random seed, CPU/GPU backend, and package version;
- `W` and `Q` shapes and orientations.

## Equality policy

The included tiny CPU fixture uses exact equality in the tested manuscript
environment. For GPU reruns, report maximum absolute and relative Frobenius
differences plus matched component correlations. Tolerance agreement is not
equivalent to bitwise identity.

For a corrected workflow, non-equality to an output generated with a different
graph-row mapping is expected and must not be hidden by component relabeling.
Compare fits with explicit component matching and rerun the downstream
quantities used in the paper.

The full manuscript parameter sets are recorded in Supplementary Table 4 and
the companion Zenodo analysis archive. These records
record analysis settings; they do not embed private input paths.
