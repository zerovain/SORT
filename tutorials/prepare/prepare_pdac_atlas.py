#!/usr/bin/env python3
"""Assemble the complete GSE282302 count atlas for the SORT tutorial."""

import argparse
from pathlib import Path
import re

import anndata as ad
import scanpy as sc

PATIENT_RE = re.compile(r"_(C\d+)_(D\d+)(?:_|$)")


def patient_id_from_sample(sample_name: str) -> str:
    match = PATIENT_RE.search(str(sample_name))
    if match is None:
        raise ValueError(f"cannot parse cohort/donor from section name: {sample_name}")
    return f"{match.group(1)}_{match.group(2)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visium-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    samples = []
    sample_keys = []
    for directory in sorted(path for path in args.visium_root.iterdir() if path.is_dir()):
        matrix = directory / "filtered_feature_bc_matrix.h5"
        if not matrix.is_file():
            continue
        sample = sc.read_visium(directory, count_file=matrix.name, load_images=False)
        sample.var_names_make_unique()
        sample.obs["sample_name"] = directory.name
        sample.obs["patient_id"] = patient_id_from_sample(directory.name)
        samples.append(sample)
        sample_keys.append(str(len(sample_keys)))
    if len(samples) != 108:
        raise ValueError(f"expected 108 GSE282302 sections, found {len(samples)}")

    atlas = ad.concat(
        samples, join="inner", label="sample_id", keys=sample_keys,
        index_unique="-",
    )
    if atlas.obs["patient_id"].nunique() != 39:
        raise ValueError("expected 39 GSE282302 patients after parsing section names")
    atlas.uns["tutorial_provenance"] = {
        "accession": "GSE282302",
        "scope": "complete 108-section atlas",
        "processed_data_redistributed": False,
        "matrix": "unmodified merged counts before SORT preprocessing",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atlas.write_h5ad(args.output, compression="gzip")
    print(f"Saved count atlas: {atlas.n_obs:,} locations x {atlas.n_vars:,} genes")


if __name__ == "__main__":
    main()
