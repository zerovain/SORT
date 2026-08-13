#!/usr/bin/env python3
"""Assemble all eight MOSTA stages for the SORT tutorial."""

import argparse
from pathlib import Path

import anndata as ad
import scanpy as sc


STAGES = ("E9.5", "E10.5", "E11.5", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    datasets = []
    for stage in STAGES:
        path = args.input_dir / f"{stage}_original.h5ad"
        if not path.is_file():
            raise FileNotFoundError(path)
        stage_data = sc.read_h5ad(path)
        stage_data.var_names_make_unique()
        stage_data.obs["slice_id"] = stage
        datasets.append(stage_data)

    atlas = ad.concat(datasets, join="outer", index_unique="-")
    atlas.uns["tutorial_provenance"] = {
        "source": "MOSTA",
        "scope": "complete eight-stage atlas",
        "processed_data_redistributed": False,
        "matrix": "outer-joined counts before SORT preprocessing",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atlas.write_h5ad(args.output, compression="gzip")
    print(f"Saved count atlas: {atlas.n_obs:,} locations x {atlas.n_vars:,} genes")


if __name__ == "__main__":
    main()
