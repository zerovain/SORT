#!/usr/bin/env python3
"""Build paired Jupyter notebooks and optional static tutorial pages."""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "tutorials" / "notebooks").glob("[0-9][0-9]_*.py"))
DOC_NOTEBOOKS = ROOT / "docs" / "tutorials"


def require(module: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise SystemExit(
            f"{module!r} is required; install the isolated docs extra with "
            "python -m pip install -e '.[docs]'"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sphinx", action="store_true")
    parser.add_argument("--format", choices=("none", "markdown", "html"), default="none")
    parser.add_argument("--output-dir", default="docs/_build/standalone")
    args = parser.parse_args()

    require("jupytext")
    notebooks = []
    for source in SOURCES:
        notebook = source.with_suffix(".ipynb")
        subprocess.run(
            [sys.executable, "-m", "jupytext", "--sync", str(source)],
            check=True,
            cwd=ROOT,
        )
        if not notebook.is_file():
            raise SystemExit(f"Jupytext did not create the paired notebook: {notebook}")
        notebooks.append(notebook)

    if args.execute or args.format != "none":
        require("nbconvert")
    if args.execute:
        for notebook in notebooks:
            subprocess.run(
                [
                    sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=3600", str(notebook),
                ],
                check=True,
                cwd=notebook.parent,
            )

    DOC_NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for notebook in notebooks:
        shutil.copy2(notebook, DOC_NOTEBOOKS / notebook.name)

    if args.format != "none":
        output = ROOT / args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        template_args = ["--template", "classic"] if args.format == "html" else []
        subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert", "--to", args.format,
                *template_args, "--output-dir", str(output), *map(str, notebooks),
            ],
            check=True,
            cwd=ROOT,
        )
    if args.sphinx:
        require("sphinx")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-W",
                "-b",
                "html",
                str(ROOT / "docs"),
                str(ROOT / "docs" / "_build" / "html"),
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
