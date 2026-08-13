"""Sphinx configuration for the SORT tutorial site."""


project = "SORT"
author = "SORT authors"
copyright = "2026, SORT authors"

extensions = ["nbsphinx", "myst_parser"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_title = "SORT tutorial"

nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_requirejs_path = ""

master_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Keep notebook links relative and portable on Read the Docs.
html_context = {"display_github": False}
