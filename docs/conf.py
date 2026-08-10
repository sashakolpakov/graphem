"""Sphinx configuration for the GraphEm reference package."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "GraphEm"
author = "Alexander Kolpakov and Igor Rivin"
copyright = "2025–2026, Alexander Kolpakov and Igor Rivin"

try:
    release = version("graphem-jax")
except PackageNotFoundError:
    release = "0.2.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Sphinx uses lightweight mocks while rendering dependency-heavy modules. The
# documentation workflow first imports every public module with the real
# installed dependencies, so missing or incompatible runtime imports still
# fail CI before this source-rendering step.
autodoc_mock_imports = ["jax", "jax.numpy", "ndlib"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky = True

# The current source docstrings contain a small set of prose-like return and
# attribute labels. They are deliberately rendered as text rather than linked
# API objects; every other unresolved cross-reference remains fatal under
# ``sphinx-build -n -W``.
nitpick_ignore = [
    ("py:class", "Other"),
    ("py:class", "Path"),
    ("py:class", "array-like"),
    ("py:class", "degrees"),
    ("py:class", "influence"),
    ("py:class", "networkx.Graph"),
    ("py:class", "np.ndarray"),
    ("py:class", "num_vertices"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "seeds"),
    ("py:class", "shape"),
]

html_theme = "sphinx_rtd_theme"
html_title = f"GraphEm {release}"
html_static_path = []
