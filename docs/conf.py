# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import os
import sys
from datetime import date

# Make the package importable when it is not pip-installed (local builds).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import surface_renewal  # noqa: E402

# -- Project information -----------------------------------------------------

project = "surface_renewal"
author = "surface_renewal developers"
copyright = f"{date.today().year}, {author}"
release = surface_renewal.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- MyST (Markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",  # $...$ and $$...$$ math (used by theory.md)
    "amsmath",
]
# Generate GitHub-style anchors for headings so intra-page links in
# theory.md keep working.
myst_heading_anchors = 3

# -- autodoc / autosummary / napoleon ----------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True


def _skip_namedtuple_field_aliases(app, what, name, obj, skip, options):
    """Skip NamedTuple field descriptors ("Alias for field number N").

    The result NamedTuples document their fields in an ``Attributes`` docstring
    section; autodoc would otherwise emit the same attribute a second time.
    """
    doc = getattr(obj, "__doc__", None) or ""
    if doc.startswith("Alias for field number"):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", _skip_namedtuple_field_aliases)

# -- intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- HTML output ---------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_static_path = []
