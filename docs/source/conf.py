# Get meta information of OpenAirClim package

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import openairclim as oac

# Project information

project = oac.__title__
copyright = f"%Y, {oac.__author__}"
author = oac.__author__
release = oac.__version__

# General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "jupyter_sphinx",
    "myst_parser",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# allow both rst and md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = ["colon_fence"]

# autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_mock_imports = ["cf_units"]

templates_path = ["_templates"]
exclude_patterns = []


# Options for HTML output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "style_external_links": False,
    "includehidden": False,
    "version_selector": True,
}
html_context = {  # footer
    "footer_links": [
        ("Imprint", "imprint.html"),
        ("Privacy Policy", "privacy-policy.html"),
        ("Terms of Use", "terms-of-use.html"),
        ("Accessibility Statement", "accessibility-statement.html"),
    ]
}


# Other options

intersphinx_mapping = {
    "cartopy": ("https://scitools.org.uk/cartopy/docs/latest", None),
    "gedai": ("https://liammegill.github.io/gedai", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}
