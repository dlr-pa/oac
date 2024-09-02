# Get meta information of OpenAirClim package
about = {}
with open("../../openairclim/__about__.py", mode="r", encoding="utf8") as fp:
    exec(fp.read(), about)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = about["__title__"]
copyright = about["__copyright__"]
author = about["__author__"]
release = about["__version__"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

# autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_mock_imports = ["cf_units"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "classic"
# html_theme = "sphinxdoc"
# html_theme = "bizstyle"
html_static_path = ["_static"]
