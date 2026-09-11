"""Configure sphinx documentation setup."""

import os
import re
import sys
import locale

locale.setlocale(locale.LC_ALL, "C")

sys.path.insert(0, os.path.abspath("../.."))
import openairclim as oac

# -- Project information ------------------------------------------------------

project = oac.__title__
copyright = f"%Y, {oac.__author__}"
author = oac.__author__
release = oac.__version__

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.bibtex",
    "sphinx_rtd_theme",
    "sphinx_design",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# .rst and plain/notebook .md files (myst_nb registers .md/.ipynb itself)
source_suffix = {
    ".rst": "restructuredtext",
}

myst_enable_extensions = ["colon_fence", "linkify"]

# Demo pages under docs/source/demos are MyST Markdown notebooks (see their
# `file_format: mystnb` front matter). "cache" only (re-)executes a notebook
# when its content changes, keyed by content hash in the jupyter-cache store
# below
nb_execution_mode = "cache"
nb_execution_cache_path = "../build/.jupyter_cache"
nb_execution_timeout = 300

# bibtex options
bibtex_bibfiles = ["bibliography.bib"]
bibtex_default_style = "plain"

# autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_mock_imports = []

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "style_external_links": False,
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


# -- Other options ------------------------------------------------------------

intersphinx_mapping = {
    "gedai": ("https://liammegill.github.io/gedai", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}


# -- Auto-link @mentions and #issue/PR references in CHANGELOG.md -------------

_GH_REPO_URL = "https://github.com/dlr-pa/oac"
_ISSUE_RE = re.compile(r"(?<![\w#])#(\d+)\b")
_PR_URL_RE = re.compile(re.escape(_GH_REPO_URL) + r"/pull/(\d+)\b")
_MENTION_RE = re.compile(
    r"(?<!\w)@([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}[A-Za-z0-9])?)(\[bot\])?"
)


def _linkify_changelog(text: str) -> str:
    # bare "#123" (old entries) first, before any "#123"-looking text below
    # gets introduced, which must not be re-matched by this same pass
    text = _ISSUE_RE.sub(rf"[#\1]({_GH_REPO_URL}/issues/\1)", text)

    # full pull URLs (new entries) -> shortened "#123" display text
    text = _PR_URL_RE.sub(lambda m: f"[#{m.group(1)}]({m.group(0)})", text)

    def _mention_sub(match: "re.Match[str]") -> str:
        username, bot_suffix = match.group(1), match.group(2) or ""
        profile = (
            f"https://github.com/apps/{username}"
            if bot_suffix
            else f"https://github.com/{username}"
        )
        return f"[@{username}{bot_suffix}]({profile})"

    return _MENTION_RE.sub(_mention_sub, text)


def _source_read(app, docname, source):
    if docname != "changelog":
        return
    changelog_path = os.path.join(app.srcdir, "..", "..", "CHANGELOG.md")
    with open(changelog_path, encoding="utf-8") as f:
        source[0] = _linkify_changelog(f.read())


def setup(app):
    """Register the changelog auto-linking hook."""
    app.connect("source-read", _source_read)
