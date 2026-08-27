"""
Resolves which release of OpenAirClim's repository data (background
concentration scenarios and response-surface lookup tables) this installed
version of OpenAirClim expects.

That data is published independently of this package, in
https://github.com/dlr-pa/oac-repository, with its own Zenodo-backed
versioning. Caching and download logic will be introduced in a follow-up
change; this module currently only pins the default data release.
"""

#: Concept DOI for the dlr-pa/oac-repository Zenodo deposition (always
#: resolves to its most recent version). Placeholder until the repository
#: is published and Zenodo's GitHub integration is enabled for it.
REPOSITORY_DATA_CONCEPT_DOI = "10.5281/zenodo.PLACEHOLDER"

#: Data repository release that this installed version of openairclim expects
#: by default. Deliberately independent of openairclim's own version number.
DEFAULT_REPOSITORY_DATA_VERSION = "0.1.0"
