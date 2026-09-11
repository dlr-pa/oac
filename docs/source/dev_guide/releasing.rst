Releasing
=========

This page covers how a new OpenAirClim release actually gets published -
both the ``openairclim`` package itself and its separately-versioned
`repository data <https://github.com/dlr-pa/oac-data>`__. For how day-to-day
development flows through GitHub (issues, branches, pull requests, the checks
that run on them), see :doc:`workflows`.


Releasing a new version
-----------------------

1. An admin manually triggers ``prepare-release.yml`` through
   `GitHub Actions <https://github.com/dlr-pa/oac/actions>`__ with the new
   semantic version number. It:

   - generates release notes via GitHub's "Generate release notes" API using
     the label mapping in ``.github/release.yml`` - and prepends them under a
     new ``## [version] - date`` heading in ``CHANGELOG.md``;
   - bumps ``__version__`` in ``openairclim/__about__.py`` and the
     version/date in ``CITATION.cff``;
   - opens a pull request (``release/v<version>`` into ``main``) with these
     changes.

2. A maintainer reviews that pull request, updates the automatically
   generated CHANGELOG if necessary, and merges it.

3. A maintainer creates a new GitHub Release from ``main`` (tagging
   ``vX.Y.Z`` as part of that). Publishing the Release automatically
   triggers ``publish.yml``, which builds the sdist/wheel and publishes them
   to PyPI using Trusted Publishing (OIDC).

4. Once the release is on PyPI, conda-forge's autotick bot notices and opens
   a pull request against the `openairclim-feedstock
   <https://github.com/conda-forge/openairclim-feedstock>`__ repository
   automatically. A feedstock maintainer just needs to review and merge it
   (usually a straightforward version/hash bump) for the release to be picked
   up by conda-forge. A larger review may be necessary if new dependencies were
   added in the new release.

``publish.yml`` can also be triggered manually at any time, independent of an
actual release, to publish a dry-run build to `TestPyPI
<https://test.pypi.org/project/openairclim/>`__ instead of the real PyPI -
useful for testing packaging changes before cutting a release.

.. warning::

  TestPyPI, like PyPI, refuses to re-accept a version that has already been
  uploaded. For that reason, the ``publish.yml`` workflow optionally accepts a
  version override (:pep:`440`, e.g. ``0.17.1.dev0``) so that repeat dry runs
  don't collide. Always make sure that you double-check the input to manual
  ``publish.yml`` triggers!


Releasing repository data
-------------------------

Since ``v0.18.0``, the response surface and background concentration files that
``oac-download-data`` fetches no longer live in this repository. Instead, they
are published from a separate one, `dlr-pa/oac-data
<https://github.com/dlr-pa/oac-data>`__, and versioned independently of
``openairclim``. A new data release does not automatically change what any
existing or new ``openairclim`` installation downloads by default - that is a
deliberate, separate opt-in step (below).

1. Publishing a **GitHub Release** on ``dlr-pa/oac-data`` (or running its
   ``zenodo-sync.yml`` workflow manually via ``workflow_dispatch``) uploads
   the current ``repository/*.nc`` files as a new version under the
   existing Zenodo record and publishes it. Concretely, the workflow:

   - looks up the latest already-published version under the target Zenodo
     record (configured via a repo *variable*, ``ZENODO_RECORD_ID`` - this
     must be an actual, already-published version's own record ID. It should
     not need to be updated for later versions);
   - creates a new draft version at Zenodo and uploads this repo's current
     ``repository/*.nc`` files into that draft, overwriting any same-named
     files carried over from the previous version;
   - updates only the ``version`` field in the draft's metadata (title,
     authors, licence and description are left untouched); and
   - publishes the draft.

   This needs a repo *secret*, ``ZENODO_TOKEN`` (a Zenodo personal access
   token with ``deposit:write`` and ``deposit:actions`` scopes), configured
   on ``dlr-pa/oac-data``. An optional ``ZENODO_BASE_URL`` variable (e.g.
   ``sandbox.zenodo.org``) lets you dry-run the whole thing against Zenodo's
   sandbox instead of production.

   The upload is done with the custom ``zenodo-sync.yml`` workflow rather than
   Zenodo's GitHub integration (like for the main ``openairclim`` repository)
   because the integration stores the full repository on Zenodo as a ZIP file,
   whereas OpenAirClim needs individual repository files in the cache.

2. This alone does **not** make ``openairclim`` start fetching the new data
   version. To point OpenAirClim at new data by default, update the
   :data:`openairclim.repository.DEFAULT_REPOSITORY_DATA_VERSION` variable in
   ``openairclim/repository.py``. Note that you can still download a previous
   version of the data using ``oac-download-data --version``/``--record``.


Permissions
-----------

The following permissions are required by the admin team to keep the full
OpenAirClim workflow running. If someone leaves the team, make sure to run
through this list to ensure that any permissions are transferred.

- **Write access to** `dlr-pa/oac <https://github.com/dlr-pa/oac>`__ **on
  GitHub**, to trigger ``prepare-release.yml``, merge the release PR, and
  publish the GitHub Release.
- **Owner/maintainer role on the** ``openairclim`` **PyPI (and TestPyPI)
  project**. This is not strictly necessary to release a new version, since
  publication is done using OIDC, but it is required to manage the project
  listing and settings.
- **Maintainer status on** `conda-forge/openairclim-feedstock
  <https://github.com/conda-forge/openairclim-feedstock>`__  to review and
  merge the autotick bot's pull requests.
- **Write access to** `dlr-pa/oac-data
  <https://github.com/dlr-pa/oac-data>`__, to publish the GitHub Releases
  that trigger a repository data sync, plus a **Zenodo account holding the**
  ``ZENODO_TOKEN`` **secret** configured there (``deposit:write`` and
  ``deposit:actions`` scopes).
- **Access to the openairclim.org DNS** (currently: DLR-PA IT). This does not
  need active maintenance. It just needs to remain pointed at GitHub Pages.
- **Access to the openairclim@dlr.de functional email**. This is the main 
  point of contact for outsiders to the Steering Committee and Scientific &
  Technical Boards.
