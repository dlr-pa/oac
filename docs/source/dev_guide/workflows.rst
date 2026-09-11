Workflows
=========

This page covers how development actually flows through GitHub: how issues,
branches and pull requests fit together, and what the automated checks on a
pull request do (and which of them can block a merge). For how a new
OpenAirClim version actually gets released, see :doc:`releasing`. For the
full contribution policy - code of conduct, commit message conventions, do's
and don'ts - see
`CONTRIBUTING.md <https://github.com/dlr-pa/oac/blob/main/CONTRIBUTING.md>`__.


Working with GitHub
-------------------

.. image:: https://img.shields.io/github/issues/dlr-pa/oac
   :target: https://github.com/dlr-pa/oac/issues
   :alt: GitHub Issues
.. image:: https://img.shields.io/github/issues-pr/dlr-pa/oac
   :target: https://github.com/dlr-pa/oac/pulls
   :alt: GitHub Pull Requests


**Issues.** Before starting work, open a GitHub issue to discuss what you are
planning to do, using the bug report or feature request template. This
applies even if you already know how you would implement it. Doing it this way
gives the Steering Committee and other contributors a chance to weigh in before
any code is written.

**Branches.** ``main`` and ``dev`` are protected and can only be changed via
pull requests. New OpenAirClim versions are released from ``main``. Everyday
development happens on unprotected branches, named by what they contain:
``feature/<name>``, ``bug/<name>``, ``task/<name>`` and ``docs/<name>``.

.. image:: https://github.com/user-attachments/assets/47030a9b-f4dd-4350-a4f1-32836f8492bf
    :alt: OpenAirClim branching model

Contributors with write access to the repository (the core development team,
or external collaborators the Steering Committee has approved) can create
these branches directly. Everyone else works from a fork instead.

**Pull requests.** Reference the issue you opened (``Closes #123``) and fill
out the pull request template - type of change, how it was tested, and the
checklist (including confirming that ``example/example.toml`` still runs
against a clean installation). Also **apply a type label** (``type:
feature``, ``type: bug``/``type: fix``, ``type: docs``, ``type: chore``,
``type: testing``, or ``breaking``): this is what both the automated checks
below and the release changelog (see :doc:`releasing`) use to categorise
your change. The
`.github/release.yml <https://github.com/dlr-pa/oac/blob/main/.github/release.yml>`__
file defines the mapping.


GitHub Actions workflows
------------------------

.. image:: https://github.com/dlr-pa/oac/actions/workflows/pip-install-test.yml/badge.svg
  :target: https://github.com/dlr-pa/oac/actions
  :alt: pip installation status
.. image:: https://github.com/dlr-pa/oac/actions/workflows/conda-install-test.yml/badge.svg
  :target: https://github.com/dlr-pa/oac/actions
  :alt: conda installation status
.. image:: https://github.com/dlr-pa/oac/actions/workflows/pixi-install-test.yml/badge.svg
  :target: https://github.com/dlr-pa/oac/actions
  :alt: pixi installation status
.. image:: https://github.com/dlr-pa/oac/actions/workflows/build-docs.yml/badge.svg
  :target: https://github.com/dlr-pa/oac/actions
  :alt: docs status

.. |dependabot-icon| image:: https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/dependabot.svg
   :width: 16px
   :alt: Dependabot

.. list-table::
    :header-rows: 1
    :widths: 15, 25, 30
    :align: center

    * - Workflow
      - Runs on
      - What it does
    * - ``build-docs``
      - Every push to main
      - Builds and deploys this site
    * - | ``check-env-exports``
      - | PR or push touching
        | pixi-related files
      - | Checks pixi, pyproject and conda envs
        | are in sync
    * - | ``conda-``, ``pip`` & ``pixi-install-test``
      - | Push to main or dev, weekly
        | or on demand
      - | Tests pip, conda and pixi installations
    * - ``lint``
      - Every pull request
      - Correctness + style checks (see below)
    * - ``prepare-release``
      - On demand
      - Prepares for new oac release
    * - ``publish``
      - On release
      - Publishes to (Test)PyPI
    * - ``quick-test``
      - Every push/PR to main or dev
      - Fast pip sanity test
    * - |dependabot-icon| Dependabot
      - Weekly
      - Checks for dependency updates


Learn more about the individual checks in the dropdown below.


.. dropdown:: ``build-docs.yml``

  ``build-docs.yml`` deploys to `openairclim.org <https://openairclim.org>`__
  on every push to ``main`` - i.e. on every merge, not only at release time.
  This means **the live documentation can be slightly ahead of the latest
  published release** on PyPI/conda-forge, reflecting whatever has already
  landed on ``main`` since. We are exploring ways to host documentation
  per-release instead of only for the latest ``main`` - see `issue #104
  <https://github.com/dlr-pa/oac/issues/104>`__. This workflow also calculates
  the ``pytest`` code coverage.


.. dropdown:: ``check-env-exports.yml``

  ``pixi.lock`` and ``environment_minimal.yaml``/``environment_dev.yaml`` are
  all derived from the dependencies declared in ``pyproject.toml`` - the lock
  file by pixi itself, the yaml files by ``pixi run export-envs`` (see
  ``scripts/export-envs.sh``). Nothing keeps them in sync automatically if
  someone edits a dependency and forgets to regenerate them, so this workflow
  checks it instead:

  - ``pixi lock --check`` fails if ``pixi.lock`` is out of date with
    ``pyproject.toml``.
  - Regenerating ``environment_minimal.yaml``/``environment_dev.yaml`` and
    diffing the result against what is committed fails if they are stale.

  It only runs when a commit touches ``pyproject.toml``, ``pixi.lock``,
  ``scripts/export-envs.sh`` or the ``environment_*.yaml`` files themselves.


.. dropdown:: ``pip-`` & ``conda-install-test.yml``

  ``pip-install-test.yml`` and ``conda-install-test.yml`` test packaging rather
  than code correctness, so unlike ``quick-test.yml`` they don't run on every
  PR - only on push to ``main``/``dev``, weekly (to catch dependency drift
  during quieter development periods), and manually via ``workflow_dispatch``.


.. dropdown:: ``pixi-install-test.yml``

  ``pixi-install-test.yml`` runs on the same schedule as the ``pip`` and
  ``conda`` checks above, but checks something different. Since ``pixi.lock``
  pins every dependency, there is no drift to catch. The risk this test is
  designed to mitigate is instead that the locked environment doesn't actually
  work on a platform other than the one you last ran ``pixi install`` on. This
  matters because some dependencies have a separate wheel per operating system.


.. dropdown:: ``lint.yml``

  The linting workflow runs two independent jobs on Python files changed in the
  PR. Note that these checks will likely be replaced soon
  (see `#149 <https://github.com/dlr-pa/oac/issues/149>`__).

  - **correctness** (``pyflakes`` + ``mypy`` at medium strictness) is a
    **required check** - a PR cannot be merged while it fails.
  - **style** (``black --check`` plus the full Prospector report: pylint,
    pycodestyle, pydocstyle, pyroma, mccabe, dodgy) is **not** required. This
    job always completes successfully regardless of what it finds, so it can
    never block a merge on its own.

  You can run the same checks locally before opening a PR. See
  :ref:`running-code-quality-checks` in the installation guide.


.. dropdown:: ``prepare-release.yml``

  See :doc:`releasing`.


.. dropdown:: ``publish.yml``

  See :doc:`releasing`.


.. dropdown:: ``quick-test.yml``

  A fast, single-configuration sanity check that runs on every push/PR to
  ``main`` and ``dev``, for quick feedback on code correctness. It installs
  OpenAirClim with the ``gui`` and ``test`` extras, generates the test fixtures
  and runs the full ``pytest`` test suite.


.. dropdown:: |dependabot-icon| Dependabot

  The GitHub dependabot keeps things patched automatically. It opens weekly
  pull requests (up to 10 open at a time) bumping Python dependencies in
  ``pyproject.toml`` and GitHub Actions versions in ``.github/workflows/``.
  These carry Dependabot's own ``dependencies``/``github_actions`` labels.
