Installation
============

.. image:: https://img.shields.io/github/v/tag/dlr-pa/oac?label=github&logo=github
    :target: https://github.com/dlr-pa/oac
    :alt: Latest GitHub tag
.. image:: https://img.shields.io/pypi/v/openairclim?color=orange&label=pypi&logo=pypi&logoColor=white
    :target: https://pypi.org/project/openairclim/
    :alt: Latest PyPI release
.. image:: https://img.shields.io/conda/vn/conda-forge/openairclim?label=conda-forge&logo=conda-forge&logoColor=white
    :target: https://anaconda.org/conda-forge/openairclim
    :alt: Latest conda-forge release

.. note::

    This guide describes how to set up OpenAirClim as a *developer*, i.e. if
    you are planning on contributing code or documentation. If you only want
    to use OpenAirClim, see the :doc:`user guide <../user_guide/installation>`
    instead - it is lighter-weight and does not require ``git``.


Cloning the repository
----------------------

Unlike a user installation, development requires the full repository,
including the test suite, lock files and CI configuration - none of which
are published to PyPI or conda-forge. Clone it from GitHub:

.. code-block:: bash

    cd path/to/working/dir
    git clone https://github.com/dlr-pa/oac.git
    cd oac

See :doc:`workflows` for the branching model and how to open a pull request.


Setting up your environment
---------------------------

OpenAirClim supports Python ``>=3.11`` (``requires-python`` in
``pyproject.toml``). Any supported Python version can be used for
development. ``pixi.lock`` pins one specific Python version, so that all
contributors and CI use the same one.

.. dropdown:: Installation with pixi (recommended)

    `pixi <https://pixi.sh>`__ installs the environments this repository
    defines from the committed ``pixi.lock``, for reproducible dependency
    versions across contributors and CI. There are two environments:
    ``default`` (the dependencies needed to run OpenAirClim, including the
    GUI) and ``dev`` (additionally including the test, documentation and
    linting tools). Packages are installed from conda-forge where available,
    and from PyPI otherwise. To install the ``dev`` environment, run:

    .. code-block:: bash

        pixi install -e dev

    Run commands inside the ``dev`` environment with ``pixi run -e dev``,
    e.g.:

    .. code-block:: bash

        pixi run -e dev test

    The ``-e dev`` flag (denoting the environment to use for a given
    command) is optional for tasks that are only defined in the ``dev``
    environment, such as ``test`` or ``docs-build`` - pixi selects that
    environment automatically.

    You can also activate the shell directly:

    .. code-block:: bash

        pixi shell -e dev

    ``pixi task list`` shows every task available (linting, building the
    docs, checking the packaged distribution, ...) - the sections below
    give the ones you will use most.


.. dropdown:: Installation with conda

    .. code-block:: bash

        conda env create -f environment_dev.yaml
        conda activate oac
        pip install -e .

    ``environment_dev.yaml`` is generated from the pixi ``dev`` environment
    and includes all test, documentation and linting dependencies.


.. dropdown:: Installation with venv

    The development dependencies are defined as
    `dependency groups <https://packaging.python.org/en/latest/specifications/dependency-groups/>`__
    in ``pyproject.toml``, which requires pip 25.1 or later. To create a
    virtual environment and install an editable version of OpenAirClim with
    the ``dev`` dependency group (test, documentation and linting tools),
    you will need a supported Python interpreter already installed:

    .. code-block:: bash

        python -m venv .venv
        source .venv/bin/activate  # for Windows: .venv\Scripts\activate
        pip install --upgrade pip
        pip install -e . --group dev

    The ``dev`` group includes the narrower ``test`` and ``docs`` groups, so
    install one of those instead if you only need part of it, e.g.
    ``pip install -e . --group test``. With
    `uv <https://docs.astral.sh/uv/>`__, ``uv sync`` creates the virtual
    environment and installs the ``dev`` group by default.


Creating test fixture data
--------------------------

The test suite relies on small, synthetic fixture files (emission
inventories, response surfaces, background concentrations, example config
files) rather than the real repository data described below. These are not
committed to the repository and must be generated once, into
``tests/core/repository/``:

.. code-block:: bash

    python -m openairclim.utils.create_test_files -o tests/core/repository/

With pixi, this happens automatically whenever you run the test suite (see
below), so you do not need to run it yourself unless you want the fixture
files without also running the tests:

.. code-block:: bash

    pixi run create-test-files

This is what ``tests/conftest.py``'s shared ``valid_config``/``working_dir``
fixtures resolve against. If you add code that requires new kinds of test
data, extend ``openairclim/utils/create_test_data.py`` (the underlying
dataset builders) and ``create_test_files.py`` (which writes them to disk)
rather than creating new fixtures elsewhere.


Running the test suite
----------------------

First, run ``create-test-files`` as described above, so that the test fixtures
are generated. Then use one of the following commands to run the tests.

With pixi:

.. code-block:: bash

    pixi run test

Without pixi:

.. code-block:: bash

    pytest tests/

Tests are named ``*_test.py`` and mirror the ``openairclim`` source tree 1:1,
e.g. ``tests/gui/tabs/scenario_test.py`` tests ``openairclim/gui/tabs/scenario.py``.


Downloading repository data
---------------------------

Unlike the test suite, actually *running* OpenAirClim (e.g. the bundled
example, the demo notebooks, or manual testing) requires the real response
surface and background concentration data. This works exactly as for users -
see :ref:`downloading-repository-data` - and is not required just to develop
or test code.


Building the documentation
--------------------------

The ``docs`` dependency group (included in the ``dev`` environment set up
with the ``pixi``/``conda``/``pip`` commands above, or on its own via
``pip install -e . --group docs``) installs
`Sphinx <https://www.sphinx-doc.org/>`__ and the theme/extensions this site
uses. Build it locally with:

.. code-block:: bash

    pixi run docs-build

    # or, without pixi:
    sphinx-build -M html docs/source docs/build

To remove a previous build (e.g. to force a full rebuild):

.. code-block:: bash

    pixi run docs-clean

    # or, without pixi:
    sphinx-build -M clean docs/source docs/build

Open ``docs/build/html/index.html`` in a browser to view it. The
demonstration pages under ``docs/source/demos`` are reStructuredText files
with ``jupyter-execute`` blocks. They are executed during the build, so the
figures and dataset outputs appear on the page, and a failing path or a
renamed output file breaks the build. They are not covered by ``pytest``.
The build writes ``results/`` and the generated ``rnd_inv_*.nc`` files into
the demo folders.


.. _running-code-quality-checks:

Running code quality checks
---------------------------

Pull requests are checked with `ruff <https://docs.astral.sh/ruff/>`__ and
`mypy <https://mypy.readthedocs.io/en/stable/>`__. With ``pixi``, run them
against the whole ``openairclim`` and ``tests`` trees:

.. code-block:: bash

    # checks
    pixi run correctness  # mypy & pyflakes (from ruff "F")
    pixi run style        # ruff check and format

    # fixes
    pixi run lint-fix     # auto-fixes ruff errors where possible
    pixi run format       # auto-formats

The ``openairclim/gui/`` and ``tests/gui/`` folders are not checked, because
the code in those folders is not yet clean. This applies to the ``lint.yml``
workflow too: changes to files in those folders are not linted on pull
requests. If you are stuck with any linting issues, please talk to the core
dev team - we're happy to help!

The ``lint.yml`` workflow runs these same checks in the same locked ``dev``
environment, but only against the files changed in a pull request. To do the
same locally, pass a space-separated list of files to the correctness tasks:

.. code-block:: bash

    pixi run corr-mypy "<changed files>"
    pixi run corr-ruff "<changed files>"

You can also run the underlying tools directly (without pixi):

.. code-block:: bash

    # required check (correctness)
    mypy <changed files>
    ruff check --select F <changed files>

    # non-required check (style)
    ruff check --ignore F <changed files>
    ruff format --check <changed files>

See the pixi setup in ``pyproject.toml`` for the complete set of commands used
by the core dev team. The ``ruff`` and ``mypy`` configurations can also be
found there.
