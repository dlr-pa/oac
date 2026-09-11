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

We recommend Python 3.13 for development. Development tooling (`Black
<https://pypi.org/project/black/>`__, `Prospector
<https://prospector.landscape.ai/>`__, which wraps
`pylint <https://pylint.readthedocs.io/en/stable/>`__,
`mypy <https://mypy.readthedocs.io/en/stable/>`__
and `pyroma <https://pypi.org/project/pyroma/>`__) is pinned to Python
``<3.14`` because Prospector's mypy integration currently crashes outright for
3.14 (an upstream incompatibility unrelated to OpenAirClim). This pin is thus
developer-tooling-only and doesn't affect which Python versions are supported
by OpenAirClim (defined by ``requires-python`` in ``pyproject.toml``). If you
set up your environment with ``pip`` rather than ``conda`` or ``pixi``, you
are responsible for picking a 3.11-3.13 interpreter yourself for the same
reason. Note that we are considering moving towards
`ruff <https://docs.astral.sh/ruff/>`__ (see
`#149 <https://github.com/dlr-pa/oac/issues/149>`__), which would solve this
problem.


.. dropdown:: Installation with pixi (recommended)

    `pixi <https://pixi.sh>`__ installs every environment this repository
    defines (``default``, ``docs`` and ``dev``) from the committed
    ``pixi.lock``, for reproducible dependency versions across contributors
    and CI - conda-forge for compiled/system packages, PyPI for everything
    else. It also picks the pinned Python version automatically, so you do
    not need to worry about the note above:

    .. code-block:: bash

        pixi install --all

    Run commands inside the ``dev`` environment with ``pixi run -e dev``,
    e.g.:

    .. code-block:: bash

        pixi run -e dev test

    Alternatively, activate it directly:

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

    ``environment_dev.yaml`` includes the ``gui``, ``docs`` and ``test``
    extras, so you do not need to update the environment with the other yaml
    files.


.. dropdown:: Installation with venv

    To create a virtual environment and install an editable version of
    OpenAirClim with all development extras (``gui``, ``docs``, ``test`` and
    linting tools), you will need a 3.11-3.13 interpreter already installed
    (see the note above):

    .. code-block:: bash

        python3.13 -m venv .venv
        source .venv/bin/activate  # for Windows: .venv\Scripts\activate
        pip install --upgrade pip
        pip install -e ".[dev]"

    The ``dev`` extra pulls in the narrower ``gui``, ``docs`` and ``test``
    extras too, so install one of those instead if you only need part of it,
    e.g. ``pip install -e ".[test]"``.


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

    pixi run -e dev create-test-files

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

    pixi run -e dev test

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

The ``docs`` extra (included by default with the ``pixi``/``conda`` commands
above, or via ``pip install -e ".[docs]"``) installs
`Sphinx <https://www.sphinx-doc.org/>`__ and the theme/extensions this site
uses. Build it locally with:

.. code-block:: bash

    pixi run -e docs docs-build

    # or, without pixi:
    sphinx-build -M html docs/source docs/build

To remove a previous build (e.g. to force a full rebuild):

.. code-block:: bash

    pixi run -e docs docs-clean

    # or, without pixi:
    sphinx-build -M clean docs/source docs/build

Open ``docs/build/html/index.html`` in a browser to view it. The
demonstration notebooks under ``docs/source/demos`` are MyST Markdown
notebooks executed by ``myst_nb``, not covered by ``pytest`` - verify any
change touching file resolution in those notebooks with an actual docs
build, not just the test suite. Execution is cached and only reruns when a
notebook's content changes.


.. _running-code-quality-checks:

Running code quality checks
---------------------------

Pull requests are checked with Black and Prospector. With pixi, run them
against the whole ``openairclim``/``tests`` trees:

.. code-block:: bash

    pixi run -e dev style        # black --check + full prospector report
    pixi run -e dev correctness  # faster pyflakes+mypy subset (the required check)

    pixi run -e dev black         # reformats in place, not just --check

Note the pixi tasks always lint the full ``openairclim``/``tests`` trees.

You can also run the underlying tools directly (without pixi), e.g. against
just your changed files:

.. code-block:: bash

    black --check --diff <changed files>
    prospector <changed files>

    # or, the faster subset which is part of the lint.yml workflow
    prospector --tool pyflakes --tool mypy -s medium <changed files>
