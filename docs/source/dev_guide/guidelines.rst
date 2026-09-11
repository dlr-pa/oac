Guidelines
==========

This page covers the conventions contributions are expected to follow: code
style, documentation, testing, and managing dependencies. For setting up a
development environment, see :doc:`installation`; for how contributions flow
through GitHub, see :doc:`workflows`.


Code style
----------

`PEP8 <https://peps.python.org/pep-0008/>`__ is our gold standard for Python
code, currently formatted with `Black <https://pypi.org/project/black/>`__ and
checked with `Prospector <https://prospector.landscape.ai/>`__.
:doc:`workflows` shows which correctness and style checks are run automatically
on a pull request. See also :ref:`running-code-quality-checks` in
:doc:`installation` for how to run the same checks locally before you open one.

Beyond tooling: source code is written once and read often, so prioritise
clarity for the reader over cleverness or brevity. Use in-line comments
where the *why* isn't obvious from the code itself. Prefer a modular,
functional style, and reuse or extend existing code rather than
reimplementing it.


Documentation conventions
-------------------------

All code should be **well documented**. Document Python functions, classes
and modules with `Google style docstrings
<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google>`__,
and cross-reference other modules with Sphinx roles (e.g.
``:func:`~pkg.mod.func``` ) rather than plain text. Docstrings render
directly into this site via ``autodoc`` and ``napoleon``, so a working
cross-reference here is also a working link there.

See :doc:`installation` for building this documentation site locally to
check your changes, including the caveat around the demo notebooks under
``docs/source/demos``.


Testing conventions
-------------------

New code should be accompanied by automated ``pytest`` test functionality.
Tests live in ``tests/``, which mirrors the ``openairclim`` source tree 1:1
(e.g. ``tests/gui/tabs/scenario_test.py`` tests
``openairclim/gui/tabs/scenario.py``). Test files are named ``*_test.py``
(not ``test_*.py``) and use class-based ``TestXxx``/``test_yyy`` grouping,
one class per function being tested.

``tests/conftest.py`` holds a shared ``valid_config``/``working_dir``
fixture pair, backed by the fixture files in ``tests/core/repository/``. Please
reuse it rather than creating for example other valid config dicts. If you add
code that requires new kinds of test data, extend
``openairclim/utils/create_test_data.py`` (the underlying dataset builders)
and ``create_test_files.py`` (which writes them to disk). If your change
touches the example config or response surfaces, update those too (or, if it
introduces a new kind of input file, extend the relevant ``openairclim/utils``
script to generate example input files for debugging and testing).

Do not hesitate to `contact <mailto:openairclim@dlr.de>`__ the Technical Board
for assistance with tricky test cases.


Managing dependencies
---------------------

Before introducing a new dependency, check that its licence (and those of
its own dependencies) is compatible with the Apache 2.0 licence that applies
to OpenAirClim. To add, remove or update a dependency:

- Update the PyPI dependency in ``[project.dependencies]`` (for the minimal
  installation) or in ``[project.optional-dependencies]`` (for an extra
  installation);
- Update the conda-forge dependency in ``[tool.pixi.dependencies]`` (for the
  minimal ("default") environment) or in one of the feature environments (e.g.
  ``[tool.pixi.feature.docs.dependencies]``);
- In a bash shell, run ``pixi run export-envs`` or directly
  ``bash scripts/export-envs.sh`` to update the conda environment YAML files
- Update ``pixi.lock`` by running ``pixi lock``.
- For a release: note that you will have to update the `openairclim-feedstock
  <https://github.com/conda-forge/openairclim-feedstock>`__ as well - see
  :doc:`releasing`.
