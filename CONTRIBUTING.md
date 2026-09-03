# How to contribute to OpenAirClim

OpenAirClim is a joint, open-source effort. We greatly value contributions of
any kind! However, please familiarise yourself with the process described in
this document before submitting issues and pull requests. This ensures that the
development proceeded as smoothly and quickly as possible.

We value the time you invest in contributing and strive to make this process as
easy as possible. Therefore, if you have ideas on how to make this process
smoother, please do let us know for example through the
[discussions](https://github.com/dlr-pa/oac/discussions) tab on GitHub.

Also, if you are interested in becoming part of the core development team, feel
free to reach out.

---

## :classical_building: OpenAirClim Governance

OpenAirClim's governance structure - the Steering Committee, the Scientific
and Technical Boards, and the organisations making up the core development
group - is described on the
[Governance](https://openairclim.org/governance.html) page of the
documentation.


## :book: Code of Conduct

Please review our
[Code of Conduct](https://github.com/dlr-pa/oac/blob/main/CODE_OF_CONDUCT.md).
It is in effect at all times. We expect it to be honoured by everyone who
contributes to this project. 


## :bulb: Asking Questions

Use the [discussions](https://github.com/dlr-pa/oac/discussions) tab on GitHub
for general questions or requests for help with your specific project. GitHub
issues are reserved for bug reporting and feature requests.

---

## :inbox_tray: Opening an Issue

Before
[creating an issue](https://help.github.com/en/github/managing-your-work-on-github/creating-an-issue),
check if you are using the latest version of the project. If you are not
up-to-date, see if updating fixes your issue first. We differentiate between
two issue types: bug reports and feature requests.

### Did you find a bug?

- **Do not open a GitHub issue if the bug relates to a security vulnerability.**
    Please review our
    [Security Policy](https://github.com/dlr-pa/oac/blob/main/SECURITY.md).
- **Ensure that a related issue has not already been reported** on GitHub. If
    it has, use this issue to add comments or further details. Please refrain
    from adding "+1" to issues - use reactions instead.
- If you cannot identify an open issue addressing your bug, please open a new
    issue. Use our `bug_report.md` template (GitHub should automatically prompt
    you to do so) and answer the questions there. Be sure to include a title
    and clear description, as much relevant information as possible and ideally
    a code sample or test case demonstrating the current and expected
    behaviour. Use
    [GitHub-flavoured Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
    for code blocks and console outputs.

### Do you have an idea for a new feature?

- Feature requests are always welcome. Since we are a small team, we cannot
    guarantee that your request will be accepted and we cannot make any
    commitments regarding the timeline for implementation and release. If you
    are able to help out and develop (part of) the feature yourself, please let
    us know in the request.
- **Do not open a duplicate feature request**. Search for existing feature
    requests first. If you find your requested feature, or one very similar,
    please use this issue.
- If you cannot find your feature request in the open issues, please open a new
    issue. Use our `feature_request.md` template (GitHub should automatically
    prompt you to do so) and answer the questions there.

---

## :flight_departure: Working with the Repository

OpenAirClim is hosted and shared on GitHub. The development of the software
follows the branching model shown below and uses both _protected_ and
_unprotected_ branches.

![branches_oac](https://github.com/user-attachments/assets/47030a9b-f4dd-4350-a4f1-32836f8492bf)

Protected branches include `main` and `dev` and can only be modified using Pull
Requests (see
[Contributing Code and Documentation](#hammer_and_wrench-contributing-code-and-documentation)
). New OpenAirClim versions are released on `main`.

Software development is carried out on the unprotected branches as well as on
local repositories and forks. The branches are labeled `feature/<feature-name>`
for features, `bug/<bug-fix-name>` for bug fixes, `task/<task-name>` for
specific tasks and `docs/<doc-name>` for changes in documentation. Unprotected
branches can be created by developers who have at least write persmissions for
the repository. These developers can be part of the core development team or
external collaborators. Before write permissions are granted, the Steering
Committee will ensure that the planned work is in line with the overall short-
and long-term project planning.

Possible contributors should contact the
[OpenAirClim team](mailto:openairclim@dlr.de). Please note that we cannot
guarantee that pull requests are considered or accepted if the work was not
discussed with the Steering Committee.

---

## :dart: Releases and Versioning

Software releases of OpenAirClim follow the
[Semantic Versioning](https://semver.org) numbering scheme with three digits
denoting MAJOR, MINOR and PATCH changes.  There are two competing needs which
have to be balanced: The occasional need for improvements or maintenance which
breaks backward compatibility and the need for stability for existing users and
developers. Decisions about backward incompatibilities have to be taken within
the OpenAirClim organisation. Please address questions about backward
compatibility and release scheduling to the Scientific and Technical Boards. 

### Repository data releases

The response surfaces and background concentration files ("repository data")
are published in a separate repository
[dlr-pa/oac-data](https://github.com/dlr-pa/oac-data), with its own versioning
that is not tied to `openairclim`'s release numbering. **Only if** your change
updates or adds repository data:

1. Publish a new release on `dlr-pa/oac-data` (its own version scheme, e.g.
    `0.4.0`).
2. If you want `openairclim` to pick up that new release by default, bump
    `DEFAULT_REPOSITORY_DATA_VERSION` in `openairclim/repository.py` in a
    pull request in the regular `dlr-pa/oac` repository.

Please note: This is **not** a required step on every `openairclim` release. It
is only required when the repository data has actually changed and it makes
sense to adopt the new version. Users can always fetch a specific version
explicitly via `oac-download-data --version`/`--record`, regardless of what's
pinned by default.

### Publishing a release

*(TODO: document the automated GitHub release workflow.)*

---

## :hammer_and_wrench: Contributing Code and Documentation

You'd like to help us add a new feature, add documentation or fix a bug? That's
amazing, you're the best! :heart:

Before starting work, please carefully read this section. Please also open an
issue to discuss your proposed changes before
[forking the repo](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)
and
[creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests). 

_Note: All contributions will be licenced under the project's [licence](https://github.com/dlr-pa/oac/blob/main/LICENSE)._

### Development environment

Set up a development environment with conda:

```bash
conda env create -f environment_dev.yaml   # or environment_minimal.yaml
conda env update -f environment_gui.yaml -n <env>   # optional: add GUI deps
```

or with pip:

```bash
pip install -e ".[dev]"
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

`environment_dev.yaml` pins `python<3.14`: Prospector's mypy integration
crashes outright under Python 3.14 (an upstream Prospector/mypy/argparse
incompatibility, unrelated to this codebase). If setting up a dev environment
via `pip install -e ".[dev]"` instead of conda, use a 3.11-3.13 interpreter for
the same reason - `pip` won't manage/select this for you. This pin is
dev-tooling-only, not a statement about which Python versions OpenAirClim
itself supports (see `requires-python` in `pyproject.toml`). `uv sync` handles
this automatically via the committed `.python-version` file, and builds a
`.venv` from the committed `uv.lock` for reproducible dependency versions -
run commands inside it with `uv run` (e.g. `uv run pytest tests/`).

### General considerations

- **The default git branch is `main`.** Unless otherwise discussed with the
    core development team, use `main` to fork the repository or create a new
    feature branch and make a pull request against.
- **Smaller is better.** Submit **one** pull request per bug fix or feature. It
    is better to submit many small pull requests than a single large one, which
    would take a very large time to review. **Do not refactor or reformat code
    unrelated to your change.**
- **Coordinate bigger changes**. For large and non-trivial changes, use an
    issue to discuss a strategy with the maintainers. This is particularly
    important if your pull request is related to other open issues.
- **Prioritise understanding.** Write code clearly and concisely, but remember
    that source code usually gets written once and read often. Therefore,
    ensure that your code is clear to the reader. Use in-line comments where
    necessary.
- **Update input files.** If your new code require changes to the input files
    (e.g. the example `config` file) or to the response surfaces, please make
    sure to also update these. If your new code introduces new input files,
    please also extend the `openairclim/utils` scripts to generate example
    input files for debugging and testing.
- **Update the CHANGELOG** for all enhancements and bug fixes. Include the
    corresponding issue number and your GitHub username. Example: "Fixed error
    in scaling methodology. #123 @liammegill"
- **Use the pull request template** available on GitHub and ensure you have
    completed the checklist.

### Scientific Relevance

New features and changes to the software should be relevant to the larger
scientific community. The implementations should be scientifically sound and
the used formulae should be checked for corectness, e.g. by a scientific
review. Wherever possible and meaningful, stick to
[CF conventions](https://cfconventions.org/). Please approach the Scientific
Board with any issues or questions related to scientific relevance.

### Code Quality

In order to ensure readability, maintainability and a sustainable development
of OpenAirClim, best practices and coding standards are a crucial part of our
software development.  For Python coding,
[PEP8](https://peps.python.org/pep-0008/) is our gold standard. We recommend
the use of an automatic code formatter such as
[Black](https://pypi.org/project/black/), which can also be used with your
choice of IDE. Pull requests are also checked with
[Prospector](https://prospector.landscape.ai/) (wrapping pylint, mypy,
pydocstyle and pyroma) - see
[Development environment](#development-environment) above for the Python
version this requires.

### Documentation

All code should be **well documented**. In order to document python functions,
classes and modules, we use
[Google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google)
and cross-reference other modules with Sphinx roles (e.g.
`` :func:`~pkg.mod.func` ``), since docstrings render directly into the Sphinx
docs. From these docstrings, documentation in HTML format can be
[generated automatically](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).

The documentation source is found in the `docs` directory and is built with
[Sphinx](https://www.sphinx-doc.org/). The `docs` extra is included by default
in `environment_dev.yaml`, or install it directly with `pip install ".[docs]"`.
To build the documentation locally, run from within `docs`:

```bash
sphinx-build -b html source build/html
```

The demonstration notebooks under `docs/source/demos` are MyST Markdown
notebooks (`file_format: mystnb`), executed by `myst_nb` and not covered by
`pytest` - verify any change touching file resolution in those notebooks with
an actual docs build, not just the test suite. Execution only happens when a
notebook's content actually changes (`nb_execution_mode = "cache"`); an
unrelated docs change reuses the cached output instead of re-running
everything (and, for `03_multi_inv`, re-downloading its input data).

### Software testing

New code should be accompanied by automated pytest test functionality. Tests
should be placed in the `tests` directory, which mirrors the `openairclim`
source tree 1:1 (e.g. `tests/gui/tabs/scenario_test.py` tests
`openairclim/gui/tabs/scenario.py`). Test files are named `*_test.py` (not
`test_*.py`) and use class-based `TestXxx`/`test_yyy` grouping, one class per
function under test. Run the full suite with:

```bash
pytest tests/
```

`tests/conftest.py` holds a shared `valid_config`/`working_dir` fixture pair,
backed by real fixture files in `tests/core/repository/` - reuse it rather
than hand-rolling another valid config dict. Those fixture files (and other
test data) can be regenerated with the dev-only scripts
`openairclim/utils/create_test_data.py` (a library of dataset builders used
directly in test code) and `create_test_files.py` (writes fixture files to
disk, e.g. `python openairclim/utils/create_test_files.py -o
tests/core/repository/`); neither is installed as a console-script.

Do not hesitate to contact the Technical Board for assistance.

### Dependencies

Before considering the introduction of a new dependency, ensure that the
licence of the dependence and any of its dependencies are compatible with the
Apache 2.0 licence that applies to OpenAirClim. Remember that all contributions
to OpenAirClim will be licenced under the project's licence. When adding or
removing dependencies, ensure that the corresponding files describing these
dependencies are updated, i.e. `pyproject.toml` and the `environment.yaml`
files. Note that `environment_dev.yaml` pins `python<3.14` for
Prospector/mypy's benefit; it is a dev-tooling environment, not the set of
Python versions OpenAirClim itself supports (see `requires-python` in
`pyproject.toml`), so CI's conda install test builds its test environment from
`environment_minimal.yaml` + `environment_gui.yaml` instead, to keep testing
the full supported Python range.

### Backwards compatibility

Balancing a good user experience with fast development is not trivial. We
strive to maintain backward compatibility where possible, but note that this
cannot always be guaranteed. If your code is not backwards compatible or
introduces a breaking change, this must be clearly indicated in the pull
request.

### Automatic checks

To check that a pull request is up to standard, a number of automatic checks
are run by GitHub Actions when you make a (draft) pull request. A ✅ means the
checks were successful; a ❌ that the checks were unsuccessful. If the checks
are broken because of something unrelated to the current pull request, please
check if there is an open issue on this problem and otherwise create one. This
problem will have to be resolved in a separate pull request before the current
pull request can be merged.

### List of authors

If you contribute to OpenAirClim and would like to be listed as an author
(e.g. on Zenodo), please add your name to the list of authors in
`CITATION.cff`.

---

## :memo: Commit Messages

Please [write a great commit message](https://chris.beams.io/posts/git-commit/). 

1. Separate the subject from the body with a blank line
2. Limit the subject line to 50 characters
3. Capitalise the first character of the subject line (e.g. "Fix xxx" rather than "fix xxx")
4. Do not end the subject line with a full stop
5. Use the imperative tense (not past tense) in the subject line (e.g. "Fix xxx", "Add yyy" rather than "Fixed xxx" or "Added yyy")
6. Wrap the body at 72 characters
7. Use the body to explain why, not what and how

---

## :medal_sports: Certificate of Origin

*Developer's Certificate of Origin 1.1*

By making a contribution to this project, I certify that:

> 1. The contribution was created in whole or in part by me and I have the
    right to submit it under the open source license indicated in the file; or
> 2. The contribution is based upon previous work that, to the best of my
    knowledge, is covered under an appropriate open source license and I have
    the right under that license to submit that work with modifications,
    whether created in whole or in part by me, under the same open source
    license (unless I am permitted to submit under a different license), as
    indicated in the file; or
> 3. The contribution was provided directly to me by some other person who
    certified (1), (2) or (3) and I have not modified it.
> 4. I understand and agree that this project and the contribution are public
    and that a record of the contribution (including all personal information I
    submit with it, including my sign-off) is maintained indefinitely and may
    be redistributed consistent with this project or the open source license(s)
    involved.

---

## :sun_with_face: Do-s and Don't-s

Finally, some do-s and don't-s.

**Do:**
- Before starting work, open a GitHub issue to discuss what you are going to do.
- Create and use a new branch for each new development.
- Comment your code clearly in English using in-line comments and docstrings
- Use short but self-explanatory variable names (e.g. `model_input` rather than
    `xm`)
- Consider using a modular/functional programming style.
- Consider reusing or extending existing code.
- Synchronise your development branch with the protected branch(es) regularly
    in order to integrate the latest updates into your work. 
- Have fun!

**Don't:**
- Use other programming languages than the ones currently supported: Python.
- Develop without proper version control.
- Use large (memory, disk space) intermediate results
- Use hard-coded pathnames or filenames.

---

## Thanks! :heart:
The OpenAirClim Team
