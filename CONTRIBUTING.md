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

OpenAirClim is hosted and shared on GitHub, using both _protected_ (`main`,
`dev`) and _unprotected_ branches. See
[Working with GitHub](https://openairclim.org/dev_guide/workflows) in the
developer guide for the branching model, branch naming conventions, and how
forks and write access work.

Possible contributors should contact the
[OpenAirClim team](mailto:openairclim@dlr.de). Please note that we cannot
guarantee that pull requests are considered or accepted if the work was not
discussed with the Steering Committee.

---

## :dart: Releases and Versioning

Software releases of OpenAirClim follow the
[Semantic Versioning](https://semver.org) numbering scheme with three digits
denoting MAJOR, MINOR and PATCH changes. Decisions about backward
incompatibilities have to be taken within the OpenAirClim organisation.
Please address questions about backward compatibility and release scheduling
to the Scientific and Technical Boards.

See [Releasing](https://openairclim.org/dev_guide/releasing) in the
developer guide for exactly how a release - of the software, and separately
of the repository data published in
[dlr-pa/oac-data](https://github.com/dlr-pa/oac-data) - gets published, and
what access it requires.

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

See [Installation](https://openairclim.org/dev_guide/installation) in the
developer guide for cloning the repository and setting up a development
environment with conda, venv or pixi, generating test fixture data,
running the test suite, building the documentation, and running code
quality checks locally.

### General considerations

- **The default git branch is `main`**. Unless otherwise discussed with the
    core development team, use `main` to fork the repository or create a new
    feature branch and make a pull request against.
- **Smaller is better**. Submit **one** pull request per bug fix or feature. It
    is better to submit many small pull requests than a single large one, which
    would take a very large time to review. **Do not refactor or reformat code
    unrelated to your change.**
- **Coordinate bigger changes**. For large and non-trivial changes, use an
    issue to discuss a strategy with the maintainers. This is particularly
    important if your pull request is related to other open issues.
- **Prioritise understanding**. Write code clearly and concisely, but remember
    that source code usually gets written once and read often. Therefore,
    ensure that your code is clear to the reader. Use in-line comments where
    necessary.
- **Don't hand-edit `CHANGELOG.md`**. It is generated automatically from
    merged pull requests when a release is prepared (see
    [Releases and Versioning](#dart-releases-and-versioning) above) - just
    make sure your pull request has a clear title and the right label(s).
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

For Python coding, [PEP8](https://peps.python.org/pep-0008/) is our gold
standard, formatted with [Black](https://pypi.org/project/black/) and
checked with [Prospector](https://prospector.landscape.ai/) (pylint, mypy,
pydocstyle and pyroma). See
[Guidelines](https://openairclim.org/dev_guide/guidelines) for our full
conventions, and
[Workflows](https://openairclim.org/dev_guide/workflows) for exactly what
runs automatically on a pull request.

### Documentation

All code should be **well documented**, using
[Google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google)
that cross-reference other modules with Sphinx roles rather than plain text.
See [Guidelines](https://openairclim.org/dev_guide/guidelines) for the full
conventions, and
[Installation](https://openairclim.org/dev_guide/installation) for building
the documentation locally.

### Software testing

New code should be accompanied by automated pytest test functionality. See
[Guidelines](https://openairclim.org/dev_guide/guidelines) for our testing
conventions, and
[Installation](https://openairclim.org/dev_guide/installation) for
generating test fixture data and running the suite. Do not hesitate to
contact the Technical Board for assistance.

### Dependencies

Before considering the introduction of a new dependency, ensure that the
licence of the dependency and any of its dependencies are compatible with
the Apache 2.0 licence that applies to OpenAirClim. Remember that all
contributions to OpenAirClim will be licenced under the project's licence.
See [Guidelines](https://openairclim.org/dev_guide/guidelines) for the full
policy on managing dependencies.

### Backwards compatibility

Balancing a good user experience with fast development is not trivial. We
strive to maintain backward compatibility where possible, but note that this
cannot always be guaranteed. If your code is not backwards compatible or
introduces a breaking change, this must be clearly indicated in the pull
request.

### Automatic checks

A number of automatic checks run on GitHub Actions when you open a (draft)
pull request. See [Workflows](https://openairclim.org/dev_guide/workflows)
for exactly which checks run, when, and which of them are required. If a
check is broken for reasons unrelated to your pull request, check whether
there's already an open issue about it and otherwise create one - it will
need to be fixed in a separate pull request before yours can be merged.

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

- Use short but self-explanatory variable names (e.g. `model_input` rather than
    `xm`)
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
