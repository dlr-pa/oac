# How to contribute to OpenAirClim

OpenAirClim is a joint, open-source effort. We greatly value contributions of any kind.
Feel free to make use of OpenAirClim, fork the repository, submit issues and contribute through pull requests!
We value the time you invest in contributing and strive to make this process as easy as possible.
Therefore, if you have ideas on how to make this process smoother, please do let us know for example through the [discussions](https://github.com/dlr-pa/oac/discussions) tab on GitHub.

Also, if you are interested in becoming part of the core development team, feel free to reach out.

## :classical_building: OpenAirClim Governance

The development of OpenAirClim is overseen by a Steering Committee, which meets every 6 months to define high-level roadmaps, define the core development team, decide on partners and agree on governance procedures.
Contributions to the OpenAirClim code base are overseen by the Scientific and Technical Boards.
The Scientific Board is responsible for deciding on model extensions, new processes and verification approaches.
The Technical Board decides on versioning, releases, data structures and code styles.
Engagement from users and the industry is obtained at regular user meetings and workshops as well as through discussions on GitHub.

## :book: Code of Conduct

Please review our [Code of Conduct](https://github.com/dlr-pa/oac/blob/main/CODE_OF_CONDUCT.md).
It is in effect at all times.
We expect it to be honoured by everyone who contributes to this project. 

## :bulb: Asking Questions

Use the [discussions](https://github.com/dlr-pa/oac/discussions) tab on GitHub for general questions or requests for help with your specific project.
GitHub issues are reserved for bug reporting and feature requests.

## :inbox_tray: Opening an Issue

Before [creating an issue](https://help.github.com/en/github/managing-your-work-on-github/creating-an-issue), check if you are using the latest version of the project.
If you are not up-to-date, see if updating fixes your issue first.
We differentiate between two issue types: bug reports and feature requests.

### Did you find a bug?

- **Do not open a GitHub issue if the bug relates to a security vulnerability.** Please review our [Security Policy](https://github.com/dlr-pa/oac/blob/main/SECURITY.md).
- **Ensure that a related issue has not already been reported** on GitHub. If it has, use this issue to add comments or further details. Please refrain from adding "+1" to issues - use reactions instead.
- If you cannot identify an open issue addressing your bug, please open a new issue. Use our `bug_report.md` template (GitHub should automatically prompt you to do so) and answer the questions there. Be sure to include a title and clear description, as much relevant information as possible and ideally a code sample or test case demonstrating the current and expected behaviour. Use [GitHub-flavoured Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) for code blocks and console outputs.

### Do you have an idea for a new feature?

- Feature requests are always welcome. Since we are a small team, we cannot guarantee that your request will be accepted and we cannot make any commitments regarding the timeline for implementation and release. If you are able to help out and develop (part of) the feature yourself, please let us know in the request.
- **Do not open a duplicate feature request**. Search for existing feature requests first. If you find your requested feature, or one very similar, please use this issue.
- If you cannot find your feature request in the open issues, please open a new issue. Use our `feature_request.md` template (GitHub should automatically prompt you to do so) and answer the questions there.

## :repeat: Pull Requests

You've written some code to add a new feature or fix a bug?
That's amazing, you're the best! :heart:

Before [forking the repo](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) and [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests), please first open an issue to discuss your proposed changes
The default git branch is `main` - use this branch to fork the repository or create a new feature branch and make a pull request against.

To check that a pull request is up to standard, a number of automatic checks are run when you make a (draft) pull request.
A ✅ means the checks were successful; a ❌ that the checks were unsuccessful.
If the checks are broken because of something unrelated to the current pull request, please check if there is an open issue on this problem and otherwise create one.
This problem will have to be resolved in a separate pull request before the current pull request can be merged.

_Note: All contributions will be licenced under the project's [licence](https://github.com/dlr-pa/oac/blob/main/LICENSE)._

- **Smaller is better.** Submit **one** pull request per bug fix or feature. It is better to submit many small pull requests than a single large one, which would take a very large time to review. **Do not refactor or reformat code unrelated to your change.**
- **Coordinate bigger changes**. For large and non-trivial changes, use an issue to discuss a strategy with the maintainers. This is particularly important if your pull request is related to other open issues.
- **Prioritise understanding.** Write code clearly and concisely, but remember that source code usually gets written once and read often. Therefore, ensure that your code is clear to the reader. Use in-line comments where necessary.
- **Follow coding style and conventions.** Keep your code consistent with the style, formatting and conventions present in the rest of the code base.
- **Include test coverage.** Add unit tests where possible and stay consistent with existing tests. 
- **Update example files.** If your new code require changes to the input files (e.g. the example `config` file) or to the response surfaces, please make sure to also update these.
- **Add/update documentation.** 
- **Update the CHANGELOG** for all enhancements and bug fixes. Include the corresponding issue number and your GitHub username. Example: "Fixed error in scaling methodology. #123 @liammegill"
- **Use the pull request template** available on GitHub and ensure you have completed the checklist.

## :memo: Commit Messages

Please [write a great commit message](https://chris.beams.io/posts/git-commit/). 

1. Separate the subject from the body with a blank line
2. Limit the subject line to 50 characters
3. Capitalise the subject line
4. Do not end the subject line with a full stop
5. Use the imperative tense (not past tense) in the subject line (e.g. "Fix xxx", "Add yyy" rather than "Fixed xxx" or "Added yyy")
6. Wrap the body at 72 characters
7. Use the body to explain why, not what and how


## :medal_sports: Certificate of Origin

*Developer's Certificate of Origin 1.1*

By making a contribution to this project, I certify that:

> 1. The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
> 2. The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
> 3. The contribution was provided directly to me by some other person who certified (1), (2) or (3) and I have not modified it.
> 4. I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.


## Thanks! :heart:
The OpenAirClim Team
