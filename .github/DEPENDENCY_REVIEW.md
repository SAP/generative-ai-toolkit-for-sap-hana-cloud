# Dependency snapshots and review

The [workflow](workflows/dependency-review.yml) submits dependency inventories for
the base and head commits of a pull request, waits for GitHub to index them,
and runs GitHub's official actions/dependency-review-action.

## Enable it for PR #167

Commit the workflow, its [Python helper](scripts/dependency_review.py),
[helper tests](scripts/test_dependency_review.py), and this guide to
fix/security-dependency-upgrade, then push that branch. The existing PR
updates automatically and starts the workflow; merging first is unnecessary.

No personal access token or HANA/model credentials are needed. The workflow uses
the automatically supplied GITHUB_TOKEN. The two snapshot jobs request
contents: write, which the dependency submission API requires. The review job
uses only contents: read. Repository or organization policy must permit those
permissions and the official review action.

The workflow supports branches in this repository, including PR #167.
It fails explicitly for fork PRs, whose tokens cannot submit these snapshots.
It uses the pull_request event, never pull_request_target.

## What to check in GitHub

Open the pull request's **Checks** tab and inspect:

1. **Dependency snapshot (base)** and **Dependency snapshot (head)**: both must
   succeed and report four resolved manifests and an accepted snapshot ID.
2. **Official Dependency Review**: the completeness step must finish with no
   snapshot warnings; the official action then checks newly introduced vulnerable
   dependencies, including development dependencies and low severity.
3. For this remediation, inspect the dependency changes for pylint.in and the
   packages behind alerts #55–60. A green job alone is not proof that those
   individual alerts were closed.

Pushing another commit runs the checks again. For the same commit, use
**Re-run all jobs** on this workflow's Actions run.

## What is submitted

On Ubuntu with Python 3.11, the helper explicitly resolves:

- requirements.txt
- .github/requirements/pylint.in, including its referenced requirements
- .github/requirements/pylint-py311-linux.txt
- .github/requirements/release-build-py311-linux.txt

It uses pip's dry-run, ignore-installed, and report options to capture all
resolved package versions without installing them. Pip may build metadata for
source-only dependencies such as pydotplus. It verifies hashes for the two lock
files. Resolution failure,
missing files, an empty report, or a rejected submission fails the job.
Inventories retain original manifest paths and direct/indirect package
relationships; they do not reconstruct dependency edges.

The submission API returns SUCCESS when it also updates the repository graph,
or ACCEPTED when it creates a snapshot without updating that graph (for example,
for a PR branch). Both are successful submissions; INVALID and unknown results
fail the job. The separate comparison step waits for GitHub to index snapshots.

The helper submits inventories through GitHub's official dependency submission
API. It does not decide whether a package is vulnerable; GitHub does that.
Both revisions use the same detector name and correlator so they can be compared.
The base snapshot is associated with the base branch and commit and can refresh
the dependency inventory for that commit.

The generic Component Detection action was not selected because its current
PipReport detector automatically looks for setup.py and requirements.txt;
that alone does not explicitly cover pylint.in and the named lock files.

## Meaning of a passing check

The snapshots were submitted and GitHub found no newly introduced vulnerable
dependencies in the comparison. This does not mean the default branch's alerts
are already closed, or that unchanged dependencies have no vulnerabilities.
Alert closure still requires default-branch reanalysis.

Coverage is limited to the four listed manifests on Python 3.11/Linux. The
workflow does not separately resolve Python 3.10, published runtime ranges,
or optional all dependencies. Existing dnspython/Python 3.10 and optional SAP SDK
findings remain separate follow-up work.

The official review action can continue after its own snapshot retry timeout.
Our preceding completeness check instead fails if GitHub still reports missing
snapshots, preventing a partial comparison from silently passing.

## References

- [Dependency submission API](https://docs.github.com/en/rest/dependency-graph/dependency-submission)
- [GitHub Dependency Review](https://docs.github.com/en/code-security/concepts/supply-chain-security/dependency-review)
- [Official review action](https://github.com/actions/dependency-review-action)
