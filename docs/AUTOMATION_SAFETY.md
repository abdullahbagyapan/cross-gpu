# Automation safety

The repository currently contains two GitHub Actions workflows:

- `codeql.yml` runs CodeQL for C/C++ on pushes and pull requests targeting
  `master`, plus a weekly schedule. Its manual build step runs `make check` on
  Ubuntu.
- `dependency-review.yml` runs dependency review on pull requests targeting
  `master` and fails on high-severity findings.

There is no checked-in multi-platform CI workflow, release workflow, artifact
publishing workflow, or agentic-workflow configuration. Do not describe those as
active repository automation until their files are added and reviewed.

## Safe changes to automation

- Keep workflow permissions minimal and job-specific.
- Treat issue text, pull-request text, logs, and external content as untrusted
  input when an automated agent reads them.
- Do not grant unattended automation permission to merge pull requests, publish
  releases, change branch protections, or modify repository secrets.
- Review workflow permissions, action references, generated files, and network
  access before enabling a new automation.
- Do not run untrusted fork code with repository secrets or with a
  `pull_request_target` configuration that grants write authority.

Dependabot is configured to check GitHub Actions dependencies weekly. Dependency
review requires GitHub's dependency graph to be enabled for the repository.
