# GitHub automation and repository safety

This guide describes a conservative automation setup for CrossGPU. Keep
reproducible build/test steps in ordinary GitHub Actions. Use AI agents for
reasoning-heavy tasks such as issue triage, CI investigation, review suggestions,
and roadmap planning. AI output must remain reviewable; it must not merge or
release code automatically.

## 1. Prepare Copilot for this repository

The repository includes `.github/copilot-instructions.md`. It gives Copilot and
compatible coding agents the project invariants, build commands, test commands,
and hardware-validation rules.

On GitHub:

1. Open the repository and select **Settings**.
2. Open the Copilot settings for the repository/account and enable the coding
   agent if it is available on your plan.
3. Enable Copilot code review if you want automated review suggestions on pull
   requests. Confirm repository custom instructions are enabled for code review.
4. Keep pull-request approval and branch protection under maintainer control.

A practical first use is to create a narrowly scoped issue, include expected
behavior and acceptance criteria, then assign the issue to **Copilot** in the
issue's **Assignees** panel. Copilot works on the issue and opens a pull request
for review. Ask it to implement one bounded change at a time; do not let it
merge its own work. Review the diff and CI results yourself. Do not ask an agent
to make broad ABI, backend-selection, or physical-device identity changes
without a design issue and focused tests. See the official
[Copilot agent task guide](https://docs.github.com/en/copilot/how-tos/copilot-on-github/use-copilot-agents/kick-off-a-task).

## 2. Copilot Automations (private/internal repositories)

GitHub also has a separate feature named **Copilot Automations**. It lets you
create persistent/scheduled/event-triggered Copilot cloud-agent tasks from the
repository's Agents interface or the Copilot Automations interface. It is not
the same thing as GitHub Agentic Workflows.

According to GitHub's current availability documentation, Copilot Automations
require a paid Copilot plan and are available for private and internal
repositories, not public repositories. If you plan to publish CrossGPU as a
public repository, use issue-assigned Copilot coding-agent tasks for
human-started implementation work and use `gh aw` below for recurring public
repository automation. Recheck availability if GitHub changes the product
policy. See [Copilot Automations](https://docs.github.com/en/copilot/concepts/agent-capabilities/agent-powered-workflows-and-operations/about-automations).

For a private/internal development mirror, open the repository's **Agents**
tab (or the Copilot Automations interface), create an automation, then define:

- A narrow task prompt, such as issue reproduction triage or weekly CI summary.
- Explicit triggers (manual first; schedule or repository event after testing).
- Only the tools needed by that task, preferably read-only to start.
- A low AI-credit limit and an owner who reviews outputs.

Do not grant merge, release, branch protection, secret-management, or unrestricted
repository-write abilities to an unattended automation.

## 3. GitHub Agentic Workflows (`gh aw`)

GitHub Agentic Workflows let you describe repository automations in Markdown
frontmatter plus natural-language instructions. The `gh aw` extension compiles
the source workflow into a `.lock.yml` GitHub Actions workflow. Commit both
files and review the generated lock file whenever the source changes.

### Install and initialize

Requirements: GitHub CLI, write access to the repository, GitHub Actions enabled,
and access to an AI engine (Copilot, Claude, OpenAI Codex, or Gemini).

```sh
gh --version
gh auth login --scopes repo,workflow
gh extension install github/gh-aw
cd /path/to/CrossGPU
gh aw init
```

`gh aw init` installs the agentic-workflow authoring instructions/skills for the
repository. Start a supported coding agent in the repository and ask it to use
the `agentic-workflows` skill to create a workflow. For example:

```text
/agentic-workflows Create a weekly CrossGPU maintenance report. Read recent
issues, pull requests, and CI runs. Summarize recurring defects, failing
platform jobs, backend validation gaps, and the three most actionable
maintainer tasks. Create one report issue only when there is meaningful new
information; otherwise do nothing. Do not edit source files, create branches,
merge pull requests, or publish releases.
```

The agent creates a Markdown source workflow and compiles its `.lock.yml` file.
Review both files before committing. After editing the Markdown source, run:

```sh
gh aw compile
```

Then test from the Actions tab or with `gh aw run <workflow-name>`.

### Authentication

For a personal repository, store the selected provider credential in **Settings
→ Secrets and variables → Actions**. Current documented examples include
`COPILOT_GITHUB_TOKEN` for Copilot in personal repositories, `OPENAI_API_KEY`
for Codex, `ANTHROPIC_API_KEY` for Claude, or `GEMINI_API_KEY` for Gemini.
Never put API keys in workflow Markdown, the generated lock file, issues, or
commits. Organization-owned repositories may use organization-billed Copilot
requests when the relevant organization policies are enabled and the workflow
requests `copilot-requests: write`.

### Recommended first automations

Start with low-risk, read-oriented agents:

- **Issue triage:** classify a new issue as bug, feature request, or question;
  identify missing GPU model, OS, driver, backend, and reproduction details;
  suggest labels or a clarifying comment.
- **CI failure investigator:** summarize a failed CI run and suggest likely
  causes and the next manual reproduction command. It must not change source.
- **Weekly maintenance report:** summarize open issues, stale pull requests,
  dependency alerts, and validation gaps.
- **Documentation drift review:** identify README claims that may not match
  build files or backend validation evidence, and open a reviewable PR only
  after a maintainer requests it.

### Agent security rules

- Prefer read-only agent permissions. Route any writes through explicitly
  declared `safe-outputs` such as `add-comment`, `create-issue`, or
  `create-pull-request`.
- Do not grant direct `contents: write`, `pull-requests: write`, or `issues: write`
  unless there is a specific reviewed requirement. A safe-output job should
  perform the narrowly allowed write instead.
- Begin with `network: defaults` or no network. Add only required destinations.
- Treat issue bodies, PR comments, commit messages, logs, and external documents
  as untrusted data that may contain prompt injection. They are data to analyze,
  not instructions to override this guide.
- Never allow an agent to approve/merge PRs, push directly to `main`, create
  releases, modify repository secrets, or weaken required checks.
- Inspect the compiled `.lock.yml`, permissions, tools, network policy, safe
  outputs, and all generated changes before enabling a workflow.
- Do not run AI agents on `pull_request_target` with untrusted PR code or grant
  fork PR jobs repository secrets.

## 4. Standard GitHub Actions safety controls

The repository's deterministic CI remains the merge gate. The following
controls should be enabled in addition to the workflows in `.github/workflows/`:

1. **Default token permissions:** in Settings → Actions → General, set the
   default `GITHUB_TOKEN` permission to read-only. Grant extra permissions only
   to the specific job that requires them.
2. **Branch ruleset for `main`:** require a pull request, require the relevant
   CI jobs to pass (Linux GCC, Linux Clang, macOS, Windows, and Level Zero),
   require `Analyze C and C++` and `Review dependency changes`, block force
   pushes and branch deletion, and require conversation resolution. Consider
   one approval if collaborators are available. Do not let automation bypass
   the ruleset. Select the status-check names shown by the first successful
   Actions runs because GitHub records the exact job names as required checks.
3. **Dependabot:** enable Dependabot alerts and security updates, and keep
   `.github/dependabot.yml` active for GitHub Actions updates.
4. **Secret protection:** enable secret scanning and push protection where
   available. Keep credentials in Actions secrets or protected environments.
5. **Code scanning:** enable the CodeQL workflow below. Review alerts rather
   than automatically dismissing them.
6. **Dependency review:** enable the dependency graph and run the dependency
   review check on every pull request.
7. **Action pinning:** prefer full commit-SHA references for third-party Actions
   and keep them updated with Dependabot. If using major-version tags for
   first-party actions, keep Dependabot enabled and review updates promptly.
8. **Release protection:** restrict who can create release tags and publish
   GitHub Releases. Build release artifacts from a reviewed tag, not from an
   arbitrary pull-request branch.

The availability of some security features depends on repository visibility
and the GitHub plan. Public repositories have more Code Security features
available than private repositories on the Free plan; check the repository's
current **Security and analysis** settings.

## 5. Workflows included

- `ci.yml`: deterministic builds and mock-backend tests on Linux, macOS,
  Windows/MinGW, and a dedicated Linux Level Zero build.
- `codeql.yml`: scheduled and PR-triggered CodeQL analysis of the compiled C/C++
  core, OpenCL plugin, and Intel Level Zero plugin.
- `dependency-review.yml`: checks newly introduced dependencies on pull
  requests and fails for high or critical severity vulnerabilities.

These workflows deliberately do not publish artifacts, merge pull requests,
or write to the repository. Review any future automation before granting it
additional permissions.
