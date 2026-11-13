# Security Policy

## Reporting a vulnerability

Please do not publish exploit details, crash payloads, or information that could
put users at risk in a public issue.

If private vulnerability reporting is enabled for this repository, use
**Security → Advisories → Report a vulnerability** on GitHub. Otherwise, contact
the repository owner through their GitHub profile and request a private
reporting channel before sending technical details.

Please include the affected commit or release, operating system, backend and
GPU model when known, reproduction steps, and the potential impact. Avoid
including memory dumps or other potentially sensitive user data unless it is
strictly necessary and has been sanitized.

## Scope

The scope includes the CrossGPU CLI, core backend manager, public plugin ABI,
bundled backend plugins, and build tooling. Third-party GPU drivers, SDKs, and
runtime libraries are outside this project's control; report vulnerabilities
in those components to their respective vendors.

There is currently no formal supported-release policy. Until one is published,
please report issues against the latest source on the default branch.
