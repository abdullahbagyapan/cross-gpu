# Architecture

CrossGPU consists of a portable C11 core and independently built shared-library
backends. The core owns command parsing, plugin discovery, device grouping,
backend selection, workload scheduling, and terminal reporting. A backend owns
runtime-specific discovery and the implementation of each workload.

## Build targets and platform adapters

The Makefile exposes these targets:

- `all` builds the executable and enabled plugins.
- `plugins` builds enabled plugins only.
- `check` builds `all`, registry tests, and mock plugins, then runs the test suite.
- `install` installs the executable and enabled plugins.
- `uninstall` removes the executable and platform-native plugin filenames.
- `print-config` prints the resolved build configuration.
- `clean` removes `BUILD_DIR` (default: `build`).

The Makefile selects one platform adapter: `src/platform_linux.c`,
`src/platform_apple.c`, or `src/platform_windows.c`. The adapters provide the
dynamic loader, directory enumeration, executable-path lookup, timing, mutexes,
and worker threads. CUDA, HIP, and Level Zero builds are restricted to Linux;
Metal builds are restricted to macOS.

## Discovery and device grouping

The loader accepts only files with the platform's `crossgpu_backend_` prefix and
shared-library suffix. It resolves `crossgpu_backend_get_api`, requests ABI
version 1, validates the plugin descriptor, and optionally calls `initialize`.
Plugin search paths are documented in the [README](../README.md#plugin-paths).

Each initialized backend is enumerated with the ABI's two-pass device contract.
Candidates are grouped into one public device only when all of these fields
match exactly:

- `identity_kind`
- `identity_namespace`
- `identity_value`

Candidates without a complete physical identity remain separate. The core does
not merge devices by product name, vendor name, device index, or a backend-local
identifier.

## Backend selection and fallback

Every candidate must receive a positive score from `backend_preferences` in
`src/registry.c`. A preference row matches the PCI vendor ID, plugin ABI kind,
and runtime plugin name. Exact vendor rows take precedence over the wildcard
OpenCL row. A candidate with no matching row is excluded, even if the plugin was
successfully loaded.

Candidates for one grouped physical device are ordered by score. The worker tries
the highest-ranked backend first; if `device_open` fails, it tries the next
candidate. Supplying `--backend NAME` filters loaded plugins to that exact name,
so no cross-backend fallback is available for that run.

This policy has an important maintainer consequence: a third-party plugin with a
new name or backend kind cannot become usable only by being placed in a plugin
directory. Add an appropriate preference row in `src/registry.c` and, when
needed, a public vendor constant and vendor descriptor. The CLI parser itself
does not require a backend-specific branch.

## Workloads

The test registry in `src/registry.c` defines the current workload names:
`compute`, `memory`, and `bandwidth`. All are included by `--test all`.

The core creates one worker per selected public GPU and runs selected workloads
serially within that worker. The specified duration is per device and is divided
equally among selected workloads. The public CLI does not expose workload memory
size; it passes `memory_bytes = 0`, which asks each backend to choose its
conservative default.

To add a workload, add a public `CROSSGPU_TEST_*` value, a registry descriptor,
and an implementation in every backend that will advertise it. The CLI parser,
help text, `all` expansion, and rate units are driven by the registry.

## Further reading

- [Backend ABI](BACKEND_ABI.md) describes the public plugin contract.
- [Validation scope](VALIDATION.md) distinguishes mock coverage from hardware validation.
- [Automation safety](AUTOMATION_SAFETY.md) describes the checked-in GitHub workflows.
- [Outstanding work](TODO.md) records source-derived maintainer tasks.
