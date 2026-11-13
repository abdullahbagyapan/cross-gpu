<p align="left">
  <img src="./assets/image.png" width="280" alt="CrossGPU Logo">
</p>

# CrossGPU

CrossGPU is a C11 command-line GPU stress tester built around a versioned,
dynamically loaded backend plugin ABI. The core executable does not link against
CUDA, ROCm, OpenCL, Metal, or Level Zero runtimes. Backends are independently
built shared libraries and are discovered at runtime.

**Initial release target: `v1.0.0`.** The roadmap below describes planned follow-up
releases. Version targets are planning milestones, not delivery-date commitments.

## Current status

- C11 CLI: `list`, `info`, `run`
- Runtime backend discovery through a platform adapter (POSIX loader on Linux/Apple; Win32 loader on Windows)
- Versioned C plugin ABI (`CROSSGPU_BACKEND_ABI_VERSION` 1)
- Typed, namespaced physical-device identities; the core compares identity
  metadata rather than interpreting backend-provided string prefixes
- OpenCL compatibility backend dynamically loads the system OpenCL runtime
- Optional CUDA, HIP, Metal, and Intel Level Zero backend plugins are implemented
  as independent shared libraries; each is included only when enabled at build time
- Automatic backend preference is per physical device
- Backend candidates are merged only when identity kind, namespace, and value
  all match; product-name similarity is never used as physical identity
- Each merged device retains all compatible backend candidates; the preferred backend is tried first and the others remain fallbacks
- Device memory metadata distinguishes physical memory, recommended working-set size, and maximum single allocation
- Backend selection and fallback are per device
- Backend and device registries use dynamically allocated storage (no fixed
  application-wide plugin, device, or per-device-backend count limits)
- Test selection and display metadata are registry-driven; `--test all` is
  generated from the registry rather than a fixed array
- Backend override accepts a plugin's runtime name, including future plugins
  unknown to the core executable
- Multiple selected devices are tested concurrently
- Terminal output only

See [Build and Validation Status](docs/VALIDATION.md) for the distinction
between mock-test coverage, CI builds, and native vendor-hardware validation.
The GitHub Actions workflow runs on pull-request creation, updates, reopening,
edits, and conversion from draft to ready-for-review, as well as pushes. It
builds and tests Linux (GCC and Clang), Windows (MinGW-w64 UCRT64), a separate
Intel Level Zero SDK/OCLOC plugin build on Ubuntu 24.04, and macOS (OpenCL and
Metal plugin builds). Separate pull-request workflows also run CodeQL analysis
on the compiled C/C++ core, OpenCL, and Level Zero code and review dependency
changes. Dependabot checks GitHub Actions references weekly.
Mock-backend tests do not claim physical-GPU workload coverage.

CUDA, HIP, Metal, and Intel Level Zero backend plugins are available as
optional independent plugins. The loader continues to accept plugins with custom
backend kinds and plugin-defined names.

## Build

The core and OpenCL plugin require a C compiler and GNU Make. Linux and Apple
use their native POSIX loader/threading APIs; Windows uses the Win32 loader,
thread, synchronization, filesystem, and timing APIs (the supported Windows
build environment is MinGW-w64 UCRT64 under MSYS2). OpenCL development headers
and the OpenCL SDK are not required. CUDA, HIP, Metal, and Level Zero toolchains
are optional and are needed only when building their corresponding plugins.

```sh
make
make check
```

This builds the platform-native executable and OpenCL plugin:

| Platform | Executable | OpenCL plugin |
|---|---|---|
| Linux | `build/cross-gpu` | `build/backends/libcrossgpu_backend_opencl.so` |
| Windows | `build/cross-gpu.exe` | `build/backends/crossgpu_backend_opencl.dll` |
| Apple macOS | `build/cross-gpu` | `build/backends/libcrossgpu_backend_opencl.dylib` |

Build the optional vendor plugins independently:

```sh
# NVIDIA CUDA (Linux + CUDA Toolkit)
make WITH_CUDA=1

# AMD HIP (Linux + ROCm)
make WITH_HIP=1

# Intel Level Zero (Linux + Level Zero SDK, loader, and OCLOC)
make WITH_LEVEL_ZERO=1

# Apple Metal (macOS + Xcode Command Line Tools)
make WITH_METAL=1

# All Linux plugins when all toolchains are installed
make WITH_OPENCL=1 WITH_CUDA=1 WITH_HIP=1 WITH_LEVEL_ZERO=1
```

The optional plugin flags can be combined. Run `make clean` when changing the
enabled backend set so disabled plugins from an earlier build are not left in
`build/backends`. CUDA, HIP, and Level Zero plugins are currently built on Linux; Metal is built
on macOS. The core executable does not link against OpenCL, CUDA, HIP/ROCm,
Metal, or Level Zero. The OpenCL plugin loads the system
OpenCL runtime dynamically (`libOpenCL.so.1`/`libOpenCL.so` on Linux and the
OpenCL framework on macOS); on macOS it also links the OpenGL framework for the
CGL-based OpenCL-to-IORegistry identity bridge.
CUDA/HIP plugins link their respective runtime libraries, the Metal plugin
links the Apple Metal and Foundation frameworks, and Level Zero links only the
Level Zero loader (`libze_loader`). Missing vendor toolchains do not affect the
core-only build.

To build only the core executable:

```sh
make clean
make WITH_OPENCL=0
```

Install the executable and enabled backend plugins:

```sh
sudo make install
```

Use `PREFIX=/opt/cross-gpu` to install under a different prefix.

## CLI

```sh
# List detected GPUs
cross-gpu list

# Detailed information
cross-gpu info

# Compute stress
cross-gpu run --test compute --duration 60s

# VRAM pattern/write-read integrity test
cross-gpu run --test memory --duration 5m

# Device-memory bandwidth
cross-gpu run --test bandwidth --duration 2m

# All registered tests; duration is split evenly
cross-gpu run --test all --duration 5m

# All detected GPUs concurrently
cross-gpu run --device all --test all --duration 10m

# One detected GPU
cross-gpu run --device 0 --test compute --duration 60s

# Force a backend plugin by its runtime name
cross-gpu run --backend opencl --test compute
cross-gpu run --backend cuda --test compute
cross-gpu run --backend hip --test compute
cross-gpu run --backend metal --test compute
cross-gpu run --backend level-zero --test compute
```

`--backend auto` is the default. The override accepts the plugin's `name` from
its `crossgpu_backend_plugin_t` descriptor; it is not restricted to backend
names compiled into the main executable. This allows future plugins to be used
without changing the CLI parser.

Duration accepts `ms`, `s`, `m`, and `h` suffixes, for example `500ms`, `60s`,
`5m`, and `1h`. The default duration is 60 seconds. The maximum is seven days. The displayed total duration is divided equally among selected workloads; the CLI reports both total and per-workload duration. Device information reports physical memory, recommended working-set size, and maximum single allocation separately; unavailable properties are shown as `not exposed`.

## Backend discovery and selection

The executable scans these locations in order:

1. Directories in `CROSSGPU_PLUGIN_PATH` (colon-separated on Linux/macOS; semicolon-separated on Windows)
2. `backends/` beside the executable
3. `./backends`
4. The configured installation directory, normally `/usr/local/lib/cross-gpu/backends`
5. `/usr/lib/cross-gpu/backends`

Set `CROSSGPU_PLUGIN_PATH_ONLY=1` to use only the directories listed in
`CROSSGPU_PLUGIN_PATH` and skip executable-relative and system plugin locations.
This is useful for isolated testing and troubleshooting plugin selection.

Automatic selection is performed separately for each physical GPU. When multiple
plugins identify the same GPU using the same typed identity, namespace, and value,
the host ranks candidates using the vendor preference registry. The preferred
backend is tried first; if its `device_open()` fails, the next compatible candidate
is tried. Devices without a trustworthy shared identity remain separate rather
than being merged by model name or an API-local index.
An explicit `--backend PLUGIN_NAME` filters discovery to that exact plugin name
and disables cross-backend fallback for that run.

Backend preference policy has a single source of truth:
`backend_preferences` in `src/registry.c`. Each row matches a vendor, the
plugin's declared ABI kind, and its runtime name. This prevents an unrelated
plugin from borrowing another backend's preference by reusing its name. Exact
vendor entries take precedence over the `CROSSGPU_VENDOR_ANY` wildcard entry;
a missing preference rejects that plugin
for the device. The generic wildcard row allows OpenCL on otherwise unlisted
vendors. Backend discovery in `src/backend_manager.c` only asks the registry
for a score and contains no vendor/API-specific selection branches.

The current preference policy is:

| Vendor | Preferred plugin | Score | Compatibility plugin | Score |
|---|---|---:|---|---:|
| NVIDIA | CUDA | 100 | OpenCL | 50 |
| AMD | HIP | 100 | OpenCL | 50 |
| Intel | Level Zero | 100 | OpenCL | 50 |
| Apple | Metal | 100 | OpenCL | 50 |
| Qualcomm, Arm, Broadcom, Imagination | OpenCL | 100 | — | — |
| Other / unknown vendors | OpenCL | 50 | — | — |


Only discovered and initialized plugins participate in selection. On systems
where a preferred API is not built or installed, the next available backend is
tried. Device-level `device_open()` failures also trigger fallback to the next
candidate for that physical GPU.

Plugins may use one of the built-in backend kinds or
`CROSSGPU_BACKEND_CUSTOM`. The host does not reject unknown nonzero backend kind
values; automatic preference is controlled by rows in the vendor/backend
preference registry, keyed by vendor ID, expected backend kind, and plugin
runtime name. The explicit backend override only needs the plugin's runtime
name.

Plugin filenames must use the form `libcrossgpu_backend_NAME.so` on Linux,
`libcrossgpu_backend_NAME.dylib` on macOS, or `crossgpu_backend_NAME.dll` on
Windows. A plugin must export
`crossgpu_backend_get_api()` and report `CROSSGPU_BACKEND_ABI_VERSION`.

For development:

```sh
CROSSGPU_PLUGIN_PATH="$PWD/build/backends" ./build/cross-gpu list
```

## Registries and adding features

Core metadata is centralized in `src/registry.c`:

- Backend descriptors define built-in backend kind names.
- Vendor descriptors name PCI vendors; a separate preference table maps vendor IDs, expected backend kinds, and runtime plugin names to scores.
- Test descriptors define CLI names, descriptions, default `all` inclusion, and
  result-rate units.

To add a vendor, add a named `CROSSGPU_VENDOR_*` value to the public vendor enum,
a vendor-name row, and preference rows keyed by its vendor ID, expected
backend kind, and plugin runtime name. To make a new backend preferred for an
existing vendor, add a preference row; no backend enum or CLI parser branch is
required. Vendor matching and backend scoring do not need new `if/else` branches.

To add a built-in test, add a `CROSSGPU_TEST_*` enum value and a descriptor row.
The CLI parser, `--test all`, help text, test-name reporting, and result-rate
formatting are driven by that registry. Each backend must then implement the new
operation. The OpenCL backend uses a workload-operation table rather than a
central test-type conditional chain.

To add a backend, implement the public plugin ABI and build a shared library
named `libcrossgpu_backend_NAME`. A new plugin can use
`CROSSGPU_BACKEND_CUSTOM`, provide its own runtime name, and be selected with
`--backend NAME`. The core CLI does not need a new parser branch. If automatic
vendor preference is desired, add the corresponding vendor enum and preference
row.

See [docs/backend-plugin.md](docs/backend-plugin.md) and
[docs/extending-crossgpu.md](docs/extending-crossgpu.md).

## CUDA, HIP, Metal, and Level Zero plugins

The CUDA and HIP plugins share a small runtime-adapter implementation in
`backends/common/runtime_backend_impl.hpp`, but are compiled independently into
`libcrossgpu_backend_cuda` and `libcrossgpu_backend_hip`. Each plugin uses its
own vendor runtime, stream, device enumeration, memory allocation, and kernel
launch path. Both use the runtime PCI bus-ID query and report the result using
the shared `pci-bdf` identity namespace also used by OpenCL. If a runtime cannot
provide a real PCI address, the plugin reports a backend-local index instead
of synthesizing a potentially incorrect physical identity.

The Metal plugin is implemented in Objective-C in
`backends/metal/metal_backend.m`. Its Metal Shading Language kernels are embedded
and compiled when a Metal device is opened. It reports Apple's recommended
working-set size as its memory estimate, not a hardware-reported total VRAM
capacity. Metal devices report their OS registry ID in the
`apple.iokit.registry-id` namespace. On macOS, the OpenCL plugin uses Apple's
CGL renderer enumeration to associate a CGL renderer with its OpenCL device
(the renderer-ID-to-`cl_device_id` conversion described in Apple TN2335), then
reads CGL's 64-bit registry-ID properties. This produces the same registry ID
space used by `MTLDevice.registryID`; no GPU-name or array-index heuristic is
involved. If the CGL association is unavailable or conflicting, OpenCL leaves
its physical identity unset and the host keeps the candidates separate.

The CUDA, HIP, Metal, and Level Zero plugins implement the `compute`, `memory`,
and `bandwidth` operations. The working set is conservative (approximately 1/16 of device
memory, bounded to 4–128 MiB and the reported maximum allocation). These are
initial stress workloads, not exhaustive VRAM diagnostics.

## Intel Level Zero implementation details

The Intel plugin is implemented in C++ in
`backends/levelzero/levelzero_backend.cpp`. It uses the Level Zero loader API
for GPU discovery, device memory allocation, command queues/lists, memory
copies, kernel dispatch, and synchronization. It filters discovery to Intel GPU
root devices (vendor ID `0x8086`) and does not expose CPU or NPU devices as GPUs.

The three workload tests are implemented as follows:

- **Compute:** repeated FP32 fused multiply-add operations on a device buffer.
- **Memory:** device-side pattern generation and verification with an error
  counter copied back to the host.
- **Bandwidth:** repeated device-to-device copies between two GPU allocations.

The OpenCL C kernel source in `backends/levelzero/levelzero_kernels.cl` is
compiled to generic SPIR-V at build time by Intel's OCLOC tool. GNU `ld` then
converts that binary into a relocatable object, which is linked into the plugin.
Level Zero's `zeModuleCreate` builds the module for the selected GPU at runtime.
The Level Zero plugin does not load or call the OpenCL runtime. OCLOC and GNU
binutils are build-time prerequisites, not runtime dependencies. No Python
script or separate kernel file is required at runtime.

Build on Linux with Level Zero 1.10+ development headers, the loader library,
an Intel GPU driver, and OCLOC installed:

```sh
make WITH_LEVEL_ZERO=1
```

If the SDK or loader is installed outside system paths, pass the corresponding
include and linker flags, for example:

```sh
make WITH_LEVEL_ZERO=1 \
  LEVEL_ZERO_CPPFLAGS="-I/opt/level-zero/include" \
  LEVEL_ZERO_LIBS="-L/opt/level-zero/lib -lze_loader" \
  OCLOC=/opt/intel/ocloc/bin/ocloc
```

The Level Zero plugin reports the PCI domain:bus:device.function address using
the shared `pci-bdf` namespace when the driver exposes the PCI properties
extension. That allows the manager to merge it with OpenCL's matching PCI
identity and prefer Level Zero while retaining OpenCL as a fallback. If PCI
identity is unavailable, the plugin reports a Level Zero-namespaced device UUID;
that UUID is intentionally not considered equivalent to an OpenCL UUID.

## OpenCL implementation details

The OpenCL plugin enumerates GPU devices (not CPU OpenCL devices), compiles its
OpenCL C kernels at runtime, and currently provides:

- `compute`: repeated FP32 fused multiply-add operations with on-device
  finite/range validation
- `memory`: deterministic address-dependent pattern writes and verification
- `bandwidth`: repeated device-buffer copies

The default working set is conservative: approximately 1/16 of reported global
memory, with a 4 MiB minimum and 128 MiB maximum, further limited by the device's
maximum single-allocation size. This is not yet a full-VRAM test; a configurable
memory percentage/limit is planned.

OpenCL platform and device enumeration uses dynamically allocated storage; the
previous fixed platform/device count ceilings have been removed.

## Testing without a GPU

```sh
make check
```

The test target builds mock CUDA-like, OpenCL-like, Intel Level Zero-like, and
custom backend plugins. It verifies that Level Zero wins for Intel while OpenCL
remains a fallback. It also verifies runtime discovery, device info,
registry-driven test selection, preferred-backend fallback, preference scoring,
identity namespace isolation, Intel Level Zero vendor restrictions, and rejection
of plugin names without a matching preference entry. It does not simulate or
validate physical GPU execution.

## Platform architecture and support

Compiler/OS detection is centralized in `include/crossgpu_platform_config.h`.
The core platform API is declared in `src/platform.h`, and the Makefile selects
one OS adapter:

| Platform | Adapter | Plugin suffix | Dynamic loader | Default OpenCL runtime |
|---|---|---|---|---|
| Linux | `src/platform_linux.c` | `.so` | `dlopen` / `dlsym` | `libOpenCL.so.1`, `libOpenCL.so` |
| Windows | `src/platform_windows.c` | `.dll` | `LoadLibraryA` / `GetProcAddress` | `OpenCL.dll` |
| Apple macOS | `src/platform_apple.c` | `.dylib` | `dlopen` / `dlsym` | OpenCL framework |

The adapters provide directory enumeration, executable-path resolution,
dynamic-library operations, monotonic time, sleep, mutexes, and worker threads.
Code outside the configuration header uses `CROSSGPU_PLATFORM_LINUX`,
`CROSSGPU_PLATFORM_WINDOWS`, and `CROSSGPU_PLATFORM_APPLE` instead of checking
compiler predefined OS macros directly.
The core and OpenCL plugin use the same selected adapter. OpenCL's CGL renderer
association remains Apple-only because it relies on Apple's CGL and IORegistry
APIs; it is not part of the cross-platform core.

The Windows build is currently targeted at MinGW-w64 UCRT64 under MSYS2. It is
not an MSVC project. Build the portable core and OpenCL plugin with:

```sh
make clean
make
make check
```

CUDA, HIP, and Level Zero plugins are currently Linux-only. Metal is Apple-only.
The Level Zero SPIR-V embedding rule uses GNU `ld -r -b binary`, so it is
intentionally limited to Linux. Vendor SDKs and actual GPU execution still
require validation on their native OS and hardware.

## Current limitations

- The generic CI and mock tests do not validate physical GPU workloads. Native
  compilation and execution status for each backend is recorded in
  [docs/VALIDATION.md](docs/VALIDATION.md); validate vendor SDK compatibility
  and hardware behavior before relying on a backend in production.
- The macOS OpenCL-to-Metal identity bridge depends on Apple's CGL renderer
  association and IORegistry properties. The portable test environment cannot
  validate the bridge on physical Apple Silicon hardware; if the runtime does
  not expose the mapping, CrossGPU deliberately leaves the devices separate.
- Temperature, power, fan speed, clocks, and utilization monitoring are not
  implemented yet.
- The memory test validates the application's working set; it is not a
  substitute for manufacturer diagnostics or exhaustive VRAM testing.

## Roadmap

The initial `v1.0.0` release establishes the plugin architecture and the three
current GPU workloads. Future versions are grouped into milestones below. The
order and scope may change as hardware testing, driver compatibility, and
community feedback reveal priorities.

### v1.0.0 — Initial release (current)

- [x] C11 command-line application with `list`, `info`, and `run` commands.
- [x] Versioned plugin ABI v1 and dynamically loaded backend libraries.
- [x] OpenCL compatibility backend and optional CUDA, HIP, Metal, and Intel
      Level Zero plugins.
- [x] Registry-driven vendor/backend preference policy and per-device fallback.
- [x] Typed physical-device identities and conservative cross-backend merging.
- [x] Linux, Apple macOS, and Windows platform adapters (Windows targets
      MinGW-w64 UCRT64 under MSYS2).
- [x] Compute, memory-integrity, and device-memory bandwidth workloads.
- [x] Mock-backend tests and pull-request CI for the supported build targets.

### v1.1.0 — Reliability and correctness

- [ ] Validate plugin structure sizes and all required ABI fields before
      reading plugin-provided structures; reject malformed plugins cleanly.
- [ ] Propagate backend-discovery and initialization errors through CLI exit
      codes instead of silently presenting an empty device list.
- [ ] Retry device enumeration when the device count changes between the
      count-query and enumeration calls.
- [ ] Replace fixed-size OpenCL extension buffers with length-aware queries and
      exact extension-token matching.
- [ ] Report actual CUDA/HIP maximum allocation limits; do not equate total
      device memory with the maximum size of one allocation.
- [ ] Correct Level Zero memory-bank accounting and clamp allocations to the
      capacity of the selected memory region.
- [ ] Harden signal handling and inter-thread cancellation with explicit
      synchronization.
- [ ] Add fault-injection tests for malformed plugins, allocation failures,
      changing device counts, and workload initialization failures.
- [ ] Define and test fallback behavior for workload initialization failures,
      while never hiding failures that occur after a workload has started.

### v1.2.0 — Configurable stress workloads

- [ ] Add user-configurable memory sizes and percentages, with backend-reported
      limits and safety headroom for the driver and operating system.
- [ ] Add chunked memory sweeps so larger portions of device memory can be
      tested without requiring one enormous allocation.
- [ ] Expand memory patterns with walking-bit, randomized (seeded),
      address-dependent, and multi-pattern verification modes.
- [ ] Add explicit host-to-device, device-to-host, and bidirectional transfer
      bandwidth workloads, separate from device-to-device copy bandwidth.
- [ ] Expose workload configuration consistently across OpenCL, CUDA, HIP,
      Metal, and Level Zero where the hardware API supports it.
- [ ] Add optional queue/stream concurrency controls and document how they
      affect repeatability and resource usage.

### v1.3.0 — Telemetry and thermal safety

- [ ] Define an optional telemetry capability in the backend interface without
      making unsupported sensors mandatory.
- [ ] Report supported temperature, power, utilization, clock, and fan metrics
      through vendor-specific providers where available.
- [ ] Add configurable monitoring intervals and time-series terminal output.
- [ ] Add thermal and power stop thresholds when reliable sensor data exists;
      never treat unavailable telemetry as a safe temperature reading.
- [ ] Record sensor source, units, and unsupported fields explicitly.

### v1.4.0 — Reports and reproducible runs

- [ ] Add machine-readable JSON and CSV output while keeping terminal output as
      the default interface.
- [ ] Record OS, driver/runtime versions, GPU identity, selected backend,
      workload configuration, random seed, and timing metadata for each run.
- [ ] Add stable exit codes and a documented result schema for automation.
- [ ] Add repeatable run profiles and the ability to replay a saved profile.
- [ ] Add comparison summaries for repeated runs without collapsing correctness
      errors, coverage, and throughput into a misleading single score.

### v1.5.0 — Platform parity and distribution

- [ ] Add Unicode-safe Windows filesystem and dynamic-library loading paths.
- [ ] Validate the Windows adapter with native Windows builds and real hardware;
      evaluate MSVC support separately from the current MinGW-w64 target.
- [ ] Evaluate native Windows builds for vendor plugins where the corresponding
      SDKs and runtime APIs support them.
- [ ] Expand macOS validation across Apple Silicon and supported Intel Macs,
      including the Metal/OpenCL physical-identity bridge.
- [ ] Provide reproducible release archives, package-manager recipes, and
      documented installation/uninstallation flows for each supported OS.

### v2.0.0 — Advanced diagnostics and plugin capabilities

- [ ] Design an ABI v2 only when needed for negotiated capabilities, optional
      telemetry, richer workload configuration, and forward-compatible
      structure sizing; retain a documented compatibility strategy for ABI v1.
- [ ] Add advanced memory diagnostics such as longer retention tests,
      multi-allocation testing, and broader address/data-pattern coverage.
- [ ] Add multi-queue and concurrent compute/memory stress scenarios with clear
      isolation and reproducibility controls.
- [ ] Add optional PCIe/interface diagnostics and distinguish interface
      throughput from device-local memory bandwidth.
- [ ] Add broader hardware regression coverage and a published backend/driver
      compatibility matrix based on actual device testing.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and pull
request checklist. Repository AI-agent setup and security automation are
documented in [docs/GITHUB_AUTOMATION.md](docs/GITHUB_AUTOMATION.md). Security
issues should be reported according to [SECURITY.md](SECURITY.md), not in public
issue reports.


## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).
See the `LICENSE` file for full details regarding permissions, modifications, and redistribution. 
