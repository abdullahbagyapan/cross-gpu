# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-11-24

### Fixed
- Add `__main__` entrypoint guard in `src/cli.py` so direct module execution (`python3 -m src.cli`) dispatches to `main()` instead of exiting 0 silently.
- Set `DEB_PYTHON_INSTALL_LAYOUT=deb` in `Makefile` when `PREFIX=/usr` so FHS-staged packaging on Debian and Ubuntu installs binaries into `/usr/bin` rather than `/usr/local/bin`.
- Assert actionable `stderr` diagnostics on GPU-less emulation targets in `scripts/emulate.sh`, achieving 100% pass rate (17/17 targets) across Linux distributions, CPU architectures (amd64, arm64, riscv64, s390x), and emulated GPU backends.

## [0.1.2] - 2026-11-23

### Added
- Docker-only multi-backend & multi-architecture emulation harness (`scripts/emulate.sh`) covering 17 targets across 4 suites (`arch`, `backends`, `distros`, `adversarial`), with Bash 3.2+ compatibility, automatic `tonistiigi/binfmt` registration, per-step diagnostics, and Markdown summary reports.
- Container-compiled C11 fake-driver ABI shims (`scripts/shims/`) covering CUDA + NVRTC, HIP + HIPRTC, Level Zero, OpenCL, and a portable pure-C11 Apple Metal Objective-C `objc_msgSend` shim (`fake_metal.c`) with atomic resource/pool/encoder leak checks (`exit 99`) and targeted fault injection (`COMPUTE_ERROR=3`, `HARDWARE_ERROR=4`).
- GitHub Actions Environment Emulation workflow (`.github/workflows/emulation.yml`) running all 17 `./scripts/emulate.sh` targets in parallel with `::error::` annotations, `$GITHUB_STEP_SUMMARY` failure diagnostics, and log artifact uploads.
- Testing and Docker emulation guide (`docs/testing.md`) linked from `README.md`, `docs/README.md`, and `docs/contributing.md`.

### Fixed
- Set `PIP_BREAK_SYSTEM_PACKAGES=1` automatically in `Makefile` `install` when `DESTDIR` is specified so FHS staging succeeds cleanly on PEP 668 externally-managed Linux distributions.

## [0.1.1] - 2026-11-22

### Added
- Strict `stdout`/`stderr` separation across `list`, `info`, and `run`, ensuring `--json` output on `stdout` never mixes with diagnostics.
- `-v` / `--verbose` diagnostic logging to `stderr` covering shared library SONAME/framework probes, physical GPU deduplication decisions, `importlib.resources` kernel lookups, cache hits/misses, compile flags, and compiler diagnostics.
- Actionable per-backend discovery diagnostics (including Linux `/dev/dri/renderD*` and `/dev/kfd` permission hints) and explicit backend/architecture skip `reason` strings for every `unsupported` workload in both `text` and JSON outputs.
- `difflib`-powered typo suggestions for invalid `--test` tokens (including comma-separated lists), `--backend` names, and subcommands.
- Copy-pasteable `Examples:` epilogs in `--help` output, interactive ASCII banner (`src/banner.py`, `--no-banner`), colorful status badges (`PASSED`, `FAILED`, `UNSUPPORTED`), and final pass/fail/unsupported/hardware-error summary tallies.
- `CROSS_GPU_NO_CACHE` environment override (bypassing disk cache reads/writes) and explicit `XDG_CACHE_HOME` override support across both Linux and macOS.
- Static shell completion scripts for Bash (`completions/cross-gpu.bash`), Zsh (`completions/_cross-gpu`), and Fish (`completions/cross-gpu.fish`), along with FHS-compliant `Makefile` install targets (`PREFIX` / `DESTDIR`).
- Automated multi-platform GitHub release workflow (`.github/workflows/release.yml`) producing sdist, wheels, and SHA-256 manifests on release tags.
- GitHub issue templates (`.github/ISSUE_TEMPLATE/`) for bug reports and feature requests.
- `SPDX-License-Identifier: MIT` headers across Python sources, native GPU kernels, shell completions, and `Makefile`.
- Millisecond (`ms`) timing support for `--duration` and `--warmup` (e.g., `250ms`, `500ms`), sub-second progress formatting (`250ms/500ms`), and short-duration batch calibration capping.

### Changed
- Simplified `cross-gpu run --json` to emit a single strict JSON document on `stdout` and removed the live background hardware sensor polling subsystem.
- Refactored `src/embed_binary.py` to load raw kernel source files directly via Python standard library `importlib.resources` with plain tuple in-memory cache keys and sanitized, metadata-validated disk cache artifacts (removing all opaque base64 blobs and digest hashing).
- Renamed source directory directly to `src/` with relative imports while preserving the collision-free `cross_gpu` installed namespace.

### Fixed
- Replaced fragile `hipDeviceProp_t` struct offsets in `src/backends/hip_backend.py` with ROCm-stable `hipDeviceGetName`, `hipDeviceTotalMem`, and `hipDeviceGetAttribute` APIs plus an oversized opaque buffer scan for `gcnArchName`.
- Fixed SONAME probing order across `HIP_LIB_CANDIDATES`, `NVRTC_LIB_CANDIDATES`, and `HIPRTC_LIB_CANDIDATES` so versioned `.so.X` runtime libraries are always tried before unversioned `-dev` `.so` symlinks.
- Corrected Level Zero `ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET` (`0x70000006`) and `ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE` (`0x70000007`) fault constants and filtered frontend `-D` macros from SPIR-V `zeModuleCreate` build flags.
- Hardened CUDA/HIP cleanup on poisoned contexts, OpenCL `size_t` work-group queries, Metal `launch_kernel` encoder/command-buffer exception cleanup and worker-thread autorelease pooling, read-only `memoryview` buffer copies, and `format_duration(0.0)` formatting (`0s`).
- Ensured Metal blit command encoder ends encoding within a `try...finally` block to prevent command buffer hangs on exception, and deduplicated `MTLCopyAllDevices()` retains.
- Normalized status badge labels and suppressed banner output on initialization errors.

## [0.1.0] - 2026-11-19

### Added
- **CLI & Contracts:** Zero-dependency Python 3.10+ standard-library CLI (`cross-gpu`) with `list`, `info`, and `run` subcommands, duration/warmup parsing, text and JSON output, and deterministic severity-ordered exit codes (`0=SUCCESS`, `1=USAGE_ERROR`, `2=INIT_ERROR`, `3=COMPUTE_ERROR`, `4=HARDWARE_ERROR`).
- **Native `ctypes` Backends:** 5 runtime plugins with explicit `argtypes`/`restype` declarations: CUDA (`libcuda.so.1`), HIP (`libamdhip64.so.6`), oneAPI Level Zero (`libze_loader.so.1`), Apple Metal (`libobjc.A.dylib` + `Metal.framework`), and OpenCL (`libOpenCL.so.1` / `OpenCL.framework`).
- **Physical GPU Deduplication:** Cross-backend deduplication by PCIe BDF, hardware UUID / macOS IOKit `registryID`, and fallback fingerprint, prioritizing native backends (`100`) over OpenCL (`50`).
- **Native GPU Kernels:** 90 standalone kernel source files (`18` compilation units × `5` backends) under `src/kernels/` with zero fake datatype or tensor emulation and explicit `#error "UNSUPPORTED: ..."` hardware gates.
- **20 Stress Workloads & Golden Verification:** Scalar integer (`int64`..`int4`), floating-point/ML (`fp64`..`fp4`, `bf16`), hardware matrix/tensor (`fp32`, `fp16`, `bf16`, `int8`), VRAM pattern/sweep, and D2D/H2D/D2H bandwidth executors with fixed SI unit scaling (`GOP/s`, `GFLOP/s`, `GB/s`), safe VRAM headroom sizing, `50–200 ms` watchdog-safe batch calibration, and host-side golden verification.
- **Distro-Friendly Packaging:** Offline PEP 517/621 `setuptools` build configuration (`pyproject.toml`), `MANIFEST.in`, MIT `LICENSE`, `cross-gpu(1)` man page, `Makefile` staging install, and GitHub CI/Release workflows.
