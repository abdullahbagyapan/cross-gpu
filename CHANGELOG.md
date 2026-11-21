# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Strict `stdout`/`stderr` separation across `list`, `info`, and `run`, ensuring `--json` output on `stdout` never mixes with diagnostics.
- `-v` / `--verbose` diagnostic logging to `stderr` covering shared library SONAME/framework probes, physical GPU deduplication decisions, `importlib.resources` kernel lookups, cache hits/misses, compile flags, and compiler diagnostics.
- Actionable per-backend discovery diagnostics (including Linux `/dev/dri/renderD*` and `/dev/kfd` permission hints) and explicit backend/architecture skip `reason` strings for every `unsupported` workload in both `text` and JSON outputs.
- `difflib`-powered typo suggestions for invalid `--test` tokens (including comma-separated lists), `--backend` names, and subcommands.
- Copy-pasteable `Examples:` epilogs in `--help` output, interactive ASCII banner (`src/banner.py`, `--no-banner`), colorful status badges (`PASSED`, `FAILED`, `UNSUPPORTED`), and final pass/fail/unsupported/hardware-error summary tallies.
- `CROSS_GPU_NO_CACHE` environment override (bypassing disk cache reads/writes) and explicit `XDG_CACHE_HOME` override support across both Linux and macOS.
- Static shell completion scripts for Bash (`completions/cross-gpu.bash`), Zsh (`completions/_cross-gpu`), and Fish (`completions/cross-gpu.fish`), along with FHS-compliant `Makefile` install targets (`PREFIX` / `DESTDIR`).
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

## [0.1.0] - 2026-11-19

### Added
- **CLI & Contracts:** Zero-dependency Python 3.10+ standard-library CLI (`cross-gpu`) with `list`, `info`, and `run` subcommands, duration/warmup parsing, text and JSON output, and deterministic severity-ordered exit codes (`0=SUCCESS`, `1=USAGE_ERROR`, `2=INIT_ERROR`, `3=COMPUTE_ERROR`, `4=HARDWARE_ERROR`).
- **Native `ctypes` Backends:** 5 runtime plugins with explicit `argtypes`/`restype` declarations: CUDA (`libcuda.so.1`), HIP (`libamdhip64.so.6`), oneAPI Level Zero (`libze_loader.so.1`), Apple Metal (`libobjc.A.dylib` + `Metal.framework`), and OpenCL (`libOpenCL.so.1` / `OpenCL.framework`).
- **Physical GPU Deduplication:** Cross-backend deduplication by PCIe BDF, hardware UUID / macOS IOKit `registryID`, and fallback fingerprint, prioritizing native backends (`100`) over OpenCL (`50`).
- **Native GPU Kernels:** 90 standalone kernel source files (`18` compilation units × `5` backends) under `src/kernels/` with zero fake datatype or tensor emulation and explicit `#error "UNSUPPORTED: ..."` hardware gates.
- **20 Stress Workloads & Golden Verification:** Scalar integer (`int64`..`int4`), floating-point/ML (`fp64`..`fp4`, `bf16`), hardware matrix/tensor (`fp32`, `fp16`, `bf16`, `int8`), VRAM pattern/sweep, and D2D/H2D/D2H bandwidth executors with fixed SI unit scaling (`GOP/s`, `GFLOP/s`, `GB/s`), safe VRAM headroom sizing, `50–200 ms` watchdog-safe batch calibration, and host-side golden verification.
- **Distro-Friendly Packaging:** Offline PEP 517/621 `setuptools` build configuration (`pyproject.toml`), `MANIFEST.in`, MIT `LICENSE`, `cross-gpu(1)` man page, `Makefile` staging install, and GitHub CI/Release workflows.
