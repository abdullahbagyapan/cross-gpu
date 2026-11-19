# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Strict `stdout`/`stderr` separation across `list`, `info`, and `run`, ensuring `--json` / NDJSON output on `stdout` never mixes with diagnostics.
- `-v` / `--verbose` diagnostic logging to `stderr` covering shared library SONAME/framework probes, physical GPU deduplication decisions, `importlib.resources` kernel lookups, cache hits/misses, compile flags, and compiler diagnostics.
- Actionable per-backend discovery diagnostics (including Linux `/dev/dri/renderD*` and `/dev/kfd` permission hints) and explicit backend/architecture skip `reason` strings for every `unsupported` workload in both `text` and JSON/NDJSON outputs.
- `difflib`-powered typo suggestions for invalid `--test` tokens (including comma-separated lists), `--backend` names, and subcommands.
- Copy-pasteable `Examples:` epilogs in `--help` output and `text`-mode zero-padded workload progress prefixes (`[04/20 compute-fp32 | warmup 3s/5s]`, `[04/20 compute-fp32 | 18s/60s]`) plus final pass/fail/unsupported/hardware-error summary tallies.
- `CROSS_GPU_NO_CACHE` environment override (bypassing disk cache reads/writes) and explicit `XDG_CACHE_HOME` override support across both Linux and macOS.
- Static shell completion scripts for Bash (`completions/cross-gpu.bash`), Zsh (`completions/_cross-gpu`), and Fish (`completions/cross-gpu.fish`), along with FHS-compliant `Makefile` install targets (`PREFIX` / `DESTDIR`).
- `SPDX-License-Identifier: MIT` headers across Python sources, native GPU kernels, shell completions, and `Makefile`.
- Millisecond (`ms`) timing support for `--duration` and `--warmup` (e.g., `250ms`, `500ms`), sub-second progress formatting (`250ms/500ms`), and short-duration batch calibration capping.
- Telemetry sensor sanitization (`NaN`/`Inf`, sentinel/out-of-range filtering) that omits unusable fields from `text` and JSON/NDJSON outputs and suppresses empty telemetry ticks.

### Changed
- Refactored `src/embed_binary.py` to load raw kernel source files directly via Python standard library `importlib.resources` with plain tuple in-memory cache keys and sanitized, metadata-validated disk cache artifacts (removing all opaque base64 blobs and digest hashing).
- Renamed source directory directly to `src/` with relative imports while preserving the collision-free `cross_gpu` installed namespace.

## [0.1.0] - 2026-11-19

### Added
- **CLI & Contracts:** Zero-dependency Python 3.10+ standard-library CLI (`cross-gpu`) with `list`, `info`, and `run` subcommands, duration/warmup parsing, text and JSON/NDJSON output, and deterministic severity-ordered exit codes (`0=SUCCESS`, `1=USAGE_ERROR`, `2=INIT_ERROR`, `3=COMPUTE_ERROR`, `4=HARDWARE_ERROR`).
- **Native `ctypes` Backends:** 5 runtime plugins with explicit `argtypes`/`restype` declarations: CUDA (`libcuda.so.1`), HIP (`libamdhip64.so.6`), oneAPI Level Zero (`libze_loader.so.1`), Apple Metal (`libobjc.A.dylib` + `Metal.framework`), and OpenCL (`libOpenCL.so.1` / `OpenCL.framework`).
- **Physical GPU Deduplication:** Cross-backend deduplication by PCIe BDF, hardware UUID / macOS IOKit `registryID`, and fallback fingerprint, prioritizing native backends (`100`) over OpenCL (`50`).
- **Native GPU Kernels:** 90 standalone kernel source files (`18` compilation units × `5` backends) under `src/kernels/` with zero fake datatype or tensor emulation and explicit `#error "UNSUPPORTED: ..."` hardware gates.
- **20 Stress Workloads & Golden Verification:** Scalar integer (`int64`..`int4`), floating-point/ML (`fp64`..`fp4`, `bf16`), hardware matrix/tensor (`fp32`, `fp16`, `bf16`, `int8`), VRAM pattern/sweep, and D2D/H2D/D2H bandwidth executors with fixed SI unit scaling (`GOP/s`, `GFLOP/s`, `GB/s`), safe VRAM headroom sizing, `50–200 ms` watchdog-safe batch calibration, and host-side golden verification.
- **1 Hz Hardware Telemetry:** Drift-compensated background sampling via NVML (`libnvidia-ml.so.1`), Linux DRM/hwmon `sysfs`, and unprivileged macOS `ioreg`.
- **Distro-Friendly Packaging:** Offline PEP 517/621 `setuptools` build configuration (`pyproject.toml`), `MANIFEST.in`, MIT `LICENSE`, `cross-gpu(1)` man page, `Makefile` staging install, and GitHub CI/Release workflows.
