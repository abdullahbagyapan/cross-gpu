# CrossGPU

Cross-platform GPU discovery and stress-testing CLI targeting **Linux** (`x86_64`, `aarch64`) and **macOS** (`arm64`, `x86_64`), built exclusively with the **Python 3.10+ standard library** and **zero external runtime dependencies**.

![Terminal Demo](./assets/demo.gif)

## Highlights

- **5 native `ctypes` backends:** CUDA (`libcuda.so.1`), HIP (`libamdhip64.so.6`), oneAPI Level Zero (`libze_loader.so.1`), Apple Metal (`libobjc.A.dylib` + `Metal.framework`), and OpenCL (`libOpenCL.so.1` / `OpenCL.framework`).
- **Priority & physical deduplication:** Native backends (`100`) take precedence over OpenCL (`50`) using normalized PCIe BDF, hardware UUID / macOS `registryID`, and fallback fingerprints.
- **90 native kernels & 20 verified workloads:** Raw kernels under `src/kernels/` loaded via `importlib.resources` (zero fake emulation), host-side golden verification, fixed SI unit rates (`GOP/s`, `GFLOP/s`, `GB/s`), and safe VRAM sizing.
- **Clean CLI & stream hygiene:** Interactive ASCII banner (`--no-banner`), colorful status badges (`PASSED`, `FAILED`, `UNSUPPORTED`), single-document `--json` output across `list`, `info`, and `run`, strict `stdout`/`stderr` separation, `-v`/`--verbose`, `difflib` typo hints, and `ms`/`s`/`m`/`h` durations.

## Installation

```sh
# Offline PEP 517 build/install (Debian, Fedora, Arch, Nix, Homebrew)
pip install --no-build-isolation --no-deps .

# Stage with Makefile (installs CLI, cross-gpu(1) man page, and bash/zsh/fish completions)
make check
make install PREFIX=/usr DESTDIR=/tmp/stage
```

## Quickstart

```sh
cross-gpu list
cross-gpu info --json
cross-gpu run --device 0 --test compute-fp32,memory-pattern --warmup 5s --duration 30s
cross-gpu run --device 0 --test compute-fp32 --warmup 250ms --duration 500ms --json -v
```

## Exit Codes

Exit codes are severity-ordered and aggregated across GPUs and workloads via `max(codes)`:

| Code | Name | Meaning |
|---|---|---|
| `0` | `SUCCESS` | All executed workloads passed without compute or hardware faults |
| `1` | `USAGE_ERROR` | Invalid CLI arguments, unknown commands/tests/backends, malformed durations, or conflicting flags |
| `2` | `INIT_ERROR` | No compatible GPU/backend found, requested target unavailable, context/kernel init or compile error |
| `3` | `COMPUTE_ERROR` | Golden verification mismatch, silent data corruption (SDC), `NaN`/`Inf`, or memory fault |
| `4` | `HARDWARE_ERROR` | Device-lost error, GPU hang/timeout/reset, uncorrectable ECC, or fatal hardware fault |

## Documentation

Full documentation lives under [`docs/`](docs/README.md):

- [Documentation Index](docs/README.md) — overview and quick navigation
- [CLI Reference](docs/cli.md) — commands, flags, `ms`/`s`/`m`/`h` timing, JSON schemas, exit codes, and recipes
- [Architecture](docs/architecture.md) — system diagram, `ctypes` ABI layer, deduplication, `importlib.resources` kernel loader, caching, and signal cleanup
- [Backends & Capability Matrix](docs/backends.md) — CUDA, HIP, Level Zero, Metal, and OpenCL runtimes, error mapping, and 5×20 support table
- [Workloads & Verification](docs/workloads.md) — all 20 workloads, SI unit accounting, verification rules, calibration, and VRAM guardrails
- [Packaging & Distribution](docs/packaging.md) — offline distro builds, FHS install paths, `sdist` verification, and environment overrides
- [Contributing](docs/contributing.md) — kernel authoring, `embed_binary verify`/`list`, coding standards, and Conventional Commits
- [Manual Page (`cross-gpu.1`)](docs/cross-gpu.1) — Unix `man(1)` page
- [Changelog](CHANGELOG.md) — release history

## License

MIT License. Copyright (c) 2026 CrossGPU Contributors. See [LICENSE](LICENSE).
