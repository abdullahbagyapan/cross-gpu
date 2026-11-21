# CrossGPU Documentation

CrossGPU is a cross-platform GPU discovery and stress-testing CLI targeting **Linux** (`x86_64`, `aarch64`) and **macOS** (`arm64`, `x86_64`), implemented exclusively with the **Python 3.10+ standard library** and **zero external runtime dependencies**.

## Documentation Index

| Document | Scope |
|---|---|
| [CLI Reference](cli.md) | `list`, `info`, and `run` options, `ms`/`s`/`m`/`h` durations, `--warmup` rules, JSON schemas, `text` mode, `stdout`/`stderr` separation, `-v`/`--verbose`, `difflib` hints, exit codes, and operational recipes |
| [Architecture](architecture.md) | System flow diagram, zero-dependency `ctypes` ABI strategy, backend priority (`100` vs. `50`), 3-tier physical GPU deduplication, `importlib.resources` raw kernel loading, runtime compilation, caching, and signal-safe teardown |
| [Backends & Capabilities](backends.md) | CUDA, HIP, Level Zero, Metal, and OpenCL library SONAMEs/frameworks, compiler pipelines, error-code classification (`INIT_ERROR` vs. `HARDWARE_ERROR`), and the complete 5-backend × 20-workload capability matrix |
| [Workloads & Verification](workloads.md) | All 20 workloads across scalar integer, scalar FP/ML, tensor MMA, and memory/bandwidth categories, fixed SI unit math (`GOP/s`, `GFLOP/s`, `GB/s`), golden verification semantics, calibration, and VRAM guardrails |
| [Packaging & Distribution](packaging.md) | Offline PEP 517/621 builds for Debian, Fedora, Arch, Nix, and Homebrew, `Makefile` FHS targets (`PREFIX`/`DESTDIR`), `sdist` self-containment, XDG/macOS cache paths, and read-only install safety |
| [Contributing](contributing.md) | Editing raw GPU kernels (`embed_binary verify`/`list`), zero fake emulation rule, coding/SPDX standards, offline & hardware test suites, and Conventional Commits |
| [Unix Manual Page](cross-gpu.1) | Standard `man(1)` troff page installed to `$(PREFIX)/share/man/man1/cross-gpu.1` |
