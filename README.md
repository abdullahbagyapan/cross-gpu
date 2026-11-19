# cross-gpu

Cross-platform GPU stress-testing CLI targeting Linux and macOS, built exclusively with the **Python 3.10+ standard library** and **zero external runtime dependencies**.

## Features

- **5 native `ctypes` backends:** CUDA (`libcuda.so.1`), HIP (`libamdhip64.so.6`), oneAPI Level Zero (`libze_loader.so.1`), Apple Metal (`libobjc.A.dylib` + `Metal.framework`), and OpenCL (`libOpenCL.so.1` / `OpenCL.framework`).
- **Deterministic deduplication & priority:** Native backends (`priority=100`) take precedence over OpenCL (`priority=50`) with physical GPU deduplication by PCIe BDF, hardware UUID / macOS `registryID`, and fallback fingerprint.
- **1 Hz hardware telemetry:** Drift-compensated background sampling via NVML (`libnvidia-ml.so.1`), Linux DRM/hwmon `sysfs`, and unprivileged macOS `ioreg`.
- **Packaged raw kernels & runtime compiler cache:** Raw kernel source files under `cross_gpu/kernels/` are bundled as package data, loaded directly via `importlib.resources`, verified with `python3 -m cross_gpu.embed_binary verify`, and cached using plain tuple keys in memory and metadata-validated files under OS-standard cache directories (`$XDG_CACHE_HOME/cross-gpu` / `~/.cache/cross-gpu` on Linux, `~/Library/Caches/cross-gpu` on macOS, or in-memory only with `CROSS_GPU_NO_CACHE=1`).
- **Operator-friendly CLI:** Strict `stdout`/`stderr` separation (valid JSON/NDJSON on `stdout` even with `-v`/`--verbose`), actionable backend-discovery & skip reasons, `difflib` typo hints, `text`-mode workload progress/tallies, and Bash/Zsh/Fish shell completions.
- **Deterministic exit codes:** `0` (`SUCCESS`), `1` (`USAGE_ERROR`), `2` (`INIT_ERROR`), `3` (`COMPUTE_ERROR`), `4` (`HARDWARE_ERROR`).

## Usage

```sh
cross-gpu list [--format text|json] [--json] [-v|--verbose]
cross-gpu info [--format text|json] [--json] [-v|--verbose]
cross-gpu run [--device all|N] [--test TEST|all] \
              [--duration TIME] [--warmup [TIME]] [--backend auto|PLUGIN_NAME] \
              [--format text|json] [--json] [-v|--verbose]
```

## Offline & Distro Installation

```sh
# Offline PEP 517 build/install (Debian, Fedora, Arch, Nix, Homebrew)
pip install --no-build-isolation --no-deps .

# Or using the Makefile with PREFIX and DESTDIR (installs man page + bash/zsh/fish completions)
make check
make install PREFIX=/usr DESTDIR=/tmp/stage
```

## Documentation

- [CLI Reference](docs/cli.md) — flags, `--verbose` stream hygiene, `difflib` typo hints, environment variables (`XDG_CACHE_HOME`, `CROSS_GPU_NO_CACHE`), and shell completions
- [Architecture](docs/architecture.md) — framework modules, native backends, `importlib.resources` kernel loading, and runtime compile caching
- [Workloads](docs/workloads.md) — all 20 stress workloads, fixed SI unit rates, VRAM guardrails, and golden verification
- [Packaging](docs/packaging.md) — offline distro builds, `sdist` verification, FHS install paths, and read-only safety
- [Contributing](docs/contributing.md) — developer workflow, editing raw kernels, and testing
- [Changelog](CHANGELOG.md) — release history and notable changes

## License

MIT License. See [LICENSE](LICENSE).
