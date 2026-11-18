# cross-gpu

Cross-platform GPU stress-testing CLI targeting Linux and macOS, built exclusively with the **Python 3.10+ standard library** and **zero external runtime dependencies**.

## Features

- **5 native `ctypes` backends:** CUDA (`libcuda.so.1`), HIP (`libamdhip64.so`), oneAPI Level Zero (`libze_loader.so.1`), Apple Metal (`libobjc.A.dylib` + `Metal.framework`), and OpenCL (`libOpenCL.so.1` / `OpenCL.framework`).
- **Deterministic deduplication & priority:** Native backends (`priority=100`) take precedence over OpenCL (`priority=50`) with physical GPU deduplication by PCIe BDF, hardware UUID / macOS `registryID`, and fallback fingerprint.
- **1 Hz hardware telemetry:** Drift-compensated background sampling via NVML (`libnvidia-ml.so.1`), Linux DRM/hwmon `sysfs`, and unprivileged macOS `ioreg`.
- **Runtime kernel packer & compiler cache:** `python3 -m cross_gpu.embed_binary` with SHA-256 caching under OS-standard cache directories (`$XDG_CACHE_HOME/cross-gpu` / `~/.cache/cross-gpu` on Linux, `~/Library/Caches/cross-gpu` on macOS).
- **Deterministic exit codes:** `0` (`SUCCESS`), `1` (`USAGE_ERROR`), `2` (`INIT_ERROR`), `3` (`COMPUTE_ERROR`), `4` (`HARDWARE_ERROR`).

## Usage

```sh
cross-gpu list [--format text|json] [--json]
cross-gpu info [--format text|json] [--json]
cross-gpu run [--device all|N] [--test TEST|all] \
              [--duration TIME] [--warmup [TIME]] [--backend auto|PLUGIN_NAME] \
              [--format text|json] [--json]
```

## Offline & Distro Installation

```sh
# Offline PEP 517 build/install (Debian, Fedora, Arch, Nix, Homebrew)
pip install --no-build-isolation --no-deps .

# Or using the Makefile with PREFIX and DESTDIR
make check
make install PREFIX=/usr DESTDIR=/tmp/stage
```

## License

MIT License. See [LICENSE](LICENSE).
