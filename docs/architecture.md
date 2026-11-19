# `cross-gpu` Architecture

`cross-gpu` is a cross-platform GPU discovery, stress-testing, and telemetry CLI targeting **Linux** and **macOS**, built exclusively with the **Python 3.10+ standard library** and **zero external runtime dependencies**.

---

## 1. Package Layout

| Module / Directory | Responsibility |
|---|---|
| `cross_gpu/cli.py`, `cross_gpu/__main__.py` | Argument and duration parsing, `list`/`info`/`run` subcommands, text & JSON/NDJSON formatters |
| `cross_gpu/models.py` | Frozen/structured dataclasses for `DeviceCapabilities`, `DeviceInfo`, `TelemetrySample`, and run results |
| `cross_gpu/exit_codes.py` | Severity-ordered `ExitCode` (`0..4`), domain exceptions, and `aggregate_exit_codes` (`max`) |
| `cross_gpu/backends/` | 5 native `ctypes` plugins (`cuda`, `hip`, `levelzero`, `metal`, `opencl`) and backend registry |
| `cross_gpu/dedup.py` | Backend priority weighting and cross-backend physical GPU deduplication |
| `cross_gpu/kernels/` | Raw native GPU stress kernel source files (`18` compilation units × `5` backends = `90` files) |
| `cross_gpu/embed_binary.py` | `importlib.resources` raw kernel loader, registry verifier CLI, runtime compilers, and compile cache |
| `cross_gpu/workloads/` | Metadata registry, golden verification math, safe VRAM sizing, calibration, and 20 `WorkloadExecutor` classes |
| `cross_gpu/runner.py` | Multi-GPU execution lifecycle, warmup exclusion, verification checkpoints, signal handling, and reporting |
| `cross_gpu/telemetry/` | 1 Hz drift-compensated background telemetry sampler (NVML, Linux DRM/hwmon `sysfs`, macOS `ioreg`) |

---

## 2. Native Backends & Physical GPU Deduplication

- Every backend binds directly to native shared libraries or macOS frameworks using `ctypes` with explicit `argtypes`/`restype` declarations:
  - **CUDA** (`libcuda.so.1`, priority `100`)
  - **HIP** (`libamdhip64.so`, priority `100`)
  - **oneAPI Level Zero** (`libze_loader.so.1`, priority `100`)
  - **Apple Metal** (`libobjc.A.dylib` + `Metal.framework`, priority `100`)
  - **OpenCL** (`libOpenCL.so.1` / `OpenCL.framework`, priority `50`)
- `cross_gpu/dedup.py` merges duplicate backend handles representing the same physical GPU using normalized PCIe BDF (`DDDD:BB:DD.F`), hardware UUID / macOS IOKit `registryID`, and fallback device fingerprints.
- In `--backend auto` mode, each deduplicated GPU uses its highest-priority native backend (`100`) over OpenCL (`50`).

---

## 3. Raw Kernel Loading (`importlib.resources`) & Runtime Compilation

### Single-Source-of-Truth Kernel Resources

- Raw kernel source files under `cross_gpu/kernels/{cuda,hip,levelzero,metal,opencl}/` are bundled directly as package data in `pyproject.toml` and loaded at runtime using Python's standard library:

  ```python
  resource = importlib.resources.files("cross_gpu").joinpath("kernels", backend, filename)
  source_bytes = resource.read_bytes()
  ```

- `cross_gpu/embed_binary.py` maintains a lightweight `KERNEL_SPECS` metadata catalog mapping `(backend, workload)` to `filename`, `format`, `entry_point`, and `entry_points`—without duplicating kernel source text into Python modules or storing opaque binary blobs.
- `load_kernel_source(backend, workload)` decodes each file as strict UTF-8 and memoizes the `KernelSource` in memory (`clear_kernel_source_cache()` clears the memoization table for tests).
- Missing specs, missing resource files, or invalid UTF-8 raise `KernelNotFoundError`, which maps directly to `ExitCode.INIT_ERROR (2)`.
- `python3 -m cross_gpu.embed_binary verify` checks that every registered kernel resolves via `importlib.resources`, is non-empty valid UTF-8, declares every expected entry point symbol, and has no orphan files under `cross_gpu/kernels/`.

### Runtime Compilation Pipelines

`compile_kernel_runtime(...)` validates device capability flags before compilation (raising `UnsupportedKernelError` when a GPU lacks native support) and dispatches to the backend compiler pipeline:

| Backend | Compiler Pipeline | Output Format |
|---|---|---|
| `cuda` | `libnvrtc` (`ctypes`) | `cubin` (when `sm_*` supported) or `ptx` |
| `hip` | `libhiprtc` (`ctypes`) | `hsaco` |
| `levelzero` | `clang` + `llvm-spirv` or `ocloc` | `spirv` |
| `metal` | `xcrun -sdk macosx metal` + `metallib` (or MSL passthrough) | `metallib` or `msl_source` |
| `opencl` | Source passthrough + `CL_PROGRAM_BINARIES` extract/reload helpers | `opencl_source` / `opencl_binary` |

### Plain Tuple & Metadata-Validated Caching

`RuntimeBinaryCache` uses two cache tiers with plain Python keys and structured metadata:

1. **In-memory cache:** Keyed directly on the plain tuple:
   `(backend, target_arch, driver_version, workload, tuple(canonical_compile_flags), source_bytes)`.
2. **Optional disk cache:** Persists compiled `.bin` artifacts alongside a `.json` metadata sidecar under OS-standard cache directories:
   - Linux / POSIX: `$XDG_CACHE_HOME/cross-gpu` (if absolute), fallback `~/.cache/cross-gpu`
   - macOS: `~/Library/Caches/cross-gpu`
   - Override on any platform: `$CROSS_GPU_CACHE_DIR`
   - Filenames are deterministic, sanitized, human-readable stems containing package `__version__`, `backend`, `workload`, `target_arch`, `driver_version`, source `size_bytes`, and sanitized/compacted compile flags (staying well within POSIX `NAME_MAX`).
   - Disk reads validate `.bin` byte length and `.json` metadata fields (`package_version`, `backend`, `workload`, `target_arch`, `driver_version`, `source_size_bytes`, `input_compile_flags`).
   - Ad-hoc `source_override` compiles bypass the disk cache completely and stay strictly in memory.
   - If the cache directory is unavailable or read-only, `RuntimeBinaryCache` degrades cleanly to in-memory-only caching without raising.

---

## 4. Execution Lifecycle, Telemetry & Exit Codes

- **Workloads (`cross_gpu/workloads/`):** All 20 workloads calibrate batch size during `prepare()` to target `~100 ms` per dispatch (`50–200 ms` watchdog-safe window), run optional excluded warmup, execute timed steps with periodic golden verification, and clean up device/host buffers in `finally` blocks. See [workloads.md](workloads.md) for per-workload accounting formulas.
- **Telemetry (`cross_gpu/telemetry/`):** A drift-compensated 1 Hz sampler reads temperature, power, clocks, utilization, and VRAM usage via NVML (`libnvidia-ml.so.1`), Linux DRM/hwmon `sysfs`, or unprivileged macOS `ioreg`.
- **Exit codes (`cross_gpu/exit_codes.py`):** Aggregated across all targeted devices and workloads using `max(codes)`:
  - `0` (`SUCCESS`)
  - `1` (`USAGE_ERROR`)
  - `2` (`INIT_ERROR`)
  - `3` (`COMPUTE_ERROR`)
  - `4` (`HARDWARE_ERROR`)
