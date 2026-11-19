# `cross-gpu` Architecture

`cross-gpu` is a cross-platform GPU discovery, hardware telemetry, and stress-testing CLI targeting **Linux** (`x86_64`, `aarch64`) and **macOS** (`arm64`, `x86_64`), written exclusively with the **Python 3.10+ standard library** and **zero external Python dependencies**.

---

## 1. High-Level System Flow

```text
                       cross-gpu CLI (cross_gpu/cli.py)
               [list | info | run]  stdout=data  stderr=diagnostics
                                     |
          +--------------------------+--------------------------+
          v                                                     v
backends/registry.py                                  workloads/registry.py
  probe 5 ctypes plugins                               20 WorkloadSpec entries
  (cuda, hip, levelzero, metal, opencl)                20 WorkloadExecutor impls
          |                                                     |
          v                                                     |
       dedup.py                                                 |
  3-tier physical GPU merge                                     |
  priority: native (100) > opencl (50)                          |
          |                                                     |
          +--------------------------+--------------------------+
                                     v
                         runner.py (StressRunner)
               multi-GPU worker threads + SIGINT/SIGTERM trap
                      |                               |
                      v                               v
               embed_binary.py                   telemetry/
       importlib.resources raw loader        1 Hz TelemetrySampler
       runtime compilers (NVRTC/hipRTC/      (NVML, Linux DRM/hwmon,
        clang+llvm-spirv/ocloc/xcrun)         macOS ioreg) + sanitizer
       plain-tuple memory + XDG disk cache
```

---

## 2. Package Layout

| Path | Responsibility |
|---|---|
| `cross_gpu/cli.py`, `cross_gpu/__main__.py` | `DeterministicArgumentParser`, `ms`/`s`/`m`/`h` duration parser, `difflib` typo hints, `text`/JSON/NDJSON formatters, `-v`/`--verbose` logger, strict `stdout`/`stderr` routing |
| `cross_gpu/models.py` | Dataclasses for `DeviceCapabilities`, `BackendDeviceInfo`, `BackendBinding`, `GPUDevice`, `TelemetrySample`, `StepMetrics`, `VerificationResult`, and `WorkloadResult` |
| `cross_gpu/exit_codes.py` | Severity-ordered `ExitCode` (`0..4`), typed exceptions, and `aggregate_exit_codes` (`max`) |
| `cross_gpu/backends/` | `BackendPlugin` ABC + `ctypes` ABI helpers (`base.py`), plugin discovery (`registry.py`), and 5 native plugins (`cuda`, `hip`, `levelzero`, `metal`, `opencl`) |
| `cross_gpu/dedup.py` | Backend priority ordering and 3-tier physical GPU deduplication |
| `cross_gpu/kernels/` | 90 raw standalone native GPU stress kernels (`18` compilation units × `5` backends) bundled as package data |
| `cross_gpu/embed_binary.py` | `importlib.resources` raw kernel loader, `verify`/`list` CLI, 5 runtime compiler adapters, and 2-tier compile cache |
| `cross_gpu/workloads/` | 20-workload registry, golden verification math, safe VRAM sizing, dynamic batch calibration, and 20 concrete `WorkloadExecutor` classes |
| `cross_gpu/runner.py` | Multi-GPU execution coordinator, warmup exclusion, periodic verification checkpoints, and signal-safe cleanup |
| `cross_gpu/telemetry/` | 1 Hz drift-compensated `TelemetrySampler` (`__init__.py`) and platform readers + sensor sanitization (`readers.py`) |

---

## 3. Zero-Dependency `ctypes` Strategy

- **Versioned SONAMEs & Standard Frameworks:** Shared libraries are resolved dynamically via `ctypes.CDLL` using standard versioned SONAMEs (`libcuda.so.1`, `libnvrtc.so.12`, `libamdhip64.so.6`, `libhiprtc.so.6`, `libze_loader.so.1`, `libOpenCL.so.1`, `libnvidia-ml.so.1`) and standard macOS paths (`/usr/lib/libobjc.A.dylib`, `/System/Library/Frameworks/Metal.framework/Metal`, `/System/Library/Frameworks/OpenCL.framework/OpenCL`). No hardcoded distro library paths are used on Linux, so `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` work naturally.
- **Explicit `argtypes` / `restype` ABI Safety:** Every C symbol is bound via `bind_c_symbol` or `bind_optional_c_symbol` (and `_bind_objc_msg` with `ctypes.CFUNCTYPE` for Objective-C `objc_msgSend` on Apple Silicon `arm64` and Intel `x86_64`). Explicit pointer widths (`c_void_p`, `c_size_t`, `c_uint64`, `c_int32`) prevent 64-bit pointer truncation and variadic calling-convention mismatches across `x86_64` and `arm64`.
- **Isolated Plugin Failure Handling:** `enumerate_all_backend_devices_with_diagnostics()` probes every plugin independently inside guarded `try`/`except` blocks. A missing runtime library, failed driver init, or Linux `/dev/dri/renderD*` / `/dev/kfd` permission error records a concise diagnostic reason (`backend -> reason`) without impacting healthy plugins.

---

## 4. Backend Priority & 3-Tier Physical GPU Deduplication

When a GPU is exposed by multiple runtimes (for example, CUDA + OpenCL on NVIDIA, HIP + OpenCL on AMD, Level Zero + OpenCL on Intel, or Metal + OpenCL on Apple Silicon), `cross_gpu/dedup.py` merges them into a single canonical `GPUDevice`:

1. **Priority Weighting:**
   - Native backends (`cuda`, `hip`, `levelzero`, `metal`): `priority = 100`
   - Cross-vendor fallback (`opencl`): `priority = 50`
   - Inputs are sorted descending by `(-priority, backend, backend_device_index)` so the highest-priority native backend becomes the primary binding (`active_backend`) while lower-priority backends remain selectable via `--backend <name>`.
2. **3-Tier Physical Identity Hierarchy:**
   - **Tier 1 — Normalized PCIe BDF:** Canonicalized lowercase `DDDD:BB:DD.F` (e.g., `0000:01:00.0`). Strongest match on Linux PCIe GPUs.
   - **Tier 2 — Hardware UUID / macOS IOKit `registryID`:** Normalized lowercase 32-hex-digit hyphenated UUID (`GPU-` prefix stripped, all-zero UUIDs rejected) or 64-bit IOKit `registryID`.
   - **Tier 3 — Ordinal Fallback Hardware Fingerprint:** `(canonical_vendor, normalized_device_name, memory_bucket)` used only when a backend already present in the group is not duplicated, neither side has conflicting strong IDs, and at least one side lacks the strong ID present on the other (for example, merging macOS Metal `registryID` with macOS OpenCL on Apple Silicon).
3. **Metadata Backfill:** Missing secondary attributes (e.g., `free_memory_bytes`, `compute_units`, `pci_bdf`, `uuid`) on the primary descriptor are backfilled from lower-priority bindings without downgrading native capabilities.

---

## 5. `embed_binary.py` Raw Kernel & Runtime Compilation Lifecycle

### Single-Source-of-Truth Kernel Loading (`importlib.resources`)

- Raw kernel source files under `cross_gpu/kernels/{cuda,hip,levelzero,metal,opencl}/` are bundled directly as package data in `pyproject.toml` and loaded via Python's standard library:

  ```python
  resource = importlib.resources.files("cross_gpu").joinpath("kernels", backend, filename)
  source_bytes = resource.read_bytes()
  ```

- `KERNEL_SPECS` stores lightweight metadata (`backend`, `workload`, `filename`, `format`, `entry_point`, `entry_points`) for all 90 compilation units—with **no** duplicated source blobs or digest hashes inside Python source files.
- `load_kernel_source(backend, workload)` decodes each kernel as strict UTF-8 and memoizes the `KernelSource` in memory (`clear_kernel_source_cache()` resets the cache in tests). Missing specs, missing resource files, or invalid UTF-8 raise `KernelNotFoundError` (`ExitCode.INIT_ERROR = 2`).
- `python3 -m cross_gpu.embed_binary verify` validates every registered resource resolves cleanly, is non-empty UTF-8, declares every expected entry-point symbol, and detects orphan files under `cross_gpu/kernels/`.

### Per-Backend Runtime Compilation Flows

Before compilation, `compile_kernel_runtime(...)` checks `DeviceCapabilities` (raising `UnsupportedKernelError` immediately if unsupported) and prepends `-DCROSS_GPU_HAS_<CAP>=<0|1>` capability macros:

| Backend | Runtime Compiler Pipeline | Artifact Format |
|---|---|---|
| `cuda` | `libnvrtc` (`ctypes`) with `--gpu-architecture=sm_XY` / `compute_XY` | `cubin` (when `nvrtcGetCUBIN` available for `sm_*`) or `ptx` |
| `hip` | `libhiprtc` (`ctypes`) with `--offload-arch=gfx...` | `hsaco` |
| `levelzero` | `clang -cc1 -triple spir64-unknown-unknown` + `llvm-spirv`, or `ocloc compile -spirv_only` | `spirv` |
| `metal` | `xcrun -sdk macosx metal` + `metallib`, falling back cleanly to MSL source passthrough when Xcode Metal CLI utilities are unavailable | `metallib` or `msl_source` |
| `opencl` | Source passthrough (`clBuildProgram`), plus `CL_PROGRAM_BINARIES` extract/reload helpers (`clCreateProgramWithBinary`) | `opencl_source` / `opencl_binary` |

### 2-Tier Plain Tuple & Metadata-Validated Cache

1. **In-Memory Cache:** Keyed directly on the plain Python tuple:
   `(backend, target_arch, driver_version, workload, tuple(canonical_compile_flags), source_bytes)`.
2. **Optional Disk Cache:** Stores `.bin` artifacts alongside a `.json` metadata sidecar under:
   - `CROSS_GPU_NO_CACHE=1|true|yes`: disk cache completely bypassed (in-memory only).
   - `$CROSS_GPU_CACHE_DIR`: explicit path override.
   - `$XDG_CACHE_HOME/cross-gpu`: honored when set to an absolute path on both Linux and macOS.
   - Defaults: `~/.cache/cross-gpu` (Linux/POSIX), `~/Library/Caches/cross-gpu` (macOS).
   - Filename stems are deterministic, sanitized, `NAME_MAX`-safe human-readable keys encoding `__version__`, `backend`, `workload`, `target_arch`, `driver_version`, `size_bytes`, and compacted/sanitized compile flags.
   - Disk hits verify `.bin` byte size and exact `.json` metadata (`package_version`, `backend`, `workload`, `target_arch`, `driver_version`, `source_size_bytes`, `input_compile_flags`). Ad-hoc `source_override` compiles bypass disk caching completely; read-only or unavailable cache paths degrade cleanly to in-memory-only caching.

---

## 6. Concurrency, Signals & Clean Resource Teardown

- **Multi-GPU Execution:** When targeting multiple GPUs (`--device all`), `StressRunner.run()` spawns one worker thread per physical GPU so all devices stress concurrently, synchronizing event emission through a shared lock (`_emit_lock`).
- **Signal Handling (`SIGINT` / `SIGTERM`):** When invoked on the main thread, `StressRunner` installs signal handlers that set a shared `threading.Event` (`_stop_event`) and restores previous handlers in a `finally` block. Worker loops check `_stop_event` between calibrated dispatches (`50–200 ms`, or shorter for sub-second `--duration`), stop `TelemetrySampler` immediately via `Event.wait()`, and exit promptly.
- **Deterministic Teardown:** Every `WorkloadExecutor` releases device buffers, pinned host buffers, and loaded kernel modules inside idempotent `cleanup()` called from a `finally` block; every device worker stops its `TelemetrySampler` and calls `plugin.destroy_context(backend_ctx)` inside its outer `finally` block, even after compute errors, hardware faults, or signals.
