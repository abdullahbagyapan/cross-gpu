# Testing & Docker Emulation

CrossGPU separates verification into three layers so every contributor can run fast offline checks locally while exercising native `ctypes` ABIs, multi-GPU deduplication, foreign CPU endianness, distro packaging, and locked-down containers via Docker—without inventing fake GPU compute metrics in production code.

---

## 1. Offline Unit Test Suite (`make check`)

All offline tests use only the Python 3.10+ standard library (`unittest`) and require no GPU hardware, drivers, or network access:

```sh
make check
bash -n completions/cross-gpu.bash
```

`make check` runs:

1. `python3 -m compileall -q src tests` — bytecode syntax check across all sources and tests.
2. `python3 -m unittest discover -s tests -v` — offline unit suite (`72` offline tests pass; `4` opt-in real-GPU tests skip unless `CROSS_GPU_HW_TESTS=1` is set).
3. `python3 -m src.embed_binary verify` — validates all `90` packaged raw kernel source files (`18` workloads × `5` backends) via `importlib.resources`, checking UTF-8 encoding, declared entry points, and absence of orphan files.

### Test Modules (`tests/`)

| Module | Coverage |
|---|---|
| `tests/test_cli.py` | `list`/`info`/`run` CLI parsing, `ms`/`s`/`m`/`h` durations, `--warmup`, `--no-banner` and color guards, single-document JSON, `stdout`/`stderr` separation under `-v`, `difflib` typo hints, shell completion coverage, and subprocess entry points |
| `tests/test_dedup.py` | PCIe BDF / UUID / macOS `registryID` normalization, 3-tier physical GPU deduplication, native (`100`) vs. OpenCL (`50`) priority resolution, and multi-GPU ordinal pairing |
| `tests/test_embed_binary.py` | `importlib.resources` raw kernel loading, `verify`/`list` CLI, tuple in-memory cache keys, metadata-validated disk cache, `CROSS_GPU_NO_CACHE` / `XDG_CACHE_HOME` overrides, kernel headers/`#error "UNSUPPORTED:"` gates, plus 3 opt-in live-GPU compile/golden tests |
| `tests/test_exit_codes.py` | Exact numeric exit codes (`0`–`4`), severity ordering, exception mapping, and `max(codes)` aggregation |
| `tests/test_runner.py` | Multi-GPU concurrency, warmup throughput exclusion, sub-second `200ms` termination, missing device/backend `INIT_ERROR`, and severity aggregation |
| `tests/test_workloads.py` | All 20 workload executors end-to-end on mock backends, fixed SI units (`GOP/s`, `GFLOP/s`, `GB/s`), golden math/decoders, fault injection (`COMPUTE_ERROR=3`), dynamic batch calibration, safe VRAM sizing, step-down allocation, and `SIGINT` cleanup |
| `tests/test_audit_regressions.py` | Versioned-before-unversioned SONAME ordering, ROCm-stable HIP attribute enumeration and opaque `gcnArchName` scan, Level Zero fault codes and SPIR-V `-D` filtering, poisoned-context cleanup, OpenCL `size_t` queries, Metal blit/compute exception cleanup, read-only `memoryview` handling, flat `src/` layout, and SPDX headers |
| `tests/test_hardware_smoke.py` | Opt-in real-GPU end-to-end smoke test across all supported workloads (`CROSS_GPU_HW_TESTS=1`) |

---

## 2. Opt-In Real-GPU Hardware Tests

On machines equipped with a real NVIDIA, AMD, Intel, or Apple GPU, enable hardware tests with `CROSS_GPU_HW_TESTS=1`:

```sh
# Short end-to-end smoke test across every workload supported by detected GPUs
CROSS_GPU_HW_TESTS=1 python3 -m unittest tests.test_hardware_smoke -v

# Live kernel compilation and golden execution tests (Metal, OpenCL, NVRTC/hipRTC)
CROSS_GPU_HW_TESTS=1 python3 -m unittest tests.test_embed_binary.TestPackagedKernels -v

# Full suite including hardware tests
CROSS_GPU_HW_TESTS=1 python3 -m unittest discover -s tests -v
```

---

## 3. Docker-Only Emulation Harness (`scripts/emulate.sh`)

`scripts/emulate.sh` is a **testing-only** harness (not shipped in sdist/wheel packages) that tests all **5 CPU architectures**, all **5 GPU backends**, **4 Linux distributions**, and **adversarial container constraints** using **Docker as the sole host dependency**.

### Design Rules

- **Zero host toolchain assumptions:** No host Python, C compiler, `make`, or QEMU installation is required. Every build, C11 shim compilation, and test runs inside containers.
- **Read-only repository mount:** The repository is mounted strictly as `/repo:ro`; containers copy into `/tmp/work` and never modify your working tree.
- **Automatic QEMU `binfmt_misc` setup:** Missing foreign-architecture handlers (`amd64`, `arm64`, `riscv64`, `ppc64le`, `s390x`) are registered on demand via `tonistiigi/binfmt` without disturbing existing native or Rosetta handlers (disable with `--skip-binfmt`).
- **Bash 3.2+ compatible:** Runs cleanly on macOS default `/bin/bash` as well as modern Linux Bash, with `INT`/`TERM` traps that stop active containers and clean temporary Docker volumes.

### Prerequisites

Ensure a Docker daemon is reachable:

```sh
docker info
```

On macOS, any standard Docker provider works (**Docker Desktop**, **OrbStack**, or **Colima** via `brew install colima docker && colima start`).

### CLI Options

```sh
./scripts/emulate.sh [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--list` | — | Print all 17 emulation targets grouped by suite |
| `--suite SUITE[,SUITE...]` | `all` | Select one or more suites: `arch`, `backends`, `distros`, `adversarial`, `all` |
| `--target NAME[,NAME...]` | — | Select specific target(s) (repeatable or comma-separated) |
| `--distro-platform PLAT` | `linux/amd64` | Docker `--platform` for `backends`, `distros`, and `adversarial` targets |
| `--duration TIME` | `150ms` | Timed CLI workload duration passed to `cross-gpu run` inside containers |
| `--log-dir DIR` | `~/.cache/cross-gpu-emulation/<timestamp>` | Output directory for per-step logs and `summary.md` |
| `--skip-binfmt` | off | Skip `tonistiigi/binfmt` multi-arch registration (e.g., when already configured or on rootless Docker) |
| `--fail-fast` | off | Stop immediately after the first failing target |
| `--dry-run` | off | Print resolved targets and settings without launching containers |
| `-v`, `--verbose` | off | Print container `stderr` live in addition to log files |
| `-h`, `--help` | — | Show help message |

### Suites & Targets (17 Total)

#### `arch` — Foreign CPU Architectures & Endianness (5 targets)

Runs in `debian:trixie-slim` across 5 platforms, verifying `platform.machine()`, `sys.byteorder` (including big-endian `s390x`), `compileall`, `embed_binary verify` (`90` kernels), all `76` unit tests, strict JSON + empty `stderr` checks, `-v` verbose separation, and offline PEP 517 wheel/sdist builds:

| Target | Docker Platform | Arch / Endianness |
|---|---|---|
| `arch-amd64` | `linux/amd64` | `x86_64` (little-endian) |
| `arch-arm64` | `linux/arm64` | `aarch64` (little-endian) |
| `arch-riscv64` | `linux/riscv64` | `riscv64` (little-endian) |
| `arch-ppc64le` | `linux/ppc64le` | `ppc64le` (little-endian) |
| `arch-s390x` | `linux/s390x` | `s390x` (big-endian) |

#### `backends` — All 5 GPU Backends, Multi-GPU Dedup & PoCL LLVM (7 targets)

| Target | Libraries / Runtime | What It Verifies |
|---|---|---|
| `backend-cuda` | `libcuda.so.1`, `libnvrtc.so.12` | Versioned SONAME loading (no `.so` symlinks), 2-GPU `list`/`info` JSON, NVRTC compile + module load/unload, 2-GPU H2D/D2H pass (`rc=0`), bitflip (`rc=3`), `CUDA_ERROR_ECC_UNCORRECTABLE` (`rc=4`), destructor leak checks |
| `backend-hip` | `libamdhip64.so.6`, `libhiprtc.so.6` | Versioned SONAME loading, ROCm-stable attribute APIs, opaque `gcnArchName` (`gfx1100`) scan, HIPRTC compile, 2-GPU pass (`rc=0`), bitflip (`rc=3`), `hipErrorECCNotCorrectable` (`rc=4`), leak checks |
| `backend-levelzero` | `libze_loader.so.1` | Versioned SONAME loading, `ze_device_properties_t`/`ze_pci_ext_properties_t` ABI layout, SPIR-V module/kernel lifecycle with `-D` flag filtering, 2-GPU pass (`rc=0`), bitflip (`rc=3`), `ZE_RESULT_ERROR_DEVICE_LOST` (`rc=4`), leak checks |
| `backend-opencl` | `libOpenCL.so.1` | Versioned SONAME loading, `cl_khr_pci_bus_info`, `clEnqueueMapBuffer`/`Unmap` pinned host path, program build & `CL_PROGRAM_BINARIES` extraction, 2-GPU pass (`rc=0`), bitflip (`rc=3`), `CL_DEVICE_NOT_AVAILABLE` (`rc=4`), leak checks |
| `backend-metal` | `libfake_metal.so` (pure C11) | Production Linux `macOS only` guard (`is_available() == False`), full `MetalBackend` Objective-C `objc_msgSend` bridge, `metallib` + `msl_source` module loads, 24-byte `_MTLSize` struct-by-value dispatch, 2-GPU blit pass (`rc=0`), GPU 0 bitflip (`rc=3`), GPU 1 `MTLCommandBufferErrorTimeout` (`rc=4`), atomic object/encoder/autorelease-pool leak checks |
| `backend-multigpu` | All 6 Linux SONAMEs together | Heterogeneous 9-handle → 5-GPU physical dedup (`cuda:2`, `hip:1`, `levelzero:1`, `opencl:1`) with native priority `100 > 50`, homogeneous 5-GPU CUDA rig, concurrent 5-GPU H2D/D2H pass (`10/10`), and isolated faults across HIP (`rc=3`), Level Zero (`rc=4`), and OpenCL (`rc=3`) with zero leaks |
| `backend-pocl` | Real `ocl-icd-libopencl1` + `pocl-opencl-icd` | Production no-CPU-fallback gate (`list` returns `0` GPUs, `run --backend opencl` returns `INIT_ERROR=2`), compilation of all `36` packaged `.cl` kernels (`src/kernels/{opencl,levelzero}/*.cl`) verifying baseline kernels compile (`rc=0`) and `#error "UNSUPPORTED:"` gates trigger cleanly, plus live PoCL execution of `compute_int32` and `compute_fp32` verified against host golden math |

#### `distros` — Multi-Distro Libc, Python & FHS Packaging (4 targets)

| Target | Container Image | Notes |
|---|---|---|
| `distro-alpine` | `alpine:latest` | musl libc, `compileall`, `76` unit tests, `embed_binary verify`, `bash -n completions/cross-gpu.bash`, offline wheel/sdist build, wheel layout check, and `make install DESTDIR=/tmp/stage PREFIX=/usr` + staged FHS tree verification |
| `distro-debian` | `debian:trixie-slim` | glibc + PEP 668 `DESTDIR` staging verification |
| `distro-fedora` | `fedora:latest` | glibc + `dnf` toolchain + FHS staging verification |
| `distro-arch` | `archlinux:latest` | glibc + `pacman --disable-sandbox` + FHS staging verification |

#### `adversarial` — Locked-Down Container Constraints (1 target)

Builds a pure-Python wheel in a stage-1 `python:3.12-slim` container into a temporary Docker volume, then runs a stage-2 container with:

- `--network=none` (strictly offline)
- `--read-only` (immutable root filesystem)
- `--tmpfs /tmp:rw,noexec,nosuid,size=256m` (writable but non-executable `/tmp`)
- `--user 65534:65534` (`nobody`, non-root)
- No `/dev/dri` or GPU device nodes

Verifies offline wheel install to `--target /tmp/site`, `--version`, `embed_binary verify`, `list --json` and `info --json` with empty `stderr`, `CROSS_GPU_NO_CACHE=1`, clean `cross_gpu/` package layout (no `src/` or `telemetry/`), and actively proves `/tmp` rejects direct script execution (`noexec`).

### Usage Recipes

```sh
# List all targets
./scripts/emulate.sh --list

# Preview resolved targets without running Docker
./scripts/emulate.sh --suite all --dry-run

# Run all GPU backend targets (fastest feedback loop for backend/ABI changes)
./scripts/emulate.sh --suite backends

# On Apple Silicon (arm64), run Linux containers natively without amd64 QEMU overhead
./scripts/emulate.sh --suite backends,distros,adversarial --distro-platform linux/arm64

# Run specific targets with live stderr and fail-fast
./scripts/emulate.sh --target backend-metal,backend-multigpu,arch-s390x --fail-fast -v

# Run full emulation matrix and save logs to a custom directory
./scripts/emulate.sh --suite all --log-dir ./emulation-logs
```

---

## 4. C11 Fake-Driver ABI Shims (`scripts/shims/`)

The C11 shims under `scripts/shims/` are compiled inside Docker containers with `-std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC`:

| File | Built As | Purpose |
|---|---|---|
| `scripts/shims/shim_common.h` | header | Shared C11 atomics, env parsing (`FAKE_GPU_VRAM_MIB`, fault targeting, leak check flag) |
| `scripts/shims/fake_cuda.c` | `libcuda.so.1` | CUDA Driver API shim (thread-local device context, atomic leak checks, bitflip & `CUDA_ERROR_ECC_UNCORRECTABLE`) |
| `scripts/shims/fake_nvrtc.c` | `libnvrtc.so.12` | NVRTC runtime compiler shim emitting valid PTX with program leak tracking |
| `scripts/shims/fake_hip.c` | `libamdhip64.so.6` | HIP runtime shim (ROCm-stable attribute IDs, opaque `gfx1100:xnack-` property buffer, atomic leak checks, bitflip & `hipErrorECCNotCorrectable`) |
| `scripts/shims/fake_hiprtc.c` | `libhiprtc.so.6` | hipRTC runtime compiler shim emitting ELF/HSACO header with program leak tracking |
| `scripts/shims/fake_levelzero.c` | `libze_loader.so.1` | Level Zero C API shim (exact `ze_*_properties_t` struct layouts, atomic device pointer tracking, SPIR-V module/kernel lifecycle, bitflip & `ZE_RESULT_ERROR_DEVICE_LOST`) |
| `scripts/shims/fake_opencl.c` | `libOpenCL.so.1` | OpenCL C API shim (`cl_khr_pci_bus_info`, `clEnqueueMapBuffer`/`Unmap`, program binaries, bitflip & `CL_DEVICE_NOT_AVAILABLE`) |
| `scripts/shims/fake_metal.c` | `libfake_metal.so` | Portable C11 Objective-C runtime shim exporting `objc_msgSend` (non-variadic ABI-safe dispatcher), `MTLCopyAllDevices`, `dispatch_data_create`/`release`, per-thread autorelease pools, 24-byte `_MTLSize` dispatch, D2H blit bitflip, and `MTLCommandBufferErrorTimeout` |

### Shim Environment Controls

| Variable | Default | Meaning |
|---|---|---|
| `FAKE_CUDA_COUNT` / `FAKE_HIP_COUNT` / `FAKE_L0_COUNT` / `FAKE_CL_COUNT` / `FAKE_METAL_COUNT` | `2` / `1` / `1` / `5` / `2` | Number of simulated GPU handles exposed by each shim (`0`–`16`) |
| `FAKE_GPU_VRAM_MIB` | `256` (`64` in harness) | Simulated VRAM per GPU in MiB (keeps concurrent 5-GPU host RAM usage small) |
| `FAKE_GPU_FAULT` | unset | Fault mode to inject: `bitflip` (`COMPUTE_ERROR=3`), `ecc_ue` (`HARDWARE_ERROR=4`), `device_lost` (`HARDWARE_ERROR=4`), or `gpu_timeout` (`HARDWARE_ERROR=4`, Metal) |
| `FAKE_GPU_FAULT_BACKEND` | unset (all) | Restrict fault injection to one backend (`cuda`, `hip`, `levelzero`, `opencl`, `metal`) |
| `FAKE_GPU_FAULT_DEVICE` | `-1` (all) | Restrict fault injection to a specific backend device ordinal |
| `FAKE_GPU_LEAK_CHECK` | `0` | When `1`, each shim's `__attribute__((destructor))` asserts all allocated buffers, contexts, queues, modules, encoders, and autorelease pools reached `0`; any leak aborts with exit code **`99`** |

### Scope Boundary: Emulation vs. Real Hardware

- **Verified by Docker emulation:** `ctypes` symbol/struct/pointer ABIs, versioned SONAME resolution, multi-backend physical GPU deduplication, H2D/D2H stamped-payload CRC32 verification, severity-ordered exit codes (`0`–`4`), resource-leak-free teardown under faults, big-endian host safety (`s390x`), OpenCL `.cl` compilation & golden execution on PoCL, multi-distro FHS installs, and locked-down container safety.
- **Requires real GPU hardware (`CROSS_GPU_HW_TESTS=1`):** Executing native CUDA `.cu` (NVRTC), HIP `.hip` (hipRTC), Level Zero SPIR-V (`clang` + `llvm-spirv` / `ocloc`), and Apple Metal `.metal` (`xcrun metal`) compute/tensor kernels on physical silicon, verifying GPU memory bandwidth/thermal behavior, and testing vendor-specific driver edge cases.

---

## 5. Logs, Exit Codes & Reports

Each run writes structured diagnostics to `--log-dir` (default `~/.cache/cross-gpu-emulation/<timestamp>/`):

```text
~/.cache/cross-gpu-emulation/20261123-163000/
├── summary.md                       # Markdown table + expanded <details open> failure blocks
├── 00_binfmt/                       # tonistiigi/binfmt probe/install logs (unless --skip-binfmt)
├── backend-cuda/
│   ├── _container.{cmd,rc,stdout.log,stderr.log}
│   ├── 01_build_fake_cuda.{cmd,expected_rc,rc,stdout.log,stderr.log}
│   └── ...
└── ...
```

Non-zero exit codes are decoded automatically in both terminal output and `summary.md`:

| Exit Code | Decoded Meaning |
|---|---|
| `99` | `SHIM_RESOURCE_LEAK` (destructor detected leaked GPU/ObjC allocation or unended encoder) |
| `124` | `TIMEOUT` |
| `132` | `SIGILL` |
| `134` | `SIGABRT` |
| `136` | `SIGFPE` |
| `137` | `SIGKILL/OOM` |
| `139` | `SIGSEGV` |
| `143` | `SIGTERM` |

---

## 6. GitHub Actions Workflows

CrossGPU maintains two testing workflows under `.github/workflows/`:

1. **CI (`.github/workflows/ci.yml`):** Runs on every `push` and `pull_request` across `ubuntu-latest` and `macos-latest` × Python `3.10`–`3.13`. Executes `compileall`, `unittest`, `embed_binary verify`, `bash -n completions/cross-gpu.bash`, sdist tarball content/extraction verification, and clean isolated venv installation checks.
2. **Environment Emulation (`.github/workflows/emulation.yml`):** Runs on manual `workflow_dispatch`, weekly on Sundays (`05:00 UTC`), on `v*.*.*` release tags, and on pull requests modifying `.github/workflows/emulation.yml`, `scripts/emulate.sh`, `scripts/shims/**`, or `Makefile`. Runs all `17` `./scripts/emulate.sh` targets in parallel across 4 matrix jobs (`backends-emulation`, `arch-emulation` with `docker/setup-qemu-action@v3`, `distro-emulation`, `adversarial-emulation`), emits `::error::` annotations on failures, appends `summary.md` directly to `$GITHUB_STEP_SUMMARY`, and uploads per-target log artifacts for 14 days.
