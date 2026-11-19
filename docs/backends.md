# Backends & Hardware Capability Matrix

CrossGPU ships 5 zero-dependency `ctypes` backend plugins under `src/backends/`. Unsupported datatypes or matrix instructions are **never** silently emulated with scalar loops—each kernel enforces explicit capability and compile-time `#error "UNSUPPORTED: ..."` hardware gates.

---

## 1. Per-Backend Runtime & Error Classification

### CUDA (`cuda`, priority `100`)

- **Runtime Libraries:** `libcuda.so.1`, fallback `libcuda.so` (CUDA Driver API).
- **Runtime Compiler:** `libnvrtc.so.12`, fallback `libnvrtc.so`, `libnvrtc.so.11.2` (`ctypes`). Emits native `cubin` when `nvrtcGetCUBIN` is available for `sm_XY`, falling back to `ptx`.
- **Hardware Identification:** PCIe BDF (`cuDeviceGetPCIBusId` / domain+bus+device attributes) and 16-byte GPU UUID (`cuDeviceGetUuid_v2`).
- **Error Classification:**
  - `HARDWARE_ERROR (4)`: `CUDA_ERROR_ECC_UNCORRECTABLE` (`214`), `CUDA_ERROR_NVLINK_UNCORRECTABLE` (`217`), `CUDA_ERROR_ILLEGAL_ADDRESS` (`700`), `CUDA_ERROR_LAUNCH_TIMEOUT` (`702`), `CUDA_ERROR_CONTEXT_IS_DESTROYED` (`709`), `CUDA_ERROR_ASSERT` (`710`), `CUDA_ERROR_HARDWARE_STACK_ERROR` (`714`), `CUDA_ERROR_ILLEGAL_INSTRUCTION` (`715`), `CUDA_ERROR_MISALIGNED_ADDRESS` (`716`), `CUDA_ERROR_INVALID_ADDRESS_SPACE` (`717`), `CUDA_ERROR_INVALID_PC` (`718`), `CUDA_ERROR_LAUNCH_FAILED` (`719`).
  - `INIT_ERROR (2)`: missing `libcuda.so.1` / `libnvrtc.so.12`, `cuInit` failure, allocation failure, or NVRTC compilation error.

### AMD HIP (`hip`, priority `100`)

- **Runtime Libraries:** `libamdhip64.so.6`, fallback `libamdhip64.so`, `libamdhip64.so.5` (supports both ROCm 6.0+ `hipDeviceProp_tR0600` and legacy ROCm 5.x ABI layouts).
- **Runtime Compiler:** `libhiprtc.so.6`, fallback `libhiprtc.so`, `libhiprtc.so.5`, `libamdhip64.so.6`, `libamdhip64.so` (`ctypes`). Emits `hsaco` compiled with `--offload-arch=<gfx...>`.
- **Hardware Identification:** PCIe BDF (`hipDeviceGetPCIBusId` / property fields) and 16-byte UUID (`hipDeviceGetUuid`).
- **Linux Permissions:** Requires read/write access to `/dev/kfd` and `/dev/dri/renderD*` (`render` and/or `video` group membership).
- **Error Classification:**
  - `HARDWARE_ERROR (4)`: `hipErrorECCNotCorrectable` (`214`), `hipErrorIllegalAddress` (`700`), `hipErrorLaunchTimeOut` (`702`), `hipErrorContextIsDestroyed` (`709`), `hipErrorAssert` (`710`), `hipErrorHardwareStackError` (`714`), `hipErrorIllegalInstruction` (`715`), `hipErrorMisalignedAddress` (`716`), `hipErrorLaunchFailure` (`719`).
  - `INIT_ERROR (2)`: missing `libamdhip64.so.6` / `libhiprtc.so.6`, permission denial, device enumeration failure, or hipRTC compilation error.

### oneAPI Level Zero (`levelzero`, priority `100`)

- **Runtime Libraries:** `libze_loader.so.1`, fallback `libze_loader.so`.
- **Runtime Compiler:** Translates `src/kernels/levelzero/*.cl` to SPIR-V (`ZE_MODULE_FORMAT_IL_SPIRV`) using `clang -cc1 -triple spir64-unknown-unknown` + `llvm-spirv` (preferred when both are in `PATH`), falling back to Intel `ocloc compile -spirv_only`. Pre-built SPIR-V binaries passed via `source_override` bypass compilation.
- **Hardware Identification:** PCIe BDF (`zeDevicePciGetPropertiesExt`) and 16-byte device UUID (`zeDeviceGetProperties`).
- **Linux Permissions:** Requires read/write access to `/dev/dri/renderD*` (`render` / `video` group membership); permission denials are detected and surfaced on `stderr`.
- **Error Classification:**
  - `HARDWARE_ERROR (4)`: `ZE_RESULT_ERROR_DEVICE_LOST` (`0x70000001`), `ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET` (`0x70000005`), `ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE` (`0x70000006`).
  - `INIT_ERROR (2)`: missing `libze_loader.so.1`, missing SPIR-V compiler toolchain (`clang`+`llvm-spirv` and `ocloc`), `zeInit` failure, or module build error.

### Apple Metal (`metal`, priority `100`)

- **Runtime Frameworks (macOS only):** `/usr/lib/libobjc.A.dylib`, `/System/Library/Frameworks/Metal.framework/Metal`, and `/usr/lib/libSystem.B.dylib`. Reports `metal: macOS only` on Linux.
- **Runtime Compiler:** Uses `xcrun -sdk macosx metal` + `metallib` to emit `.metallib` when Xcode Metal CLI utilities are installed; cleanly falls back to MSL source passthrough (`newLibraryWithSource:options:error:`) when `xcrun metal` is unavailable (for example, macOS with Command Line Tools only).
- **Hardware Identification:** 64-bit IOKit `registryID` enriched with unprivileged `ioreg` GPU core count, vendor ID, and driver version.
- **Error Classification:**
  - `HARDWARE_ERROR (4)`: `MTLCommandBufferErrorTimeout` (`2`), `Blacklisted` (`4`), `NotPermitted` (`5`), `OutOfMemory` (`6`), `PageFault` (`7`), `AccessRevoked` (`8`), `StackOverflow` (`10`).
  - `INIT_ERROR (2)`: non-macOS host, framework load error, pipeline state creation failure, or MSL compile error.

### OpenCL (`opencl`, priority `50`)

- **Runtime Libraries:** `libOpenCL.so.1`, fallback `libOpenCL.so` on Linux; `/System/Library/Frameworks/OpenCL.framework/OpenCL` on macOS.
- **Runtime Compiler:** Passes OpenCL C source directly to `clCreateProgramWithSource` + `clBuildProgram`; also exposes `CL_PROGRAM_BINARIES` extraction and `clCreateProgramWithBinary` reload helpers.
- **Hardware Identification:** PCIe BDF (`cl_khr_pci_bus_info`, `cl_nv_device_attribute_query`, `cl_amd_device_attribute_query`), UUID (`cl_khr_device_uuid`), and fallback fingerprint.
- **Error Classification:**
  - `HARDWARE_ERROR (4)`: `CL_DEVICE_NOT_AVAILABLE` (`-2`), `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST` (`-14`), and runtime-phase `CL_OUT_OF_RESOURCES` (`-5`).
  - `INIT_ERROR (2)`: missing `libOpenCL.so.1`, zero platforms/GPU devices, allocation failure, or `CL_BUILD_PROGRAM_FAILURE` (`-11`).

---

## 2. Backend × Workload Hardware Capability Matrix

All 20 canonical workloads mapped across all 5 backends directly from the device capability probes and kernel `#if` / `#error "UNSUPPORTED: ..."` guards:

| Workload | CUDA (`sm_*`) | HIP (`gfx*`) | Level Zero (SPIR-V) | Metal (MSL) | OpenCL (CL C) |
|---|---|---|---|---|---|
| `compute-int64` | All (`u64` MAD) | All (`u64` MAD) | All (`ulong` MAD) | All (`ulong` MAD) | All (`ulong` MAD) |
| `compute-int32` | All (`u32` MAD) | All (`u32` MAD) | All (`uint` MAD) | All (`uint` MAD) | All (`uint` MAD) |
| `compute-int16` | All (`u16` MAD) | All (`u16` MAD) | All (`ushort` MAD) | All (`ushort` MAD) | All (`ushort` MAD) |
| `compute-int8` | SM 6.1+ (`dp4a.u32.u32`) | `gfx906+` / RDNA (`udot4`) | All (`uchar4` MAD) | All (`uchar4` MAD) | All (`uchar4` MAD) |
| `compute-int4` | SM 7.5+ (`s4` MMA `m8n8k32`) | `gfx906` / CDNA1–2 / RDNA2 (`sdot8`)* | **Unsupported** | **Unsupported** | **Unsupported** |
| `compute-fp64` | SM 1.3+ (native `double`) | All (native `double`) | `cl_khr_fp64` | **Unsupported** | `cl_khr_fp64` |
| `compute-fp32` | All (native `fmaf`) | All (native `fmaf`) | All (native `fma`) | All (native `fma`) | All (native `fma`) |
| `compute-fp16` | SM 5.3+ (`fma.rn.f16x2`) | `gfx900+` (packed `half2`) | `cl_khr_fp16` | All (`half2` FMA) | `cl_khr_fp16` |
| `compute-fp8` | SM 8.9+ (`e4m3` MMA `m16n8k32`) | CDNA3 (`gfx94x`) / RDNA4 (`gfx12xx`) | **Unsupported** | **Unsupported** | **Unsupported** |
| `compute-fp4` | SM 10.0+ (`e2m1` MMA `m16n8k64`) | CDNA4 (`gfx950+` FP4 MFMA) | **Unsupported** | **Unsupported** | **Unsupported** |
| `compute-bf16` | SM 8.0+ (`fma.rn.bf16x2`) | CDNA2+ (`gfx90a+`) / RDNA3+ (`gfx11xx+`) | **Unsupported** (scalar) | Metal 3.1+ (`__METAL_VERSION__ >= 310`, Apple9+ `bfloat2`) | **Unsupported** |
| `compute-tensor-fp32` | SM 8.0+ (TF32 MMA `m16n8k8`) | CDNA (`gfx908+` FP32 MFMA) | Intel XMX (`cl_intel_subgroup_matrix_multiply_accumulate_tf32`) | Apple7+ (`simdgroup_float8x8`) | Intel XMX (`..._tf32`) |
| `compute-tensor-fp16` | SM 7.5+ (FP16 MMA `m16n8k8` / `m16n8k16`) | CDNA (`gfx908+`) / RDNA3+ (`gfx11xx+`) | Intel XMX (`cl_intel_subgroup_matrix_multiply_accumulate`) | Apple7+ (`simdgroup_half8x8`) | Intel XMX (`cl_intel_subgroup_matrix_multiply_accumulate`) |
| `compute-tensor-bf16` | SM 8.0+ (BF16 MMA `m16n8k16`) | CDNA (`gfx908+`) / RDNA3+ (`gfx11xx+`) | Intel XMX (`cl_intel_subgroup_matrix_multiply_accumulate`) | Metal 3.1+ / Apple9+ (`simdgroup_bfloat8x8`) | Intel XMX (`cl_intel_subgroup_matrix_multiply_accumulate`) |
| `compute-tensor-int8` | SM 8.0+ (`s8` MMA `m16n8k32`) | CDNA (`gfx908+`) / RDNA3+ (`gfx11xx+`) | Intel XMX (`cl_intel_subgroup_matrix_multiply_accumulate`) | **Unsupported** | Intel XMX (`cl_intel_subgroup_matrix_multiply_accumulate`) |
| `memory-pattern` | All | All | All | All | All |
| `memory-sweep` | All | All | All | All | All |
| `bandwidth-d2d` | All (`uint4` kernel / `cuMemcpyDtoD`) | All (`uint4` kernel / `hipMemcpy`) | All (`uint4` kernel / `zeCommandListAppendMemoryCopy`) | All (`uint4` kernel / blit copy) | All (`uint4` kernel / `clEnqueueCopyBuffer`) |
| `bandwidth-h2d` | All (`cuMemcpyHtoD`) | All (`hipMemcpy` H2D) | All (`zeCommandListAppendMemoryCopy`) | All (shared staging / blit) | All (`clEnqueueWriteBuffer`) |
| `bandwidth-d2h` | All (`cuMemcpyDtoH`) | All (`hipMemcpy` D2H) | All (`zeCommandListAppendMemoryCopy`) | All (blit / shared staging) | All (`clEnqueueReadBuffer`) |

*\*Note:* `_hip_capabilities()` conservatively defaults `int4=False` across HIP devices unless explicitly enabled for `sdot8`-capable targets; the HIP `compute_int4.hip` kernel additionally enforces compile-time `gfx906`/CDNA1–2/RDNA2 guards.
