# `cross-gpu` Workload Reference & Accounting Specification

All 20 canonical stress workloads use **fixed, non-auto-scaling SI units** (`10^9` scaling):

| Category | Rate Unit | Total Unit | Scale |
|---|---|---|---|
| Scalar Integer & Tensor INT8 | `GOP/s` | `GOP` | `1,000,000,000` ops |
| Scalar FP/ML & Tensor FP/BF16 | `GFLOP/s` | `GFLOP` | `1,000,000,000` FLOPs |
| Memory & Bandwidth | `GB/s` | `GB` | `1,000,000,000` bytes (SI) |

---

## 1. Operation Accounting (`OPS_PER_THREAD_PER_ITER`)

For every compute and tensor workload, each dispatch records:

```text
step_operations = total_threads * iterations * OPS_PER_THREAD_PER_ITER
```

| Workload | Unit | Capability | `OPS_PER_THREAD_PER_ITER` by Backend / Architecture | Verification |
|---|---|---|---|---|
| `compute-int64` | `GOP/s` | `int64` | `256` (all backends: `8 accs * 16 MADs * 2`) | Bit-exact modular LCG (`O(log N)` affine solver) |
| `compute-int32` | `GOP/s` | `int32` | `256` (all backends: `8 accs * 16 MADs * 2`) | Bit-exact modular LCG (`O(log N)` affine solver) |
| `compute-int16` | `GOP/s` | `int16` | `256` (all backends: `8 accs * 16 MADs * 2`) | Bit-exact modular LCG (`O(log N)` affine solver) |
| `compute-int8` | `GOP/s` | `int8` | `1024` (CUDA `dp4a`/HIP `udot4`: `8*16*8`; L0/Metal/OCL `uchar4`: `8*4*16*2`) | Bit-exact (`dp4a`/`udot4` reference or `uchar4` affine solver) |
| `compute-int4` | `GOP/s` | `int4` | `4096` (CUDA SM 7.5+ `s4` MMA); `2048` (HIP `sdot8`); unsupported on L0/Metal/OCL | Bit-exact (`8*base + 36` on CUDA; `sdot8` on HIP) |
| `compute-fp64` | `GFLOP/s` | `fp64` | `256` (CUDA/HIP/L0/OCL); unsupported on Metal | Reject `NaN`/`Inf`; `rel_tol=1e-12, abs_tol=1e-12` |
| `compute-fp32` | `GFLOP/s` | `fp32` | `256` (all backends) | Reject `NaN`/`Inf`; `rel_tol=1e-5, abs_tol=1e-5` |
| `compute-fp16` | `GFLOP/s` | `fp16` | `512` (all backends: `8 half2 * 16 FMAs * 2 lanes * 2`) | Reject `NaN`/`Inf`; `rel_tol=2e-3, abs_tol=2e-3` |
| `compute-bf16` | `GFLOP/s` | `bf16` | `512` (CUDA/HIP/Metal); unsupported on L0/OCL | Reject `NaN`/`Inf`; `rel_tol=1.5e-2, abs_tol=3e-2` |
| `compute-fp8` | `GFLOP/s` | `fp8` | `8192` (CUDA SM 8.9+ / HIP CDNA3); `4096` (HIP RDNA4); unsupported on L0/Metal/OCL | Reject `NaN`/`Inf`; exact paired MMA sum (`16*base + 15.0`), `tol=1e-5` |
| `compute-fp4` | `GFLOP/s` | `fp4` | `16384` (CUDA SM 10.0+); `32768` (HIP CDNA4); unsupported on L0/Metal/OCL | Reject `NaN`/`Inf`; exact paired MMA sum (`16*base + 15.0`), `tol=1e-5` |
| `compute-tensor-fp32` | `GFLOP/s` | `tensor_fp32` | `2048` (CUDA TF32 / L0 & OCL XMX); `1024` (HIP CDNA MFMA / Metal `simdgroup_float8x8`) | Reject `NaN`/`Inf`; exact paired MMA sum (`8*base + 3.0` Metal, `16*base + 15.0` others), `tol=1e-4` |
| `compute-tensor-fp16` | `GFLOP/s` | `tensor_fp16` | `4096` (CUDA SM 8.0+ / HIP / L0 / OCL); `2048` (CUDA SM 7.5); `1024` (Metal `simdgroup_half8x8`) | Reject `NaN`/`Inf`; exact paired MMA sum, `tol=1e-4` |
| `compute-tensor-bf16` | `GFLOP/s` | `tensor_bf16` | `4096` (CUDA / HIP CDNA2+ & RDNA3-4 / L0 / OCL); `2048` (HIP CDNA1 `gfx908`); `1024` (Metal `simdgroup_bfloat8x8`) | Reject `NaN`/`Inf`; exact paired MMA sum, `tol=1e-4` |
| `compute-tensor-int8` | `GOP/s` | `tensor_int8` | `8192` (CUDA SM 8.0+ / L0 & OCL XMX); `4096` (HIP CDNA / RDNA3-4); unsupported on Metal | Bit-exact analytical `(16*base + 136) mod 2^32` |

---

## 2. Memory & Bandwidth Byte Accounting (`GB/s`, `GB`)

| Workload | Method | Bytes per Cycle / Transfer | Buffer Sizing Guardrails | Verification |
|---|---|---|---|---|
| `memory-pattern` | `kernel_memory_pattern_write` + sync + `kernel_memory_pattern_verify` across 7 adversarial modes (`0x00000000`, `0xFFFFFFFF`, `0x55555555`, `0xAAAAAAAA`, walking-1s, walking-0s, deterministic PRNG) | `2 * buffer_bytes` (full VRAM write + full VRAM read) | 60% of usable VRAM (leaving >= 512 MiB–1 GiB discrete headroom, or >= 1 GiB / 25% unified headroom), 4096-byte aligned, 10% step-down fallback | GPU atomic `error_count == 0` every pass + host spot-check of word 0 |
| `memory-sweep` | `kernel_memory_sweep_write` + sync + `kernel_memory_sweep_verify` with rotating 64-bit SplitMix64 seed per pass | `2 * buffer_bytes` (full VRAM write + full VRAM read) | Up to 85% of usable VRAM with headroom, `max_buffer_length` clamp, and 10% step-down fallback | GPU atomic `error_count == 0` every pass + host spot-check of qword 0 |
| `bandwidth-d2d` | `kernel_bandwidth_d2d` (128-bit coalesced `uint4` copy; falls back to backend `memcpy_d2d` if no compiled binary) | `2 * copy_bytes` per copy (`BYTES_PER_ELEMENT = 32`: 16-byte read + 16-byte write effective DRAM traffic) | Up to 512 MiB discrete / 256 MiB unified per buffer (`<= 25%` usable VRAM each for equal `src` & `dst`), 10% step-down fallback | Full-payload `zlib.crc32` + head/mid/tail 4 KiB sentinel check after scrubbing `dst` |
| `bandwidth-h2d` | Backend `memcpy_h2d` from pinned host buffer (or standard `bytearray` fallback) | `1 * copy_bytes` per transfer (unidirectional H2D payload) | Up to 512 MiB discrete / 256 MiB unified, clamped to `<= 25%` usable VRAM and `<= 25%` physical host RAM | Round-trip `memcpy_d2h` readback checked via full-payload `zlib.crc32` + sentinels |
| `bandwidth-d2h` | Backend `memcpy_d2h` into pinned host buffer (or standard `bytearray` fallback) | `1 * copy_bytes` per transfer (unidirectional D2H payload) | Up to 512 MiB discrete / 256 MiB unified, clamped to `<= 25%` usable VRAM and `<= 25%` physical host RAM | Host buffer scrubbed before run; verified zero-copy via full-payload `zlib.crc32` + sentinels |

---

## 3. Watchdog-Safe Batch Calibration & Lifecycle

- During `prepare()`, compute, tensor, and bandwidth executors launch a short timing probe (`probe_iters`, stepping up if `< 5 ms`) and scale `iterations` / transfers per step proportionally to target **`~100 ms`** per dispatch (clamped inside the **`50 ms–200 ms`** watchdog-safe window and bounded by strict min/max ceilings).
- `StressRunner` runs an excluded warmup phase (`--warmup > 0`, telemetry `phase="warmup"`), synchronizes the device, invokes `reset_run_state()`, and then runs the timed loop (`--duration`, telemetry `phase="run"`) with periodic verification checkpoints and responsive `SIGINT`/`SIGTERM` checks between dispatches.
- Any verification mismatch, `NaN`/`Inf`, or nonzero GPU `error_count` fails immediately with `ExitCode.COMPUTE_ERROR (3)`. Fatal driver/device-lost/timeout errors report `ExitCode.HARDWARE_ERROR (4)`.
