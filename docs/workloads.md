# `cross-gpu` Workload Reference & Verification Specification

All 20 canonical stress workloads report throughput and totals in **fixed, non-auto-scaling SI units** (`10^9` scaling):

| Category | Workloads | Rate Unit | Total Unit | Scale |
|---|---|---|---|---|
| Scalar Integer | `compute-int64`, `compute-int32`, `compute-int16`, `compute-int8`, `compute-int4` | `GOP/s` | `GOP` | `1,000,000,000` ops |
| Scalar Floating-Point & ML | `compute-fp64`, `compute-fp32`, `compute-fp16`, `compute-fp8`, `compute-fp4`, `compute-bf16` | `GFLOP/s` | `GFLOP` | `1,000,000,000` FLOPs |
| Hardware Matrix / Tensor | `compute-tensor-fp32`, `compute-tensor-fp16`, `compute-tensor-bf16` (`GFLOP/s`), `compute-tensor-int8` (`GOP/s`) | `GFLOP/s` / `GOP/s` | `GFLOP` / `GOP` | `1,000,000,000` ops |
| Memory & Bandwidth | `memory-pattern`, `memory-sweep`, `bandwidth-d2d`, `bandwidth-h2d`, `bandwidth-d2h` | `GB/s` | `GB` | `1,000,000,000` bytes (SI) |

---

## 1. Operation Accounting (`OPS_PER_THREAD_PER_ITER`)

For every compute and tensor workload, each calibrated dispatch records:

```text
step_operations = total_threads * iterations * OPS_PER_THREAD_PER_ITER
```

For hardware matrix / tensor workloads, each tile multiply-accumulate counts **`2 * M * N * K`** operations (one multiply + one add per inner product term), divided across the active warp/wavefront/subgroup width (`32`, `64`, or `8` lanes) to obtain per-thread `OPS_PER_THREAD_PER_ITER`.

### Group 1 — Scalar Integer (`GOP/s`, `GOP`)

| Workload | Capability | `OPS_PER_THREAD_PER_ITER` | Verification Rule |
|---|---|---|---|
| `compute-int64` | `int64` | `256` (`8 accs × 16 MADs × 2 ops`) | Bit-exact unsigned `uint64` check via `O(log N)` modular affine solver |
| `compute-int32` | `int32` | `256` (`8 accs × 16 MADs × 2 ops`) | Bit-exact unsigned `uint32` check via `O(log N)` modular affine solver |
| `compute-int16` | `int16` | `256` (`8 accs × 16 MADs × 2 ops`) | Bit-exact unsigned `uint16` check via `O(log N)` modular affine solver |
| `compute-int8` | `int8` | `1024` (CUDA `dp4a` / HIP `udot4`: `8 × 16 × 8`; L0/Metal/OCL `uchar4`: `8 × 4 lanes × 16 MADs × 2`) | Bit-exact (`dp4a`/`udot4` reference solver or `uchar4` modular affine solver) |
| `compute-int4` | `int4` | `4096` (CUDA SM 7.5+ `s4` MMA `m8n8k32`: `32 × 2*8*8*32/32`); `2048` (HIP `sdot8`: `8 × 16 × 16`); unsupported on L0/Metal/OCL | Bit-exact analytical (`(8 * base + 36) mod 2^32` on CUDA; `sdot8` reference on HIP) |

### Group 2 — Scalar Floating-Point & ML (`GFLOP/s`, `GFLOP`)

| Workload | Capability | `OPS_PER_THREAD_PER_ITER` | Verification Rule |
|---|---|---|---|
| `compute-fp64` | `fp64` | `256` (`8 accs × 8 FMA pairs × 2 FMAs × 2 FLOPs`); unsupported on Metal | Bounded contractive FMA recurrence; reject `NaN`/`Inf`; `rel_tol=1e-12, abs_tol=1e-12` |
| `compute-fp32` | `fp32` | `256` (`8 accs × 8 FMA pairs × 2 FMAs × 2 FLOPs`, all backends) | Bounded contractive FMA recurrence; reject `NaN`/`Inf`; `rel_tol=1e-5, abs_tol=1e-5` |
| `compute-fp16` | `fp16` | `512` (`8 half2 accs × 16 FMAs × 2 lanes × 2 FLOPs`, all backends) | Bounded contractive `half2` recurrence; reject `NaN`/`Inf`; `rel_tol=2e-3, abs_tol=2e-3` |
| `compute-bf16` | `bf16` | `512` (`8 bf16x2 accs × 16 FMAs × 2 lanes × 2 FLOPs`, CUDA/HIP/Metal); unsupported on L0/OCL | Bounded contractive `bf16x2` recurrence; reject `NaN`/`Inf`; `rel_tol=1.5e-2, abs_tol=3e-2` |
| `compute-fp8` | `fp8` | `8192` (CUDA SM 8.9+ `m16n8k32` / HIP CDNA3); `4096` (HIP RDNA4); unsupported on L0/Metal/OCL | Reject `NaN`/`Inf`; analytical paired MMA sum (`16 * base + 15.0`), `rel_tol=1e-5, abs_tol=1e-5` |
| `compute-fp4` | `fp4` | `16384` (CUDA SM 10.0+ `m16n8k64`); `32768` (HIP CDNA4 `m16n16k128`); unsupported on L0/Metal/OCL | Reject `NaN`/`Inf`; analytical paired MMA sum (`16 * base + 15.0`), `rel_tol=1e-5, abs_tol=1e-5` |

### Group 3 — Hardware Matrix / Tensor (`GFLOP/s` & `GOP/s`)

| Workload | Unit | Capability | `OPS_PER_THREAD_PER_ITER` (`2 * M * N * K` per tile) | Verification Rule |
|---|---|---|---|---|
| `compute-tensor-fp32` | `GFLOP/s` | `tensor_fp32` | `2048` (CUDA TF32 `m16n8k8` / L0 & OCL XMX `k8`); `1024` (HIP CDNA `m16n16k4` / Metal `8×8×8`) | Reject `NaN`/`Inf`; analytical paired MMA tile sum (`8*base + 3.0` Metal, `16*base + 15.0` others), `tol=1e-4` |
| `compute-tensor-fp16` | `GFLOP/s` | `tensor_fp16` | `4096` (CUDA SM 8.0+ / HIP / L0 / OCL); `2048` (CUDA SM 7.5); `1024` (Metal `8×8×8`) | Reject `NaN`/`Inf`; analytical paired MMA tile sum, `tol=1e-4` |
| `compute-tensor-bf16` | `GFLOP/s` | `tensor_bf16` | `4096` (CUDA / HIP CDNA2+ & RDNA3–4 / L0 / OCL); `2048` (HIP CDNA1 `gfx908`); `1024` (Metal `8×8×8`) | Reject `NaN`/`Inf`; analytical paired MMA tile sum, `tol=1e-4` |
| `compute-tensor-int8` | `GOP/s` | `tensor_int8` | `8192` (CUDA SM 8.0+ `m16n8k32` / L0 & OCL XMX `k32`); `4096` (HIP CDNA / RDNA3–4); unsupported on Metal | Bit-exact analytical tile checksum `(16 * base + 136) mod 2^32` |

---

## 2. Memory & Bandwidth Byte Accounting (`GB/s`, `GB`)

| Workload | Execution Method | Effective Bytes Counted | Safe VRAM Sizing Guardrails | Verification Rule |
|---|---|---|---|---|
| `memory-pattern` | `kernel_memory_pattern_write` + sync + `kernel_memory_pattern_verify` rotating through 7 adversarial modes (`0x00000000`, `0xFFFFFFFF`, `0x55555555`, `0xAAAAAAAA`, walking-1s, walking-0s, deterministic Murmur/PRNG hash) | `2 * buffer_bytes` per write+verify cycle (`BYTES_PER_ELEMENT = 4` per pass) | `60%` of usable VRAM (leaving `>= 512 MiB–1 GiB` discrete headroom or `>= 1 GiB / 25%` unified headroom), 4096-byte aligned, `10%` allocation step-down fallback | GPU atomic `error_count == 0` every pass + host spot-check of word `0` |
| `memory-sweep` | `kernel_memory_sweep_write` + sync + `kernel_memory_sweep_verify` with rotating 64-bit SplitMix64 seed per pass | `2 * buffer_bytes` per write+verify cycle (`BYTES_PER_ELEMENT = 8` per pass) | Up to `85%` of usable VRAM with OS headroom, `max_buffer_length` clamp, and `10%` step-down fallback | GPU atomic `error_count == 0` every pass + host spot-check of qword `0` |
| `bandwidth-d2d` | `kernel_bandwidth_d2d` (128-bit coalesced `uint4` copy; falls back to backend `memcpy_d2d` if no compiled binary) | `2 * copy_bytes` per copy (`BYTES_PER_ELEMENT = 32`: 16-byte `uint4` read + 16-byte `uint4` write effective DRAM traffic) | Up to `512 MiB` discrete / `256 MiB` unified per buffer (`<= 25%` usable VRAM each for equal `src` & `dst`), `10%` step-down fallback | Destination scrubbed before timed run; verified via full-payload `zlib.crc32` + head/mid/tail 4 KiB sentinel check |
| `bandwidth-h2d` | Backend `memcpy_h2d` from pinned host buffer (or standard `bytearray` fallback if pinned allocation fails) | `1 * copy_bytes` per transfer (unidirectional H2D payload) | Up to `512 MiB` discrete / `256 MiB` unified, clamped to `<= 25%` usable VRAM and `<= 25%` physical host RAM | Round-trip `memcpy_d2h` readback checked via full-payload `zlib.crc32` + 4 KiB sentinels |
| `bandwidth-d2h` | Backend `memcpy_d2h` into pinned host buffer (or standard `bytearray` fallback) | `1 * copy_bytes` per transfer (unidirectional D2H payload) | Up to `512 MiB` discrete / `256 MiB` unified, clamped to `<= 25%` usable VRAM and `<= 25%` physical host RAM | Host buffer scrubbed before run; verified zero-copy via full-payload `zlib.crc32` + 4 KiB sentinels |

---

## 3. Warmup, Calibration & Verification Lifecycle

1. **Dynamic Batch Calibration (`prepare()`):**
   - For standard runs (`--duration >= 1s`), compute, tensor, and bandwidth executors run a short timing probe (stepping up if `< 5 ms`) and scale `iterations` / transfers per step to target **`~100 ms`** per dispatch (clamped inside the **`50 ms–200 ms`** watchdog-safe window).
   - For sub-second runs (`--duration < 1s`, e.g., `250ms` or `500ms`), `calibrate_iterations` automatically caps the target below the requested duration (`20%` of `duration_s`, floored at `5 ms`) to prevent excessive timing overshoot.
2. **Excluded Warmup Phase (`--warmup > 0`):**
   - Executes real GPU steps with telemetry tagged `phase="warmup"`; operations and elapsed time during warmup are completely excluded from reported throughput and totals.
   - Synchronizes the GPU and invokes `executor.reset_run_state(exec_ctx)` before starting the timed clock.
3. **Timed Measurement & Periodic Checkpoints (`--duration`):**
   - Telemetry is tagged `phase="run"`. Every `5.0 s` (`DEFAULT_VERIFY_INTERVAL_S`) and at the end of the timed phase, `executor.verify()` validates results; checkpoint verification overhead is subtracted from timed `elapsed_s` so reported throughput reflects pure GPU execution.
   - Any numerical/bitwise mismatch, `NaN`/`Inf`, or nonzero GPU `error_count` immediately stops the workload and reports `ExitCode.COMPUTE_ERROR (3)`. Fatal driver/device-lost/timeout errors report `ExitCode.HARDWARE_ERROR (4)`.
