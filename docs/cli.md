# CrossGPU CLI Reference

Complete command-line interface, output schemas, exit codes, and operational recipes for CrossGPU.

---

## 1. Synopsis

```sh
cross-gpu [-v|--verbose] list [--format text|json] [--json] [-v|--verbose]
cross-gpu [-v|--verbose] info [--format text|json] [--json] [-v|--verbose]
cross-gpu [-v|--verbose] run [--device all|N] [--test TEST|all] \
                             [--duration TIME] [--warmup [TIME]] \
                             [--backend auto|PLUGIN_NAME] [--no-banner] \
                             [--format text|json] [--json] [-v|--verbose]
cross-gpu --help
cross-gpu --version
```

The CLI can also be invoked directly as a Python module: `python3 -m cross_gpu <command> [options]` (when installed) or `python3 -m src <command> [options]` (from source repository).

---

## 2. Subcommands & Options

### Global Options

| Flag | Default | Description |
|---|---|---|
| `--help`, `-h` | — | Print usage and copy-pasteable `Examples:` epilog to `stdout` and exit `0`. |
| `--version` | — | Print `cross-gpu <version>` to `stdout` and exit `0`. |
| `-v`, `--verbose` | off | Stream structured `[verbose]` diagnostics (SONAME/framework probes, deduplication decisions, kernel lookups, cache hits/misses, compile flags, and compiler logs) exclusively to `stderr`. Accepted before or after the subcommand. |

### Shared Formatting Options (`list`, `info`, `run`)

| Flag | Default | Description |
|---|---|---|
| `--format text\|json` | `text` | Output format. `list`, `info`, and `run` each emit a single strict JSON document on `stdout` in `json` mode. |
| `--json` | off | Convenience alias for `--format json`. Combining `--format text` with `--json` is an error and exits with `ExitCode.USAGE_ERROR (1)`. |

### `run` Options

| Flag | Default | Description |
|---|---|---|
| `--device all\|N` | `all` | Target all detected physical GPUs (`all`) or a single canonical non-negative integer index `N` from `cross-gpu list`. |
| `--test TEST\|all` | `all` | Single canonical workload name, comma-separated list (e.g., `compute-fp32,memory-pattern`), or `all` (all 20 workloads). Duplicate entries in a list are deduplicated preserving order; combining `all` with specific names exits `1`. |
| `--duration TIME` | `60s` | Timed measurement duration per workload. Accepts suffixes `ms`, `s`, `m`, and `h` (case-insensitive, e.g., `250ms`, `500ms`, `1.5s`, `5m`, `1h`). Must be strictly `> 0` (`0ms`, `0s`, negative, or malformed inputs exit `1`). |
| `--warmup [TIME]` | `0s` | Pre-measurement warmup duration per workload. Three edge-case rules: **omitted** -> `0s` (no warmup); **bare `--warmup`** (no value, including right before another flag like `--warmup --json`) -> `5s`; **explicit value** (`--warmup 10s`, `--warmup=250ms`, `--warmup 0ms`) -> parsed duration (`>= 0`). |
| `--backend auto\|PLUGIN` | `auto` | `auto` selects each GPU's highest-priority backend (native `100` over `opencl` `50`), or explicitly force `cuda`, `hip`, `levelzero`, `metal`, or `opencl`. |
| `--no-banner` | off | Suppress the interactive ASCII banner in text mode. |

### Typo Hints (`difflib`)

Invalid subcommands, unknown `--backend` names, and unknown `--test` tokens (including individual elements inside comma-separated lists) emit `difflib` suggestions (`n=3`, `cutoff=0.5`) to `stderr` and exit with `ExitCode.USAGE_ERROR (1)`:

```text
cross-gpu: error: unknown command 'lsit'. Did you mean: list?
cross-gpu: error: unknown backend 'cdua'. Did you mean: cuda?
cross-gpu: error: unknown test 'fp32'. Did you mean: compute-fp32?; unknown test 'mem-sweep'. Did you mean: memory-sweep?
```

---

## 3. Stream Hygiene (`stdout` vs. `stderr`)

- **`stdout`** is strictly reserved for formatted command output (`text` tables, JSON documents, `--help`, `--version`).
- **`stderr`** receives all errors, per-backend discovery reasons when zero GPUs or a requested backend are unavailable (including Linux `/dev/dri/renderD*` and `/dev/kfd` permission hints), compiler logs, and `-v`/`--verbose` diagnostics.
- `cross-gpu list --json -v`, `cross-gpu info --json -v`, and `cross-gpu run --json -v` always produce **100% valid JSON on `stdout`**.

---

## 4. Exit Codes & Multi-GPU Aggregation

Exit codes are ordered strictly by ascending severity and aggregated across all targeted devices and workloads using `max(codes)`:

| Code | Name | Condition |
|---|---|---|
| `0` | `SUCCESS` | All executed workloads passed without compute or hardware faults. (During `--test all`, unsupported workloads on a device are skipped with `0` unless *every* workload on every selected device is unsupported.) |
| `1` | `USAGE_ERROR` | Invalid CLI arguments, unknown subcommands/tests/backends, malformed durations, or conflicting `--format text --json`. |
| `2` | `INIT_ERROR` | No compatible GPU/backend found, requested `--device` or `--backend` unavailable, context creation failed, kernel missing/corrupt, runtime compilation failed, or an explicitly requested `--test` is unsupported on all targeted devices. |
| `3` | `COMPUTE_ERROR` | Numerical/bitwise golden verification mismatch, silent data corruption (SDC), unexpected `NaN`/`Inf`, or VRAM pattern/sweep error (`error_count > 0`). |
| `4` | `HARDWARE_ERROR` | Device-lost error, GPU hang/timeout/reset, uncorrectable ECC/NVLink fault, illegal address/instruction, or fatal thermal fault. |

Example: if GPU 0 encounters `COMPUTE_ERROR (3)` and GPU 1 encounters `HARDWARE_ERROR (4)`, the process exits with `max(3, 4) = 4`.

---

## 5. Output Formats & Verified Examples

### `cross-gpu list` (`text`)

```text
INDEX  NAME      VENDOR  VRAM       ACTIVE  BACKENDS      ID / BDF
-----  --------  ------  ---------  ------  ------------  --------------
0      Apple M2  Apple   11.84 GiB  metal   metal,opencl  reg:0x1000003e8
```

### `cross-gpu list --json`

Single JSON document written to `stdout`:

```json
{
  "devices": [
    {
      "index": 0,
      "name": "Apple M2",
      "vendor": "Apple",
      "active_backend": "metal",
      "available_backends": [
        "metal",
        "opencl"
      ],
      "total_memory_bytes": 12713115648,
      "compute_units": 10,
      "pci_bdf": null,
      "uuid": null,
      "registry_id": 4294968296
    }
  ]
}
```

### `cross-gpu info --json`

Single JSON document exposing top-level deduplicated hardware properties, capability flags, and every bound backend descriptor:

```json
{
  "devices": [
    {
      "index": 0,
      "name": "Apple M2",
      "vendor": "Apple",
      "vendor_id": "0x106b",
      "architecture": "applegpu_g14g",
      "compute_capability": "metal-3",
      "active_backend": "metal",
      "available_backends": [
        "metal",
        "opencl"
      ],
      "total_memory_bytes": 12713115648,
      "free_memory_bytes": 12713050112,
      "compute_units": 10,
      "driver_version": "345.20.4",
      "pci_bdf": null,
      "uuid": null,
      "registry_id": 4294968296,
      "capabilities": {
        "fp64": false,
        "fp32": true,
        "fp16": true,
        "bf16": false,
        "fp8": false,
        "fp4": false,
        "int64": true,
        "int32": true,
        "int16": true,
        "int8": true,
        "int4": false,
        "tensor_fp32": true,
        "tensor_fp16": true,
        "tensor_bf16": false,
        "tensor_int8": false
      },
      "backends": [
        {
          "backend": "metal",
          "priority": 100,
          "backend_device_index": 0,
          "name": "Apple M2",
          "vendor": "Apple",
          "vendor_id": "0x106b",
          "architecture": "applegpu_g14g",
          "compute_capability": "metal-3",
          "total_memory_bytes": 12713115648,
          "free_memory_bytes": 12713050112,
          "compute_units": 10,
          "driver_version": "345.20.4",
          "pci_bdf": null,
          "uuid": null,
          "registry_id": 4294968296,
          "capabilities": {
            "fp64": false,
            "fp32": true,
            "fp16": true,
            "bf16": false,
            "fp8": false,
            "fp4": false,
            "int64": true,
            "int32": true,
            "int16": true,
            "int8": true,
            "int4": false,
            "tensor_fp32": true,
            "tensor_fp16": true,
            "tensor_bf16": false,
            "tensor_int8": false
          },
          "extra": {
            "has_unified_memory": true,
            "is_low_power": false,
            "max_buffer_length": 9534832640
          }
        }
      ]
    }
  ]
}
```

### `cross-gpu run` (`text`)

In interactive TTY text mode, `cross-gpu run` prints the CrossGPU ASCII banner (unless `--no-banner` is set) followed by a live aligned workload table with colorful uppercase status badges (`PASSED`, `FAILED`, `UNSUPPORTED`, `INIT_ERROR`, `HARDWARE_ERROR`) and a closing summary line:

```text
Devices: 0:Apple M2 (metal) | Tests: 2 | Backend: auto | Duration: 500ms | Warmup: 200ms

WORKLOAD               DEV  BACKEND     STATUS                     RATE            TOTAL  DETAIL
------------------------------------------------------------------------------------------------
compute-fp32           0    metal       PASSED          3135.89 GFLOP/s    1587.39 GFLOP  506ms
compute-fp64           0    metal       UNSUPPORTED                   -                -  Metal lacks native fp64
------------------------------------------------------------------------------------------------
Summary: 1 passed, 0 failed, 1 unsupported | Exit: INIT_ERROR (2)
```

### `cross-gpu run --json`

Emits a single strict JSON document on `stdout` containing run metadata, target devices, configuration, per-workload results (including a concrete `"reason"` field for every `unsupported` workload), and a summary object:

```json
{
  "command": "run",
  "version": "0.1.0",
  "exit_code": 2,
  "exit_name": "INIT_ERROR",
  "interrupted": false,
  "total_workloads": 2,
  "devices": [
    {
      "index": 0,
      "name": "Apple M2",
      "backend": "metal"
    }
  ],
  "config": {
    "device": "all",
    "backend": "auto",
    "workloads": [
      "compute-fp32",
      "compute-fp64"
    ],
    "duration_s": 0.5,
    "warmup_s": 0.0
  },
  "results": [
    {
      "device": 0,
      "device_name": "Apple M2",
      "backend": "metal",
      "workload": "compute-fp32",
      "category": "compute-fp",
      "status": "passed",
      "duration_s": 0.500152,
      "rate": 3108.7119,
      "rate_unit": "GFLOP/s",
      "total": 1554.8285,
      "total_unit": "GFLOP",
      "errors": 0,
      "message": "",
      "exit_code": 0
    },
    {
      "device": 0,
      "device_name": "Apple M2",
      "backend": "metal",
      "workload": "compute-fp64",
      "category": "compute-fp",
      "status": "unsupported",
      "duration_s": 0.0,
      "rate": null,
      "rate_unit": "GFLOP/s",
      "total": null,
      "total_unit": "GFLOP",
      "errors": 0,
      "message": "Metal lacks native fp64",
      "exit_code": 2,
      "reason": "Metal lacks native fp64"
    }
  ],
  "summary": {
    "passed": 1,
    "failed": 0,
    "unsupported": 1,
    "init_error": 0,
    "hardware_error": 0,
    "total_workloads": 2,
    "interrupted": false,
    "exit_code": 2,
    "exit_name": "INIT_ERROR"
  }
}
```

---

## 6. Operational Recipes

```sh
# 1. Single-GPU thermal burn-in with 10s warmup and 10m duration
cross-gpu run --device 0 --test compute-fp32,memory-pattern,memory-sweep --warmup 10s --duration 10m

# 2. Sub-second CI smoke check extracting pass rates via jq (diagnostics isolated on stderr)
CROSS_GPU_NO_CACHE=1 cross-gpu run --test compute-fp32 --warmup 250ms --duration 500ms --json -v \
  | jq -r '.results[] | "\(.device) \(.workload) \(.status) \(.rate) \(.rate_unit)"'

# 3. Force OpenCL instead of auto-selected native backend (CUDA/HIP/Level Zero/Metal)
cross-gpu run --backend opencl --test all --duration 60s --json

# 4. Warmup calibration with default 5s warmup (bare --warmup before another flag)
cross-gpu run --device all --test compute-tensor-fp16 --warmup --duration 30s
```

---

## 7. Environment Variables & Shell Completions

| Variable | Effect |
|---|---|
| `XDG_CACHE_HOME` | Stores compiled kernel cache files under `$XDG_CACHE_HOME/cross-gpu` when set to an absolute path (on both Linux and macOS). Defaults to `~/.cache/cross-gpu` on Linux and `~/Library/Caches/cross-gpu` on macOS. |
| `CROSS_GPU_NO_CACHE` | When set to `1`, `true`, or `yes` (case-insensitive), completely bypasses disk cache reads and writes (in-memory caching only). |
| `CROSS_GPU_CACHE_DIR` | Explicitly overrides the disk cache directory on any platform (ignored when `CROSS_GPU_NO_CACHE` is truthy). |
| `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` | Honored by the OS dynamic linker when probing versioned SONAMEs (`libcuda.so.1`, `libnvrtc.so.12`, `libamdhip64.so.6`, `libze_loader.so.1`, `libOpenCL.so.1`). |

Static completions are installed by `make install` (see [packaging.md](packaging.md)):

- Bash: `completions/cross-gpu.bash` -> `$(PREFIX)/share/bash-completion/completions/cross-gpu`
- Zsh: `completions/_cross-gpu` -> `$(PREFIX)/share/zsh/site-functions/_cross-gpu`
- Fish: `completions/cross-gpu.fish` -> `$(PREFIX)/share/fish/vendor_completions.d/cross-gpu.fish`
