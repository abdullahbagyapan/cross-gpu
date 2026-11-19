# `cross-gpu` CLI Reference

`cross-gpu` provides deterministic device discovery, hardware inspection, and GPU stress testing with strict stream hygiene and operator-friendly diagnostics.

---

## 1. Commands & Synopsis

```sh
cross-gpu [-v|--verbose] list [--format text|json] [--json] [-v|--verbose]
cross-gpu [-v|--verbose] info [--format text|json] [--json] [-v|--verbose]
cross-gpu [-v|--verbose] run [--device all|N] [--test TEST|all] \
                             [--duration TIME] [--warmup [TIME]] \
                             [--backend auto|PLUGIN_NAME] \
                             [--format text|json] [--json] [-v|--verbose]
cross-gpu --help
cross-gpu --version
```

### Examples

```sh
cross-gpu list
cross-gpu info --json
cross-gpu run --device 0 --test compute-fp32,memory-pattern --warmup 5s --duration 30s
cross-gpu run --backend opencl --test all --duration 60s --json
```

---

## 2. Strict `stdout` / `stderr` Separation & `--verbose` (`-v`)

- **`stdout` is strictly reserved for command output:** formatted `text` tables/telemetry, valid JSON (`list`/`info`), valid NDJSON (`run`), `--help`, and `--version`.
- **`stderr` is strictly reserved for diagnostics:** CLI errors, per-backend unavailability reasons (including Linux `/dev/dri/renderD*` and `/dev/kfd` permission hints when zero GPUs or a requested backend are unavailable), compiler diagnostics, and `-v`/`--verbose` traces.
- Combining `--json` with `-v`/`--verbose` (e.g., `cross-gpu list --json -v`, `cross-gpu info --json -v`, `cross-gpu run --json -v`) always emits **100% valid JSON/NDJSON on `stdout`** while streaming structured `[verbose]` diagnostics to `stderr`:
  - Shared library SONAME and macOS framework probes (`libcuda.so.1`, `libamdhip64.so.6`, `libze_loader.so.1`, `Metal.framework`, `libOpenCL.so.1`).
  - Physical GPU deduplication decisions (matched PCIe BDF, UUID/`registryID`, or fallback fingerprint, and priority winner).
  - `importlib.resources` kernel lookups, disk/in-memory cache hits and misses, compile flags, and full backend compiler logs.

---

## 3. Typo Hints, Progress Display & Skip Reasons

- **`difflib` Typo Suggestions:** Unknown subcommands, `--backend` plugin names, and `--test` values (including individual tokens inside comma-separated lists like `--test fp32,mem-sweep`) exit with `ExitCode.USAGE_ERROR (1)` and print close suggestions on `stderr`:
  - `cross-gpu: error: unknown command 'lsit'. Did you mean: list?`
  - `cross-gpu: error: unknown backend 'cdua'. Did you mean: cuda?`
  - `cross-gpu: error: unknown test 'fp32'. Did you mean: compute-fp32?`
- **`text`-Mode Progress & Summary Tally:**
  - Live 1 Hz telemetry and completion lines include zero-padded workload counters and elapsed/total time:
    - Warmup phase: `[04/20 compute-fp32 | warmup 3s/5s]`
    - Timed phase: `[04/20 compute-fp32 | 18s/60s]`
  - Runs end with an aligned summary table and a final tally line reporting `passed`, `failed`, `unsupported`, and `hardware_error` counts alongside the resolved exit code.
- **Explicit Unsupported Reasons:** Every workload marked `status="unsupported"` includes a specific `reason` string in both `text` output and JSON/NDJSON `workload_result` events (for example, `requires CUDA SM >= 8.9 (detected SM 8.6)`, `Metal lacks native fp64`, or `OpenCL device missing cl_khr_fp16 extension`).

---

## 4. Environment Variables

| Variable | Effect |
|---|---|
| `XDG_CACHE_HOME` | On Linux, stores compiled kernel cache files under `$XDG_CACHE_HOME/cross-gpu` (defaulting to `~/.cache/cross-gpu`). On macOS, defaults to `~/Library/Caches/cross-gpu` unless an absolute `XDG_CACHE_HOME` is explicitly set. |
| `CROSS_GPU_NO_CACHE` | When set to a truthy value (`1`, `true`, `yes`, case-insensitive), completely bypasses disk cache reads and writes (in-memory caching only) for ephemeral containers, CI, and read-only chroots. |
| `CROSS_GPU_CACHE_DIR` | Explicitly overrides the disk cache directory on any platform (unless `CROSS_GPU_NO_CACHE` is truthy). |
| `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` | Honored by the OS dynamic linker when probing versioned SONAMEs (`libcuda.so.1`, `libnvrtc.so.12`, `libamdhip64.so.6`, `libze_loader.so.1`, `libOpenCL.so.1`, `libnvidia-ml.so.1`). |

---

## 5. Shell Completions

Static, zero-dependency completion scripts are shipped under `completions/` and installed by `make install`:

| Shell | Source File | FHS Install Path (`PREFIX ?= /usr/local`) |
|---|---|---|
| Bash | `completions/cross-gpu.bash` | `$(PREFIX)/share/bash-completion/completions/cross-gpu` |
| Zsh | `completions/_cross-gpu` | `$(PREFIX)/share/zsh/site-functions/_cross-gpu` |
| Fish | `completions/cross-gpu.fish` | `$(PREFIX)/share/fish/vendor_completions.d/cross-gpu.fish` |

---

## 6. Deterministic Exit Codes

Exit codes are ordered by ascending severity and aggregated across devices and workloads via `max(codes)`:

| Code | Name | Meaning |
|---|---|---|
| `0` | `SUCCESS` | All executed workloads passed without compute or hardware faults. |
| `1` | `USAGE_ERROR` | Invalid CLI arguments, unknown commands/tests/backends, malformed durations, or conflicting flags. |
| `2` | `INIT_ERROR` | No compatible GPU/backend found, requested device/backend unavailable, context creation failed, or runtime kernel load/compile failed. |
| `3` | `COMPUTE_ERROR` | Numerical/bitwise verification mismatch, silent data corruption (SDC), unexpected `NaN`/`Inf`, or memory fault. |
| `4` | `HARDWARE_ERROR` | Device-lost error, GPU hang/reset, uncorrectable ECC fault, Xid/GPU fault, or thermal fault. |
