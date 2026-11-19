# `cross-gpu` Telemetry Reference

`cross-gpu` collects hardware telemetry during `cross-gpu run` via a lightweight 2-file subsystem under `cross_gpu/telemetry/` with zero external Python dependencies.

---

## 1. Two-File Telemetry Architecture

- **`cross_gpu/telemetry/__init__.py`:**
  - `read_telemetry(device)`: routes each device to its platform-native reader (macOS `ioreg`, Linux NVIDIA NVML merged with Linux DRM/hwmon `sysfs`, or Linux AMD/Intel DRM/hwmon `sysfs`).
  - `TelemetrySampler`: drift-compensated 1 Hz background polling thread built on `time.monotonic()` and `threading.Event.wait(timeout)`. Skips missed intervals without burst catch-up, and wakes/joins immediately when `stop()` is called so sub-second runs (`250ms`, `500ms`) never stall on a 1-second sleep.
- **`cross_gpu/telemetry/readers.py`:**
  - `read_nvml(pci_bdf)`: lazy `ctypes` client for `libnvidia-ml.so.1` (fallback `libnvidia-ml.so`) with explicit `argtypes`/`restype`, matched by normalized PCIe BDF.
  - `read_linux_sysfs(pci_bdf)`: scans `/sys/class/drm/card*/device/uevent` for matching `PCI_SLOT_NAME=`, reading DRM utilization/clocks and `hwmon/hwmon*/` sensors.
  - `read_macos(registry_id)` / `parse_ioreg_output(...)`: runs unprivileged `ioreg -r -d 1 -w 0 -c IOAccelerator` and parses `PerformanceStatistics` for the matching Metal `registryID` (`IORegistryEntryID`).
  - `sanitize_sample(sample)` / `sanitize_metrics_dict(metrics)`: validates every metric before emission and normalizes missing, non-finite, out-of-range, or sentinel values to `None`.

---

## 2. Telemetry Metric Reference (All 8 Sensors)

| JSON Key | Type | Unit | Valid Sanitized Range | NVIDIA Linux (NVML) | AMD / Intel Linux (`sysfs`) | macOS (`ioreg`) |
|---|---|---|---|---|---|---|
| `temperature_c` | `float` | `°C` | `-40.0 < val < 200.0` | `nvmlDeviceGetTemperature` (`NVML_TEMPERATURE_GPU`) | `hwmon/hwmon*/temp1_input` (`/ 1000.0`) | `Temperature(C)` (dGPUs only) |
| `power_w` | `float` | `W` | `val >= 0.0`, finite | `nvmlDeviceGetPowerUsage` (`/ 1000.0`) | `hwmon/hwmon*/power1_average` or `power1_input` (`/ 1e6`) | `GPU Power(W)` / `Total Power(W)` (dGPUs only) |
| `gpu_util_pct` | `float` | `%` | `0.0 <= val <= 100.0` | `nvmlDeviceGetUtilizationRates().gpu` | `gpu_busy_percent` | `Device Utilization %` / `GPU Activity(%)` / `Renderer Utilization %` |
| `mem_util_pct` | `float` | `%` | `0.0 <= val <= 100.0` | `nvmlDeviceGetUtilizationRates().memory` | `mem_busy_percent` | `Memory Utilization %` / `VRAM Utilization %` |
| `core_clock_mhz` | `int` | `MHz` | `val > 0` | `nvmlDeviceGetClockInfo` (`NVML_CLOCK_GRAPHICS`) | Active `*` entry in `pp_dpm_sclk`, or `gt_cur_freq_mhz`, or `gt/gt0/rps_cur_freq_mhz` | `Core Clock(MHz)` |
| `mem_clock_mhz` | `int` | `MHz` | `val > 0` | `nvmlDeviceGetClockInfo` (`NVML_CLOCK_MEM`) | Active `*` entry in `pp_dpm_mclk` | `Memory Clock(MHz)` |
| `fan_speed_pct` | `float` | `%` | `0.0 <= val <= 100.0` | `nvmlDeviceGetFanSpeed` | `hwmon/hwmon*/pwm1` ÷ `pwm1_max` (`255` default) | `Fan Speed(%)` |
| `fan_speed_rpm` | `int` | `RPM` | `0 <= val < 65535` | *Not exposed by NVML* (backfilled from `sysfs` if present) | `hwmon/hwmon*/fan1_input` | `Fan Speed(RPM)` (dGPUs only) |

*Note:* `0` RPM is intentionally preserved as a valid reading for zero-RPM fan-stop GPUs, whereas negative values and `>= 65535` unsigned sentinels are rejected as unusable. `0 MHz` clocks under active load are treated as unusable/unavailable.

---

## 3. Clean Output Filtering (No `N/A` or `null` Noise)

- **`text` mode:** Live telemetry lines include **only** usable (non-`None`) sensors, separated by ` | `, e.g.:
  - Full discrete GPU: `[01/04 compute-fp32 | 18s/60s] dev=0 | 65.0°C | 200.0W | GPU 98% | Mem 45% | Core 2100MHz | MemClk 9000MHz | Fan 55%/1900RPM`
  - Unprivileged Apple Silicon: `[01/01 compute-fp32 | 250ms/500ms] dev=0 | GPU 48%`
  - `N/A` placeholders are **never** printed on telemetry lines.
- **`json` / NDJSON mode:** `"metrics"` inside each `{"type": "telemetry", ...}` record contains **only** keys with usable values (no `null` fields).
- **Empty tick suppression:** If zero telemetry sensors are usable on a given tick (for example, an isolated container without NVML or `/sys/class/drm`), that telemetry tick is suppressed completely in both `text` and JSON/NDJSON outputs while preserving `run_start`, `workload_result`, and `run_summary`.

---

## 4. Non-Root Behavior & Troubleshooting

- **macOS (Apple Silicon & Intel):** `cross-gpu` intentionally uses unprivileged `ioreg` rather than privileged `powermetrics` or private SMC APIs so operators never need `sudo`. On Apple Silicon (`AGXAccelerator`), `PerformanceStatistics` exposes GPU utilization (`gpu_util_pct`), whereas die temperature, wattage, clock frequencies, and fan RPM are intentionally unavailable without root privileges and are therefore omitted cleanly.
- **Linux (`sysfs` & containers):**
  - If `/sys/class/drm` is not bind-mounted into a container or a virtual GPU does not expose `hwmon` nodes, missing sensors are silently omitted without causing errors.
  - Ensure your user has read permissions on `/sys/class/drm/card*/device/` (standard on Linux) and read/write permissions on `/dev/dri/renderD*` (and `/dev/kfd` for AMD ROCm) via the `render` and `video` groups (`sudo usermod -aG render,video "$USER"`).
  - For NVIDIA GPUs on Linux, ensure `libnvidia-ml.so.1` is reachable via standard linker paths or `LD_LIBRARY_PATH`.
