# `cross-gpu` Telemetry Reference

During `cross-gpu run`, a drift-compensated 1 Hz background sampler (`TelemetrySampler`) polls hardware sensors via platform-native readers with zero external Python dependencies.

---

## 1. Platform Readers

| Platform / Vendor | Source | Notes |
|---|---|---|
| Linux (NVIDIA) | NVML (`libnvidia-ml.so.1` via `ctypes`), merged with DRM/hwmon `sysfs` fallback | Matched by normalized PCIe BDF (`DDDD:BB:DD.F`) |
| Linux (AMD / Intel) | `/sys/class/drm/card*/device/` (`gpu_busy_percent`, `mem_busy_percent`, `pp_dpm_sclk`, `pp_dpm_mclk`, `gt_cur_freq_mhz`, `hwmon/hwmon*/`) | Unprivileged `sysfs` reads |
| macOS | Unprivileged `ioreg -r -d 1 -w 0 -c IOAccelerator` | Matched by Metal `registryID` (`IORegistryEntryID`) |

---

## 2. Sensor Sanitization Rules

Every sensor value is validated inside `cross_gpu/telemetry/readers.py` (`sanitize_sample` / `sanitize_metrics_dict`) before emission. Non-finite (`NaN`, `Inf`), missing, out-of-range, and sentinel readings are normalized to `None`:

| Metric Key | Unit | Valid Range |
|---|---|---|
| `temperature_c` | `°C` | Strictly between `-40.0` and `200.0` (`-40.0 < val < 200.0`) |
| `power_w` | `W` | `val >= 0.0` and finite |
| `gpu_util_pct` | `%` | `0.0 <= val <= 100.0` |
| `mem_util_pct` | `%` | `0.0 <= val <= 100.0` |
| `core_clock_mhz` | `MHz` | `val > 0` (`0 MHz` under active load is rejected) |
| `mem_clock_mhz` | `MHz` | `val > 0` |
| `fan_speed_pct` | `%` | `0.0 <= val <= 100.0` |
| `fan_speed_rpm` | `RPM` | `0 <= val < 65535` (preserves valid `0` RPM on zero-RPM fan-stop GPUs; rejects negative and sentinel values) |

---

## 3. Output Filtering (No `N/A` Placeholders or `null` Keys)

- **`text` mode:** Live telemetry lines render **only** fields that have usable (non-`None`) readings, joined cleanly with ` | ` (for example, `[01/04 compute-fp32 | 250ms/500ms] dev=0 | 64.0°C | GPU 98%`). Unavailable fields are omitted completely—`N/A` placeholders are never printed.
- **`json` / NDJSON mode:** `"metrics"` inside each `{"type": "telemetry", ...}` event contains **only** keys with usable values (no `null` entries).
- **Empty tick suppression:** If zero telemetry fields are usable on a given tick, the entire tick is suppressed in both `text` and JSON/NDJSON outputs so logs and downstream pipelines never receive empty telemetry records.
- **Prompt shutdown:** `TelemetrySampler` waits on `threading.Event.wait(timeout)`, so short sub-second workloads (e.g., `--duration 250ms` or `500ms`) wake and join the sampler thread immediately without stalling for a full 1-second interval.
