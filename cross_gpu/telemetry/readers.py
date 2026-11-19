# SPDX-License-Identifier: MIT
"""Platform telemetry readers: NVML (ctypes), Linux DRM/hwmon sysfs, and macOS ioreg."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

from cross_gpu.backends.base import bind_c_symbol, bind_optional_c_symbol, load_shared_library
from cross_gpu.dedup import normalize_pci_bdf
from cross_gpu.models import TelemetrySample

NVML_LIB_CANDIDATES: tuple[str, ...] = ("libnvidia-ml.so.1", "libnvidia-ml.so")
NVML_SUCCESS = 0
NVML_TEMPERATURE_GPU = 0
NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_MEM = 2

_DRM_SYSFS_ROOT = Path("/sys/class/drm")
_TEMP_MIN_EXCLUSIVE_C = -40.0
_TEMP_MAX_EXCLUSIVE_C = 200.0
_FAN_RPM_SENTINEL_MAX = 65535


def _as_finite_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _sanitize_temperature_c(val: Any) -> float | None:
    num = _as_finite_float(val)
    if num is None or not (_TEMP_MIN_EXCLUSIVE_C < num < _TEMP_MAX_EXCLUSIVE_C):
        return None
    return num


def _sanitize_power_w(val: Any) -> float | None:
    num = _as_finite_float(val)
    if num is None or num < 0.0:
        return None
    return num


def _sanitize_pct(val: Any) -> float | None:
    num = _as_finite_float(val)
    if num is None or not (0.0 <= num <= 100.0):
        return None
    return num


def _sanitize_clock_mhz(val: Any) -> int | None:
    num = _as_finite_float(val)
    if num is None or num <= 0.0:
        return None
    return int(round(num))


def _sanitize_fan_rpm(val: Any) -> int | None:
    num = _as_finite_float(val)
    if num is None or num < 0.0 or num >= _FAN_RPM_SENTINEL_MAX:
        return None
    return int(round(num))


def sanitize_sample(sample: TelemetrySample) -> TelemetrySample:
    """Normalize any missing, non-finite, out-of-range, or sentinel sensor reading to None."""
    return TelemetrySample(
        temperature_c=_sanitize_temperature_c(sample.temperature_c),
        power_w=_sanitize_power_w(sample.power_w),
        gpu_util_pct=_sanitize_pct(sample.gpu_util_pct),
        mem_util_pct=_sanitize_pct(sample.mem_util_pct),
        core_clock_mhz=_sanitize_clock_mhz(sample.core_clock_mhz),
        mem_clock_mhz=_sanitize_clock_mhz(sample.mem_clock_mhz),
        fan_speed_pct=_sanitize_pct(sample.fan_speed_pct),
        fan_speed_rpm=_sanitize_fan_rpm(sample.fan_speed_rpm),
    )


def sanitize_metrics_dict(metrics: Mapping[str, Any] | None) -> dict[str, float | int]:
    """Sanitize a raw telemetry metrics mapping and omit all unusable/None keys."""
    if not metrics:
        return {}
    sample = TelemetrySample(
        temperature_c=metrics.get("temperature_c"),
        power_w=metrics.get("power_w"),
        gpu_util_pct=metrics.get("gpu_util_pct"),
        mem_util_pct=metrics.get("mem_util_pct"),
        core_clock_mhz=metrics.get("core_clock_mhz"),
        mem_clock_mhz=metrics.get("mem_clock_mhz"),
        fan_speed_pct=metrics.get("fan_speed_pct"),
        fan_speed_rpm=metrics.get("fan_speed_rpm"),
    )
    return sanitize_sample(sample).to_usable_dict()


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


class _NvmlClient:
    """Lazy singleton wrapper for libnvidia-ml.so.1 with explicit ctypes signatures."""

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = None
        self._initialized = False
        self._failed = False

    def _ensure_init(self) -> bool:
        if self._initialized:
            return True
        if self._failed:
            return False
        try:
            lib = load_shared_library(NVML_LIB_CANDIDATES)
            self.nvmlInit = bind_optional_c_symbol(
                lib, "nvmlInit_v2", ctypes.c_int, []
            ) or bind_c_symbol(lib, "nvmlInit", ctypes.c_int, [])
            self.nvmlDeviceGetHandleByPciBusId = bind_optional_c_symbol(
                lib,
                "nvmlDeviceGetHandleByPciBusId_v2",
                ctypes.c_int,
                [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
            ) or bind_c_symbol(
                lib,
                "nvmlDeviceGetHandleByPciBusId",
                ctypes.c_int,
                [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
            )
            self.nvmlDeviceGetTemperature = bind_c_symbol(
                lib,
                "nvmlDeviceGetTemperature",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
            )
            self.nvmlDeviceGetPowerUsage = bind_c_symbol(
                lib,
                "nvmlDeviceGetPowerUsage",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
            )
            self.nvmlDeviceGetUtilizationRates = bind_c_symbol(
                lib,
                "nvmlDeviceGetUtilizationRates",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(_NvmlUtilization)],
            )
            self.nvmlDeviceGetClockInfo = bind_c_symbol(
                lib,
                "nvmlDeviceGetClockInfo",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
            )
            self.nvmlDeviceGetFanSpeed = bind_c_symbol(
                lib,
                "nvmlDeviceGetFanSpeed",
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
            )
            if self.nvmlInit() != NVML_SUCCESS:
                self._failed = True
                return False
            self._lib = lib
            self._initialized = True
            return True
        except Exception:
            self._failed = True
            return False


_NVML_CLIENT = _NvmlClient()


def read_nvml(pci_bdf: str | None) -> TelemetrySample:
    """Read NVIDIA GPU telemetry via minimal ctypes calls to libnvidia-ml.so.1."""
    norm_bdf = normalize_pci_bdf(pci_bdf)
    if not norm_bdf or not _NVML_CLIENT._ensure_init():
        return TelemetrySample()

    handle = ctypes.c_void_p()
    try:
        rc = _NVML_CLIENT.nvmlDeviceGetHandleByPciBusId(
            norm_bdf.encode("ascii"), ctypes.byref(handle)
        )
        if rc != NVML_SUCCESS or not handle.value:
            return TelemetrySample()
    except Exception:
        return TelemetrySample()

    dev = handle.value
    temp_c: float | None = None
    power_w: float | None = None
    gpu_util: float | None = None
    mem_util: float | None = None
    core_mhz: int | None = None
    mem_mhz: int | None = None
    fan_pct: float | None = None

    u_val = ctypes.c_uint(0)
    if _NVML_CLIENT.nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, ctypes.byref(u_val)) == NVML_SUCCESS:
        temp_c = float(u_val.value)

    if _NVML_CLIENT.nvmlDeviceGetPowerUsage(dev, ctypes.byref(u_val)) == NVML_SUCCESS:
        power_w = round(float(u_val.value) / 1000.0, 3)

    util = _NvmlUtilization()
    if _NVML_CLIENT.nvmlDeviceGetUtilizationRates(dev, ctypes.byref(util)) == NVML_SUCCESS:
        gpu_util = float(util.gpu)
        mem_util = float(util.memory)

    if _NVML_CLIENT.nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, ctypes.byref(u_val)) == NVML_SUCCESS:
        core_mhz = int(u_val.value)

    if _NVML_CLIENT.nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, ctypes.byref(u_val)) == NVML_SUCCESS:
        mem_mhz = int(u_val.value)

    if _NVML_CLIENT.nvmlDeviceGetFanSpeed(dev, ctypes.byref(u_val)) == NVML_SUCCESS:
        fan_pct = float(u_val.value)

    return sanitize_sample(
        TelemetrySample(
            temperature_c=temp_c,
            power_w=power_w,
            gpu_util_pct=gpu_util,
            mem_util_pct=mem_util,
            core_clock_mhz=core_mhz,
            mem_clock_mhz=mem_mhz,
            fan_speed_pct=fan_pct,
            fan_speed_rpm=None,
        )
    )


def _read_sysfs_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8", errors="ignore").strip())
    except (OSError, ValueError):
        return None


def _parse_dpm_active_mhz(path: Path) -> int | None:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in content.splitlines():
        match = re.search(r"(\d+)\s*mhz\s*\*", line, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _find_drm_device_dir(norm_bdf: str, drm_root: Path) -> Path | None:
    if not drm_root.is_dir():
        return None
    for card_dir in sorted(drm_root.iterdir()):
        if not re.match(r"^card\d+$", card_dir.name):
            continue
        dev_dir = card_dir / "device"
        uevent_path = dev_dir / "uevent"
        try:
            uevent_text = uevent_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in uevent_text.splitlines():
            if line.startswith("PCI_SLOT_NAME="):
                slot_bdf = normalize_pci_bdf(line.split("=", 1)[1])
                if slot_bdf == norm_bdf:
                    return dev_dir
    return None


def read_linux_sysfs(
    pci_bdf: str | None,
    *,
    drm_root: Path = _DRM_SYSFS_ROOT,
) -> TelemetrySample:
    """Read AMD/Intel GPU telemetry from Linux DRM and hwmon sysfs entries."""
    norm_bdf = normalize_pci_bdf(pci_bdf)
    if not norm_bdf:
        return TelemetrySample()

    dev_dir = _find_drm_device_dir(norm_bdf, drm_root)
    if dev_dir is None:
        return TelemetrySample()

    gpu_busy = _read_sysfs_int(dev_dir / "gpu_busy_percent")
    mem_busy = _read_sysfs_int(dev_dir / "mem_busy_percent")

    core_mhz = _parse_dpm_active_mhz(dev_dir / "pp_dpm_sclk")
    if core_mhz is None:
        core_mhz = _read_sysfs_int(dev_dir / "gt_cur_freq_mhz")
    if core_mhz is None:
        core_mhz = _read_sysfs_int(dev_dir / "gt" / "gt0" / "rps_cur_freq_mhz")
    mem_mhz = _parse_dpm_active_mhz(dev_dir / "pp_dpm_mclk")

    temp_c: float | None = None
    power_w: float | None = None
    fan_pct: float | None = None
    fan_rpm: int | None = None

    hwmon_root = dev_dir / "hwmon"
    if hwmon_root.is_dir():
        for hw_dir in sorted(hwmon_root.iterdir()):
            if not hw_dir.is_dir():
                continue
            if temp_c is None:
                raw_temp = _read_sysfs_int(hw_dir / "temp1_input")
                if raw_temp is not None:
                    temp_c = round(raw_temp / 1000.0, 2)
            if power_w is None:
                raw_power = _read_sysfs_int(hw_dir / "power1_average")
                if raw_power is None:
                    raw_power = _read_sysfs_int(hw_dir / "power1_input")
                if raw_power is not None:
                    power_w = round(raw_power / 1_000_000.0, 3)
            if fan_pct is None:
                raw_pwm = _read_sysfs_int(hw_dir / "pwm1")
                if raw_pwm is not None:
                    pwm_max = _read_sysfs_int(hw_dir / "pwm1_max") or 255
                    if pwm_max > 0:
                        fan_pct = round(raw_pwm * 100.0 / pwm_max, 1)
            if fan_rpm is None:
                fan_rpm = _read_sysfs_int(hw_dir / "fan1_input")

    return sanitize_sample(
        TelemetrySample(
            temperature_c=temp_c,
            power_w=power_w,
            gpu_util_pct=float(gpu_busy) if gpu_busy is not None else None,
            mem_util_pct=float(mem_busy) if mem_busy is not None else None,
            core_clock_mhz=core_mhz,
            mem_clock_mhz=mem_mhz,
            fan_speed_pct=fan_pct,
            fan_speed_rpm=fan_rpm,
        )
    )


def _extract_ioreg_number(block: str, *keys: str) -> float | None:
    for key in keys:
        escaped = re.escape(key)
        match = re.search(rf'"{escaped}"\s*=\s*(-?\d+(?:\.\d+)?)', block)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def parse_ioreg_output(output: str, registry_id: int | None) -> TelemetrySample:
    """Parse `ioreg -r -d 1 -w 0 -c IOAccelerator` text into a TelemetrySample."""
    if not output:
        return TelemetrySample()

    blocks = re.split(r"(?=^\+-o\s+)", output, flags=re.MULTILINE)
    target_block: str | None = None
    for block in blocks:
        if "+-o" not in block:
            continue
        if registry_id is None:
            target_block = block
            break
        id_match = re.search(r"\bid\s+(0x[0-9a-fA-F]+|\d+)", block)
        block_reg_id: int | None = None
        if id_match:
            raw = id_match.group(1)
            block_reg_id = int(raw, 16) if raw.startswith("0x") else int(raw)
        else:
            entry_val = _extract_ioreg_number(block, "IORegistryEntryID")
            if entry_val is not None:
                block_reg_id = int(entry_val)
        if block_reg_id == registry_id:
            target_block = block
            break

    if not target_block:
        return TelemetrySample()

    perf_match = re.search(r'"PerformanceStatistics"\s*=\s*\{([^}]*)\}', target_block)
    scope = perf_match.group(1) if perf_match else target_block

    temp_c = _extract_ioreg_number(scope, "Temperature(C)")
    power_w = _extract_ioreg_number(scope, "GPU Power(W)", "Total Power(W)")
    gpu_util = _extract_ioreg_number(
        scope, "Device Utilization %", "GPU Activity(%)", "Renderer Utilization %"
    )
    mem_util = _extract_ioreg_number(scope, "Memory Utilization %", "VRAM Utilization %")
    core_clk = _extract_ioreg_number(scope, "Core Clock(MHz)")
    mem_clk = _extract_ioreg_number(scope, "Memory Clock(MHz)")
    fan_pct = _extract_ioreg_number(scope, "Fan Speed(%)")
    fan_rpm = _extract_ioreg_number(scope, "Fan Speed(RPM)")

    return sanitize_sample(
        TelemetrySample(
            temperature_c=temp_c,
            power_w=power_w,
            gpu_util_pct=gpu_util,
            mem_util_pct=mem_util,
            core_clock_mhz=int(core_clk) if core_clk is not None else None,
            mem_clock_mhz=int(mem_clk) if mem_clk is not None else None,
            fan_speed_pct=fan_pct,
            fan_speed_rpm=int(fan_rpm) if fan_rpm is not None else None,
        )
    )


def read_macos(registry_id: int | None) -> TelemetrySample:
    """Read macOS GPU telemetry via unprivileged `ioreg` IOAccelerator query."""
    if sys.platform != "darwin":
        return TelemetrySample()
    try:
        proc = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return TelemetrySample()
    if proc.returncode != 0 or not proc.stdout:
        return TelemetrySample()
    return parse_ioreg_output(proc.stdout, registry_id)


def merge_samples(primary: TelemetrySample, fallback: TelemetrySample) -> TelemetrySample:
    """Fill missing fields in sanitized primary TelemetrySample using sanitized fallback values."""
    clean_primary = sanitize_sample(primary)
    clean_fallback = sanitize_sample(fallback)
    return TelemetrySample(
        temperature_c=(
            clean_primary.temperature_c
            if clean_primary.temperature_c is not None
            else clean_fallback.temperature_c
        ),
        power_w=(
            clean_primary.power_w
            if clean_primary.power_w is not None
            else clean_fallback.power_w
        ),
        gpu_util_pct=(
            clean_primary.gpu_util_pct
            if clean_primary.gpu_util_pct is not None
            else clean_fallback.gpu_util_pct
        ),
        mem_util_pct=(
            clean_primary.mem_util_pct
            if clean_primary.mem_util_pct is not None
            else clean_fallback.mem_util_pct
        ),
        core_clock_mhz=(
            clean_primary.core_clock_mhz
            if clean_primary.core_clock_mhz is not None
            else clean_fallback.core_clock_mhz
        ),
        mem_clock_mhz=(
            clean_primary.mem_clock_mhz
            if clean_primary.mem_clock_mhz is not None
            else clean_fallback.mem_clock_mhz
        ),
        fan_speed_pct=(
            clean_primary.fan_speed_pct
            if clean_primary.fan_speed_pct is not None
            else clean_fallback.fan_speed_pct
        ),
        fan_speed_rpm=(
            clean_primary.fan_speed_rpm
            if clean_primary.fan_speed_rpm is not None
            else clean_fallback.fan_speed_rpm
        ),
    )


__all__ = [
    "merge_samples",
    "parse_ioreg_output",
    "read_linux_sysfs",
    "read_macos",
    "read_nvml",
    "sanitize_metrics_dict",
    "sanitize_sample",
]
