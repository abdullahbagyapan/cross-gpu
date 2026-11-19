# SPDX-License-Identifier: MIT
"""Unit tests for NVML/sysfs/IOKit telemetry readers and 1 Hz sampler cadence."""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from src.models import GPUDevice, TelemetrySample
from src.telemetry import TelemetrySampler, read_telemetry
from src.telemetry.readers import (
    parse_ioreg_output,
    read_linux_sysfs,
    read_nvml,
    sanitize_metrics_dict,
    sanitize_sample,
)


SAMPLE_IOREG_APPLE = """+-o AGXAcceleratorG14G  <class AGXAcceleratorG14G, id 0x1000003e8, registered, matched, active, busy 0 (606 ms), retain 55>
    {
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=2338521088,"Tiler Utilization %"=40,"Renderer Utilization %"=39,"Device Utilization %"=42}
      "gpu-core-count" = 10
      "model" = "Apple M2"
    }
"""

SAMPLE_IOREG_INTEL_MAC = """+-o AMDRadeonX6000Accelerator  <class AMDRadeonX6000Accelerator, id 0x1000004aa, registered>
    {
      "PerformanceStatistics" = {"Device Utilization %"=87,"Memory Utilization %"=65,"Temperature(C)"=73,"Core Clock(MHz)"=1750,"Memory Clock(MHz)"=1000,"Fan Speed(%)"=55,"Fan Speed(RPM)"=2400,"GPU Power(W)"=112}
    }
"""


class TestTelemetry(unittest.TestCase):
    def test_linux_sysfs_reader_amd_and_intel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            drm = Path(tmp)
            card0 = drm / "card0" / "device"
            hwmon = card0 / "hwmon" / "hwmon3"
            hwmon.mkdir(parents=True)
            (card0 / "uevent").write_text(
                "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:03:00.0\n",
                encoding="utf-8",
            )
            (card0 / "gpu_busy_percent").write_text("91\n", encoding="utf-8")
            (card0 / "mem_busy_percent").write_text("44\n", encoding="utf-8")
            (card0 / "pp_dpm_sclk").write_text(
                "0: 500Mhz \n1: 2150Mhz *\n2: 2500Mhz \n",
                encoding="utf-8",
            )
            (card0 / "pp_dpm_mclk").write_text(
                "0: 96Mhz \n1: 1200Mhz *\n",
                encoding="utf-8",
            )
            (hwmon / "temp1_input").write_text("68500\n", encoding="utf-8")
            (hwmon / "power1_average").write_text("245000000\n", encoding="utf-8")
            (hwmon / "pwm1").write_text("128\n", encoding="utf-8")
            (hwmon / "fan1_input").write_text("1850\n", encoding="utf-8")

            sample = read_linux_sysfs("0000:03:00.0", drm_root=drm)
            self.assertEqual(sample.temperature_c, 68.5)
            self.assertEqual(sample.power_w, 245.0)
            self.assertEqual(sample.gpu_util_pct, 91.0)
            self.assertEqual(sample.mem_util_pct, 44.0)
            self.assertEqual(sample.core_clock_mhz, 2150)
            self.assertEqual(sample.mem_clock_mhz, 1200)
            self.assertAlmostEqual(sample.fan_speed_pct or 0.0, 50.2, places=1)
            self.assertEqual(sample.fan_speed_rpm, 1850)

            # Intel gt_cur_freq_mhz path
            card1 = drm / "card1" / "device"
            card1.mkdir(parents=True)
            (card1 / "uevent").write_text(
                "DRIVER=i915\nPCI_SLOT_NAME=0000:00:02.0\n",
                encoding="utf-8",
            )
            (card1 / "gt_cur_freq_mhz").write_text("1450\n", encoding="utf-8")
            intel_sample = read_linux_sysfs("00:02.0", drm_root=drm)
            self.assertEqual(intel_sample.core_clock_mhz, 1450)
            self.assertIsNone(intel_sample.temperature_c)

    def test_macos_ioreg_parser(self) -> None:
        m2 = parse_ioreg_output(SAMPLE_IOREG_APPLE, 0x1000003E8)
        self.assertEqual(m2.gpu_util_pct, 42.0)
        self.assertIsNone(m2.temperature_c)
        self.assertIsNone(m2.power_w)

        combined = SAMPLE_IOREG_APPLE + "\n" + SAMPLE_IOREG_INTEL_MAC
        dgpu = parse_ioreg_output(combined, 0x1000004AA)
        self.assertEqual(dgpu.temperature_c, 73.0)
        self.assertEqual(dgpu.power_w, 112.0)
        self.assertEqual(dgpu.gpu_util_pct, 87.0)
        self.assertEqual(dgpu.mem_util_pct, 65.0)
        self.assertEqual(dgpu.core_clock_mhz, 1750)
        self.assertEqual(dgpu.mem_clock_mhz, 1000)
        self.assertEqual(dgpu.fan_speed_pct, 55.0)
        self.assertEqual(dgpu.fan_speed_rpm, 2400)

        missing = parse_ioreg_output(combined, 0xDEADBEEF)
        self.assertEqual(missing, TelemetrySample())

    def test_nvml_reader_never_raises_when_missing(self) -> None:
        with mock.patch("src.telemetry.readers._NVML_CLIENT._ensure_init", return_value=False):
            sample = read_nvml("0000:01:00.0")
            self.assertEqual(sample, TelemetrySample())

    def test_nvml_reader_mocked_values(self) -> None:
        class FakeClient:
            def _ensure_init(self) -> bool:
                return True

            def nvmlDeviceGetHandleByPciBusId(self, bdf: bytes, out_ptr: object) -> int:
                self.bdf = bdf
                out_ptr._obj.value = 12345  # type: ignore[attr-defined]
                return 0

            def nvmlDeviceGetTemperature(self, dev: int, sensor: int, out: object) -> int:
                out._obj.value = 64  # type: ignore[attr-defined]
                return 0

            def nvmlDeviceGetPowerUsage(self, dev: int, out: object) -> int:
                out._obj.value = 315500  # type: ignore[attr-defined]
                return 0

            def nvmlDeviceGetUtilizationRates(self, dev: int, out: object) -> int:
                out._obj.gpu = 99  # type: ignore[attr-defined]
                out._obj.memory = 88  # type: ignore[attr-defined]
                return 0

            def nvmlDeviceGetClockInfo(self, dev: int, kind: int, out: object) -> int:
                out._obj.value = 2520 if kind == 0 else 10501  # type: ignore[attr-defined]
                return 0

            def nvmlDeviceGetFanSpeed(self, dev: int, out: object) -> int:
                out._obj.value = 60  # type: ignore[attr-defined]
                return 0

        with mock.patch("src.telemetry.readers._NVML_CLIENT", FakeClient()):
            sample = read_nvml("01:00.0")
            self.assertEqual(sample.temperature_c, 64.0)
            self.assertEqual(sample.power_w, 315.5)
            self.assertEqual(sample.gpu_util_pct, 99.0)
            self.assertEqual(sample.mem_util_pct, 88.0)
            self.assertEqual(sample.core_clock_mhz, 2520)
            self.assertEqual(sample.mem_clock_mhz, 10501)
            self.assertEqual(sample.fan_speed_pct, 60.0)
            self.assertIsNone(sample.fan_speed_rpm)

    def test_sampler_cadence_and_clean_stop(self) -> None:
        gpu = GPUDevice(index=0, name="Mock GPU", vendor="NVIDIA", pci_bdf="0000:01:00.0")
        timestamps: list[float] = []
        done = threading.Event()

        def on_sample(sample: TelemetrySample, elapsed_s: float) -> None:
            timestamps.append(elapsed_s)
            if len(timestamps) >= 3:
                done.set()

        sampler = TelemetrySampler(
            gpu,
            on_sample,
            interval_s=0.04,
            reader_fn=lambda d: TelemetrySample(temperature_c=55.0),
        )
        t0 = time.monotonic()
        sampler.start()
        self.assertTrue(done.wait(timeout=1.0))
        sampler.stop(timeout_s=1.0)
        elapsed_total = time.monotonic() - t0
        self.assertGreaterEqual(len(timestamps), 3)
        self.assertLess(elapsed_total, 0.9)

    def test_read_telemetry_router_handles_none(self) -> None:
        self.assertEqual(read_telemetry(None), TelemetrySample())

    def test_sanitize_sample_rejects_bogus_and_preserves_valid_values(self) -> None:
        bogus = TelemetrySample(
            temperature_c=float("nan"),
            power_w=-12.5,
            gpu_util_pct=105.0,
            mem_util_pct=-1.0,
            core_clock_mhz=0,
            mem_clock_mhz=-400,
            fan_speed_pct=float("inf"),
            fan_speed_rpm=-10,
        )
        self.assertEqual(sanitize_sample(bogus), TelemetrySample())
        self.assertEqual(sanitize_sample(bogus).to_usable_dict(), {})

        # Boundary checks: temp (-40, 200) exclusive, sentinel RPM rejected, valid 0 RPM kept.
        for bad_temp in (-273.0, -40.0, 200.0, 999.0, float("-inf")):
            self.assertIsNone(sanitize_sample(TelemetrySample(temperature_c=bad_temp)).temperature_c)
        self.assertEqual(sanitize_sample(TelemetrySample(temperature_c=64.0)).temperature_c, 64.0)
        self.assertEqual(sanitize_sample(TelemetrySample(power_w=0.0)).power_w, 0.0)
        self.assertEqual(sanitize_sample(TelemetrySample(fan_speed_rpm=0)).fan_speed_rpm, 0)
        self.assertIsNone(sanitize_sample(TelemetrySample(fan_speed_rpm=65535)).fan_speed_rpm)
        self.assertEqual(
            sanitize_metrics_dict({"temperature_c": 64.0, "power_w": -1.0, "core_clock_mhz": 0}),
            {"temperature_c": 64.0},
        )

        # Readers sanitize sysfs and ioreg bogus/sentinel values automatically
        with tempfile.TemporaryDirectory() as tmp:
            drm = Path(tmp)
            card0 = drm / "card0" / "device"
            hwmon = card0 / "hwmon" / "hwmon0"
            hwmon.mkdir(parents=True)
            (card0 / "uevent").write_text("PCI_SLOT_NAME=0000:05:00.0\n", encoding="utf-8")
            (card0 / "gpu_busy_percent").write_text("150\n", encoding="utf-8")
            (card0 / "gt_cur_freq_mhz").write_text("0\n", encoding="utf-8")
            (hwmon / "temp1_input").write_text("-273000\n", encoding="utf-8")
            (hwmon / "power1_average").write_text("-5000000\n", encoding="utf-8")
            (hwmon / "fan1_input").write_text("0\n", encoding="utf-8")
            sample = read_linux_sysfs("0000:05:00.0", drm_root=drm)
            self.assertIsNone(sample.temperature_c)
            self.assertIsNone(sample.power_w)
            self.assertIsNone(sample.gpu_util_pct)
            self.assertIsNone(sample.core_clock_mhz)
            self.assertEqual(sample.fan_speed_rpm, 0)

        bad_ioreg = """+-o AGXAccelerator  <class AGXAccelerator, id 0x1000, registered>
            {
              "PerformanceStatistics" = {"Temperature(C)"=250,"GPU Power(W)"=-5,"Device Utilization %"=95,"Core Clock(MHz)"=0,"Fan Speed(RPM)"=0}
            }
        """
        parsed = parse_ioreg_output(bad_ioreg, 0x1000)
        self.assertIsNone(parsed.temperature_c)
        self.assertIsNone(parsed.power_w)
        self.assertIsNone(parsed.core_clock_mhz)
        self.assertEqual(parsed.gpu_util_pct, 95.0)
        self.assertEqual(parsed.fan_speed_rpm, 0)

    def test_sampler_stops_promptly_without_waiting_full_1s_interval(self) -> None:
        gpu = GPUDevice(index=0, name="Mock GPU", vendor="NVIDIA", pci_bdf="0000:01:00.0")
        got_first = threading.Event()
        sampler = TelemetrySampler(
            gpu,
            lambda _s, _e: got_first.set(),
            interval_s=1.0,
            reader_fn=lambda _d: TelemetrySample(temperature_c=55.0),
        )
        sampler.start()
        self.assertTrue(got_first.wait(timeout=0.5))
        t_stop = time.monotonic()
        sampler.stop(timeout_s=1.0)
        stop_elapsed = time.monotonic() - t_stop
        self.assertLess(stop_elapsed, 0.25)


if __name__ == "__main__":
    unittest.main()
