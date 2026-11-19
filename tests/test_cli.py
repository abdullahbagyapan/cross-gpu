# SPDX-License-Identifier: MIT
"""Unit tests for CLI argument parsing, --warmup edge cases, JSON/NDJSON, and exit code 4."""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

from cross_gpu.backends.base import (
    BackendContext,
    BackendPlugin,
    DeviceBuffer,
    HostBuffer,
    KernelModule,
)
from cross_gpu.cli import (
    build_parser,
    format_telemetry_line,
    main,
    parse_duration,
    parse_test_arg,
    resolve_output_format,
)
from cross_gpu.exit_codes import BackendHardwareError, ExitCode
from cross_gpu.models import BackendBinding, BackendDeviceInfo, DeviceCapabilities, GPUDevice
from cross_gpu.workloads import list_workload_names


class _FaultyHardwarePlugin(BackendPlugin):
    name = "cuda"
    priority = 100

    def is_available(self) -> bool:
        return True

    def enumerate_devices(self) -> list[BackendDeviceInfo]:
        return []

    def create_context(self, device: BackendDeviceInfo) -> BackendContext:
        raise BackendHardwareError("Xid 79: GPU has fallen off the bus")

    def destroy_context(self, ctx: BackendContext) -> None:
        pass

    def synchronize(self, ctx: BackendContext) -> None:
        pass

    def alloc_device(self, ctx: BackendContext, size_bytes: int) -> DeviceBuffer:
        return DeviceBuffer(backend=self.name, handle=1, size_bytes=size_bytes)

    def free_device(self, ctx: BackendContext, buf: DeviceBuffer) -> None:
        pass

    def alloc_pinned_host(self, ctx: BackendContext, size_bytes: int) -> HostBuffer:
        return HostBuffer(backend=self.name, handle=1, size_bytes=size_bytes)

    def free_pinned_host(self, ctx: BackendContext, buf: HostBuffer) -> None:
        pass

    def memcpy_h2d(self, ctx: BackendContext, dst: DeviceBuffer, src: object, size_bytes: int) -> None:
        pass

    def memcpy_d2h(self, ctx: BackendContext, dst: object, src: DeviceBuffer, size_bytes: int) -> None:
        pass

    def memcpy_d2d(self, ctx: BackendContext, dst: DeviceBuffer, src: DeviceBuffer, size_bytes: int) -> None:
        pass

    def load_module(self, ctx: BackendContext, compiled_binary: object) -> KernelModule:
        return KernelModule(backend=self.name, handle=1)

    def unload_module(self, ctx: BackendContext, module: KernelModule) -> None:
        pass

    def launch_kernel(self, ctx: BackendContext, module: KernelModule, entry_point: str, grid: tuple[int, int, int], block: tuple[int, int, int], args: object, shared_mem: int = 0) -> None:
        pass


def _sample_device() -> GPUDevice:
    caps = DeviceCapabilities(fp64=True, fp16=True)
    info = BackendDeviceInfo(
        backend="cuda",
        priority=100,
        backend_device_index=0,
        native_handle=0,
        name="NVIDIA RTX 4090",
        vendor="NVIDIA",
        vendor_id=0x10DE,
        architecture="sm_89",
        compute_capability="8.9",
        total_memory_bytes=24 * 1024**3,
        compute_units=128,
        driver_version="550.54",
        pci_bdf="0000:01:00.0",
        capabilities=caps,
    )
    return GPUDevice(
        index=0,
        name=info.name,
        vendor=info.vendor,
        vendor_id=info.vendor_id,
        architecture=info.architecture,
        compute_capability=info.compute_capability,
        total_memory_bytes=info.total_memory_bytes,
        compute_units=info.compute_units,
        driver_version=info.driver_version,
        pci_bdf=info.pci_bdf,
        capabilities=caps,
        bindings=[BackendBinding(backend="cuda", priority=100, device_info=info)],
    )


class TestCLI(unittest.TestCase):
    def test_duration_parser(self) -> None:
        self.assertEqual(parse_duration("30s"), 30.0)
        self.assertEqual(parse_duration("5m"), 300.0)
        self.assertEqual(parse_duration("1h"), 3600.0)
        self.assertEqual(parse_duration("0.5m"), 30.0)
        self.assertEqual(parse_duration("0s", allow_zero=True), 0.0)

        for bad in ("0s", "-5s", "60", "5d", "abc", "", "nans", "infs"):
            with self.subTest(bad=bad):
                with self.assertRaises(Exception):
                    parse_duration(bad, allow_zero=False)

    def test_warmup_flag_edge_cases(self) -> None:
        parser = build_parser()

        # Omitted -> 0s
        a1 = parser.parse_args(["run"])
        self.assertEqual(a1.warmup, "0s")

        # Passed without value immediately before another flag -> 5s
        a2 = parser.parse_args(["run", "--warmup", "--json"])
        self.assertEqual(a2.warmup, "5s")
        self.assertTrue(a2.json)

        # Passed at end without value -> 5s
        a3 = parser.parse_args(["run", "--warmup"])
        self.assertEqual(a3.warmup, "5s")

        # Passed with explicit value (--warmup 10s / --warmup=10s)
        a4 = parser.parse_args(["run", "--warmup", "10s"])
        self.assertEqual(a4.warmup, "10s")
        a5 = parser.parse_args(["run", "--warmup=10s"])
        self.assertEqual(a5.warmup, "10s")

    def test_format_and_json_conflict_returns_usage_error(self) -> None:
        self.assertEqual(resolve_output_format(None, False), "text")
        self.assertEqual(resolve_output_format(None, True), "json")
        self.assertEqual(resolve_output_format("json", True), "json")

        for cmd in ("list", "info", "run"):
            out, err = io.StringIO(), io.StringIO()
            rc = main([cmd, "--format", "text", "--json"], stdout=out, stderr=err)
            self.assertEqual(rc, ExitCode.USAGE_ERROR)

    def test_parse_test_lists_and_invalid_args_exit_1(self) -> None:
        all_names, explicit_all = parse_test_arg("all")
        self.assertEqual(all_names, list_workload_names())
        self.assertEqual(len(all_names), 20)
        self.assertFalse(explicit_all)

        subset, explicit_sub = parse_test_arg("compute-fp32,memory-pattern")
        self.assertEqual(subset, ["compute-fp32", "memory-pattern"])
        self.assertTrue(explicit_sub)

        for bad_argv in (
            [],
            ["unknown-cmd"],
            ["run", "--unknown-flag"],
            ["run", "--test", "nonexistent-test"],
            ["run", "--test", "all,compute-fp32"],
            ["run", "--device", "-1"],
            ["run", "--duration", "0s"],
        ):
            with self.subTest(bad_argv=bad_argv):
                out, err = io.StringIO(), io.StringIO()
                try:
                    rc = main(bad_argv, stdout=out, stderr=err)
                except SystemExit as exc:
                    rc = int(exc.code)
                self.assertEqual(rc, ExitCode.USAGE_ERROR)

    def test_list_and_info_json_output(self) -> None:
        dev = _sample_device()
        for cmd in ("list", "info"):
            out, err = io.StringIO(), io.StringIO()
            rc = main(
                [cmd, "--json"],
                stdout=out,
                stderr=err,
                discover_fn=lambda: ([dev], {"cuda": _FaultyHardwarePlugin()}),
            )
            self.assertEqual(rc, ExitCode.SUCCESS)
            parsed = json.loads(out.getvalue())
            self.assertEqual(len(parsed["devices"]), 1)
            self.assertEqual(parsed["devices"][0]["name"], "NVIDIA RTX 4090")

    def test_run_ndjson_and_hardware_error_exit_code_4(self) -> None:
        dev = _sample_device()
        out, err = io.StringIO(), io.StringIO()
        rc = main(
            ["run", "--device", "0", "--test", "compute-fp32", "--duration", "1s", "--json"],
            stdout=out,
            stderr=err,
            discover_fn=lambda: ([dev], {"cuda": _FaultyHardwarePlugin()}),
        )
        self.assertEqual(rc, ExitCode.HARDWARE_ERROR)
        lines = [json.loads(line) for line in out.getvalue().strip().splitlines()]
        types = [obj["type"] for obj in lines]
        self.assertIn("run_start", types)
        self.assertIn("workload_result", types)
        self.assertIn("run_summary", types)
        self.assertEqual(lines[-1]["exit_code"], 4)

    def test_text_telemetry_line_includes_all_8_fields_and_units(self) -> None:
        line = format_telemetry_line(
            {
                "elapsed_s": 1.0,
                "device": 0,
                "phase": "run",
                "workload": "compute-fp32",
                "metrics": {
                    "temperature_c": 65.0,
                    "power_w": 200.0,
                    "gpu_util_pct": 98.0,
                    "mem_util_pct": 45.0,
                    "core_clock_mhz": 2100,
                    "mem_clock_mhz": 9000,
                    "fan_speed_pct": 55.0,
                    "fan_speed_rpm": 1900,
                },
            }
        )
        for token in ("°C", "W", "%", "MHz", "RPM", "65.0°C", "200.0W", "2100MHz", "1900RPM"):
            self.assertIn(token, line)

    def test_subprocess_entry_points(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        exe = repo_root / "cross-gpu"

        for cmd in (
            [str(exe), "--help"],
            [str(exe), "--version"],
            [sys.executable, "-m", "cross_gpu", "--version"],
        ):
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
