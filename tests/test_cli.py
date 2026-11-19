# SPDX-License-Identifier: MIT
"""Unit tests for CLI argument parsing, --warmup edge cases, JSON/NDJSON, and exit code 4."""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from cross_gpu.backends.base import (
    BackendContext,
    BackendPlugin,
    DeviceBuffer,
    HostBuffer,
    KernelModule,
    check_linux_render_node_hint,
)
from cross_gpu.backends.registry import KNOWN_BACKENDS
from cross_gpu.cli import (
    build_parser,
    format_duration,
    format_progress_prefix,
    format_run_summary_text,
    format_telemetry_line,
    main,
    parse_duration,
    parse_test_arg,
    resolve_output_format,
)
from cross_gpu.exit_codes import BackendHardwareError, ExitCode
from cross_gpu.models import (
    BackendBinding,
    BackendDeviceInfo,
    DeviceCapabilities,
    GPUDevice,
    WorkloadResult,
)
from cross_gpu.workloads import (
    WORKLOAD_REGISTRY,
    explain_unsupported_workload,
    list_workload_names,
)


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
        self.assertAlmostEqual(parse_duration("1ms"), 0.001)
        self.assertAlmostEqual(parse_duration("250ms"), 0.25)
        self.assertAlmostEqual(parse_duration("500ms"), 0.5)
        self.assertAlmostEqual(parse_duration("1500ms"), 1.5)
        self.assertAlmostEqual(parse_duration("500MS"), 0.5)
        self.assertEqual(parse_duration("30s"), 30.0)
        self.assertEqual(parse_duration("5m"), 300.0)
        self.assertEqual(parse_duration("1h"), 3600.0)
        self.assertEqual(parse_duration("0.5m"), 30.0)
        self.assertEqual(parse_duration("0s", allow_zero=True), 0.0)
        self.assertEqual(parse_duration("0ms", allow_zero=True), 0.0)

        for bad in ("0ms", "0s", "-5ms", "-100ms", "-5s", "ms", "10mms", "60", "5d", "abc", "", "nans", "infs"):
            with self.subTest(bad=bad):
                with self.assertRaises(Exception):
                    parse_duration(bad, allow_zero=False)

        # CLI-level rejection returns ExitCode.USAGE_ERROR (1)
        for bad_argv in (
            ["run", "--duration", "0ms"],
            ["run", "--duration=-100ms"],
            ["run", "--duration", "-100ms"],
            ["run", "--duration", "ms"],
            ["run", "--duration", "10mms"],
        ):
            out, err = io.StringIO(), io.StringIO()
            try:
                rc = main(bad_argv, stdout=out, stderr=err)
            except SystemExit as exc:
                rc = int(exc.code)
            self.assertEqual(rc, ExitCode.USAGE_ERROR)
            self.assertEqual(out.getvalue(), "")

        # --warmup accepts 0ms and 250ms
        parser = build_parser()
        self.assertEqual(parse_duration(parser.parse_args(["run", "--warmup", "0ms"]).warmup, allow_zero=True), 0.0)
        self.assertAlmostEqual(parse_duration(parser.parse_args(["run", "--warmup", "250ms"]).warmup, allow_zero=True), 0.25)

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

    def test_text_telemetry_line_omits_unusable_fields_and_suppresses_empty(self) -> None:
        # Partial telemetry: only temp and GPU util usable -> never prints N/A or missing fields
        partial_line = format_telemetry_line(
            {
                "elapsed_s": 5.0,
                "device": 0,
                "phase": "run",
                "workload": "compute-fp32",
                "workload_index": 1,
                "workload_total": 4,
                "phase_elapsed_s": 0.25,
                "phase_duration_s": 0.5,
                "metrics": {
                    "temperature_c": 64.0,
                    "power_w": None,
                    "gpu_util_pct": 98.0,
                    "mem_util_pct": None,
                    "core_clock_mhz": 0,
                    "mem_clock_mhz": None,
                    "fan_speed_pct": None,
                    "fan_speed_rpm": None,
                },
            }
        )
        self.assertIn("[01/04 compute-fp32 | 250ms/500ms]", partial_line)
        self.assertIn("64.0°C | GPU 98%", partial_line)
        self.assertNotIn("N/A", partial_line)
        self.assertNotIn("W", partial_line)
        self.assertNotIn("MHz", partial_line)
        self.assertNotIn("Fan", partial_line)

        # Zero-RPM fan-stop GPU preserves valid 0RPM without N/A
        zero_rpm_line = format_telemetry_line(
            {
                "elapsed_s": 1.0,
                "device": 0,
                "phase": "run",
                "workload": "compute-fp32",
                "metrics": {"fan_speed_rpm": 0},
            }
        )
        self.assertIn("Fan 0RPM", zero_rpm_line)
        self.assertNotIn("N/A", zero_rpm_line)

        # All 8 fields unusable/None -> returns empty string (suppressed)
        empty_line = format_telemetry_line(
            {
                "elapsed_s": 1.0,
                "device": 0,
                "phase": "run",
                "workload": "compute-fp32",
                "metrics": {
                    "temperature_c": None,
                    "power_w": float("nan"),
                    "gpu_util_pct": 150.0,
                    "mem_util_pct": -1.0,
                    "core_clock_mhz": 0,
                    "mem_clock_mhz": -50,
                    "fan_speed_pct": None,
                    "fan_speed_rpm": -1,
                },
            }
        )
        self.assertEqual(empty_line, "")

    def test_subsecond_duration_and_progress_formatting(self) -> None:
        self.assertEqual(format_duration(0.001), "1ms")
        self.assertEqual(format_duration(0.25), "250ms")
        self.assertEqual(format_duration(0.5), "500ms")
        self.assertEqual(format_duration(1.0), "1s")
        self.assertEqual(format_duration(1.5), "1.5s")
        self.assertEqual(format_duration(60.0), "60s")

        self.assertEqual(
            format_progress_prefix(
                "compute-fp32",
                workload_index=1,
                workload_total=4,
                phase="warmup",
                phase_elapsed_s=0.25,
                phase_duration_s=0.5,
            ),
            "[01/04 compute-fp32 | warmup 250ms/500ms]",
        )
        self.assertEqual(
            format_progress_prefix(
                "compute-fp32",
                workload_index=1,
                workload_total=4,
                phase="run",
                phase_elapsed_s=0.25,
                phase_duration_s=0.5,
            ),
            "[01/04 compute-fp32 | 250ms/500ms]",
        )

    def test_cli_json_and_text_suppress_empty_telemetry_ticks(self) -> None:
        dev = _sample_device()

        class _FakeRunner:
            def __init__(self, *, event_callback=None, **kwargs):
                self._cb = event_callback

            def run(self):
                from cross_gpu.runner import RunReport

                # Empty/unusable tick -> must be suppressed in BOTH text and json
                self._cb(
                    {
                        "type": "telemetry",
                        "elapsed_s": 0.1,
                        "device": 0,
                        "phase": "run",
                        "workload": "compute-fp32",
                        "metrics": {"temperature_c": None, "core_clock_mhz": 0},
                    }
                )
                # Partial usable tick -> only non-null keys emitted
                self._cb(
                    {
                        "type": "telemetry",
                        "elapsed_s": 0.2,
                        "device": 0,
                        "phase": "run",
                        "workload": "compute-fp32",
                        "workload_index": 1,
                        "workload_total": 1,
                        "phase_elapsed_s": 0.2,
                        "phase_duration_s": 0.5,
                        "metrics": {"temperature_c": 64.0, "power_w": None, "gpu_util_pct": 98.0},
                    }
                )
                res = WorkloadResult(
                    0, "NVIDIA RTX 4090", "cuda", "compute-fp32", "compute-fp",
                    "pass", 0.5, 100.0, "GFLOP/s", 50.0, "GFLOP", 0, "", "", ExitCode.SUCCESS,
                )
                self._cb({"type": "workload_result", "workload_index": 1, "workload_total": 1, "target_duration_s": 0.5, **res.to_dict()})
                self._cb({"type": "run_summary", "exit_code": 0, "total_workloads": 1, "results": [res.to_dict()]})
                return RunReport(exit_code=ExitCode.SUCCESS, results=[res])

        out_json, err_json = io.StringIO(), io.StringIO()
        rc_json = main(
            ["run", "--device", "0", "--test", "compute-fp32", "--duration", "500ms", "--json"],
            stdout=out_json,
            stderr=err_json,
            discover_fn=lambda: ([dev], {"cuda": _FaultyHardwarePlugin()}),
            runner_factory=_FakeRunner,
        )
        self.assertEqual(rc_json, ExitCode.SUCCESS)
        objs = [json.loads(line) for line in out_json.getvalue().strip().splitlines()]
        telem_objs = [o for o in objs if o["type"] == "telemetry"]
        self.assertEqual(len(telem_objs), 1)
        self.assertEqual(telem_objs[0]["metrics"], {"temperature_c": 64.0, "gpu_util_pct": 98.0})
        self.assertNotIn("power_w", telem_objs[0]["metrics"])

        out_text, err_text = io.StringIO(), io.StringIO()
        rc_text = main(
            ["run", "--device", "0", "--test", "compute-fp32", "--duration", "500ms"],
            stdout=out_text,
            stderr=err_text,
            discover_fn=lambda: ([dev], {"cuda": _FaultyHardwarePlugin()}),
            runner_factory=_FakeRunner,
        )
        self.assertEqual(rc_text, ExitCode.SUCCESS)
        self.assertNotIn("N/A", out_text.getvalue())
        self.assertIn("[01/01 compute-fp32 | 200ms/500ms] dev=0 | 64.0°C | GPU 98%", out_text.getvalue())

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

    def test_help_epilog_examples_and_stdout_routing(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            main(["--help"], stdout=out, stderr=err)
        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(err.getvalue(), "")
        help_text = out.getvalue()
        self.assertIn("Examples:", help_text)
        self.assertIn("cross-gpu list", help_text)
        self.assertIn("cross-gpu info --json", help_text)
        self.assertIn(
            "cross-gpu run --device 0 --test compute-fp32,memory-pattern --warmup 5s --duration 30s",
            help_text,
        )
        self.assertIn(
            "cross-gpu run --backend opencl --test all --duration 60s --json",
            help_text,
        )

    def test_difflib_typo_hints_return_usage_error_1(self) -> None:
        # Unknown --test single token and comma-separated tokens
        out, err = io.StringIO(), io.StringIO()
        rc = main(["run", "--test", "fp32,mem-sweep"], stdout=out, stderr=err)
        self.assertEqual(rc, ExitCode.USAGE_ERROR)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("unknown test 'fp32'. Did you mean: compute-fp32?", err.getvalue())
        self.assertIn("unknown test 'mem-sweep'. Did you mean: memory-sweep?", err.getvalue())

        # Unknown --backend plugin name
        out, err = io.StringIO(), io.StringIO()
        rc = main(["run", "--backend", "cdua"], stdout=out, stderr=err)
        self.assertEqual(rc, ExitCode.USAGE_ERROR)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("unknown backend 'cdua'. Did you mean: cuda?", err.getvalue())

        # Unknown subcommand
        out, err = io.StringIO(), io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            main(["lsit"], stdout=out, stderr=err)
        self.assertEqual(ctx.exception.code, int(ExitCode.USAGE_ERROR))
        self.assertEqual(out.getvalue(), "")
        self.assertIn("unknown command 'lsit'. Did you mean: list?", err.getvalue())

    def test_stdout_stderr_separation_under_json_verbose(self) -> None:
        dev_cuda = _sample_device()
        dev_opencl_info = BackendDeviceInfo(
            backend="opencl",
            priority=50,
            backend_device_index=0,
            native_handle=1,
            name="NVIDIA RTX 4090",
            vendor="NVIDIA",
            vendor_id=0x10DE,
            pci_bdf="0000:01:00.0",
            total_memory_bytes=24 * 1024**3,
        )

        class _MockCuda(_FaultyHardwarePlugin):
            name = "cuda"
            priority = 100

            def is_available(self) -> bool:
                self._record_probe("probe cuda: loaded libcuda.so.1")
                return True

            def enumerate_devices(self) -> list[BackendDeviceInfo]:
                return [dev_cuda.bindings[0].device_info]

        class _MockOpenCL(_FaultyHardwarePlugin):
            name = "opencl"
            priority = 50

            def is_available(self) -> bool:
                self._record_probe("probe opencl: loaded libOpenCL.so.1")
                return True

            def enumerate_devices(self) -> list[BackendDeviceInfo]:
                return [dev_opencl_info]

        class _MockUnavailableMetal(_FaultyHardwarePlugin):
            name = "metal"
            priority = 100

            def is_available(self) -> bool:
                self._load_error = "macOS only"
                self._record_probe("probe metal: skipped (macOS only)")
                return False

        mock_plugins = [_MockCuda(), _MockOpenCL(), _MockUnavailableMetal()]
        with mock.patch(
            "cross_gpu.backends.registry.discover_backends",
            return_value=mock_plugins,
        ):
            for cmd in ("list", "info"):
                out, err = io.StringIO(), io.StringIO()
                rc = main([cmd, "--json", "-v"], stdout=out, stderr=err)
                self.assertEqual(rc, ExitCode.SUCCESS)
                parsed = json.loads(out.getvalue())
                self.assertEqual(len(parsed["devices"]), 1)
                self.assertNotIn("[verbose]", out.getvalue())
                self.assertIn("[verbose] probe cuda: loaded libcuda.so.1", err.getvalue())
                self.assertIn("[verbose] backend metal: unavailable (macOS only)", err.getvalue())
                self.assertIn("selected cuda (100) over opencl (50)", err.getvalue())

            # run --json -v with unsupported workload keeps stdout strictly valid NDJSON
            out, err = io.StringIO(), io.StringIO()
            rc = main(
                ["run", "--device", "0", "--test", "compute-fp8", "--duration", "1s", "--json", "-v"],
                stdout=out,
                stderr=err,
            )
            self.assertEqual(rc, ExitCode.HARDWARE_ERROR)
            lines = [json.loads(line) for line in out.getvalue().strip().splitlines()]
            self.assertGreaterEqual(len(lines), 3)
            self.assertNotIn("[verbose]", out.getvalue())
            self.assertIn("[verbose] probe cuda: loaded libcuda.so.1", err.getvalue())

    def test_text_mode_progress_prefix_and_summary_tally(self) -> None:
        warmup_prefix = format_progress_prefix(
            "compute-fp32",
            workload_index=4,
            workload_total=20,
            phase="warmup",
            phase_elapsed_s=3.0,
            phase_duration_s=5.0,
        )
        self.assertEqual(warmup_prefix, "[04/20 compute-fp32 | warmup 3s/5s]")

        run_prefix = format_progress_prefix(
            "compute-fp32",
            workload_index=4,
            workload_total=20,
            phase="run",
            phase_elapsed_s=18.0,
            phase_duration_s=60.0,
        )
        self.assertEqual(run_prefix, "[04/20 compute-fp32 | 18s/60s]")

        telem_line = format_telemetry_line(
            {
                "elapsed_s": 23.0,
                "device": 0,
                "phase": "run",
                "workload": "compute-fp32",
                "workload_index": 4,
                "workload_total": 20,
                "phase_elapsed_s": 18.0,
                "phase_duration_s": 60.0,
                "metrics": {"temperature_c": 70.0},
            }
        )
        self.assertTrue(telem_line.startswith("[04/20 compute-fp32 | 18s/60s] dev=0 |"))

        summary = format_run_summary_text(
            [
                WorkloadResult(0, "GPU", "cuda", "compute-fp32", "compute-fp", "pass", 1.0, 100.0, "GFLOP/s", 100.0, "GFLOP", 0, "", "", ExitCode.SUCCESS),
                WorkloadResult(0, "GPU", "cuda", "compute-fp8", "compute-fp", "unsupported", 0.0, None, "GFLOP/s", None, "GFLOP", 0, "requires CUDA SM >= 8.9 (detected SM 8.6)", "requires CUDA SM >= 8.9 (detected SM 8.6)", ExitCode.SUCCESS),
                WorkloadResult(0, "GPU", "cuda", "compute-int32", "compute-int", "fail", 1.0, 50.0, "GOP/s", 50.0, "GOP", 1, "SDC", "", ExitCode.COMPUTE_ERROR),
                WorkloadResult(0, "GPU", "cuda", "memory-sweep", "memory", "hardware_error", 0.0, None, "GB/s", None, "GB", 0, "Xid 79", "", ExitCode.HARDWARE_ERROR),
            ],
            ExitCode.HARDWARE_ERROR,
        )
        self.assertIn("requires CUDA SM >= 8.9 (detected SM 8.6)", summary)
        self.assertIn(
            "Tally: passed=1 failed=1 unsupported=1 hardware_error=1 | Exit Code: 4 (HARDWARE_ERROR)",
            summary,
        )

    def test_actionable_unsupported_reasons_and_backend_diagnostics(self) -> None:
        # Concrete backend/arch skip explanations
        cuda_sm86 = BackendDeviceInfo(
            backend="cuda",
            priority=100,
            backend_device_index=0,
            native_handle=0,
            name="NVIDIA RTX 3090",
            vendor="NVIDIA",
            architecture="sm_86",
            compute_capability="8.6",
            capabilities=DeviceCapabilities(fp8=False),
        )
        self.assertEqual(
            explain_unsupported_workload(cuda_sm86, "compute-fp8"),
            "requires CUDA SM >= 8.9 (detected SM 8.6)",
        )
        metal_m2 = BackendDeviceInfo(
            backend="metal",
            priority=100,
            backend_device_index=0,
            native_handle=0,
            name="Apple M2",
            vendor="Apple",
            architecture="applegpu_g14g",
            capabilities=DeviceCapabilities(fp64=False),
        )
        self.assertEqual(
            explain_unsupported_workload(metal_m2, "compute-fp64"),
            "Metal lacks native fp64",
        )
        opencl_dev = BackendDeviceInfo(
            backend="opencl",
            priority=50,
            backend_device_index=0,
            native_handle=0,
            name="Generic OpenCL GPU",
            vendor="Intel",
            capabilities=DeviceCapabilities(fp16=False),
        )
        self.assertEqual(
            explain_unsupported_workload(opencl_dev, "compute-fp16"),
            "OpenCL device missing cl_khr_fp16 extension",
        )

        # Unsupported reason appears in BOTH text and JSON/NDJSON run outputs
        class _HealthyStubPlugin(_FaultyHardwarePlugin):
            def create_context(self, device: BackendDeviceInfo) -> BackendContext:
                return BackendContext(backend=self.name, device_info=device, handle=1)

        gpu_sm86 = GPUDevice(
            index=0,
            name=cuda_sm86.name,
            vendor=cuda_sm86.vendor,
            architecture=cuda_sm86.architecture,
            compute_capability=cuda_sm86.compute_capability,
            capabilities=cuda_sm86.capabilities,
            bindings=[BackendBinding(backend="cuda", priority=100, device_info=cuda_sm86)],
        )
        out_json, err_json = io.StringIO(), io.StringIO()
        rc_json = main(
            ["run", "--device", "0", "--test", "compute-fp8", "--duration", "1s", "--json"],
            stdout=out_json,
            stderr=err_json,
            discover_fn=lambda: ([gpu_sm86], {"cuda": _HealthyStubPlugin()}, {}),
        )
        self.assertEqual(rc_json, ExitCode.INIT_ERROR)
        events = [json.loads(ln) for ln in out_json.getvalue().strip().splitlines()]
        wl_event = next(e for e in events if e["type"] == "workload_result")
        self.assertEqual(wl_event["status"], "unsupported")
        self.assertEqual(wl_event["reason"], "requires CUDA SM >= 8.9 (detected SM 8.6)")

        out_text, err_text = io.StringIO(), io.StringIO()
        rc_text = main(
            ["run", "--device", "0", "--test", "compute-fp8", "--duration", "1s"],
            stdout=out_text,
            stderr=err_text,
            discover_fn=lambda: ([gpu_sm86], {"cuda": _HealthyStubPlugin()}, {}),
        )
        self.assertEqual(rc_text, ExitCode.INIT_ERROR)
        self.assertIn("requires CUDA SM >= 8.9 (detected SM 8.6)", out_text.getvalue())

        # Zero GPUs detected: stdout stays valid JSON/clean text, stderr prints concrete backend reasons
        diag_map = {
            "cuda": "libcuda.so.1 not found",
            "levelzero": "/dev/dri/renderD128 permission denied — check 'render'/'video' group membership",
            "metal": "macOS only",
            "opencl": "no OpenCL platforms detected",
        }
        for cmd in ("list", "info"):
            out_z, err_z = io.StringIO(), io.StringIO()
            rc_z = main(
                [cmd, "--json"],
                stdout=out_z,
                stderr=err_z,
                discover_fn=lambda: ([], {}, diag_map),
            )
            self.assertEqual(rc_z, ExitCode.SUCCESS)
            self.assertEqual(json.loads(out_z.getvalue()), {"devices": []})
            for k, v in diag_map.items():
                self.assertIn(f"{k}: {v}", err_z.getvalue())

        # Requested unavailable backend reports concrete reason on stderr
        out_unavail, err_unavail = io.StringIO(), io.StringIO()
        rc_unavail = main(
            ["run", "--backend", "levelzero", "--test", "compute-fp32", "--duration", "1s", "--json"],
            stdout=out_unavail,
            stderr=err_unavail,
            discover_fn=lambda: ([], {}, diag_map),
        )
        self.assertEqual(rc_unavail, ExitCode.INIT_ERROR)
        self.assertIn("renderD128 permission denied", err_unavail.getvalue())
        ndjson_lines = [json.loads(ln) for ln in out_unavail.getvalue().strip().splitlines()]
        self.assertEqual(ndjson_lines[-1]["exit_code"], int(ExitCode.INIT_ERROR))

        # Linux render node permission hint helper
        with tempfile.TemporaryDirectory() as tmp:
            dri = Path(tmp)
            node = dri / "renderD128"
            node.write_bytes(b"")
            with mock.patch("os.access", return_value=False):
                hint = check_linux_render_node_hint(platform_name="linux", dri_dir=dri)
            self.assertIsNotNone(hint)
            self.assertIn("permission denied", hint or "")
            self.assertIn("'render'/'video'", hint or "")

    def test_shell_completions_cover_all_workloads_backends_and_flags(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        completion_files = (
            repo_root / "completions" / "cross-gpu.bash",
            repo_root / "completions" / "_cross-gpu",
            repo_root / "completions" / "cross-gpu.fish",
        )
        expected_workloads = list(WORKLOAD_REGISTRY.keys())
        expected_backends = ("auto", *KNOWN_BACKENDS)
        expected_subcommands = ("list", "info", "run")
        expected_long_flags = (
            "help",
            "version",
            "format",
            "json",
            "verbose",
            "device",
            "test",
            "duration",
            "warmup",
            "backend",
        )

        zsh_lines = (repo_root / "completions" / "_cross-gpu").read_text(encoding="utf-8").splitlines()
        self.assertEqual(zsh_lines[0], "#compdef cross-gpu")
        self.assertEqual(zsh_lines[1], "# SPDX-License-Identifier: MIT")

        for comp_path in completion_files:
            self.assertTrue(comp_path.is_file(), f"Missing completion file {comp_path}")
            content = comp_path.read_text(encoding="utf-8")
            self.assertIn("SPDX-License-Identifier: MIT", content)
            for wl in expected_workloads:
                self.assertIn(wl, content, f"Missing workload {wl!r} in {comp_path.name}")
            for backend in expected_backends:
                self.assertIn(backend, content, f"Missing backend {backend!r} in {comp_path.name}")
            for cmd in expected_subcommands:
                self.assertIn(cmd, content, f"Missing subcommand {cmd!r} in {comp_path.name}")
            for flag in expected_long_flags:
                self.assertIn(flag, content, f"Missing flag {flag!r} in {comp_path.name}")
            for ms_tok in ("250ms", "500ms"):
                self.assertIn(ms_tok, content, f"Missing ms duration {ms_tok!r} in {comp_path.name}")


if __name__ == "__main__":
    unittest.main()
