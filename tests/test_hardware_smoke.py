"""Opt-in hardware-in-the-loop GPU stress smoke tests (enable via CROSS_GPU_HW_TESTS=1)."""

from __future__ import annotations

import os
import unittest

from cross_gpu.backends.registry import enumerate_all_backend_devices
from cross_gpu.dedup import deduplicate_devices
from cross_gpu.exit_codes import ExitCode
from cross_gpu.runner import StressRunner
from cross_gpu.workloads import list_workload_names


@unittest.skipUnless(
    os.environ.get("CROSS_GPU_HW_TESTS") == "1",
    "Hardware smoke tests skipped unless CROSS_GPU_HW_TESTS=1 is set",
)
class TestHardwareSmoke(unittest.TestCase):
    def test_real_gpu_all_supported_workloads_short_pass(self) -> None:
        raw_devices, plugins = enumerate_all_backend_devices()
        devices = deduplicate_devices(raw_devices)
        if not devices:
            self.skipTest("No physical GPU detected on this host")

        runner = StressRunner(
            devices=[devices[0]],
            plugins=plugins,
            tests=list_workload_names(),
            duration_s=0.5,
            warmup_s=0.1,
            explicit_test_selection=False,
        )
        report = runner.run()
        self.assertEqual(report.exit_code, ExitCode.SUCCESS)

        passed = [r for r in report.results if r.status == "pass"]
        failed = [r for r in report.results if r.status not in ("pass", "unsupported")]
        self.assertEqual(failed, [], f"Unexpected hardware failures: {failed}")
        self.assertGreaterEqual(len(passed), 5)
        for res in passed:
            self.assertGreater(res.rate or 0.0, 0.0, f"{res.workload} reported zero rate")
            self.assertGreater(res.total or 0.0, 0.0, f"{res.workload} reported zero total")
            self.assertEqual(res.errors, 0)


if __name__ == "__main__":
    unittest.main()
