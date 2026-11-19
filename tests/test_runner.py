# SPDX-License-Identifier: MIT
"""Unit tests for StressRunner orchestration, warmup exclusion, units, and faults."""

from __future__ import annotations

import copy
import threading
import unittest

from cross_gpu.backends.base import (
    BackendContext,
    BackendPlugin,
    DeviceBuffer,
    HostBuffer,
    KernelModule,
)
from cross_gpu.embed_binary import CompiledKernelBinary
from cross_gpu.exit_codes import BackendHardwareError, ExitCode
from cross_gpu.models import (
    BackendBinding,
    BackendDeviceInfo,
    DeviceCapabilities,
    GPUDevice,
    StepMetrics,
    TelemetrySample,
    VerificationResult,
)
from cross_gpu.runner import StressRunner
from cross_gpu.workloads import (
    WORKLOAD_REGISTRY,
    WorkloadEntry,
    WorkloadExecutionContext,
    WorkloadExecutor,
    register_workload_executor,
)


class _StubBackend(BackendPlugin):
    name = "stub"
    priority = 100

    def __init__(self) -> None:
        super().__init__()
        self.created_contexts: int = 0
        self.destroyed_contexts: int = 0
        self.sync_calls: int = 0

    def is_available(self) -> bool:
        return True

    def enumerate_devices(self) -> list[BackendDeviceInfo]:
        return []

    def create_context(self, device: BackendDeviceInfo) -> BackendContext:
        self.created_contexts += 1
        return BackendContext(backend=self.name, device_info=device, handle=self.created_contexts)

    def destroy_context(self, ctx: BackendContext) -> None:
        self.destroyed_contexts += 1
        ctx.handle = None

    def synchronize(self, ctx: BackendContext) -> None:
        self.sync_calls += 1

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


def _make_gpu(index: int, *, fp8: bool = False) -> GPUDevice:
    caps = DeviceCapabilities(fp32=True, fp8=fp8, int32=True)
    info = BackendDeviceInfo(
        backend="stub",
        priority=100,
        backend_device_index=index,
        native_handle=index,
        name=f"Stub GPU {index}",
        vendor="Test",
        vendor_id=0x1234,
        architecture="arch1",
        total_memory_bytes=8 * 1024**3,
        compute_units=32,
        driver_version="1.0",
        capabilities=caps,
    )
    return GPUDevice(
        index=index,
        name=info.name,
        vendor=info.vendor,
        vendor_id=info.vendor_id,
        architecture=info.architecture,
        total_memory_bytes=info.total_memory_bytes,
        compute_units=info.compute_units,
        driver_version=info.driver_version,
        capabilities=caps,
        bindings=[BackendBinding(backend="stub", priority=100, device_info=info)],
    )


def _fresh_registry() -> dict[str, WorkloadEntry]:
    return {name: copy.copy(entry) for name, entry in WORKLOAD_REGISTRY.items()}


class TestRunner(unittest.TestCase):
    def test_unplugged_workload_returns_not_implemented_and_init_error(self) -> None:
        backend = _StubBackend()
        gpu = _make_gpu(0)
        events: list[dict[str, object]] = []
        reg = _fresh_registry()
        register_workload_executor("compute-fp32", None, registry=reg)
        runner = StressRunner(
            devices=[gpu],
            plugins={"stub": backend},
            tests=["compute-fp32"],
            duration_s=0.05,
            registry=reg,
            event_callback=events.append,
        )
        report = runner.run()
        self.assertEqual(report.exit_code, ExitCode.INIT_ERROR)
        self.assertEqual(len(report.results), 1)
        self.assertEqual(report.results[0].status, "not_implemented")
        self.assertIsNone(report.results[0].rate)
        self.assertIsNone(report.results[0].total)
        self.assertEqual(backend.created_contexts, 1)
        self.assertEqual(backend.destroyed_contexts, 1)

    def test_warmup_excluded_and_unit_rates_computed(self) -> None:
        events: list[dict[str, object]] = []
        reg = _fresh_registry()

        step_counter = {"total": 0}

        class _CountingExecutor(WorkloadExecutor):
            requires_kernel = True

            def __init__(self, spec):
                super().__init__(spec)
                self.cleaned = False

            def prepare(self, ctx: WorkloadExecutionContext, binary: object) -> None:
                ctx.state["prepared"] = True

            def run_step(self, ctx: WorkloadExecutionContext) -> StepMetrics:
                import time as _time

                _time.sleep(0.015)
                step_counter["total"] += 1
                # Each step yields exactly 1e9 FLOPs (= 1.0 GFLOP)
                return StepMetrics(operations=1_000_000_000.0)

            def verify(self, ctx: WorkloadExecutionContext) -> VerificationResult:
                return VerificationResult(passed=True, error_count=0)

            def cleanup(self, ctx: WorkloadExecutionContext) -> None:
                self.cleaned = True

        register_workload_executor("compute-fp32", _CountingExecutor, registry=reg)
        backend = _StubBackend()
        gpu = _make_gpu(0)

        runner = StressRunner(
            devices=[gpu],
            plugins={"stub": backend},
            tests=["compute-fp32"],
            duration_s=0.08,
            warmup_s=0.06,
            telemetry_interval_s=0.02,
            registry=reg,
            event_callback=events.append,
            compile_fn=lambda *a, **kw: CompiledKernelBinary(
                backend="stub",
                workload="compute-fp32",
                entry_point="kernel_compute_fp32",
                format="ptx",
                binary_bytes=b"ptx",
            ),
            telemetry_reader=lambda d: TelemetrySample(temperature_c=60.0),
        )
        report = runner.run()
        self.assertEqual(report.exit_code, ExitCode.SUCCESS)
        res = report.results[0]
        self.assertEqual(res.status, "pass")
        self.assertEqual(res.rate_unit, "GFLOP/s")
        self.assertEqual(res.total_unit, "GFLOP")
        self.assertGreater(res.total or 0.0, 0.0)
        self.assertLess(res.total or 0.0, float(step_counter["total"]))
        self.assertGreater(res.rate or 0.0, 0.0)

        phases = {e["phase"] for e in events if e["type"] == "telemetry"}
        self.assertIn("warmup", phases)
        self.assertIn("run", phases)

    def test_multi_gpu_concurrency_and_severity_aggregation(self) -> None:
        reg = _fresh_registry()
        barrier = threading.Barrier(2, timeout=2.0)

        class _MixedFaultExecutor(WorkloadExecutor):
            requires_kernel = False

            def prepare(self, ctx: WorkloadExecutionContext, binary: object) -> None:
                barrier.wait()

            def run_step(self, ctx: WorkloadExecutionContext) -> StepMetrics:
                if ctx.device.index == 1:
                    raise BackendHardwareError("GPU 1 Xid 79 fallen off bus")
                return StepMetrics(operations=500_000_000.0)

            def verify(self, ctx: WorkloadExecutionContext) -> VerificationResult:
                return VerificationResult(passed=False, error_count=2, message="SDC mismatch")

            def cleanup(self, ctx: WorkloadExecutionContext) -> None:
                pass

        register_workload_executor("bandwidth-d2d", _MixedFaultExecutor, registry=reg)
        backend = _StubBackend()
        gpus = [_make_gpu(0), _make_gpu(1)]

        runner = StressRunner(
            devices=gpus,
            plugins={"stub": backend},
            tests=["bandwidth-d2d"],
            duration_s=0.02,
            registry=reg,
        )
        report = runner.run()
        # GPU 0 -> COMPUTE_ERROR (3), GPU 1 -> HARDWARE_ERROR (4) -> max = 4
        self.assertEqual(report.exit_code, ExitCode.HARDWARE_ERROR)
        self.assertEqual(report.results[0].status, "fail")
        self.assertEqual(report.results[0].exit_code, ExitCode.COMPUTE_ERROR)
        self.assertEqual(report.results[1].status, "hardware_error")
        self.assertEqual(report.results[1].exit_code, ExitCode.HARDWARE_ERROR)

    def test_missing_device_or_backend_returns_init_error(self) -> None:
        backend = _StubBackend()
        gpu = _make_gpu(0)

        r1 = StressRunner(
            devices=[gpu],
            plugins={"stub": backend},
            tests=["compute-fp32"],
            duration_s=0.01,
            device_index=3,
        ).run()
        self.assertEqual(r1.exit_code, ExitCode.INIT_ERROR)

        r2 = StressRunner(
            devices=[gpu],
            plugins={"stub": backend},
            tests=["compute-fp32"],
            duration_s=0.01,
            backend_preference="cuda",
        ).run()
        self.assertEqual(r2.exit_code, ExitCode.INIT_ERROR)


if __name__ == "__main__":
    unittest.main()
