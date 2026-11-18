"""Multi-GPU stress-testing orchestrator with warmup/run phases, signals, and telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import signal
import threading
import time
from typing import Any, Callable, Sequence

from cross_gpu.backends.base import BackendContext, BackendPlugin
from cross_gpu.dedup import resolve_device_backend
from cross_gpu.embed_binary import compile_kernel_runtime
from cross_gpu.exit_codes import (
    BackendHardwareError,
    BackendInitError,
    ComputeError,
    ExitCode,
    KernelCompilationError,
    KernelNotFoundError,
    UnsupportedKernelError,
    aggregate_exit_codes,
)
from cross_gpu.models import BackendBinding, GPUDevice, TelemetrySample, WorkloadResult
from cross_gpu.telemetry import TelemetrySampler, read_telemetry
from cross_gpu.workloads import (
    WORKLOAD_REGISTRY,
    WorkloadEntry,
    WorkloadExecutionContext,
    WorkloadExecutor,
    WorkloadSpec,
)

EventCallback = Callable[[dict[str, Any]], None]
CompileFn = Callable[..., Any]


@dataclass
class RunReport:
    """Complete execution report returned by StressRunner.run()."""

    exit_code: ExitCode
    results: list[WorkloadResult]
    interrupted: bool = False


class _DeviceTelemetryState:
    """Thread-safe container tracking the active phase and workload for 1 Hz tags."""

    def __init__(self, initial_workload: str) -> None:
        self._lock = threading.Lock()
        self.phase: str = "run"
        self.workload: str = initial_workload

    def set(self, phase: str, workload: str) -> None:
        with self._lock:
            self.phase = phase
            self.workload = workload

    def get(self) -> tuple[str, str]:
        with self._lock:
            return self.phase, self.workload


class StressRunner:
    """Coordinates device/backend selection, multi-GPU execution, signals, and telemetry."""

    def __init__(
        self,
        *,
        devices: Sequence[GPUDevice],
        plugins: dict[str, BackendPlugin],
        tests: Sequence[str],
        duration_s: float,
        warmup_s: float = 0.0,
        device_index: int | None = None,
        backend_preference: str = "auto",
        explicit_test_selection: bool = True,
        event_callback: EventCallback | None = None,
        telemetry_interval_s: float = 1.0,
        registry: dict[str, WorkloadEntry] | None = None,
        compile_fn: CompileFn = compile_kernel_runtime,
        telemetry_reader: Callable[[GPUDevice | None], TelemetrySample] = read_telemetry,
    ) -> None:
        self.devices = list(devices)
        self.plugins = dict(plugins)
        self.tests = list(tests)
        self.duration_s = float(duration_s)
        self.warmup_s = float(warmup_s)
        self.device_index = device_index
        self.backend_preference = backend_preference
        self.explicit_test_selection = explicit_test_selection
        self.event_callback = event_callback
        self.telemetry_interval_s = float(telemetry_interval_s)
        self.registry = WORKLOAD_REGISTRY if registry is None else registry
        self.compile_fn = compile_fn
        self.telemetry_reader = telemetry_reader

        self._emit_lock = threading.Lock()
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        """Signal all running worker loops to stop cleanly."""
        self._stop_event.set()

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.event_callback is None:
            return
        with self._emit_lock:
            self.event_callback(payload)

    def _select_targets(self) -> list[tuple[GPUDevice, BackendBinding, BackendPlugin]]:
        if not self.devices:
            raise BackendInitError("No compatible GPU devices detected")

        if self.device_index is not None:
            matched = [d for d in self.devices if d.index == self.device_index]
            if not matched:
                raise BackendInitError(
                    f"Requested device index {self.device_index} is unavailable "
                    f"(detected indices: {[d.index for d in self.devices]})"
                )
            selected_devices = matched
        else:
            selected_devices = list(self.devices)

        targets: list[tuple[GPUDevice, BackendBinding, BackendPlugin]] = []
        for device in selected_devices:
            binding = resolve_device_backend(device, self.backend_preference)
            plugin = self.plugins.get(binding.backend)
            if plugin is None:
                raise BackendInitError(
                    f"Backend plugin {binding.backend!r} is not loaded"
                )
            targets.append((device, binding, plugin))
        return targets

    def _run_device_workloads(
        self,
        device: GPUDevice,
        binding: BackendBinding,
        plugin: BackendPlugin,
        out_results: list[WorkloadResult],
        out_lock: threading.Lock,
    ) -> None:
        telemetry_state = _DeviceTelemetryState(self.tests[0] if self.tests else "")

        def on_sample(sample: TelemetrySample, elapsed_s: float) -> None:
            phase, workload_name = telemetry_state.get()
            self._emit(
                {
                    "type": "telemetry",
                    "timestamp": time.time(),
                    "elapsed_s": round(elapsed_s, 3),
                    "phase": phase,
                    "device": device.index,
                    "workload": workload_name,
                    "metrics": sample.to_dict(),
                }
            )

        sampler = TelemetrySampler(
            device,
            on_sample,
            interval_s=self.telemetry_interval_s,
            reader_fn=self.telemetry_reader,
        )

        backend_ctx: BackendContext | None = None
        try:
            backend_ctx = plugin.create_context(binding.device_info)
        except BackendHardwareError as exc:
            for test_name in self.tests:
                spec = self.registry[test_name].spec
                res = WorkloadResult(
                    device=device.index,
                    device_name=device.name,
                    backend=binding.backend,
                    workload=test_name,
                    category=spec.category,
                    status="hardware_error",
                    duration_s=0.0,
                    rate=None,
                    rate_unit=spec.rate_unit,
                    total=None,
                    total_unit=spec.total_unit,
                    errors=0,
                    message=str(exc),
                    exit_code=ExitCode.HARDWARE_ERROR,
                )
                with out_lock:
                    out_results.append(res)
                self._emit({"type": "workload_result", **res.to_dict()})
            return
        except Exception as exc:
            for test_name in self.tests:
                spec = self.registry[test_name].spec
                res = WorkloadResult(
                    device=device.index,
                    device_name=device.name,
                    backend=binding.backend,
                    workload=test_name,
                    category=spec.category,
                    status="not_implemented",
                    duration_s=0.0,
                    rate=None,
                    rate_unit=spec.rate_unit,
                    total=None,
                    total_unit=spec.total_unit,
                    errors=0,
                    message=f"Context initialization failed: {exc}",
                    exit_code=ExitCode.INIT_ERROR,
                )
                with out_lock:
                    out_results.append(res)
                self._emit({"type": "workload_result", **res.to_dict()})
            return

        sampler.start()
        try:
            for test_name in self.tests:
                if self._stop_event.is_set():
                    break
                entry = self.registry[test_name]
                res = self._execute_single_workload(
                    device=device,
                    binding=binding,
                    plugin=plugin,
                    backend_ctx=backend_ctx,
                    entry=entry,
                    telemetry_state=telemetry_state,
                )
                with out_lock:
                    out_results.append(res)
                self._emit({"type": "workload_result", **res.to_dict()})
        finally:
            sampler.stop()
            if backend_ctx is not None:
                try:
                    plugin.destroy_context(backend_ctx)
                except Exception:
                    pass

    def _execute_single_workload(
        self,
        *,
        device: GPUDevice,
        binding: BackendBinding,
        plugin: BackendPlugin,
        backend_ctx: BackendContext,
        entry: WorkloadEntry,
        telemetry_state: _DeviceTelemetryState,
    ) -> WorkloadResult:
        spec: WorkloadSpec = entry.spec
        executor_cls = entry.executor_cls

        if executor_cls is None:
            return WorkloadResult(
                device=device.index,
                device_name=device.name,
                backend=binding.backend,
                workload=spec.name,
                category=spec.category,
                status="not_implemented",
                duration_s=0.0,
                rate=None,
                rate_unit=spec.rate_unit,
                total=None,
                total_unit=spec.total_unit,
                errors=0,
                message=f"Workload executor/kernel for {spec.name!r} is not implemented yet",
                exit_code=ExitCode.INIT_ERROR,
            )

        executor: WorkloadExecutor = executor_cls(spec)
        if not executor.check_support(binding.device_info):
            code = ExitCode.INIT_ERROR if self.explicit_test_selection else ExitCode.SUCCESS
            return WorkloadResult(
                device=device.index,
                device_name=device.name,
                backend=binding.backend,
                workload=spec.name,
                category=spec.category,
                status="unsupported",
                duration_s=0.0,
                rate=None,
                rate_unit=spec.rate_unit,
                total=None,
                total_unit=spec.total_unit,
                errors=0,
                message=(
                    f"Device {device.name!r} lacks required capability "
                    f"{spec.required_capability!r}"
                ),
                exit_code=code,
            )

        exec_ctx = WorkloadExecutionContext(
            device=device,
            binding=binding,
            backend=plugin,
            backend_context=backend_ctx,
            spec=spec,
        )

        prepared = False
        try:
            compiled = None
            requires_kernel = getattr(executor, "requires_kernel", spec.requires_kernel)
            if requires_kernel:
                compiled = self.compile_fn(
                    binding.backend,
                    spec.name,
                    binding.device_info.architecture,
                    binding.device_info.capabilities,
                    driver_version=binding.device_info.driver_version,
                    entry_point=spec.default_entry_point,
                )
            exec_ctx.compiled_binary = compiled

            executor.prepare(exec_ctx, compiled)
            prepared = True

            # Warmup phase: telemetry tagged "warmup", throughput excluded.
            if self.warmup_s > 0 and not self._stop_event.is_set():
                telemetry_state.set("warmup", spec.name)
                warmup_start = time.perf_counter()
                while not self._stop_event.is_set():
                    executor.run_step(exec_ctx)
                    plugin.synchronize(backend_ctx)
                    if time.perf_counter() - warmup_start >= self.warmup_s:
                        break

            # Timed run phase.
            telemetry_state.set("run", spec.name)
            total_ops = 0.0
            step_errors = 0
            run_start = time.perf_counter()
            while True:
                if self._stop_event.is_set():
                    break
                step = executor.run_step(exec_ctx)
                plugin.synchronize(backend_ctx)
                total_ops += float(step.operations)
                step_errors += int(step.errors)
                if time.perf_counter() - run_start >= self.duration_s:
                    break
            elapsed_s = max(time.perf_counter() - run_start, 1e-9)

            ver = executor.verify(exec_ctx)
            total_errors = step_errors + int(ver.error_count)
            total_scaled = total_ops / spec.unit_scale
            rate = total_scaled / elapsed_s

            if not ver.passed or total_errors > 0:
                return WorkloadResult(
                    device=device.index,
                    device_name=device.name,
                    backend=binding.backend,
                    workload=spec.name,
                    category=spec.category,
                    status="fail",
                    duration_s=elapsed_s,
                    rate=rate,
                    rate_unit=spec.rate_unit,
                    total=total_scaled,
                    total_unit=spec.total_unit,
                    errors=max(1, total_errors),
                    message=ver.message or "Verification mismatch",
                    exit_code=ExitCode.COMPUTE_ERROR,
                )

            return WorkloadResult(
                device=device.index,
                device_name=device.name,
                backend=binding.backend,
                workload=spec.name,
                category=spec.category,
                status="pass",
                duration_s=elapsed_s,
                rate=rate,
                rate_unit=spec.rate_unit,
                total=total_scaled,
                total_unit=spec.total_unit,
                errors=0,
                message=ver.message,
                exit_code=ExitCode.SUCCESS,
            )

        except BackendHardwareError as exc:
            return WorkloadResult(
                device=device.index,
                device_name=device.name,
                backend=binding.backend,
                workload=spec.name,
                category=spec.category,
                status="hardware_error",
                duration_s=0.0,
                rate=None,
                rate_unit=spec.rate_unit,
                total=None,
                total_unit=spec.total_unit,
                errors=0,
                message=str(exc),
                exit_code=ExitCode.HARDWARE_ERROR,
            )
        except ComputeError as exc:
            return WorkloadResult(
                device=device.index,
                device_name=device.name,
                backend=binding.backend,
                workload=spec.name,
                category=spec.category,
                status="fail",
                duration_s=0.0,
                rate=None,
                rate_unit=spec.rate_unit,
                total=None,
                total_unit=spec.total_unit,
                errors=1,
                message=str(exc),
                exit_code=ExitCode.COMPUTE_ERROR,
            )
        except UnsupportedKernelError as exc:
            return WorkloadResult(
                device=device.index,
                device_name=device.name,
                backend=binding.backend,
                workload=spec.name,
                category=spec.category,
                status="unsupported",
                duration_s=0.0,
                rate=None,
                rate_unit=spec.rate_unit,
                total=None,
                total_unit=spec.total_unit,
                errors=0,
                message=str(exc),
                exit_code=ExitCode.INIT_ERROR,
            )
        except (KernelNotFoundError, KernelCompilationError, BackendInitError) as exc:
            return WorkloadResult(
                device=device.index,
                device_name=device.name,
                backend=binding.backend,
                workload=spec.name,
                category=spec.category,
                status="not_implemented",
                duration_s=0.0,
                rate=None,
                rate_unit=spec.rate_unit,
                total=None,
                total_unit=spec.total_unit,
                errors=0,
                message=str(exc),
                exit_code=ExitCode.INIT_ERROR,
            )
        finally:
            if prepared:
                try:
                    executor.cleanup(exec_ctx)
                except Exception:
                    pass

    def run(self) -> RunReport:
        """Execute selected workloads across target GPUs with signal-safe cleanup."""
        self._stop_event.clear()
        try:
            targets = self._select_targets()
        except BackendInitError as exc:
            self._emit(
                {
                    "type": "run_summary",
                    "exit_code": int(ExitCode.INIT_ERROR),
                    "error": str(exc),
                    "total_workloads": 0,
                    "results": [],
                }
            )
            return RunReport(exit_code=ExitCode.INIT_ERROR, results=[])

        self._emit(
            {
                "type": "run_start",
                "timestamp": time.time(),
                "devices": [
                    {
                        "index": dev.index,
                        "name": dev.name,
                        "backend": binding.backend,
                    }
                    for dev, binding, _ in targets
                ],
                "workloads": list(self.tests),
                "duration_s": self.duration_s,
                "warmup_s": self.warmup_s,
                "backend": self.backend_preference,
            }
        )

        prev_sigint = None
        prev_sigterm = None
        installed_signals = threading.current_thread() is threading.main_thread()
        if installed_signals:
            def _handle_sig(_signum: int, _frame: Any) -> None:
                self._stop_event.set()

            prev_sigint = signal.getsignal(signal.SIGINT)
            prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, _handle_sig)
            signal.signal(signal.SIGTERM, _handle_sig)

        results: list[WorkloadResult] = []
        results_lock = threading.Lock()
        try:
            if len(targets) == 1:
                dev, binding, plugin = targets[0]
                self._run_device_workloads(dev, binding, plugin, results, results_lock)
            else:
                threads: list[threading.Thread] = []
                for dev, binding, plugin in targets:
                    t = threading.Thread(
                        target=self._run_device_workloads,
                        args=(dev, binding, plugin, results, results_lock),
                        name=f"cross-gpu-dev-{dev.index}",
                    )
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()
        finally:
            if installed_signals:
                if prev_sigint is not None:
                    signal.signal(signal.SIGINT, prev_sigint)
                if prev_sigterm is not None:
                    signal.signal(signal.SIGTERM, prev_sigterm)

        results.sort(key=lambda r: (r.device, self.tests.index(r.workload)))
        codes = [r.exit_code for r in results]
        if results and not self.explicit_test_selection and all(
            r.status == "unsupported" for r in results
        ):
            final_code = ExitCode.INIT_ERROR
        else:
            final_code = aggregate_exit_codes(codes) if results else ExitCode.INIT_ERROR

        self._emit(
            {
                "type": "run_summary",
                "exit_code": int(final_code),
                "interrupted": self._stop_event.is_set(),
                "total_workloads": len(results),
                "results": [r.to_dict() for r in results],
            }
        )
        return RunReport(
            exit_code=final_code,
            results=results,
            interrupted=self._stop_event.is_set(),
        )
