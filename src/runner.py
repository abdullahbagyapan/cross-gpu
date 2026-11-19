# SPDX-License-Identifier: MIT
"""Multi-GPU stress-testing orchestrator with warmup/run phases, signals, and telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
import signal
import threading
import time
from typing import Any, Callable, Sequence

from .backends.base import BackendContext, BackendPlugin, DiagCallback
from .dedup import resolve_device_backend
from .embed_binary import compile_kernel_runtime
from .exit_codes import (
    BackendHardwareError,
    BackendInitError,
    ComputeError,
    ExitCode,
    KernelCompilationError,
    KernelNotFoundError,
    UnsupportedKernelError,
    aggregate_exit_codes,
)
from .models import (
    BackendBinding,
    BackendDeviceInfo,
    GPUDevice,
    TelemetrySample,
    WorkloadResult,
)
from .telemetry import TelemetrySampler, read_telemetry, sanitize_sample
from .workloads import (
    WORKLOAD_REGISTRY,
    WorkloadEntry,
    WorkloadExecutionContext,
    WorkloadExecutor,
    WorkloadSpec,
    explain_unsupported_workload,
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
    """Thread-safe container tracking active phase, workload, and progress for 1 Hz tags."""

    def __init__(
        self,
        initial_workload: str,
        workload_total: int = 1,
        initial_phase: str = "run",
        phase_duration_s: float = 0.0,
    ) -> None:
        self._lock = threading.Lock()
        self.phase: str = initial_phase
        self.workload: str = initial_workload
        self.workload_index: int = 1
        self.workload_total: int = max(1, workload_total)
        self.phase_start_mono: float = time.perf_counter()
        self.phase_duration_s: float = phase_duration_s

    def set(
        self,
        phase: str,
        workload: str,
        *,
        workload_index: int | None = None,
        phase_duration_s: float | None = None,
        reset_timer: bool = True,
    ) -> None:
        with self._lock:
            self.phase = phase
            self.workload = workload
            if workload_index is not None:
                self.workload_index = workload_index
            if phase_duration_s is not None:
                self.phase_duration_s = float(phase_duration_s)
            if reset_timer:
                self.phase_start_mono = time.perf_counter()

    def snapshot(self) -> tuple[str, str, int, int, float, float]:
        with self._lock:
            phase_elapsed = max(0.0, time.perf_counter() - self.phase_start_mono)
            return (
                self.phase,
                self.workload,
                self.workload_index,
                self.workload_total,
                phase_elapsed,
                self.phase_duration_s,
            )


DEFAULT_VERIFY_INTERVAL_S = 5.0


_UNSUPPORTED_REASON_RE = re.compile(r"UNSUPPORTED\s*:\s*([^\r\n\"']+)", re.IGNORECASE)


def _is_unsupported_compile_error(exc: KernelCompilationError) -> bool:
    """Return True when a compiler diagnostic explicitly signals #error UNSUPPORTED."""
    haystack = f"{exc}\n{exc.compiler_log}".upper()
    return "UNSUPPORTED" in haystack


def _extract_unsupported_reason(
    exc: KernelCompilationError,
    device_info: BackendDeviceInfo,
    spec: WorkloadSpec,
) -> str:
    """Extract a clean human-readable skip reason from a compiler UNSUPPORTED diagnostic."""
    for source in (exc.compiler_log, str(exc)):
        if not source:
            continue
        match = _UNSUPPORTED_REASON_RE.search(source)
        if match:
            cleaned = match.group(1).strip(" .;:-\t\"'")
            if cleaned:
                return cleaned
    return explain_unsupported_workload(device_info, spec)


def _callable_accepts_kwarg(fn: Callable[..., Any], kwarg: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == kwarg:
            return True
    return False


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
        verify_interval_s: float = DEFAULT_VERIFY_INTERVAL_S,
        registry: dict[str, WorkloadEntry] | None = None,
        compile_fn: CompileFn = compile_kernel_runtime,
        telemetry_reader: Callable[[GPUDevice | None], TelemetrySample] = read_telemetry,
        diag_fn: DiagCallback | None = None,
        backend_diagnostics: dict[str, str] | None = None,
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
        self.verify_interval_s = float(verify_interval_s)
        self.registry = WORKLOAD_REGISTRY if registry is None else registry
        self.compile_fn = compile_fn
        self.telemetry_reader = telemetry_reader
        self.diag_fn = diag_fn
        self.backend_diagnostics = dict(backend_diagnostics or {})

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
            if self.backend_preference != "auto" and self.backend_preference in self.backend_diagnostics:
                reason = self.backend_diagnostics[self.backend_preference]
                raise BackendInitError(
                    f"Requested backend {self.backend_preference!r} is unavailable ({reason})"
                )
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
            try:
                binding = resolve_device_backend(device, self.backend_preference)
            except BackendInitError as exc:
                reason = self.backend_diagnostics.get(self.backend_preference)
                if reason:
                    raise BackendInitError(f"{exc} ({self.backend_preference}: {reason})") from exc
                raise
            plugin = self.plugins.get(binding.backend)
            if plugin is None:
                reason = self.backend_diagnostics.get(binding.backend)
                suffix = f" ({reason})" if reason else ""
                raise BackendInitError(
                    f"Backend plugin {binding.backend!r} is not loaded{suffix}"
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
        target_count: int = 1,
    ) -> None:
        telemetry_state = _DeviceTelemetryState(
            self.tests[0] if self.tests else "",
            workload_total=len(self.tests),
            initial_phase="warmup" if self.warmup_s > 0 else "run",
            phase_duration_s=self.warmup_s if self.warmup_s > 0 else self.duration_s,
        )

        def on_sample(sample: TelemetrySample, elapsed_s: float) -> None:
            usable_metrics = sanitize_sample(sample).to_usable_dict()
            if not usable_metrics:
                return
            (
                phase,
                workload_name,
                wl_idx,
                wl_total,
                phase_elapsed,
                phase_duration,
            ) = telemetry_state.snapshot()
            self._emit(
                {
                    "type": "telemetry",
                    "timestamp": time.time(),
                    "elapsed_s": round(elapsed_s, 3),
                    "phase": phase,
                    "device": device.index,
                    "workload": workload_name,
                    "workload_index": wl_idx,
                    "workload_total": wl_total,
                    "phase_elapsed_s": round(phase_elapsed, 3),
                    "phase_duration_s": round(phase_duration, 3),
                    "metrics": usable_metrics,
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
            total_tests = len(self.tests)
            for wl_idx, test_name in enumerate(self.tests, start=1):
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
                self._emit(
                    {
                        "type": "workload_result",
                        "workload_index": wl_idx,
                        "workload_total": total_tests,
                        "target_duration_s": self.duration_s,
                        **res.to_dict(),
                    }
                )
            return
        except Exception as exc:
            total_tests = len(self.tests)
            for wl_idx, test_name in enumerate(self.tests, start=1):
                spec = self.registry[test_name].spec
                res = WorkloadResult(
                    device=device.index,
                    device_name=device.name,
                    backend=binding.backend,
                    workload=test_name,
                    category=spec.category,
                    status="init_error",
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
                self._emit(
                    {
                        "type": "workload_result",
                        "workload_index": wl_idx,
                        "workload_total": total_tests,
                        "target_duration_s": self.duration_s,
                        **res.to_dict(),
                    }
                )
            return

        sampler.start()
        try:
            total_tests = len(self.tests)
            for wl_idx, test_name in enumerate(self.tests, start=1):
                if self._stop_event.is_set():
                    break
                entry = self.registry[test_name]
                telemetry_state.set(
                    "warmup" if self.warmup_s > 0 else "run",
                    test_name,
                    workload_index=wl_idx,
                    phase_duration_s=self.warmup_s if self.warmup_s > 0 else self.duration_s,
                    reset_timer=True,
                )
                res = self._execute_single_workload(
                    device=device,
                    binding=binding,
                    plugin=plugin,
                    backend_ctx=backend_ctx,
                    entry=entry,
                    telemetry_state=telemetry_state,
                    workload_index=wl_idx,
                    target_count=target_count,
                )
                with out_lock:
                    out_results.append(res)
                self._emit(
                    {
                        "type": "workload_result",
                        "workload_index": wl_idx,
                        "workload_total": total_tests,
                        "target_duration_s": self.duration_s,
                        **res.to_dict(),
                    }
                )
        finally:
            sampler.stop()
            if backend_ctx is not None:
                try:
                    plugin.destroy_context(backend_ctx)
                except Exception:
                    pass

    def _unsupported_exit_code(self, target_count: int) -> ExitCode:
        # On multi-GPU runs, explicit unsupported status is promoted in run() only if all devices lack support.
        if self.explicit_test_selection and target_count <= 1:
            return ExitCode.INIT_ERROR
        return ExitCode.SUCCESS

    def _execute_single_workload(
        self,
        *,
        device: GPUDevice,
        binding: BackendBinding,
        plugin: BackendPlugin,
        backend_ctx: BackendContext,
        entry: WorkloadEntry,
        telemetry_state: _DeviceTelemetryState,
        workload_index: int = 1,
        target_count: int = 1,
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
            skip_reason = explain_unsupported_workload(binding.device_info, spec)
            if self.diag_fn is not None:
                self.diag_fn(
                    f"skip [{binding.backend}:{spec.name}] dev={device.index}: {skip_reason}"
                )
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
                message=skip_reason,
                reason=skip_reason,
                exit_code=self._unsupported_exit_code(target_count),
            )

        exec_ctx = WorkloadExecutionContext(
            device=device,
            binding=binding,
            backend=plugin,
            backend_context=backend_ctx,
            spec=spec,
            state={"target_duration_s": self.duration_s},
        )

        try:
            compiled = None
            requires_kernel = getattr(executor, "requires_kernel", spec.requires_kernel)
            if requires_kernel:
                compile_kwargs: dict[str, Any] = {
                    "driver_version": binding.device_info.driver_version,
                    "entry_point": spec.default_entry_point,
                }
                if self.diag_fn is not None and _callable_accepts_kwarg(self.compile_fn, "diag_fn"):
                    compile_kwargs["diag_fn"] = self.diag_fn
                compiled = self.compile_fn(
                    binding.backend,
                    spec.name,
                    binding.device_info.architecture,
                    binding.device_info.capabilities,
                    **compile_kwargs,
                )
            exec_ctx.compiled_binary = compiled

            executor.prepare(exec_ctx, compiled)

            # Warmup phase: telemetry tagged "warmup", throughput excluded.
            if self.warmup_s > 0 and not self._stop_event.is_set():
                telemetry_state.set(
                    "warmup",
                    spec.name,
                    workload_index=workload_index,
                    phase_duration_s=self.warmup_s,
                    reset_timer=True,
                )
                warmup_start = time.perf_counter()
                while not self._stop_event.is_set():
                    executor.run_step(exec_ctx)
                    plugin.synchronize(backend_ctx)
                    if time.perf_counter() - warmup_start >= self.warmup_s:
                        break
                plugin.synchronize(backend_ctx)
                executor.reset_run_state(exec_ctx)

            # Timed run phase with periodic verification checkpoints and signal responsiveness.
            telemetry_state.set(
                "run",
                spec.name,
                workload_index=workload_index,
                phase_duration_s=self.duration_s,
                reset_timer=True,
            )
            total_ops = 0.0
            step_errors = 0
            timed_steps = 0
            checkpoint_overhead_s = 0.0
            checkpoint_failure = None

            plugin.synchronize(backend_ctx)
            run_start = time.perf_counter()
            last_checkpoint = run_start

            while not self._stop_event.is_set():
                step = executor.run_step(exec_ctx)
                plugin.synchronize(backend_ctx)
                timed_steps += 1
                total_ops += float(step.operations)
                step_errors += int(step.errors)
                if step.errors > 0:
                    break

                now = time.perf_counter()
                if (now - run_start - checkpoint_overhead_s) >= self.duration_s:
                    break

                if self.verify_interval_s > 0 and (now - last_checkpoint) >= self.verify_interval_s:
                    cp_start = time.perf_counter()
                    cp_ver = executor.verify(exec_ctx)
                    cp_end = time.perf_counter()
                    checkpoint_overhead_s += cp_end - cp_start
                    last_checkpoint = cp_end
                    if not cp_ver.passed or cp_ver.error_count > 0:
                        checkpoint_failure = cp_ver
                        break

            raw_elapsed = time.perf_counter() - run_start - checkpoint_overhead_s
            elapsed_s = max(raw_elapsed, 1e-9) if timed_steps > 0 else 0.0

            if timed_steps == 0 and self._stop_event.is_set():
                return WorkloadResult(
                    device=device.index,
                    device_name=device.name,
                    backend=binding.backend,
                    workload=spec.name,
                    category=spec.category,
                    status="pass",
                    duration_s=0.0,
                    rate=0.0,
                    rate_unit=spec.rate_unit,
                    total=0.0,
                    total_unit=spec.total_unit,
                    errors=0,
                    message="Interrupted before timed execution",
                    exit_code=ExitCode.SUCCESS,
                )

            ver = checkpoint_failure if checkpoint_failure is not None else executor.verify(exec_ctx)
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
            skip_reason = str(exc).strip() or explain_unsupported_workload(
                binding.device_info, spec
            )
            if self.diag_fn is not None:
                self.diag_fn(
                    f"skip [{binding.backend}:{spec.name}] dev={device.index}: {skip_reason}"
                )
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
                message=skip_reason,
                reason=skip_reason,
                exit_code=self._unsupported_exit_code(target_count),
            )
        except KernelCompilationError as exc:
            if self.diag_fn is not None and exc.compiler_log:
                for log_line in exc.compiler_log.splitlines():
                    self.diag_fn(f"runtime compiler [{binding.backend}:{spec.name}]: {log_line}")
            if _is_unsupported_compile_error(exc):
                skip_reason = _extract_unsupported_reason(exc, binding.device_info, spec)
                if self.diag_fn is not None:
                    self.diag_fn(
                        f"skip [{binding.backend}:{spec.name}] dev={device.index}: {skip_reason}"
                    )
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
                    message=skip_reason,
                    reason=skip_reason,
                    exit_code=self._unsupported_exit_code(target_count),
                )
            return WorkloadResult(
                device=device.index,
                device_name=device.name,
                backend=binding.backend,
                workload=spec.name,
                category=spec.category,
                status="init_error",
                duration_s=0.0,
                rate=None,
                rate_unit=spec.rate_unit,
                total=None,
                total_unit=spec.total_unit,
                errors=0,
                message=f"{exc}: {exc.compiler_log}".strip(": "),
                exit_code=ExitCode.INIT_ERROR,
            )
        except KernelNotFoundError as exc:
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
        except BackendInitError as exc:
            return WorkloadResult(
                device=device.index,
                device_name=device.name,
                backend=binding.backend,
                workload=spec.name,
                category=spec.category,
                status="init_error",
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
        target_count = len(targets)
        try:
            if target_count == 1:
                dev, binding, plugin = targets[0]
                self._run_device_workloads(
                    dev, binding, plugin, results, results_lock, target_count=1
                )
            else:
                threads: list[threading.Thread] = []
                for dev, binding, plugin in targets:
                    t = threading.Thread(
                        target=self._run_device_workloads,
                        args=(dev, binding, plugin, results, results_lock, target_count),
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

        # Promote explicit --test <name> to INIT_ERROR if unsupported on ALL selected devices.
        if results and self.explicit_test_selection:
            by_workload: dict[str, list[WorkloadResult]] = {}
            for r in results:
                by_workload.setdefault(r.workload, []).append(r)
            for wl_results in by_workload.values():
                if all(r.status == "unsupported" for r in wl_results):
                    for r in wl_results:
                        r.exit_code = ExitCode.INIT_ERROR

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
