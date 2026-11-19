# SPDX-License-Identifier: MIT
"""CLI argument parser, duration parser, text/JSON/NDJSON formatters, and dispatcher."""

from __future__ import annotations

import argparse
import difflib
import inspect
import json
import math
import re
import sys
import threading
from typing import Any, Callable, Mapping, Sequence, TextIO

from cross_gpu import __version__
from cross_gpu.backends.base import DiagCallback
from cross_gpu.backends.registry import (
    KNOWN_BACKENDS,
    enumerate_all_backend_devices,
    enumerate_all_backend_devices_with_diagnostics,
)
from cross_gpu.dedup import deduplicate_devices
from cross_gpu.exit_codes import ExitCode, UsageError
from cross_gpu.models import GPUDevice, WorkloadResult
from cross_gpu.runner import StressRunner
from cross_gpu.workloads import WORKLOAD_REGISTRY, list_workload_names

_DURATION_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(ms|[smh])\s*$",
    re.IGNORECASE,
)
_DURATION_MULTIPLIERS: dict[str, float] = {
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}
_SUBCOMMANDS: tuple[str, ...] = ("list", "info", "run")
_INVALID_COMMAND_RE = re.compile(
    r"(?:argument\s+command:\s*)?invalid choice:\s*'([^']+)'",
    re.IGNORECASE,
)

_ROOT_EXAMPLES_EPILOG = """\
Examples:
  cross-gpu list
  cross-gpu info --json
  cross-gpu run --device 0 --test compute-fp32,memory-pattern --warmup 5s --duration 30s
  cross-gpu run --backend opencl --test all --duration 60s --json
"""

_LIST_EXAMPLES_EPILOG = """\
Examples:
  cross-gpu list
  cross-gpu list --json
  cross-gpu list --verbose
"""

_INFO_EXAMPLES_EPILOG = """\
Examples:
  cross-gpu info
  cross-gpu info --json
  cross-gpu info --json --verbose
"""

_RUN_EXAMPLES_EPILOG = """\
Examples:
  cross-gpu run --device 0 --test compute-fp32,memory-pattern --warmup 5s --duration 30s
  cross-gpu run --backend opencl --test all --duration 60s --json
  cross-gpu run --test compute-fp32 --duration 10s --verbose
"""


def suggest_close_matches(token: str, candidates: Sequence[str]) -> list[str]:
    """Return up to 3 close matches using stdlib difflib (cutoff=0.5)."""
    return difflib.get_close_matches(token, list(candidates), n=3, cutoff=0.5)


def format_unknown_with_suggestion(
    kind: str,
    token: str,
    candidates: Sequence[str],
    *,
    fallback_hint: str = "",
) -> str:
    """Format a usage error with optional difflib 'Did you mean' suggestions."""
    matches = suggest_close_matches(token, candidates)
    if matches:
        return f"unknown {kind} '{token}'. Did you mean: {', '.join(matches)}?"
    if fallback_hint:
        return f"unknown {kind} '{token}'; {fallback_hint}"
    return f"unknown {kind} '{token}'"


class DeterministicArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that routes all parsing errors to ExitCode.USAGE_ERROR (1)."""

    def __init__(self, *args: Any, out_stream: TextIO | None = None, err_stream: TextIO | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._out_stream = out_stream
        self._err_stream = err_stream

    def _print_message(self, message: str, file: Any = None) -> None:  # type: ignore[override]
        if not message:
            return
        if file is None or file is sys.stdout:
            target = sys.stdout if self._out_stream is None else self._out_stream
        elif file is sys.stderr:
            target = sys.stderr if self._err_stream is None else self._err_stream
        else:
            target = file
        target.write(message)

    def error(self, message: str) -> None:  # type: ignore[override]
        err = sys.stderr if self._err_stream is None else self._err_stream
        formatted = message
        if self.prog == "cross-gpu":
            match = _INVALID_COMMAND_RE.search(message)
            if match and "list" in message and "info" in message and "run" in message:
                formatted = format_unknown_with_suggestion(
                    "command",
                    match.group(1),
                    _SUBCOMMANDS,
                    fallback_hint="valid commands: list, info, run",
                )
        self.print_usage(err)
        err.write(f"{self.prog}: error: {formatted}\n")
        raise SystemExit(int(ExitCode.USAGE_ERROR))

    def exit(self, status: int = 0, message: str | None = None) -> None:  # type: ignore[override]
        if message:
            target = (
                (sys.stdout if self._out_stream is None else self._out_stream)
                if status == 0
                else (sys.stderr if self._err_stream is None else self._err_stream)
            )
            target.write(message)
        normalized = 0 if status == 0 else int(ExitCode.USAGE_ERROR)
        raise SystemExit(normalized)


def parse_duration(value: str, *, allow_zero: bool = False) -> float:
    """Parse duration strings like '500ms', '60s', '5m', '1.5h' into seconds."""
    if not isinstance(value, str):
        raise UsageError(f"Invalid duration {value!r}")
    match = _DURATION_PATTERN.match(value)
    if not match:
        raise UsageError(
            f"Malformed duration {value!r}; expected positive number with ms, s, m, or h suffix"
        )
    magnitude = float(match.group(1))
    unit = match.group(2).lower()
    if not math.isfinite(magnitude):
        raise UsageError(f"Invalid duration {value!r}")
    seconds = magnitude / 1000.0 if unit == "ms" else magnitude * _DURATION_MULTIPLIERS[unit]
    if seconds < 0 or (not allow_zero and seconds == 0):
        raise UsageError(
            f"Duration must be {'>= 0' if allow_zero else '> 0'}, got {value!r}"
        )
    return seconds


def format_duration(seconds: float) -> str:
    """Format a duration in seconds cleanly using ms for sub-second values and s otherwise."""
    safe = max(0.0, float(seconds))
    if safe < 1.0:
        ms = int(round(safe * 1000.0))
        if ms >= 1000:
            return "1s"
        return f"{ms}ms"
    if abs(safe - round(safe)) < 1e-6:
        return f"{int(round(safe))}s"
    return f"{safe:.1f}s"


def parse_device_arg(value: str) -> int | None:
    """Parse --device all|N into None (all) or a non-negative integer index."""
    cleaned = value.strip()
    if cleaned == "all":
        return None
    if not cleaned.isdigit():
        raise UsageError(f"Invalid --device value {value!r}; expected 'all' or non-negative integer")
    return int(cleaned)


def parse_test_arg(value: str) -> tuple[list[str], bool]:
    """Parse --test TEST|all|a,b into (validated_workloads, is_explicit_selection)."""
    cleaned = value.strip()
    if not cleaned:
        raise UsageError("--test value cannot be empty")
    if cleaned == "all":
        return list_workload_names(), False

    parts = [part.strip() for part in cleaned.split(",")]
    if any(not part for part in parts):
        raise UsageError(f"Invalid --test list {value!r}: empty test entry")
    if "all" in parts:
        raise UsageError(f"Cannot combine 'all' with specific test names in --test {value!r}")

    selected: list[str] = []
    seen: set[str] = set()
    unknown_hints: list[str] = []
    valid_names = list_workload_names()
    for part in parts:
        if part not in WORKLOAD_REGISTRY:
            unknown_hints.append(
                format_unknown_with_suggestion(
                    "test",
                    part,
                    valid_names,
                    fallback_hint=f"valid workloads: {', '.join(valid_names)}, or 'all'",
                )
            )
            continue
        if part not in seen:
            seen.add(part)
            selected.append(part)
    if unknown_hints:
        raise UsageError("; ".join(unknown_hints))
    return selected, True


def parse_backend_arg(value: str, *, extra_backends: Sequence[str] = ()) -> str:
    """Validate --backend auto|PLUGIN_NAME, raising UsageError with typo hints."""
    cleaned = value.strip()
    if not cleaned:
        raise UsageError("--backend cannot be empty")
    valid: list[str] = ["auto", *KNOWN_BACKENDS]
    for extra in extra_backends:
        if extra not in valid:
            valid.append(extra)
    if cleaned not in valid:
        raise UsageError(
            format_unknown_with_suggestion(
                "backend",
                cleaned,
                valid,
                fallback_hint=f"valid backends: {', '.join(valid)}",
            )
        )
    return cleaned


def resolve_output_format(fmt: str | None, json_flag: bool) -> str:
    """Resolve --format text|json combined with --json, failing on conflict."""
    if fmt == "text" and json_flag:
        raise UsageError("Conflicting flags: --format text cannot be combined with --json")
    if json_flag:
        return "json"
    if fmt is not None:
        return fmt
    return "text"


def build_parser(
    *,
    out_stream: TextIO | None = None,
    err_stream: TextIO | None = None,
) -> DeterministicArgumentParser:
    """Construct the cross-gpu CLI parser."""
    parser = DeterministicArgumentParser(
        prog="cross-gpu",
        description="Cross-platform GPU stress-testing CLI.",
        epilog=_ROOT_EXAMPLES_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        out_stream=out_stream,
        err_stream=err_stream,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"cross-gpu {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Write detailed probe, deduplication, and compiler diagnostics to stderr",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd_name, help_text, cmd_epilog in (
        ("list", "List detected GPU devices", _LIST_EXAMPLES_EPILOG),
        ("info", "Show detailed information for detected GPUs", _INFO_EXAMPLES_EPILOG),
    ):
        sub = subparsers.add_parser(
            cmd_name,
            help=help_text,
            epilog=cmd_epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            out_stream=out_stream,
            err_stream=err_stream,
        )
        sub.add_argument(
            "--format",
            choices=("text", "json"),
            default=None,
            help="Output format: text or json (default: text)",
        )
        sub.add_argument(
            "--json",
            action="store_true",
            help="Convenience shortcut for --format json",
        )
        sub.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Write detailed probe, deduplication, and compiler diagnostics to stderr",
        )

    run_p = subparsers.add_parser(
        "run",
        help="Run GPU stress workloads",
        epilog=_RUN_EXAMPLES_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        out_stream=out_stream,
        err_stream=err_stream,
    )
    run_p.add_argument(
        "--device",
        default="all",
        help="Device index or 'all' (default: all)",
    )
    run_p.add_argument(
        "--test",
        default="all",
        help="Workload name, comma-separated list, or 'all' (default: all)",
    )
    run_p.add_argument(
        "--duration",
        default="60s",
        help="Duration per workload such as 500ms, 60s, 5m, 1h (default: 60s)",
    )
    run_p.add_argument(
        "--warmup",
        nargs="?",
        const="5s",
        default="0s",
        help="Warmup duration before each workload such as 250ms, 5s (default: 5s if omitted value, 0s if omitted)",
    )
    run_p.add_argument(
        "--backend",
        default="auto",
        help="auto or a loaded backend plugin name (default: auto)",
    )
    run_p.add_argument(
        "--format",
        choices=("text", "json"),
        default=None,
        help="Output format: text or json (default: text)",
    )
    run_p.add_argument(
        "--json",
        action="store_true",
        help="Convenience shortcut for --format json",
    )
    run_p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Write detailed probe, deduplication, and compiler diagnostics to stderr",
    )
    return parser


def _format_mem_gb(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "N/A"
    return f"{num_bytes / (1024**3):.2f} GiB"


def format_list_text(devices: Sequence[GPUDevice]) -> str:
    """Format detected GPU devices as an aligned fixed-width table."""
    if not devices:
        return "No compatible GPU devices detected.\n"

    headers = ("INDEX", "NAME", "VENDOR", "VRAM", "ACTIVE", "BACKENDS", "ID / BDF")
    rows: list[tuple[str, ...]] = []
    for dev in devices:
        ident = dev.pci_bdf or (
            f"reg:0x{dev.registry_id:x}" if dev.registry_id is not None else (dev.uuid or "N/A")
        )
        rows.append(
            (
                str(dev.index),
                dev.name,
                dev.vendor,
                _format_mem_gb(dev.total_memory_bytes),
                dev.active_backend,
                ",".join(dev.available_backends),
                ident,
            )
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    lines = [
        "  ".join(headers[col].ljust(widths[col]) for col in range(len(headers))),
        "  ".join("-" * widths[col] for col in range(len(headers))),
    ]
    for row in rows:
        lines.append("  ".join(row[col].ljust(widths[col]) for col in range(len(headers))))
    return "\n".join(lines) + "\n"


def format_info_text(devices: Sequence[GPUDevice]) -> str:
    """Format detailed hardware and backend properties for each detected GPU."""
    if not devices:
        return "No compatible GPU devices detected.\n"

    sections: list[str] = []
    for dev in devices:
        caps = dev.capabilities.to_dict()
        enabled_caps = [k for k, v in caps.items() if v]
        lines = [
            f"Device {dev.index}: {dev.name}",
            f"  Vendor:             {dev.vendor}"
            + (f" (0x{dev.vendor_id:04x})" if dev.vendor_id is not None else ""),
            f"  Active Backend:     {dev.active_backend}",
            f"  Available Backends: {', '.join(dev.available_backends)}",
            f"  Architecture:       {dev.architecture or 'N/A'}",
            f"  Compute Capability: {dev.compute_capability or 'N/A'}",
            f"  Total Memory:       {_format_mem_gb(dev.total_memory_bytes)} ({dev.total_memory_bytes} B)",
            f"  Compute Units:      {dev.compute_units or 'N/A'}",
            f"  Driver Version:     {dev.driver_version or 'N/A'}",
            f"  PCIe BDF:           {dev.pci_bdf or 'N/A'}",
            f"  UUID:               {dev.uuid or 'N/A'}",
            f"  Registry ID:        {f'0x{dev.registry_id:x}' if dev.registry_id is not None else 'N/A'}",
            f"  Capabilities:       {', '.join(enabled_caps) if enabled_caps else 'none'}",
        ]
        for binding in dev.bindings:
            info = binding.device_info
            lines.append(
                f"  Binding [{info.backend}] (priority={info.priority}): "
                f"arch={info.architecture or 'N/A'}, cu={info.compute_units or 'N/A'}, "
                f"driver={info.driver_version or 'N/A'}"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def _format_duration_span(elapsed_s: float, total_s: float) -> str:
    safe_total = max(0.0, float(total_s))
    safe_elapsed = max(0.0, float(elapsed_s))
    if safe_total > 0.0:
        safe_elapsed = min(safe_elapsed, safe_total)
    if safe_total == 0.0:
        return f"{format_duration(safe_elapsed)}/0s"
    if safe_total < 1.0:
        total_ms = max(1, int(round(safe_total * 1000.0)))
        elapsed_ms = min(total_ms, int(round(safe_elapsed * 1000.0)))
        return f"{elapsed_ms}ms/{total_ms}ms"
    if abs(safe_total - round(safe_total)) < 1e-6:
        return f"{int(round(safe_elapsed))}s/{int(round(safe_total))}s"
    return f"{safe_elapsed:.1f}s/{safe_total:.1f}s"


def format_progress_prefix(
    workload: str,
    *,
    workload_index: int,
    workload_total: int,
    phase: str = "run",
    phase_elapsed_s: float = 0.0,
    phase_duration_s: float = 0.0,
) -> str:
    """Format zero-padded workload progress prefix, e.g. [04/20 compute-fp32 | 18s/60s]."""
    safe_total = max(1, int(workload_total))
    safe_idx = max(1, int(workload_index))
    width = max(2, len(str(safe_total)))
    span = _format_duration_span(phase_elapsed_s, phase_duration_s)
    if phase == "warmup":
        return f"[{safe_idx:0{width}d}/{safe_total:0{width}d} {workload} | warmup {span}]"
    return f"[{safe_idx:0{width}d}/{safe_total:0{width}d} {workload} | {span}]"


def format_telemetry_line(event: dict[str, Any]) -> str:
    """Format a 1 Hz telemetry event as a compact single line with all 8 fields & units."""
    m = event.get("metrics", {})
    temp = f"{m['temperature_c']:.1f}°C" if m.get("temperature_c") is not None else "N/A °C"
    pwr = f"{m['power_w']:.1f}W" if m.get("power_w") is not None else "N/A W"
    gpu = f"{m['gpu_util_pct']:.0f}%" if m.get("gpu_util_pct") is not None else "N/A %"
    mem = f"{m['mem_util_pct']:.0f}%" if m.get("mem_util_pct") is not None else "N/A %"
    cclk = f"{m['core_clock_mhz']}MHz" if m.get("core_clock_mhz") is not None else "N/A MHz"
    mclk = f"{m['mem_clock_mhz']}MHz" if m.get("mem_clock_mhz") is not None else "N/A MHz"
    fan_p = f"{m['fan_speed_pct']:.0f}%" if m.get("fan_speed_pct") is not None else "N/A %"
    fan_r = f"{m['fan_speed_rpm']}RPM" if m.get("fan_speed_rpm") is not None else "N/A RPM"

    wl_idx = event.get("workload_index")
    wl_total = event.get("workload_total")
    if wl_idx is not None and wl_total is not None:
        prefix = format_progress_prefix(
            str(event.get("workload", "")),
            workload_index=int(wl_idx),
            workload_total=int(wl_total),
            phase=str(event.get("phase", "run")),
            phase_elapsed_s=float(event.get("phase_elapsed_s", event.get("elapsed_s", 0.0))),
            phase_duration_s=float(event.get("phase_duration_s", 0.0)),
        )
        return (
            f"{prefix} dev={event.get('device')} | "
            f"Temp={temp} Power={pwr} GPU={gpu} Mem={mem} "
            f"CoreClk={cclk} MemClk={mclk} Fan={fan_p}/{fan_r}"
        )

    return (
        f"[{event.get('elapsed_s', 0.0):6.1f}s] dev={event.get('device')} "
        f"phase={event.get('phase')} workload={event.get('workload')} | "
        f"Temp={temp} Power={pwr} GPU={gpu} Mem={mem} "
        f"CoreClk={cclk} MemClk={mclk} Fan={fan_p}/{fan_r}"
    )


def format_workload_status_line(event: dict[str, Any]) -> str:
    """Format a compact progress-prefixed workload completion status line for text mode."""
    wl_idx = int(event.get("workload_index", 1))
    wl_total = int(event.get("workload_total", 1))
    target_dur = float(event.get("target_duration_s", event.get("duration_s", 0.0)))
    elapsed = float(event.get("duration_s", 0.0))
    prefix = format_progress_prefix(
        str(event.get("workload", "")),
        workload_index=wl_idx,
        workload_total=wl_total,
        phase="run",
        phase_elapsed_s=elapsed,
        phase_duration_s=target_dur,
    )
    status = str(event.get("status", ""))
    dev = event.get("device")
    if status == "pass":
        rate = event.get("rate")
        rate_unit = event.get("rate_unit", "")
        rate_part = f" rate={rate:.2f} {rate_unit}" if rate is not None else ""
        return f"{prefix} dev={dev} status=pass{rate_part}"
    detail = str(event.get("reason") or event.get("message") or "").strip()
    suffix = f" ({detail})" if detail else ""
    return f"{prefix} dev={dev} status={status}{suffix}"


def format_run_summary_text(results: Sequence[WorkloadResult], exit_code: ExitCode) -> str:
    """Format final workload results as an aligned summary table with tally."""
    if not results:
        return (
            "Tally: passed=0 failed=0 unsupported=0 hardware_error=0 | "
            f"Exit Code: {int(exit_code)} ({exit_code.name})\n"
        )

    include_reason = any(
        res.status == "unsupported"
        or bool(res.reason)
        or res.status in ("fail", "init_error", "hardware_error", "not_implemented")
        for res in results
    )
    headers: tuple[str, ...] = (
        ("DEV", "BACKEND", "WORKLOAD", "STATUS", "RATE", "TOTAL", "ERR", "TIME", "EXIT", "REASON")
        if include_reason
        else ("DEV", "BACKEND", "WORKLOAD", "STATUS", "RATE", "TOTAL", "ERR", "TIME", "EXIT")
    )
    rows: list[tuple[str, ...]] = []
    for res in results:
        rate_str = f"{res.rate:.2f} {res.rate_unit}" if res.rate is not None else "N/A"
        total_str = f"{res.total:.2f} {res.total_unit}" if res.total is not None else "N/A"
        base_row = (
            str(res.device),
            res.backend,
            res.workload,
            res.status,
            rate_str,
            total_str,
            str(res.errors),
            format_duration(res.duration_s) if 0.0 < res.duration_s < 1.0 else f"{res.duration_s:.2f}s",
            str(int(res.exit_code)),
        )
        if include_reason:
            detail = res.reason or (res.message if res.status != "pass" else "")
            if res.status == "unsupported" and not detail:
                detail = f"unsupported on {res.backend}"
            rows.append((*base_row, detail or "-"))
        else:
            rows.append(base_row)

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    lines = [
        "",
        "Summary:",
        "  ".join(headers[col].ljust(widths[col]) for col in range(len(headers))),
        "  ".join("-" * widths[col] for col in range(len(headers))),
    ]
    for row in rows:
        lines.append("  ".join(row[col].ljust(widths[col]) for col in range(len(headers))))

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    unsupported = sum(1 for r in results if r.status == "unsupported")
    hardware_error = sum(1 for r in results if r.status == "hardware_error")
    init_error = sum(1 for r in results if r.status in ("init_error", "not_implemented"))
    tally = (
        f"Tally: passed={passed} failed={failed} unsupported={unsupported} "
        f"hardware_error={hardware_error}"
    )
    if init_error:
        tally += f" init_error={init_error}"
    lines.append(f"{tally} | Exit Code: {int(exit_code)} ({exit_code.name})")
    return "\n".join(lines) + "\n"


def _make_verbose_logger(err_stream: TextIO, enabled: bool) -> DiagCallback | None:
    if not enabled:
        return None
    lock = threading.Lock()

    def _log(message: str) -> None:
        with lock:
            err_stream.write(f"[verbose] {message}\n")
            err_stream.flush()

    return _log


def _write_backend_diagnostics(
    err_stream: TextIO,
    backend_diagnostics: Mapping[str, str],
    *,
    header: str | None = None,
    only_backend: str | None = None,
) -> None:
    if not backend_diagnostics:
        return
    if only_backend is not None and only_backend != "auto":
        reason = backend_diagnostics.get(only_backend)
        if reason:
            err_stream.write(f"cross-gpu: {only_backend}: {reason}\n")
            err_stream.flush()
        return
    if header:
        err_stream.write(f"cross-gpu: {header}\n")
    for backend_name in sorted(backend_diagnostics):
        err_stream.write(f"  {backend_name}: {backend_diagnostics[backend_name]}\n")
    err_stream.flush()


def _default_discover_fn() -> tuple[list[GPUDevice], dict[str, Any]]:
    raw_devices, plugins = enumerate_all_backend_devices()
    return deduplicate_devices(raw_devices), plugins


def _discover_with_diagnostics(
    discover_fn: Callable[..., Any] | None,
    *,
    diag_fn: DiagCallback | None = None,
) -> tuple[list[GPUDevice], dict[str, Any], dict[str, str]]:
    if discover_fn is None or discover_fn is _default_discover_fn:
        raw_devices, plugins, diagnostics = enumerate_all_backend_devices_with_diagnostics(
            diag_fn=diag_fn
        )
        return deduplicate_devices(raw_devices, diag_fn=diag_fn), plugins, diagnostics

    result = discover_fn()
    if isinstance(result, tuple) and len(result) == 3:
        return result[0], result[1], dict(result[2])
    return result[0], result[1], {}


def _runner_accepts_kwarg(runner_factory: Callable[..., Any], kwarg: str) -> bool:
    try:
        sig = inspect.signature(runner_factory)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == kwarg:
            return True
    return False


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    discover_fn: Callable[..., Any] | None = _default_discover_fn,
    runner_factory: Callable[..., Any] = StressRunner,
) -> int:
    """Main CLI entry point returning deterministic integer exit code."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    parser = build_parser(out_stream=out, err_stream=err)
    args = parser.parse_args(list(argv) if argv is not None else None)
    verbose = bool(getattr(args, "verbose", False))
    diag_fn = _make_verbose_logger(err, verbose)

    try:
        fmt = resolve_output_format(args.format, bool(args.json))
        if args.command == "list":
            devices, _, backend_diagnostics = _discover_with_diagnostics(
                discover_fn, diag_fn=diag_fn
            )
            if not devices:
                _write_backend_diagnostics(
                    err, backend_diagnostics, header="no compatible GPUs detected:"
                )
            if fmt == "json":
                payload = {"devices": [dev.to_list_dict() for dev in devices]}
                out.write(json.dumps(payload, indent=2) + "\n")
            else:
                out.write(format_list_text(devices))
            out.flush()
            return int(ExitCode.SUCCESS)

        if args.command == "info":
            devices, _, backend_diagnostics = _discover_with_diagnostics(
                discover_fn, diag_fn=diag_fn
            )
            if not devices:
                _write_backend_diagnostics(
                    err, backend_diagnostics, header="no compatible GPUs detected:"
                )
            if fmt == "json":
                payload = {"devices": [dev.to_info_dict() for dev in devices]}
                out.write(json.dumps(payload, indent=2) + "\n")
            else:
                out.write(format_info_text(devices))
            out.flush()
            return int(ExitCode.SUCCESS)

        if args.command == "run":
            device_index = parse_device_arg(args.device)
            tests, explicit_test_selection = parse_test_arg(args.test)
            duration_s = parse_duration(args.duration, allow_zero=False)
            warmup_s = parse_duration(args.warmup, allow_zero=True)
            is_default_discover = discover_fn is None or discover_fn is _default_discover_fn
            if is_default_discover:
                backend_pref = parse_backend_arg(args.backend)
            else:
                backend_pref = args.backend.strip()
                if not backend_pref:
                    raise UsageError("--backend cannot be empty")

            devices, plugins, backend_diagnostics = _discover_with_diagnostics(
                discover_fn, diag_fn=diag_fn
            )
            if not is_default_discover:
                backend_pref = parse_backend_arg(
                    args.backend, extra_backends=tuple(plugins.keys())
                )

            def emit_event(event: dict[str, Any]) -> None:
                etype = event.get("type")
                if etype == "run_summary" and event.get("error"):
                    err.write(f"cross-gpu: error: {event['error']}\n")
                    if not devices:
                        _write_backend_diagnostics(
                            err,
                            backend_diagnostics,
                            header="backend probe summary:" if backend_pref == "auto" else None,
                            only_backend=backend_pref if backend_pref != "auto" else None,
                        )
                    elif backend_pref != "auto":
                        _write_backend_diagnostics(
                            err,
                            backend_diagnostics,
                            only_backend=backend_pref,
                        )
                    err.flush()

                if fmt == "json":
                    out.write(json.dumps(event) + "\n")
                    out.flush()
                    return
                if etype == "telemetry":
                    out.write(format_telemetry_line(event) + "\n")
                    out.flush()
                elif etype == "workload_result":
                    out.write(format_workload_status_line(event) + "\n")
                    out.flush()
                elif etype == "run_start":
                    devs_desc = ", ".join(
                        f"{d['index']}:{d['name']} ({d['backend']})"
                        for d in event.get("devices", ())
                    )
                    out.write(
                        f"Starting cross-gpu run on [{devs_desc}] | "
                        f"tests={len(tests)} duration={format_duration(duration_s)} "
                        f"warmup={format_duration(warmup_s)}\n"
                    )
                    out.flush()

            runner_kwargs: dict[str, Any] = {
                "devices": devices,
                "plugins": plugins,
                "tests": tests,
                "duration_s": duration_s,
                "warmup_s": warmup_s,
                "device_index": device_index,
                "backend_preference": backend_pref,
                "explicit_test_selection": explicit_test_selection,
                "event_callback": emit_event,
            }
            if _runner_accepts_kwarg(runner_factory, "diag_fn"):
                runner_kwargs["diag_fn"] = diag_fn
            if _runner_accepts_kwarg(runner_factory, "backend_diagnostics"):
                runner_kwargs["backend_diagnostics"] = backend_diagnostics

            runner = runner_factory(**runner_kwargs)
            report = runner.run()
            if fmt == "text":
                out.write(format_run_summary_text(report.results, report.exit_code))
                out.flush()
            return int(report.exit_code)

        raise UsageError(f"Unknown command: {args.command}")

    except UsageError as exc:
        err.write(f"cross-gpu: error: {exc}\n")
        err.flush()
        return int(ExitCode.USAGE_ERROR)
