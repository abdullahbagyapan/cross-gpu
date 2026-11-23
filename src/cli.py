# SPDX-License-Identifier: MIT
"""CLI argument parser, duration parser, text/JSON formatters, and dispatcher."""

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

from . import __version__
from .backends.base import DiagCallback
from .backends.registry import (
    KNOWN_BACKENDS,
    enumerate_all_backend_devices,
    enumerate_all_backend_devices_with_diagnostics,
)
from .banner import maybe_print_banner, supports_color
from .dedup import deduplicate_devices
from .exit_codes import ExitCode, UsageError
from .models import GPUDevice, WorkloadResult
from .runner import StressRunner
from .workloads import WORKLOAD_REGISTRY, list_workload_names

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

_STATUS_LABELS: dict[str, str] = {
    "passed": "PASSED",
    "failed": "FAILED",
    "unsupported": "UNSUPPORTED",
    "init_error": "INIT_ERROR",
    "hardware_error": "HARDWARE_ERROR",
    "not_implemented": "INIT_ERROR",
    "stopped": "STOPPED",
}

_STATUS_COLORS: dict[str, str] = {
    "PASSED": "\033[32;1m",
    "FAILED": "\033[31;1m",
    "UNSUPPORTED": "\033[33m",
    "INIT_ERROR": "\033[31m",
    "HARDWARE_ERROR": "\033[31;1m",
    "STOPPED": "\033[36m",
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
    if safe == 0.0:
        return "0s"
    if safe < 1.0:
        ms = int(round(safe * 1000.0))
        if ms >= 1000:
            return "1s"
        if ms == 0:
            return "<1ms"
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
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

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
        "--no-banner",
        action="store_true",
        default=False,
        help="Suppress the interactive ASCII banner",
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


def format_status_badge(status: str, use_color: bool = False) -> str:
    """Format workload status with optional ANSI colors."""
    label = _STATUS_LABELS.get(status.lower(), status.upper())
    if not use_color:
        return label
    prefix = _STATUS_COLORS.get(label, "")
    return f"{prefix}{label}\033[0m" if prefix else label


def format_workload_status_line(event: dict[str, Any], *, use_color: bool = False) -> str:
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
    badge = format_status_badge(status, use_color=use_color)
    dev = event.get("device")
    if status == "passed":
        rate = event.get("rate")
        rate_unit = event.get("rate_unit", "")
        rate_part = f" rate={rate:.2f} {rate_unit}" if rate is not None else ""
        return f"{prefix} dev={dev} status={badge}{rate_part}"
    detail = str(event.get("reason") or event.get("message") or "").strip()
    suffix = f" ({detail})" if detail else ""
    return f"{prefix} dev={dev} status={badge}{suffix}"


def format_workload_row_text(event: Mapping[str, Any], *, use_color: bool = False) -> str:
    """Format a completed workload row for the live CrossGPU text table."""
    name = str(event.get("workload", ""))
    dev = str(event.get("device", 0))
    backend = str(event.get("backend", ""))
    status = str(event.get("status", ""))
    raw_badge = format_status_badge(status, use_color=False)
    badge = format_status_badge(status, use_color=use_color)
    pad_extra = len(badge) - len(raw_badge)

    rate = event.get("rate")
    rate_unit = str(event.get("rate_unit", ""))
    total = event.get("total")
    total_unit = str(event.get("total_unit", ""))
    rate_str = f"{float(rate):.2f} {rate_unit}".strip() if rate is not None else "-"
    total_str = f"{float(total):.2f} {total_unit}".strip() if total is not None else "-"

    dur_s = float(event.get("duration_s", 0.0))
    time_str = format_duration(dur_s) if 0.0 < dur_s < 1.0 else f"{dur_s:.2f}s"
    is_pass = status == "passed"
    reason = str(
        event.get("reason") or (event.get("message") if not is_pass else "") or ""
    ).strip()
    if status == "unsupported" and not reason:
        reason = f"unsupported on {backend}"
    if is_pass and not reason:
        detail = time_str
    elif reason and dur_s > 0.0:
        detail = f"{time_str} ({reason})"
    else:
        detail = reason or time_str

    return (
        f"{name:<22} {dev:<4} {backend:<10} {badge:<{14 + pad_extra}} "
        f"{rate_str:>16} {total_str:>16}  {detail}"
    )


def format_run_footer_text(
    results: Sequence[WorkloadResult],
    exit_code: ExitCode,
) -> str:
    """Format the closing summary line for text mode."""
    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")
    unsupported = sum(1 for r in results if r.status == "unsupported")
    hardware_error = sum(1 for r in results if r.status == "hardware_error")
    init_error = sum(1 for r in results if r.status in ("init_error", "not_implemented"))

    summary_parts = [
        f"{passed} passed",
        f"{failed} failed",
        f"{unsupported} unsupported",
    ]
    if init_error:
        summary_parts.append(f"{init_error} init_error")
    if hardware_error:
        summary_parts.append(f"{hardware_error} hardware_error")

    return (
        f"Summary: {', '.join(summary_parts)} | "
        f"Exit: {exit_code.name} ({int(exit_code)})\n"
    )


def format_run_summary_text(
    results: Sequence[WorkloadResult],
    exit_code: ExitCode,
    *,
    use_color: bool = False,
) -> str:
    """Format final workload results as an aligned summary table with tally."""
    if not results:
        return (
            "Tally: passed=0 failed=0 unsupported=0 hardware_error=0 | "
            f"Exit Code: {int(exit_code)} ({exit_code.name})\n"
        )

    include_reason = any(
        res.status == "unsupported"
        or bool(res.reason)
        or res.status in ("failed", "init_error", "hardware_error", "not_implemented")
        for res in results
    )
    headers: tuple[str, ...] = (
        ("DEV", "BACKEND", "WORKLOAD", "STATUS", "RATE", "TOTAL", "ERR", "TIME", "EXIT", "REASON")
        if include_reason
        else ("DEV", "BACKEND", "WORKLOAD", "STATUS", "RATE", "TOTAL", "ERR", "TIME", "EXIT")
    )
    raw_rows: list[tuple[str, ...]] = []
    display_rows: list[tuple[str, ...]] = []
    for res in results:
        rate_str = f"{res.rate:.2f} {res.rate_unit}" if res.rate is not None else "-"
        total_str = f"{res.total:.2f} {res.total_unit}" if res.total is not None else "-"
        raw_status = format_status_badge(res.status, use_color=False)
        color_status = format_status_badge(res.status, use_color=use_color)
        dur_str = format_duration(res.duration_s) if 0.0 < res.duration_s < 1.0 else f"{res.duration_s:.2f}s"
        raw_base = (
            str(res.device),
            res.backend,
            res.workload,
            raw_status,
            rate_str,
            total_str,
            str(res.errors),
            dur_str,
            str(int(res.exit_code)),
        )
        disp_base = (
            str(res.device),
            res.backend,
            res.workload,
            color_status,
            rate_str,
            total_str,
            str(res.errors),
            dur_str,
            str(int(res.exit_code)),
        )
        if include_reason:
            detail = res.reason or (res.message if res.status == "passed" else "")
            if res.status == "unsupported" and not detail:
                detail = f"unsupported on {res.backend}"
            raw_rows.append((*raw_base, detail or "-"))
            display_rows.append((*disp_base, detail or "-"))
        else:
            raw_rows.append(raw_base)
            display_rows.append(disp_base)

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in raw_rows))
        for col in range(len(headers))
    ]
    lines = [
        "",
        "Summary:",
        "  ".join(headers[col].ljust(widths[col]) for col in range(len(headers))),
        "  ".join("-" * widths[col] for col in range(len(headers))),
    ]
    for raw_row, disp_row in zip(raw_rows, display_rows):
        cells: list[str] = []
        for col in range(len(headers)):
            pad_extra = len(disp_row[col]) - len(raw_row[col])
            cells.append(disp_row[col].ljust(widths[col] + pad_extra))
        lines.append("  ".join(cells))

    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")
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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        parser.print_help(file=out)
        out.flush()
        return int(ExitCode.USAGE_ERROR)

    args = parser.parse_args(raw_argv)
    if not getattr(args, "command", None):
        parser.print_help(file=out)
        out.flush()
        return int(ExitCode.USAGE_ERROR)

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

            use_color = supports_color(out)
            is_tty = False
            try:
                is_tty = bool(out.isatty())
            except Exception:
                is_tty = False

            no_banner = bool(getattr(args, "no_banner", False))
            progress_active = [False]
            table_header_printed = [False]
            started_devices: list[dict[str, Any]] = []
            run_error_msg: list[str] = []

            def _clear_progress() -> None:
                if progress_active[0] and is_tty and fmt != "json":
                    out.write("\r\033[K")
                    out.flush()
                    progress_active[0] = False

            def _ensure_table_header(
                devs_list: Sequence[Mapping[str, Any]] | None = None,
            ) -> None:
                if table_header_printed[0] or fmt == "json":
                    return
                maybe_print_banner(out, no_banner=no_banner, json_mode=False)
                if devs_list:
                    devs_desc = ", ".join(
                        f"{d['index']}:{d['name']} ({d['backend']})"
                        for d in devs_list
                    )
                    warmup_part = (
                        f" | Warmup: {format_duration(warmup_s)}"
                        if warmup_s > 0.0
                        else ""
                    )
                    out.write(
                        f"Devices: {devs_desc} | Tests: {len(tests)} | "
                        f"Backend: {backend_pref} | Duration: {format_duration(duration_s)}"
                        f"{warmup_part}\n\n"
                    )
                out.write(
                    f"{'WORKLOAD':<22} {'DEV':<4} {'BACKEND':<10} {'STATUS':<14} "
                    f"{'RATE':>16} {'TOTAL':>16}  DETAIL\n"
                )
                out.write("-" * 96 + "\n")
                out.flush()
                table_header_printed[0] = True

            def emit_event(event: dict[str, Any]) -> None:
                etype = event.get("type")
                if etype == "run_summary" and event.get("error"):
                    err_str = str(event["error"])
                    run_error_msg.append(err_str)
                    err.write(f"cross-gpu: error: {err_str}\n")
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

                if etype == "run_start":
                    for d in event.get("devices", ()):
                        started_devices.append(dict(d))
                    if fmt != "json":
                        _ensure_table_header(started_devices)
                elif etype == "progress" and fmt != "json" and is_tty:
                    _ensure_table_header()
                    line = format_progress_prefix(
                        str(event.get("workload", "")),
                        workload_index=int(event.get("workload_index", 1)),
                        workload_total=int(event.get("workload_total", 1)),
                        phase=str(event.get("phase", "run")),
                        phase_elapsed_s=float(event.get("phase_elapsed_s", 0.0)),
                        phase_duration_s=float(event.get("phase_duration_s", 0.0)),
                    )
                    out.write(f"\r\033[K{line}")
                    out.flush()
                    progress_active[0] = True
                elif etype == "workload_result" and fmt != "json":
                    _clear_progress()
                    _ensure_table_header()
                    out.write(format_workload_row_text(event, use_color=use_color) + "\n")
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
            _clear_progress()
            if fmt == "json":
                counts = {
                    "passed": sum(1 for r in report.results if r.status == "passed"),
                    "failed": sum(1 for r in report.results if r.status == "failed"),
                    "unsupported": sum(1 for r in report.results if r.status == "unsupported"),
                    "init_error": sum(
                        1 for r in report.results if r.status in ("init_error", "not_implemented")
                    ),
                    "hardware_error": sum(
                        1 for r in report.results if r.status == "hardware_error"
                    ),
                }
                target_devs = started_devices or [
                    {
                        "index": d.index,
                        "name": d.name,
                        "backend": d.active_backend,
                    }
                    for d in devices
                    if device_index is None or d.index == device_index
                ]
                err_text = getattr(report, "error", "") or (run_error_msg[0] if run_error_msg else "")
                summary_obj: dict[str, Any] = {
                    **counts,
                    "total_workloads": len(report.results),
                    "interrupted": bool(report.interrupted),
                    "exit_code": int(report.exit_code),
                    "exit_name": report.exit_code.name,
                }
                if err_text:
                    summary_obj["error"] = err_text
                payload: dict[str, Any] = {
                    "command": "run",
                    "version": __version__,
                    "exit_code": int(report.exit_code),
                    "exit_name": report.exit_code.name,
                    "interrupted": bool(report.interrupted),
                    "total_workloads": len(report.results),
                    "devices": target_devs,
                    "config": {
                        "device": args.device,
                        "backend": backend_pref,
                        "workloads": list(tests),
                        "duration_s": duration_s,
                        "warmup_s": warmup_s,
                    },
                    "results": [r.to_dict() for r in report.results],
                    "summary": summary_obj,
                }
                if err_text:
                    payload["error"] = err_text
                out.write(json.dumps(payload, indent=2) + "\n")
                out.flush()
                return int(report.exit_code)

            if fmt == "text":
                if table_header_printed[0]:
                    out.write("-" * 96 + "\n")
                out.write(
                    format_run_footer_text(
                        report.results,
                        report.exit_code,
                    )
                )
                out.flush()
            return int(report.exit_code)

        raise UsageError(f"Unknown command: {args.command}")

    except UsageError as exc:
        err.write(f"cross-gpu: error: {exc}\n")
        err.flush()
        return int(ExitCode.USAGE_ERROR)


if __name__ == "__main__":
    sys.exit(main())
