#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render GitHub Actions job summaries with expanded failure diagnostics from emulation logs."""

from __future__ import annotations

from pathlib import Path
import sys

MAX_LINES_PER_STREAM = 200
MAX_CHARS_PER_STREAM = 16000


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _tail_text(text: str) -> str:
    if not text:
        return "(empty)"
    lines = text.splitlines()
    truncated_lines = False
    if len(lines) > MAX_LINES_PER_STREAM:
        lines = lines[-MAX_LINES_PER_STREAM:]
        truncated_lines = True
    joined = "\n".join(lines)
    if len(joined) > MAX_CHARS_PER_STREAM:
        joined = joined[-MAX_CHARS_PER_STREAM:]
        truncated_lines = True
    if truncated_lines:
        return f"... (truncated; see uploaded workflow artifact for full log)\n{joined}"
    return joined


def _code_fence(text: str) -> str:
    fence = "````" if "```" in text else "```"
    return f"{fence}text\n{text}\n{fence}"


def render_summary(title: str, log_dir: Path) -> str:
    lines: list[str] = [f"### {title}", ""]
    rc_files = sorted(log_dir.glob("*.rc")) if log_dir.is_dir() else []

    if not rc_files:
        lines.append("> **Warning:** No logged commands were recorded before this job stopped.")
        lines.append("> Check the runner setup / container pull step logs above for the root cause.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| Status | Step | Exit Code | Expected | Stdout | Stderr |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    failures: list[tuple[str, str, str, str, Path, Path]] = []
    for rc_file in rc_files:
        name = rc_file.stem
        actual = _read_text(rc_file).strip() or "?"
        exp_path = log_dir / f"{name}.expected_rc"
        expected = _read_text(exp_path).strip() if exp_path.is_file() else "0"
        out_path = log_dir / f"{name}.stdout.log"
        err_path = log_dir / f"{name}.stderr.log"
        cmd_path = log_dir / f"{name}.cmd"
        failed_marker = log_dir / f"{name}.failed"

        out_size = out_path.stat().st_size if out_path.is_file() else 0
        err_size = err_path.stat().st_size if err_path.is_file() else 0
        passed = (actual == expected) and not failed_marker.is_file()
        badge = "PASS" if passed else "**FAIL**"
        lines.append(
            f"| {badge} | `{name}` | `{actual}` | `{expected}` | {out_size} B | {err_size} B |"
        )
        if not passed:
            cmd = _read_text(cmd_path).strip() or "(command unavailable)"
            failures.append((name, actual, expected, cmd, out_path, err_path))

    lines.append("")
    if failures:
        lines.append(f"#### Failure Details ({len(failures)} failed step(s))")
        lines.append("")
        for name, actual, expected, cmd, out_path, err_path in failures:
            err_text = _tail_text(_read_text(err_path))
            out_text = _tail_text(_read_text(out_path))
            lines.append(
                f"<details open><summary><strong>FAIL: <code>{name}</code> "
                f"(expected rc={expected}, got rc={actual})</strong></summary>"
            )
            lines.append("")
            lines.append(f"**Command:** `{cmd}`")
            lines.append("")
            lines.append("**stderr:**")
            lines.append(_code_fence(err_text))
            lines.append("")
            lines.append("**stdout:**")
            lines.append(_code_fence(out_text))
            lines.append("")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} <title> <log_dir>", file=sys.stderr)
        return 2
    sys.stdout.write(render_summary(argv[1], Path(argv[2])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
