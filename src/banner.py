# SPDX-License-Identifier: MIT
"""Terminal banner and ANSI color guards (<80 columns)."""

from __future__ import annotations

import os
from typing import TextIO

from . import __version__

BANNER_LINES = (
    r"   ______                     __________  __  __",
    r"  / ____/________  __________/ ____/ __ \/ / / /",
    r" / /   / ___/ __ \/ ___/ ___/ / __/ /_/ / / / / ",
    r"/ /___/ /  / /_/ (__  |__  ) /_/ / ____/ /_/ /  ",
    r"\____/_/   \____/____/____/\____/_/    \____/   ",
)


def supports_color(stream: TextIO, env: dict[str, str] | None = None) -> bool:
    """Return True only when ANSI color output is appropriate for ``stream``."""
    environ = os.environ if env is None else env
    if "NO_COLOR" in environ:
        return False
    term = environ.get("TERM", "")
    if term.lower() == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def should_show_banner(
    stream: TextIO,
    *,
    no_banner: bool = False,
    json_mode: bool = False,
    env: dict[str, str] | None = None,
) -> bool:
    """Return True only when the interactive ASCII banner should be emitted."""
    if no_banner or json_mode:
        return False
    environ = os.environ if env is None else env
    if environ.get("TERM", "").lower() == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def render_banner(use_color: bool = False, version: str = __version__) -> str:
    """Render the <80-column CrossGPU ASCII banner."""
    subtitle = f"CrossGPU v{version} — Cross-Platform GPU Stress & Validation Suite"
    if use_color:
        cyan = "\033[36;1m"
        dim = "\033[2m"
        reset = "\033[0m"
        body = "\n".join(f"{cyan}{line}{reset}" for line in BANNER_LINES)
        return f"{body}\n{dim}{subtitle}{reset}\n"
    body = "\n".join(BANNER_LINES)
    return f"{body}\n{subtitle}\n"


def maybe_print_banner(
    stream: TextIO,
    *,
    no_banner: bool = False,
    json_mode: bool = False,
    env: dict[str, str] | None = None,
) -> bool:
    """Print the banner to ``stream`` if guards allow; return True if printed."""
    if not should_show_banner(stream, no_banner=no_banner, json_mode=json_mode, env=env):
        return False
    text = render_banner(use_color=supports_color(stream, env=env))
    stream.write(text + "\n")
    stream.flush()
    return True
