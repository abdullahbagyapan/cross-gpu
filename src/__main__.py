# SPDX-License-Identifier: MIT
"""Module execution entry point for `cross_gpu` (installed) or `src` (source tree)."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
