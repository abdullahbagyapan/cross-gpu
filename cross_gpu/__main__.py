# SPDX-License-Identifier: MIT
"""Module execution entry point for `python3 -m cross_gpu`."""

from __future__ import annotations

import sys

from cross_gpu.cli import main

if __name__ == "__main__":
    sys.exit(main())
