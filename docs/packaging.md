# Packaging & Distribution Guide

`cross-gpu` is designed for clean, reproducible, offline packaging on Linux distributions (Debian, Fedora, Arch Linux, Nix) and macOS (Homebrew) with **zero external runtime dependencies**.

---

## 1. PEP 517 / PEP 621 Offline Builds

`pyproject.toml` uses `setuptools` (`setuptools.build_meta`), declares `requires-python = ">=3.10"`, `dependencies = []`, SPDX license `MIT`, and exposes the `cross-gpu = "cross_gpu.cli:main"` console entry point.

Build and install strictly offline without build isolation:

```sh
# Build wheel and sdist offline
python3 -m build --no-isolation

# Or install directly into a prefix/staging root
pip install --no-build-isolation --no-deps .
```

The repository `Makefile` also provides standard `PREFIX` and `DESTDIR` staging, including installation of the `cross-gpu(1)` man page (`docs/cross-gpu.1`):

```sh
make check
make install PREFIX=/usr DESTDIR=/tmp/stage
```

---

## 2. Bundling Raw Kernel Source Files

All 90 raw GPU kernel source files under `cross_gpu/kernels/` are declared as package data in `pyproject.toml` and included in source distributions via `MANIFEST.in`:

```toml
[tool.setuptools]
include-package-data = true

[tool.setuptools.package-data]
cross_gpu = [
    "kernels/*/*.cu",
    "kernels/*/*.hip",
    "kernels/*/*.cl",
    "kernels/*/*.metal",
]
```

At runtime, `cross_gpu/embed_binary.py` reads these files exclusively through `importlib.resources.files("cross_gpu").joinpath("kernels", backend, filename)`. This works uniformly across:

- Editable development checkouts
- Installed wheels in read-only system directories (e.g., `/usr/lib/python3/dist-packages/`)
- Zipped archives / zipapps
- Virtual environments and Homebrew formulas

---

## 3. FHS & Read-Only Install Safety

- `cross-gpu` **never** writes files inside the installed Python package directory.
- Runtime-compiled GPU binaries (`.bin`) and plain metadata sidecars (`.json`) are cached strictly in OS-standard user cache locations:
  - **Linux / POSIX:** `$XDG_CACHE_HOME/cross-gpu` (when `$XDG_CACHE_HOME` is an absolute path), falling back to `~/.cache/cross-gpu`
  - **macOS:** `~/Library/Caches/cross-gpu`
  - **Explicit override:** `$CROSS_GPU_CACHE_DIR`
- If the resolved cache directory is read-only, unwritable, or `$HOME` is unavailable (e.g., minimal sandboxed containers), `RuntimeBinaryCache` automatically degrades to in-memory-only tuple caching without raising errors.

---

## 4. Verifying an Installed Package Outside the Source Tree

Packagers should verify that `importlib.resources` resolves every bundled kernel from the installed package rather than from a relative path in the source checkout:

```sh
python3 -m venv /tmp/cross-gpu-verify
/tmp/cross-gpu-verify/bin/pip install --no-build-isolation --no-deps .
(
  cd /tmp
  /tmp/cross-gpu-verify/bin/cross-gpu --version
  /tmp/cross-gpu-verify/bin/python -m cross_gpu.embed_binary verify
  /tmp/cross-gpu-verify/bin/python -m cross_gpu.embed_binary list | wc -l
)
rm -rf /tmp/cross-gpu-verify
```

Expected output:

- `Verified 90 packaged kernel source(s): OK` (exit code `0`)
- `90` lines from `embed_binary list`
