# Packaging & Distribution Guide

CrossGPU is designed for clean, reproducible, offline packaging across **Debian/Ubuntu**, **Fedora/RHEL**, **Arch Linux**, **Nix**, and **Homebrew** with **zero external Python runtime dependencies** (`requires-python = ">=3.10"`, `dependencies = []`, SPDX license `MIT`).

---

## 1. Offline PEP 517 / PEP 621 Builds

`pyproject.toml` uses universally packaged `setuptools` (`setuptools.build_meta`) and exposes the `cross-gpu = "cross_gpu.cli:main"` console entry point.

Build and install strictly offline without build isolation:

```sh
# Build wheel and sdist offline
python3 -m build --no-isolation

# Install directly into a prefix/staging root without network access
pip install --no-build-isolation --no-deps .
```

### `Makefile` FHS Staging Targets

`Makefile` honors `PREFIX` (default `/usr/local`), `DESTDIR`, `PYTHON`, `DATADIR`, `MANDIR`, `BASHCOMPDIR`, `ZSHCOMPDIR`, and `FISHCOMPDIR`:

```sh
make check
make install PREFIX=/usr DESTDIR=/tmp/stage
```

Installed FHS files:

| File | Staged Path |
|---|---|
| Console script | `$(DESTDIR)$(PREFIX)/bin/cross-gpu` |
| Man page | `$(DESTDIR)$(PREFIX)/share/man/man1/cross-gpu.1` |
| Bash completion | `$(DESTDIR)$(PREFIX)/share/bash-completion/completions/cross-gpu` |
| Zsh completion | `$(DESTDIR)$(PREFIX)/share/zsh/site-functions/_cross-gpu` |
| Fish completion | `$(DESTDIR)$(PREFIX)/share/fish/vendor_completions.d/cross-gpu.fish` |

### Distribution Packaging Notes

- **Debian / Ubuntu (`dh-python` / `pybuild`):** Uses `pybuild --system=pyproject`, `python3-setuptools` as the sole build dependency, zero runtime dependencies beyond `python3 (>= 3.10)`, and `SPDX-License-Identifier: MIT` headers across all source files for straightforward `debian/copyright` auditing. Install `docs/cross-gpu.1` via `dh_installman` and `completions/*` via `dh_install`.
- **Fedora / RHEL (RPM `%pyproject_*` macros):** Build with `%pyproject_buildrequires` / `%pyproject_wheel` / `%pyproject_install`, run `%check` with `python3 -m unittest discover -s tests -v` and `python3 -m cross_gpu.embed_binary verify`, and install `docs/cross-gpu.1` + `completions/*` into `%{_mandir}/man1/` and shell completion directories.
- **Arch Linux (`PKGBUILD`):** `makedepends=('python-build' 'python-installer' 'python-setuptools' 'python-wheel')`, `depends=('python')`, offline `python -m build --wheel --no-isolation`, `python -m installer --destdir="$pkgdir" dist/*.whl`, plus `install -Dm644` for man page and completions.
- **Nix (`buildPythonApplication`):** `pyproject = true; build-system = [ setuptools ]; dependencies = [];` with `installManPage docs/cross-gpu.1` and `installShellCompletion --bash completions/cross-gpu.bash --zsh completions/_cross-gpu --fish completions/cross-gpu.fish`.
- **Homebrew Formula:** Uses `python3` stdlib + `virtualenv_install_with_resources` (zero PyPI resources required), installing `man1.install "docs/cross-gpu.1"`, `bash_completion.install "completions/cross-gpu.bash" => "cross-gpu"`, `zsh_completion.install "completions/_cross-gpu"`, and `fish_completion.install "completions/cross-gpu.fish"`.

---

## 2. Bundled Raw Kernel Package Data

All 90 raw GPU kernel source files under `src/kernels/` are declared as package data in `pyproject.toml` (mapped via `package-dir = {"cross_gpu" = "src"}`) and included in `sdist` via `MANIFEST.in`:

```toml
[tool.setuptools]
packages = ["cross_gpu", "cross_gpu.backends", "cross_gpu.workloads"]
package-dir = {"cross_gpu" = "src"}
include-package-data = true

[tool.setuptools.package-data]
cross_gpu = [
    "kernels/*/*.cu",
    "kernels/*/*.hip",
    "kernels/*/*.cl",
    "kernels/*/*.metal",
]
```

At runtime, `src/embed_binary.py` reads kernels strictly via `importlib.resources.files((__package__ or "src").split(".")[0]).joinpath("kernels", backend, filename)`—working identically across editable checkouts, read-only system installs (`/usr/lib/python3/dist-packages/`), zipapps, virtual environments, and Homebrew kegs.

---

## 3. FHS, Read-Only Safety & Environment Variables

- CrossGPU **never** writes inside the installed Python package tree.
- Runtime-compiled GPU binaries (`.bin`) and metadata sidecars (`.json`) are stored strictly in OS-standard cache directories:
  - **Linux / POSIX:** `$XDG_CACHE_HOME/cross-gpu` (when `$XDG_CACHE_HOME` is an absolute path), defaulting to `~/.cache/cross-gpu`.
  - **macOS:** `~/Library/Caches/cross-gpu` by default, or `$XDG_CACHE_HOME/cross-gpu` when an absolute `XDG_CACHE_HOME` is explicitly set.
  - **Explicit override:** `$CROSS_GPU_CACHE_DIR`.
  - **Disable disk cache:** `CROSS_GPU_NO_CACHE=1` (or `true` / `yes`, case-insensitive) completely bypasses disk cache reads and writes (in-memory caching only) for ephemeral containers, CI, and read-only chroots.
- If the cache directory is read-only or `$HOME` is unset, `RuntimeBinaryCache` degrades automatically to in-memory-only caching without raising errors.
- Native shared libraries use versioned SONAMEs (`libcuda.so.1`, `libnvrtc.so.12`, `libamdhip64.so.6`, `libze_loader.so.1`, `libOpenCL.so.1`) with zero hardcoded Linux paths, so `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` work naturally.

---

## 4. Offline Verification (`sdist` & Installed Package)

All unit tests run 100% offline with zero GPU hardware required:

```sh
python3 -m compileall -q src tests
python3 -m unittest discover -s tests -v
python3 -m src.embed_binary verify
bash -n completions/cross-gpu.bash
```

### Verify Self-Contained `sdist` Archive

```sh
python3 -m build --sdist --no-isolation --outdir /tmp/cg-sdist .
tar -tzf /tmp/cg-sdist/cross_gpu-*.tar.gz
mkdir -p /tmp/cg-extract && tar -xzf /tmp/cg-sdist/cross_gpu-*.tar.gz -C /tmp/cg-extract
(
  cd /tmp/cg-extract/cross_gpu-*
  python3 -m unittest discover -s tests -v
  python3 -m src.embed_binary verify
)
```

### Verify Installed Wheel Outside the Source Tree

```sh
python3 -m venv /tmp/cross-gpu-verify
/tmp/cross-gpu-verify/bin/pip install --no-build-isolation --no-deps .
(
  cd /tmp
  /tmp/cross-gpu-verify/bin/cross-gpu --version
  /tmp/cross-gpu-verify/bin/python -m cross_gpu.embed_binary verify
  /tmp/cross-gpu-verify/bin/python -m cross_gpu.embed_binary list | wc -l  # expects 90
)
rm -rf /tmp/cross-gpu-verify
```
