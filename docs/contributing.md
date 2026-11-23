# Contributing to CrossGPU

Contributions are welcome. Every change must remain Python 3.10+ standard-library-only, readable, SPDX-tagged, and verified by offline unit tests.

---

## 1. Core Engineering Rules

- **Zero external Python dependencies:** Use only the Python 3.10+ standard library in `src/` and `tests/`. Do not add runtime packages to `pyproject.toml`.
- **Zero fake emulation:** Never upcast or emulate unsupported datatypes (`fp8`, `fp4`, `bf16`, `fp64`, `int4`) or matrix/tensor MMA instructions using scalar loops. Unsupported hardware paths must fail capability checks (`UnsupportedKernelError`) or emit `#error "UNSUPPORTED: <reason>"`.
- **Explicit `ctypes` ABI prototypes:** Every C symbol loaded via `ctypes` must declare explicit `argtypes` and `restype` via `bind_c_symbol` / `bind_optional_c_symbol` (or `_bind_objc_msg` for Metal).
- **Readable, flat code:** Keep functions concise and flat using guard clauses / early returns, include `from __future__ import annotations` and full type annotations, remove dead code, and include `SPDX-License-Identifier: MIT` headers on all source files (`# SPDX-License-Identifier: MIT` for Python/shell/Makefile; `// SPDX-License-Identifier: MIT` for GPU kernels; keep `#compdef cross-gpu` on line 1 of `completions/_cross-gpu` and `#!/usr/bin/env python3` on line 1 of `cross-gpu`).

---

## 2. Authoring & Verifying GPU Kernels

Raw kernel source files live under `src/kernels/` and are the single source of truth:

- `src/kernels/cuda/*.cu` (NVRTC CUDA C++)
- `src/kernels/hip/*.hip` (hipRTC HIP C++)
- `src/kernels/levelzero/*.cl` (OpenCL C targeting SPIR-V)
- `src/kernels/metal/*.metal` (Metal Shading Language)
- `src/kernels/opencl/*.cl` (OpenCL C 1.2/3.0)

Because `src/embed_binary.py` loads these raw files directly via `importlib.resources`, **edits take effect immediately**—there is no binary packing or code-generation step.

When editing or adding a kernel:

1. Start line 1 with `// SPDX-License-Identifier: MIT`, followed by the standard header comment documenting `Entry point(s):` and `OPS_PER_THREAD_PER_ITER:` (or `BYTES_PER_ELEMENT:`).
2. Keep kernel parameter ABIs and operation counts strictly synchronized with `src/workloads/` (`common.py`, `compute_scalar.py`, `compute_tensor.py`, `memory.py`).
3. If introducing a new workload or backend extension, update `KERNEL_WORKLOADS` / `BACKEND_SOURCE_CONFIG` in `src/embed_binary.py`, `WORKLOAD_SPECS` in `src/workloads/registry.py`, `completions/*`, and `[tool.setuptools.package-data]` in `pyproject.toml`.
4. Verify registry integrity, strict UTF-8 encoding, declared entry-point symbols, and absence of orphan files:

   ```sh
   python3 -m src.embed_binary verify
   python3 -m src.embed_binary list
   ```

---

## 3. Running Offline & Hardware Tests

Run the full offline verification suite before committing:

```sh
make check
bash -n completions/cross-gpu.bash
```

Which runs:

```sh
python3 -m compileall -q src tests
python3 -m unittest discover -s tests -v
python3 -m src.embed_binary verify
```

On machines with a real GPU, you can additionally run the opt-in end-to-end hardware smoke test:

```sh
CROSS_GPU_HW_TESTS=1 python3 -m unittest tests.test_hardware_smoke -v
```

To exercise all 5 GPU backends, 5 CPU architectures (including big-endian `s390x`), 4 Linux distros, and locked-down containers via Docker, see [Testing & Docker Emulation](testing.md):

```sh
./scripts/emulate.sh --list
./scripts/emulate.sh --suite backends
```

---

## 4. Atomic Commits & Conventional Commits

Keep commits small, logical, and independently green using [Conventional Commits](https://www.conventionalcommits.org/) (`<type>(<scope>): <description>`):

| Commit Pattern | Repository Example |
|---|---|
| `feat(<scope>): ...` | `feat(cli): add millisecond (ms) duration parsing and sub-second progress formatting` |
| `fix(<scope>): ...` | `fix(metal): add threadgroup_barrier to prevent tensor MMA WAR race` |
| `refactor(<scope>): ...` | `refactor(embed): load raw kernel files via importlib.resources without blobs or sha256` |
| `build(<scope>): ...` | `build(packaging): ensure complete sdist contents and verify archive in CI` |
| `ci(<scope>): ...` | `ci(github): verify packaged kernel loading from outside source tree` |
| `test(<scope>): ...` | `test(cli): cover ms durations, banner color guards, and single-document JSON output` |
| `docs(<scope>): ...` | `docs(reference): document backends, 20 workloads, and CLI JSON schemas` |
| `chore(<scope>): ...` | `chore(license): add SPDX headers across source tree and create CHANGELOG.md` |
