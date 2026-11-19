# Contributing to `cross-gpu`

Thank you for contributing to `cross-gpu`. Keep changes focused, readable, strictly standard-library-only, and backed by offline unit tests.

---

## 1. Hard Constraints

- **Python 3.10+ standard library only:** Do not add third-party Python dependencies to `pyproject.toml` or import external packages in `cross_gpu/` or `tests/`.
- **Zero fake emulation:** Never silently upcast or emulate unsupported GPU data types (`fp8`, `fp4`, `bf16`, `fp64`, `int4`) or tensor MMA instructions with scalar loops. Unsupported backends/architectures must raise `UnsupportedKernelError` or fail compilation with an explicit `#error "UNSUPPORTED: ..."` gate.
- **Explicit `ctypes` bindings:** Declare `argtypes` and `restype` for every native C symbol loaded via `ctypes`.
- **Read-only install safety:** Never write cache or temporary files into the installed `cross_gpu` package tree.

---

## 2. Development & Testing Workflow

Run the standard offline check suite (bytecode compilation, unit tests, and packaged kernel verification):

```sh
make check
```

Or invoke each step individually:

```sh
python3 -m compileall -q cross_gpu tests
python3 -m unittest discover -s tests -v
python3 -m cross_gpu.embed_binary verify
```

### Opt-in Live Hardware Smoke Tests

Unit tests are designed to run 100% offline without requiring a GPU. When testing on a machine with a supported GPU, opt into the end-to-end hardware smoke test suite:

```sh
CROSS_GPU_HW_TESTS=1 python3 -m unittest tests.test_hardware_smoke -v
```

---

## 3. Editing GPU Kernels

Raw kernel source files live directly under `cross_gpu/kernels/`:

- `cross_gpu/kernels/cuda/*.cu`
- `cross_gpu/kernels/hip/*.hip`
- `cross_gpu/kernels/levelzero/*.cl`
- `cross_gpu/kernels/metal/*.metal`
- `cross_gpu/kernels/opencl/*.cl`

Because `cross_gpu/embed_binary.py` loads raw kernel files directly via `importlib.resources`, **edits to kernel source files take effect immediately**—there is no packing or code-generation step.

After modifying or adding a kernel:

1. Keep the header comment (`Entry point(s):` and `OPS_PER_THREAD_PER_ITER:` or `BYTES_PER_ELEMENT:`), parameter ABI, and operation-count constants aligned with `cross_gpu/workloads/`.
2. If adding a new workload or backend file, update `KERNEL_SPECS` (via `KERNEL_WORKLOADS` / `BACKEND_SOURCE_CONFIG`) in `cross_gpu/embed_binary.py` and package-data patterns in `pyproject.toml` and `MANIFEST.in` if a new file extension is introduced.
3. Verify registry integrity, UTF-8 encoding, declared entry-point symbols, and absence of orphan files:

   ```sh
   python3 -m cross_gpu.embed_binary verify
   python3 -m cross_gpu.embed_binary list
   ```

---

## 4. Coding & Commit Standards

- Include `from __future__ import annotations` and full type annotations in Python modules.
- Keep functions short and flat with early returns / guard clauses.
- Use atomic [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat(workloads): ...`, `fix(metal): ...`, `refactor(embed): ...`, `test(embed): ...`, `docs(architecture): ...`).
- Keep every commit green with `make check` passing before opening a pull request.
