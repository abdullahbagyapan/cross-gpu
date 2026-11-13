# Build and Validation Status

This document distinguishes automated source/build checks from tests that
require a vendor SDK or physical GPU. A successful mock-backend test does not
prove that a native vendor runtime or a real GPU workload works.

## Portable checks

The following checks passed in the project development environment on
2026-11-13:

```sh
make clean
make -j2
make check
```

The default build compiled the core executable and OpenCL plugin. The test
suite exercised mock plugins and checked discovery, registry preference scores,
exact-vendor versus wildcard precedence, preferred-backend fallback,
cross-backend identity merging, identity namespace isolation, and rejection of
Level Zero on non-Intel devices and unregistered plugin names.

## Platform and vendor validation

| Component | Build / execution status |
|---|---|
| Core C11 application | Linux build and mock tests passed |
| OpenCL plugin | Linux build passed; physical-GPU coverage depends on the host runtime |
| Apple Metal plugin | macOS build/test job is configured; hosted CI and physical Apple GPU validation are still pending |
| Apple Metal/OpenCL identity bridge | User-confirmed working on Apple M2; macOS CI checks compilation, not identity mapping |
| NVIDIA CUDA plugin | Source is included; native SDK build and NVIDIA hardware execution remain unverified here |
| AMD HIP plugin | Source is included; native SDK build and AMD hardware execution remain unverified here |
| Intel Level Zero plugin | Dedicated Ubuntu 24.04 SDK/OCLOC CI build is configured; awaiting its first hosted run; physical Intel GPU execution remains unverified |

The Level Zero plugin additionally requires Level Zero development headers,
`libze_loader`, a compatible Intel GPU driver, OCLOC, and a C++14 compiler.
The pull-request workflow has a dedicated Ubuntu 24.04 job that installs the
`libze-dev` and `intel-ocloc` packages and builds the actual Level Zero plugin.
The mock-backend checks exercise selection and fallback; they do not execute
Level Zero kernels on a physical Intel GPU.

## What the test suite does not prove

- Correctness, thermal behavior, stability, or performance on physical GPUs.
- Compatibility with every version of CUDA, ROCm, Metal, OpenCL, or Level Zero.
- Correctness of platform-specific physical-device identity mapping on every
  driver version.
- Exhaustive VRAM coverage. The current memory workload tests a bounded working
  set and is not a substitute for vendor diagnostics.

Run vendor-specific builds and workloads on representative hardware before
publishing a release that claims support for that backend. Record the OS,
driver, SDK/compiler versions, GPU model, and exact build command when reporting
results.


## Platform adapter coverage

The core platform adapter is split into Linux, Apple macOS, and Windows
implementations, with compiler/OS macro detection centralized in
`include/crossgpu_platform_config.h`. Linux GCC/Clang checks passed in the
current development environment. The macOS and Windows hosted jobs are
configured but have not yet completed on GitHub. Windows targets MinGW-w64
UCRT64 under MSYS2; MSVC is not currently supported. Local Windows syntax was
checked with Win32 interface stubs, not a native Windows compiler. The hosted
Windows job will provide the first native MinGW build and mock-backend check; it
will not constitute physical GPU validation. CUDA, HIP, and Level Zero remain
Linux-only; Metal remains Apple-only.

## Repository security automation

The repository includes a CodeQL workflow for the compiled C/C++ core, OpenCL,
and Intel Level Zero plugins, plus a dependency-review workflow for pull
requests. Both are
configured in `.github/workflows/`. These workflows must complete on GitHub;
local `make check` does not validate hosted workflow execution. Dependency
Review requires the repository dependency graph to be enabled.
