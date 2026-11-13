# Intel oneAPI Level Zero backend

This plugin is a separate C++ shared library. It uses the Level Zero loader
(`libze_loader`) at runtime and compiles its OpenCL C kernel source to generic
SPIR-V with Intel OCLOC at build time, then embeds the SPIR-V through GNU
`ld -r -b binary`. It does not call the OpenCL runtime and does not require
Python.

## Requirements

- Linux
- Level Zero API 1.10+ development headers
- Level Zero loader library (`libze_loader`)
- Intel GPU runtime/driver implementing Level Zero
- Intel OpenCL Offline Compiler (`ocloc`) for SPIR-V generation
- A C++14 compiler and GNU binutils (`ld`)

The Level Zero loader/header packages are distributed by the
[oneAPI Level Zero project](https://github.com/oneapi-src/level-zero/releases).
The plugin uses `zeInitDrivers()` to request GPU drivers only (rather than the
legacy `zeInit()` / `zeDriverGet()` pair).
Intel's GPU runtime is distributed with the
[Intel Graphics Compute Runtime](https://github.com/intel/compute-runtime).
OCLOC is distributed separately from the DPC++ compiler.

## Build

```sh
make clean
make WITH_LEVEL_ZERO=1
```

For non-standard SDK paths, override `LEVEL_ZERO_CPPFLAGS`, `LEVEL_ZERO_LIBS`,
and `OCLOC`. For example:

```sh
make WITH_LEVEL_ZERO=1 \
  LEVEL_ZERO_CPPFLAGS="-I/opt/level-zero/include" \
  LEVEL_ZERO_LIBS="-L/opt/level-zero/lib -lze_loader" \
  OCLOC=/opt/intel/ocloc/bin/ocloc
```

The generated plugin is `build/backends/libcrossgpu_backend_level_zero.so`.
Select it explicitly with `--backend level-zero`, or use automatic selection
where it is preferred for Intel GPUs. Automatic cross-backend merging with
OpenCL requires both plugins to report the same PCI BDF identity.

## Workloads

- Compute: repeated FP32 FMA kernel with range validation.
- Memory: deterministic device-memory pattern write/verify.
- Bandwidth: device-to-device memory copies.

The first implementation deliberately enumerates Intel GPU root devices only;
CPU and NPU Level Zero devices and tile/sub-device handles are not included.
