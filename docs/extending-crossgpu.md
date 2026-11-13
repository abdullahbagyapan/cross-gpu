# Extending CrossGPU

CrossGPU centralizes core metadata so common additions do not require changes to
CLI parsing, help generation, display names, or fixed-size arrays.

## Add a vendor

1. Add the PCI vendor ID as a named value in `crossgpu_vendor_id_t` in
   `include/crossgpu_backend.h`.
2. Add one row to `vendor_descriptors` in `src/registry.c`.
3. Add preference rows to `backend_preferences` in `src/registry.c`, keyed by
   that vendor enum, expected backend kind, and plugin runtime name (for
   example, `"vulkan"`).
4. Have the backend report the matching enum value in `device_info.vendor_id`.

The preference table is the only source of backend-selection policy. A row
matches the vendor, expected backend kind, and runtime plugin name. Exact
vendor rows override wildcard rows; a missing match rejects the plugin for
that vendor. This prevents an unrelated plugin from borrowing OpenCL's
wildcard compatibility permission by reusing its name. A wildcard `opencl` row
supplies generic compatibility for vendors without an explicit preference. Do
not add vendor or backend-kind conditions
to `candidate_score()` or any other discovery code. A custom backend can be assigned a higher score than existing backends by
adding a preference row for its runtime plugin name and `CROSSGPU_BACKEND_CUSTOM`
kind.

## Add a backend plugin

1. Implement the callbacks in `crossgpu_backend_plugin_t`.
2. Export `crossgpu_backend_get_api()`.
3. Return ABI version 1 and the full plugin struct size.
4. Implement the two-pass enumeration contract.
5. Use a unique plugin name and filename:
   `libcrossgpu_backend_NAME.so` on Linux, `libcrossgpu_backend_NAME.dylib`
   on macOS, or `crossgpu_backend_NAME.dll` on Windows.
6. Place it in a discovered `backends/` directory or set
   `CROSSGPU_PLUGIN_PATH`.

The core loader accepts unrecognized nonzero backend kinds. `--backend NAME`
filters by the plugin's runtime `name`, so adding a plugin-specific API does not
require a new CLI parser branch. To make it the automatic preferred backend for
a vendor, add a vendor/backend-kind/runtime-name preference row.

## Add a workload/test operation

1. Add a named `CROSSGPU_TEST_*` enum value to `crossgpu_test_type_t` in
   `include/crossgpu_backend.h`.
2. Add one descriptor row to `test_descriptors` in `src/registry.c`.
   - Set `CROSSGPU_TEST_FLAG_DEFAULT_ALL` to include it in `--test all`.
   - Set a rate flag if its metrics should be displayed as compute or bandwidth.
   - Use `CROSSGPU_TEST_FLAG_NONE` when it should not be part of the default suite.
3. Implement the operation in every backend that claims to support it.
4. For OpenCL, add its preparation and step callbacks and one entry to
   `opencl_test_operations` in `backends/opencl/opencl_backend.c`.
5. Add mock coverage and verify duration splitting and result reporting.

The CLI test parser, help output, `all` expansion, test names, and rate units are
read from the test registry. There is no `TEST_COUNT` or positional test array
to keep in sync.

## Add a new result metric

Extend `crossgpu_metrics_t` only with an ABI version change. Define the unit and
aggregation semantics, update backend implementations, and update terminal
reporting. Prefer explicit metric fields over overloading the existing
`operations` or byte counters.

## Build the vendor plugins

Each backend is a separate shared library and is discovered by the existing
plugin loader. The core executable does not acquire a link-time dependency on
any GPU runtime.

```sh
# Linux
make WITH_OPENCL=1 WITH_CUDA=1 WITH_HIP=1 WITH_LEVEL_ZERO=1

# macOS
make WITH_OPENCL=1 WITH_METAL=1
```

CUDA requires the NVIDIA CUDA Toolkit (`nvcc`). HIP requires AMD ROCm (`hipcc`).
Metal requires macOS, Xcode Command Line Tools, and the Metal SDK. Intel Level
Zero requires the Level Zero development headers/loader and Intel OCLOC (`ocloc`)
to compile the plugin kernels to SPIR-V. The Makefile
installs every enabled plugin into `$(PREFIX)/lib/cross-gpu/backends`.

The CUDA and HIP source files include the shared implementation in
`backends/common/runtime_backend_impl.hpp`, but each is compiled and loaded as
a distinct plugin with its own runtime dependency and plugin descriptor. Metal
is an independent Objective-C plugin in `backends/metal/metal_backend.m`. Intel
Level Zero is an independent C++ plugin in
`backends/levelzero/levelzero_backend.cpp`.

Do not assume that a plugin can be built on a machine without its SDK. A core
only build remains available with `make WITH_OPENCL=0`; OpenCL can also be
independently disabled with `WITH_OPENCL=0`.
