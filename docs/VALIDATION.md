# Validation scope

Run the portable test suite with:

```sh
make check
```

This target builds the default executable and enabled default OpenCL plugin,
registry test, and mock backend plugins. It checks registry scoring, plugin
discovery, identity grouping and namespace isolation, vendor restrictions,
backend preference, and device-open fallback with mocks.

It does not validate CUDA, HIP, Metal, OpenCL, or Level Zero workloads on a
physical GPU. It also does not establish thermal behavior, stability,
performance, driver compatibility, or exhaustive VRAM coverage. The memory
workload uses a bounded backend-selected working set and is not a substitute for
vendor diagnostics.

Before claiming support for an optional backend, build it with the relevant SDK
and run it on representative hardware. Record the operating system, GPU model,
driver version, SDK/compiler version, exact build command, selected backend, and
test command with the result. The architecture and build restrictions are in
[ARCHITECTURE.md](ARCHITECTURE.md); backend-specific prerequisites are documented
next to their backend source directories.
