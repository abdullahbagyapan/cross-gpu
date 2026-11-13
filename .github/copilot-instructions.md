# CrossGPU contributor guidance

CrossGPU is a C11 command-line GPU stress tester with independently built,
dynamically loaded backend plugins. Keep the core independent of GPU vendor SDKs
and preserve the versioned backend ABI in `include/crossgpu_backend.h`.

## Engineering expectations

- Keep the core portable across Linux, macOS, and Windows. Platform-specific
  loader, filesystem, threading, and timing code belongs in the existing
  platform adapter files.
- Treat CUDA, HIP, Metal, and Level Zero as optional plugins. A missing SDK or
  runtime must not break a default core/OpenCL build.
- Do not merge devices based on product names or API-local ordinals. Use the
  typed physical-device identity fields and preserve namespace isolation.
- Preserve per-device backend preference and fallback behaviour. Explicit
  backend selection must not silently fall back to another backend.
- Avoid fixed-size registries or hard-coded lists where the registry already
  owns the relevant metadata.
- Keep C sources compatible with C11 and compile cleanly with `-Wall -Wextra
  -Wpedantic`.

## Validation

- Run `make check` after changes to the core, registry, plugin discovery, or
  mock backend behaviour.
- Validate an optional backend only with its vendor SDK and representative
  hardware; mock tests do not prove physical-GPU workload correctness.
- When changing build logic, account for the supported MinGW-w64 UCRT64
  Windows environment and macOS framework linking.

## Scope and safety

- Keep changes focused. Do not add a direct vendor-runtime dependency to the
  core executable.
- Update user-facing documentation when backend availability, platform support,
  or validation claims change.
