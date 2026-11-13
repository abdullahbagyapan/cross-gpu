# Contributing to CrossGPU

Thanks for your interest in improving CrossGPU. Contributions should preserve
its small C11 core, independently loaded backend plugins, and vendor-neutral
plugin manager.

## Before opening a pull request

- Search existing issues and pull requests for related work.
- For a substantial feature or ABI change, open an issue first to agree on the
  design and compatibility impact.
- Keep changes focused and document user-visible behavior in `README.md` or
  the relevant document under `docs/`.

## Development requirements

The core and OpenCL plugin can be built with a C11 compiler and GNU Make.
Optional backends have their own platform SDK and compiler requirements; see
`README.md` and the README in the relevant backend directory.

```sh
make clean
make -j2
make check
```

Do not commit generated output from `build/`. The Level Zero SPIR-V binary is
converted into a relocatable object by the platform linker during the build;
the generated `.spv` and `.o` files should not be edited or committed.

## Code and architecture expectations

- Keep the core independent of vendor GPU SDKs and runtimes.
- Keep each backend in its own plugin and use the public ABI in
  `include/crossgpu_backend.h`.
- Treat ABI changes as compatibility changes. Update the ABI version and
  plugin documentation whenever the ABI layout or contract changes.
- Do not merge physical devices based on product names, array positions, or
  backend-local indices. Cross-backend merging requires the same trustworthy
  identity kind, namespace, and value.
- Do not claim a shared identity when the platform API cannot establish one.
- Keep hardware-specific selection restrictions and fallback behavior covered
  by mock-backend tests.
- Use checked allocation, bounded string handling, and clear error reporting.
- Follow the existing C11/C++14 conventions and avoid adding runtime
  dependencies to the core without discussion.

## Pull request checklist

- [ ] `make clean && make -j2` succeeds on the platform tested.
- [ ] `make check` passes.
- [ ] Optional backend builds are tested with the real vendor SDK when changed.
- [ ] Physical-device behavior is tested on supported hardware when possible;
      otherwise the untested parts are stated clearly.
- [ ] Documentation and build instructions are updated.
- [ ] No generated binaries, credentials, local build output, or unrelated files
      are included.
