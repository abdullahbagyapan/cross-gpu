<p align="left">
  <img src="./assets/image.png" width="280" alt="CrossGPU logo">
</p>

# CrossGPU

CrossGPU is a C11 command-line GPU stress tester. Its core executable discovers
versioned backend plugins at runtime, so it has no link-time dependency on GPU
vendor runtimes.

## Supported backends

| Backend | Build switch | Host support |
|---|---|---|
| OpenCL | `WITH_OPENCL=1` (default) | Linux, macOS, Windows |
| CUDA | `WITH_CUDA=1` | Linux, Windows |
| HIP | `WITH_HIP=1` | Linux, Windows |
| Metal | `WITH_METAL=1` | macOS |
| Intel Level Zero | `WITH_LEVEL_ZERO=1` | Linux, Windows |

The optional backends require their respective compiler, SDK, runtime, and
compatible GPU. OpenCL uses the project's minimal declarations and loads the
system OpenCL runtime at runtime; OpenCL development headers are not required.
See the backend-specific build notes in `backends/` for CUDA, HIP, Metal, and
Level Zero requirements.

## Build and test

The default build produces the executable and OpenCL plugin:

```sh
make
make check
```

Build optional plugins only on their supported host platform:

```sh
# Linux
make WITH_CUDA=1
make WITH_HIP=1
make WITH_LEVEL_ZERO=1
make WITH_OPENCL=1 WITH_CUDA=1 WITH_HIP=1 WITH_LEVEL_ZERO=1

# macOS
make WITH_METAL=1
make WITH_OPENCL=1 WITH_METAL=1
```

On Windows, build the available vendor plugins with the same switches:

```sh
make WITH_CUDA=1
make WITH_HIP=1
make WITH_LEVEL_ZERO=1
```

CUDA on Windows requires a CUDA Toolkit and its supported MSVC host compiler;
HIP requires AMD's HIP SDK; Level Zero requires the Level Zero headers, loader,
and OCLOC. The Windows Level Zero linker input is `ze_loader.lib` by default;
override `LEVEL_ZERO_LIBS` if the SDK uses a different path or name.
Native Windows vendor-SDK and hardware validation is still required before
relying on these plugins in production.

To omit the default OpenCL plugin, use `make WITH_OPENCL=0`. Run `make clean`
before changing the enabled backend set to prevent old plugins remaining in the
build directory. `make print-config` shows the resolved compiler, platform, and
backend configuration.

Install the executable and enabled plugins with `make install`; set `PREFIX` to
choose an installation prefix. `make uninstall` removes the installed executable
and the platform-native plugin filenames. `make check` runs registry and
mock-plugin tests; it does not exercise physical GPUs. See
[validation scope](docs/VALIDATION.md).

## CLI

```text
cross-gpu list
cross-gpu info
cross-gpu run [--device all|N] [--test compute|memory|bandwidth|all]
              [--duration TIME] [--backend auto|PLUGIN_NAME]
```

Examples:

```sh
cross-gpu list
cross-gpu info
cross-gpu run --device 0 --test compute --duration 60s
cross-gpu run --device all --test all --duration 5m
cross-gpu run --backend opencl --test memory --duration 300ms
```

`list` and `info` take no options. For `run`, `--device` defaults to `all`,
`--test` defaults to `all`, `--duration` defaults to `60s`, and `--backend`
defaults to `auto`. Duration accepts a positive number with no suffix or with
`ms`, `s`, `m`, or `h`; the accepted range is 1 ms through seven days. When more
than one test is selected, the total duration is divided evenly between tests.

`--backend` matches a loaded plugin's runtime name. A backend must also be
approved for the detected vendor by the compiled preference registry, so an
arbitrary third-party plugin name is not automatically usable. See
[architecture](docs/ARCHITECTURE.md) and [the backend ABI](docs/BACKEND_ABI.md).

## Plugin paths

Plugins use these platform-native filenames:

| Platform | Filename |
|---|---|
| Linux | `libcrossgpu_backend_NAME.so` |
| macOS | `libcrossgpu_backend_NAME.dylib` |
| Windows | `crossgpu_backend_NAME.dll` |

The loader searches, in order:

1. Directories in `CROSSGPU_PLUGIN_PATH` (`:` separated on Linux/macOS; `;` on Windows).
2. A `backends/` directory beside the executable.
3. `./backends`.
4. The plugin directory compiled from `PREFIX`, normally `/usr/local/lib/cross-gpu/backends` on Linux/macOS or `C:/CrossGPU/lib/cross-gpu/backends` on Windows.
5. `/usr/lib/cross-gpu/backends` on Linux only.

Set `CROSSGPU_PLUGIN_PATH_ONLY=1` to search only directories in
`CROSSGPU_PLUGIN_PATH`. This is useful for isolated testing. For example:

```sh
CROSSGPU_PLUGIN_PATH="$PWD/build/backends" ./build/cross-gpu list
```

## License

CrossGPU is licensed under the GNU General Public License, version 3.0. See
[LICENSE](LICENSE).
