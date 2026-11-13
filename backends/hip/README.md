# HIP backend plugin

Build on Linux with AMD ROCm/HIP installed:

```sh
make WITH_HIP=1
```

This produces `build/backends/libcrossgpu_backend_hip.so`. The plugin links to
the HIP runtime, but the CrossGPU core executable does not. A supported ROCm
installation and AMD GPU are required at runtime.

The plugin provides the registered `compute`, `memory`, and `bandwidth`
workloads. It reports PCI bus identity so the host can deduplicate the HIP view
of a GPU against OpenCL and other PCI-aware backends.

Optional compiler controls:

```sh
make WITH_HIP=1 HIPCC=hipcc HIPCCFLAGS='-O2 -std=c++14 --offload-arch=gfx90a'
```

Select an offload architecture supported by the installed ROCm release and the
target GPU. No architecture flag is imposed by default.
