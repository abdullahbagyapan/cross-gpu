# CUDA backend plugin

Build on Linux with the NVIDIA CUDA Toolkit installed:

```sh
make WITH_CUDA=1
```

This produces `build/backends/libcrossgpu_backend_cuda.so`. The plugin links to
CUDA Runtime, but the CrossGPU core executable does not. At runtime, the NVIDIA
driver and a CUDA-compatible GPU are required.

The plugin provides the registered `compute`, `memory`, and `bandwidth`
workloads. It reports PCI bus identity so the host can deduplicate the CUDA
view of a GPU against OpenCL and other PCI-aware backends.

Optional compiler controls:

```sh
make WITH_CUDA=1 NVCC=nvcc NVCCFLAGS='-O2 -std=c++14 -arch=sm_80'
```

Choose an architecture supported by the installed CUDA Toolkit and deployment
GPUs. No single architecture flag is imposed by default.
