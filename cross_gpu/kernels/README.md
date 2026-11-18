# Kernel Source Placeholders

This directory contains per-backend kernel subdirectories (`cuda/`, `hip/`, `levelzero/`, `metal/`, `opencl/`).
Per the core framework boundary, no GPU kernel source files are included here yet.

When kernels are added, place workload sources or precompiled binaries under:
- `cuda/<workload>.cu` (or `.ptx` / `.cubin`)
- `hip/<workload>.hip` (or `.hsaco`)
- `levelzero/<workload>.cl` (or `.spv`)
- `metal/<workload>.metal` (or `.metallib`)
- `opencl/<workload>.cl`

Then run `python3 -m cross_gpu.embed_binary pack` to update the embedded payload registry.
