# Native GPU Stress Kernels

Self-contained kernel source files for all 5 backends (`18` compilation units per backend, `90` total):

- `cuda/*.cu` — Standalone NVRTC CUDA kernels (scalar INT/FP, inline PTX `dp4a`/`f16x2`/`bf16x2`/`mma.sync.aligned`, memory/bandwidth).
- `hip/*.hip` — Standalone hipRTC HIP kernels (scalar INT/FP, AMDGCN `udot4`/`sdot8`/`mfma`/`wmma`, memory/bandwidth).
- `levelzero/*.cl` — OpenCL C kernels targeting SPIR-V (`ocloc` / `clang` + `llvm-spirv`) with Intel XMX DPAS guards.
- `metal/*.metal` — Metal Shading Language kernels with `<metal_simdgroup_matrix>` and explicit unsupported-type `#error` gates.
- `opencl/*.cl` — Portable OpenCL C 1.2/3.0 kernels with extension guards (`cl_khr_fp64`, `cl_khr_fp16`, Intel XMX).

(`bandwidth-h2d` and `bandwidth-d2h` use backend host-device transfer APIs directly.)

Re-pack embedded payloads and SHA-256 digests after editing kernel sources:

```sh
python3 -m cross_gpu.embed_binary pack
python3 -m cross_gpu.embed_binary verify
```
