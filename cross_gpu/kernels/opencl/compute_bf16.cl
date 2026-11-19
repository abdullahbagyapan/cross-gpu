// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_bf16 (opencl)
 * Entry point(s): kernel_compute_bf16
 * ABI: (__global float* out_buf, uint num_elements, uint iterations, float seed_a, float seed_b)
 * Arch / Extension guard: UNSUPPORTED on standard OpenCL C without native scalar bf16 FMA
 * OPS_PER_THREAD_PER_ITER: 0 (explicitly gated; zero fake emulation)
 */

#error "UNSUPPORTED: OpenCL C lacks native scalar bf16 FMA instructions"
