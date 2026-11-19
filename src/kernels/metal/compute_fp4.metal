// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_fp4 (metal)
 * Entry point(s): kernel_compute_fp4
 * ABI: (device float* out_buf, constant uint& num_elements, constant uint& iterations, constant float& seed_a, constant float& seed_b)
 * Arch / Extension guard: UNSUPPORTED on Metal Shading Language
 * OPS_PER_THREAD_PER_ITER: 0 (explicitly gated; zero fake emulation)
 */

#error "UNSUPPORTED: Metal Shading Language lacks native hardware fp4 compute instructions"
