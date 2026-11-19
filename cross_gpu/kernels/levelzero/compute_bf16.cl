// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_bf16 (levelzero)
 * Entry point(s): kernel_compute_bf16
 * ABI: (__global float* out_buf, uint num_elements, uint iterations, float seed_a, float seed_b)
 * Arch / Extension guard: UNSUPPORTED on SPIR-V / Level Zero without native scalar bf16 FMA
 * OPS_PER_THREAD_PER_ITER: 0 (explicitly gated; zero fake emulation; see compute_tensor_bf16 for XMX)
 */

#error "UNSUPPORTED: Level Zero SPIR-V lacks native scalar bf16 FMA instructions"
