/*
 * cross-gpu kernel: compute_fp4 (levelzero)
 * Entry point(s): kernel_compute_fp4
 * ABI: (__global float* out_buf, uint num_elements, uint iterations, float seed_a, float seed_b)
 * Arch / Extension guard: UNSUPPORTED on SPIR-V / Level Zero without native fp4 ISA
 * OPS_PER_THREAD_PER_ITER: 0 (explicitly gated; zero fake emulation)
 */

#error "UNSUPPORTED: Level Zero SPIR-V lacks native hardware fp4 compute instructions"
