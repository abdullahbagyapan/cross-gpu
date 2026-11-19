/*
 * cross-gpu kernel: compute_int4 (levelzero)
 * Entry point(s): kernel_compute_int4
 * ABI: (__global uint* out_buf, uint num_elements, uint iterations, uint seed_a, uint seed_b)
 * Arch / Extension guard: UNSUPPORTED on SPIR-V / Level Zero without native int4 ISA
 * OPS_PER_THREAD_PER_ITER: 0 (explicitly gated; zero fake emulation)
 */

#error "UNSUPPORTED: Level Zero SPIR-V lacks native hardware int4 arithmetic instructions"
