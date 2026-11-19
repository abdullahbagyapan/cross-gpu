/*
 * cross-gpu kernel: compute_int4 (metal)
 * Entry point(s): kernel_compute_int4
 * ABI: (device uint* out_buf, constant uint& num_elements, constant uint& iterations, constant uint& seed_a, constant uint& seed_b)
 * Arch / Extension guard: UNSUPPORTED on Metal Shading Language
 * OPS_PER_THREAD_PER_ITER: 0 (explicitly gated; zero fake emulation)
 */

#error "UNSUPPORTED: Metal Shading Language lacks native hardware int4 arithmetic instructions"
