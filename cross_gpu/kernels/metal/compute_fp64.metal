/*
 * cross-gpu kernel: compute_fp64 (metal)
 * Entry point(s): kernel_compute_fp64
 * ABI: (device double* out_buf, constant uint& num_elements, constant uint& iterations, constant double& seed_a, constant double& seed_b)
 * Arch / Extension guard: UNSUPPORTED on Metal Shading Language
 * OPS_PER_THREAD_PER_ITER: 0 (explicitly gated; zero fake emulation)
 */

#error "UNSUPPORTED: Metal Shading Language does not support native fp64 (double) compute"
