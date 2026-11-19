/*
 * cross-gpu kernel: compute_tensor_int8 (metal)
 * Entry point(s): kernel_compute_tensor_int8
 * ABI: (device uint* out_buf, constant uint& num_elements, constant uint& iterations, constant uint& seed_a, constant uint& seed_b)
 * Arch / Extension guard: UNSUPPORTED on Metal simdgroup_matrix
 * OPS_PER_THREAD_PER_ITER: 0 (explicitly gated; zero fake emulation)
 */

#error "UNSUPPORTED: Metal simdgroup_matrix does not support native int8 matrix multiply-accumulate"
