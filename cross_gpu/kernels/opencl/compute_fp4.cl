/*
 * cross-gpu kernel: compute_fp4 (opencl)
 * Entry point(s): kernel_compute_fp4
 * ABI: (__global float* out_buf, uint num_elements, uint iterations, float seed_a, float seed_b)
 * Arch / Extension guard: UNSUPPORTED on standard OpenCL C without vendor fp4 ISA
 * OPS_PER_THREAD_PER_ITER: 0 (explicitly gated; zero fake emulation)
 */

#error "UNSUPPORTED: OpenCL C lacks native hardware fp4 compute instructions"
