/*
 * cross-gpu kernel: compute_fp64 (opencl)
 * Entry point(s): kernel_compute_fp64
 * ABI: (__global double* out_buf, uint num_elements, uint iterations, double seed_a, double seed_b)
 * Arch / Extension guard: cl_khr_fp64
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 2 = 256 (8 accumulators x 8 FMA pairs x 2 FMAs x 2 FLOPs)
 */

#if !defined(cl_khr_fp64) || (defined(CROSS_GPU_HAS_FP64) && !CROSS_GPU_HAS_FP64)
#error "UNSUPPORTED: OpenCL compute_fp64 requires cl_khr_fp64 extension"
#endif
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel_compute_fp64(
    __global double* restrict out_buf,
    const uint num_elements,
    const uint iterations,
    const double seed_a,
    const double seed_b) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= num_elements) {
        return;
    }

    const double a = fma((double)0.25, seed_a - floor(seed_a), (double)0.5);
    const double b_pos = fma((double)0.25, seed_b - floor(seed_b), (double)0.25);
    const double b_neg = (double)-0.5 * b_pos;

    double r0 = 0.125 + 0.0625 * (double)((gid + 0u) & 15u);
    double r1 = 0.125 + 0.0625 * (double)((gid + 1u) & 15u);
    double r2 = 0.125 + 0.0625 * (double)((gid + 2u) & 15u);
    double r3 = 0.125 + 0.0625 * (double)((gid + 3u) & 15u);
    double r4 = 0.125 + 0.0625 * (double)((gid + 4u) & 15u);
    double r5 = 0.125 + 0.0625 * (double)((gid + 5u) & 15u);
    double r6 = 0.125 + 0.0625 * (double)((gid + 6u) & 15u);
    double r7 = 0.125 + 0.0625 * (double)((gid + 7u) & 15u);

    for (uint it = 0u; it < iterations; ++it) {
        #pragma unroll
        for (uint pair = 0u; pair < 8u; ++pair) {
            r0 = fma(r0, a, b_pos);
            r1 = fma(r1, a, b_pos);
            r2 = fma(r2, a, b_pos);
            r3 = fma(r3, a, b_pos);
            r4 = fma(r4, a, b_pos);
            r5 = fma(r5, a, b_pos);
            r6 = fma(r6, a, b_pos);
            r7 = fma(r7, a, b_pos);

            r0 = fma(r0, a, b_neg);
            r1 = fma(r1, a, b_neg);
            r2 = fma(r2, a, b_neg);
            r3 = fma(r3, a, b_neg);
            r4 = fma(r4, a, b_neg);
            r5 = fma(r5, a, b_neg);
            r6 = fma(r6, a, b_neg);
            r7 = fma(r7, a, b_neg);
        }
    }

    const int ok = isfinite(r0) && isfinite(r1) && isfinite(r2) && isfinite(r3) &&
                   isfinite(r4) && isfinite(r5) && isfinite(r6) && isfinite(r7);
    const double sum = (r0 + r1) + (r2 + r3) + (r4 + r5) + (r6 + r7);
    out_buf[gid] = (ok && isfinite(sum)) ? sum : as_double(0x7FF8000000000000UL);
}
