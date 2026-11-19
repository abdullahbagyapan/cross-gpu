/*
 * cross-gpu kernel: compute_fp64 (cuda)
 * Entry point(s): kernel_compute_fp64
 * ABI: (double* out_buf, uint32_t num_elements, uint32_t iterations, double seed_a, double seed_b)
 * Arch / Extension guard: all CUDA architectures with native FP64 FMA
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 2 = 256 (8 accumulators x 8 FMA pairs x 2 FMAs x 2 FLOPs)
 */

typedef unsigned int u32;

extern "C" __global__ void kernel_compute_fp64(
    double* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    double seed_a,
    double seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) {
        return;
    }

    /* Contractive alternating FMA keeps orbits bounded in [0.1, 2.0] forever. */
    const double a = fma(0.25, seed_a - floor(seed_a), 0.5);
    const double b_pos = fma(0.25, seed_b - floor(seed_b), 0.25);
    const double b_neg = -0.5 * b_pos;

    double r0 = 0.125 + 0.0625 * (double)((gid + 0u) & 15u);
    double r1 = 0.125 + 0.0625 * (double)((gid + 1u) & 15u);
    double r2 = 0.125 + 0.0625 * (double)((gid + 2u) & 15u);
    double r3 = 0.125 + 0.0625 * (double)((gid + 3u) & 15u);
    double r4 = 0.125 + 0.0625 * (double)((gid + 4u) & 15u);
    double r5 = 0.125 + 0.0625 * (double)((gid + 5u) & 15u);
    double r6 = 0.125 + 0.0625 * (double)((gid + 6u) & 15u);
    double r7 = 0.125 + 0.0625 * (double)((gid + 7u) & 15u);

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 8u; ++pair) {
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

    const bool ok = isfinite(r0) && isfinite(r1) && isfinite(r2) && isfinite(r3) &&
                    isfinite(r4) && isfinite(r5) && isfinite(r6) && isfinite(r7);
    const double sum = (r0 + r1) + (r2 + r3) + (r4 + r5) + (r6 + r7);
    out_buf[gid] = (ok && isfinite(sum)) ? sum : __longlong_as_double(0x7FF8000000000000ULL);
}
