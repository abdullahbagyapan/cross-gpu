/*
 * cross-gpu kernel: compute_fp32 (cuda)
 * Entry point(s): kernel_compute_fp32
 * ABI: (float* out_buf, uint32_t num_elements, uint32_t iterations, float seed_a, float seed_b)
 * Arch / Extension guard: all CUDA architectures (native FP32 FMA)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 2 = 256 (8 accumulators x 8 FMA pairs x 2 FMAs x 2 FLOPs)
 */

typedef unsigned int u32;

extern "C" __global__ void kernel_compute_fp32(
    float* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    float seed_a,
    float seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) {
        return;
    }

    const float a = fmaf(0.25f, seed_a - floorf(seed_a), 0.5f);
    const float b_pos = fmaf(0.25f, seed_b - floorf(seed_b), 0.25f);
    const float b_neg = -0.5f * b_pos;

    float r0 = 0.125f + 0.0625f * (float)((gid + 0u) & 15u);
    float r1 = 0.125f + 0.0625f * (float)((gid + 1u) & 15u);
    float r2 = 0.125f + 0.0625f * (float)((gid + 2u) & 15u);
    float r3 = 0.125f + 0.0625f * (float)((gid + 3u) & 15u);
    float r4 = 0.125f + 0.0625f * (float)((gid + 4u) & 15u);
    float r5 = 0.125f + 0.0625f * (float)((gid + 5u) & 15u);
    float r6 = 0.125f + 0.0625f * (float)((gid + 6u) & 15u);
    float r7 = 0.125f + 0.0625f * (float)((gid + 7u) & 15u);

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 8u; ++pair) {
            r0 = fmaf(r0, a, b_pos);
            r1 = fmaf(r1, a, b_pos);
            r2 = fmaf(r2, a, b_pos);
            r3 = fmaf(r3, a, b_pos);
            r4 = fmaf(r4, a, b_pos);
            r5 = fmaf(r5, a, b_pos);
            r6 = fmaf(r6, a, b_pos);
            r7 = fmaf(r7, a, b_pos);

            r0 = fmaf(r0, a, b_neg);
            r1 = fmaf(r1, a, b_neg);
            r2 = fmaf(r2, a, b_neg);
            r3 = fmaf(r3, a, b_neg);
            r4 = fmaf(r4, a, b_neg);
            r5 = fmaf(r5, a, b_neg);
            r6 = fmaf(r6, a, b_neg);
            r7 = fmaf(r7, a, b_neg);
        }
    }

    const bool ok = isfinite(r0) && isfinite(r1) && isfinite(r2) && isfinite(r3) &&
                    isfinite(r4) && isfinite(r5) && isfinite(r6) && isfinite(r7);
    const float sum = (r0 + r1) + (r2 + r3) + (r4 + r5) + (r6 + r7);
    out_buf[gid] = (ok && isfinite(sum)) ? sum : __uint_as_float(0x7FC00000u);
}
