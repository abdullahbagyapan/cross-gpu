/*
 * cross-gpu kernel: compute_fp32 (metal)
 * Entry point(s): kernel_compute_fp32
 * ABI: (device float* out_buf, constant uint& num_elements, constant uint& iterations, constant float& seed_a, constant float& seed_b)
 * Arch / Extension guard: all Metal devices (native FP32 FMA)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 2 = 256 (8 accumulators x 8 FMA pairs x 2 FMAs x 2 FLOPs)
 */

#include <metal_stdlib>
using namespace metal;

kernel void kernel_compute_fp32(
    device float* out_buf [[buffer(0)]],
    constant uint& num_elements [[buffer(1)]],
    constant uint& iterations [[buffer(2)]],
    constant float& seed_a [[buffer(3)]],
    constant float& seed_b [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= num_elements) {
        return;
    }

    const float a = fma(0.25f, seed_a - floor(seed_a), 0.5f);
    const float b_pos = fma(0.25f, seed_b - floor(seed_b), 0.25f);
    const float b_neg = -0.5f * b_pos;

    float r0 = 0.125f + 0.0625f * float((gid + 0u) & 15u);
    float r1 = 0.125f + 0.0625f * float((gid + 1u) & 15u);
    float r2 = 0.125f + 0.0625f * float((gid + 2u) & 15u);
    float r3 = 0.125f + 0.0625f * float((gid + 3u) & 15u);
    float r4 = 0.125f + 0.0625f * float((gid + 4u) & 15u);
    float r5 = 0.125f + 0.0625f * float((gid + 5u) & 15u);
    float r6 = 0.125f + 0.0625f * float((gid + 6u) & 15u);
    float r7 = 0.125f + 0.0625f * float((gid + 7u) & 15u);

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

    const bool ok = isfinite(r0) && isfinite(r1) && isfinite(r2) && isfinite(r3) &&
                    isfinite(r4) && isfinite(r5) && isfinite(r6) && isfinite(r7);
    const float sum = (r0 + r1) + (r2 + r3) + (r4 + r5) + (r6 + r7);
    out_buf[gid] = (ok && isfinite(sum)) ? sum : as_type<float>(0x7FC00000u);
}
