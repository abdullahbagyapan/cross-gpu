/*
 * cross-gpu kernel: compute_bf16 (metal)
 * Entry point(s): kernel_compute_bf16
 * ABI: (device float* out_buf, constant uint& num_elements, constant uint& iterations, constant float& seed_a, constant float& seed_b)
 * Arch / Extension guard: __METAL_VERSION__ >= 310 (native MSL bfloat2)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 4 = 512 (8 bfloat2 accs x 16 mul-adds x 2 lanes x 2 FLOPs)
 */

#include <metal_stdlib>
using namespace metal;

#if !defined(__METAL_VERSION__) || (__METAL_VERSION__ < 310)
#error "UNSUPPORTED: Metal compute_bf16 requires Metal 3.1+ (__METAL_VERSION__ >= 310) for native bfloat"
#endif

kernel void kernel_compute_bf16(
    device float* out_buf [[buffer(0)]],
    constant uint& num_elements [[buffer(1)]],
    constant uint& iterations [[buffer(2)]],
    constant float& seed_a [[buffer(3)]],
    constant float& seed_b [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= num_elements) {
        return;
    }

    const bfloat a_b = bfloat(fma(0.25f, seed_a - floor(seed_a), 0.5f));
    const bfloat bp_b = bfloat(fma(0.25f, seed_b - floor(seed_b), 0.25f));
    const bfloat bn_b = bfloat(-0.5f * float(bp_b));

    const bfloat2 a = bfloat2(a_b, a_b);
    const bfloat2 b_pos = bfloat2(bp_b, bp_b);
    const bfloat2 b_neg = bfloat2(bn_b, bn_b);

    bfloat2 r0 = bfloat2(bfloat(0.125f + 0.0625f * float((gid + 0u) & 15u)), bfloat(0.1875f + 0.0625f * float((gid + 0u) & 15u)));
    bfloat2 r1 = bfloat2(bfloat(0.125f + 0.0625f * float((gid + 1u) & 15u)), bfloat(0.1875f + 0.0625f * float((gid + 1u) & 15u)));
    bfloat2 r2 = bfloat2(bfloat(0.125f + 0.0625f * float((gid + 2u) & 15u)), bfloat(0.1875f + 0.0625f * float((gid + 2u) & 15u)));
    bfloat2 r3 = bfloat2(bfloat(0.125f + 0.0625f * float((gid + 3u) & 15u)), bfloat(0.1875f + 0.0625f * float((gid + 3u) & 15u)));
    bfloat2 r4 = bfloat2(bfloat(0.125f + 0.0625f * float((gid + 4u) & 15u)), bfloat(0.1875f + 0.0625f * float((gid + 4u) & 15u)));
    bfloat2 r5 = bfloat2(bfloat(0.125f + 0.0625f * float((gid + 5u) & 15u)), bfloat(0.1875f + 0.0625f * float((gid + 5u) & 15u)));
    bfloat2 r6 = bfloat2(bfloat(0.125f + 0.0625f * float((gid + 6u) & 15u)), bfloat(0.1875f + 0.0625f * float((gid + 6u) & 15u)));
    bfloat2 r7 = bfloat2(bfloat(0.125f + 0.0625f * float((gid + 7u) & 15u)), bfloat(0.1875f + 0.0625f * float((gid + 7u) & 15u)));

    /* MSL does not overload fma() for bfloat; native bfloat2 mul-add lowers directly to bfloat ops. */
    for (uint it = 0u; it < iterations; ++it) {
        #pragma unroll
        for (uint pair = 0u; pair < 8u; ++pair) {
            r0 = r0 * a + b_pos;
            r1 = r1 * a + b_pos;
            r2 = r2 * a + b_pos;
            r3 = r3 * a + b_pos;
            r4 = r4 * a + b_pos;
            r5 = r5 * a + b_pos;
            r6 = r6 * a + b_pos;
            r7 = r7 * a + b_pos;

            r0 = r0 * a + b_neg;
            r1 = r1 * a + b_neg;
            r2 = r2 * a + b_neg;
            r3 = r3 * a + b_neg;
            r4 = r4 * a + b_neg;
            r5 = r5 * a + b_neg;
            r6 = r6 * a + b_neg;
            r7 = r7 * a + b_neg;
        }
    }

    const float s0 = float(r0.x) + float(r0.y);
    const float s1 = float(r1.x) + float(r1.y);
    const float s2 = float(r2.x) + float(r2.y);
    const float s3 = float(r3.x) + float(r3.y);
    const float s4 = float(r4.x) + float(r4.y);
    const float s5 = float(r5.x) + float(r5.y);
    const float s6 = float(r6.x) + float(r6.y);
    const float s7 = float(r7.x) + float(r7.y);

    const bool ok = isfinite(s0) && isfinite(s1) && isfinite(s2) && isfinite(s3) &&
                    isfinite(s4) && isfinite(s5) && isfinite(s6) && isfinite(s7);
    const float sum = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    out_buf[gid] = (ok && isfinite(sum)) ? sum : as_type<float>(0x7FC00000u);
}
