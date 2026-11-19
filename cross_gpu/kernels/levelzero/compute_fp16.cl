/*
 * cross-gpu kernel: compute_fp16 (levelzero)
 * Entry point(s): kernel_compute_fp16
 * ABI: (__global float* out_buf, uint num_elements, uint iterations, float seed_a, float seed_b)
 * Arch / Extension guard: cl_khr_fp16
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 4 = 512 (8 half2 accs x 16 FMAs x 2 lanes x 2 FLOPs)
 */

#ifndef cl_khr_fp16
#error "UNSUPPORTED: Level Zero compute_fp16 requires cl_khr_fp16 extension"
#endif
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void kernel_compute_fp16(
    __global float* restrict out_buf,
    const uint num_elements,
    const uint iterations,
    const float seed_a,
    const float seed_b) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= num_elements) {
        return;
    }

    const half a_h = (half)fma(0.25f, seed_a - floor(seed_a), 0.5f);
    const half bp_h = (half)fma(0.25f, seed_b - floor(seed_b), 0.25f);
    const half bn_h = (half)(-0.5f * (float)bp_h);

    const half2 a = (half2)(a_h, a_h);
    const half2 b_pos = (half2)(bp_h, bp_h);
    const half2 b_neg = (half2)(bn_h, bn_h);

    half2 r0 = (half2)((half)(0.125f + 0.0625f * (float)((gid + 0u) & 15u)), (half)(0.1875f + 0.0625f * (float)((gid + 0u) & 15u)));
    half2 r1 = (half2)((half)(0.125f + 0.0625f * (float)((gid + 1u) & 15u)), (half)(0.1875f + 0.0625f * (float)((gid + 1u) & 15u)));
    half2 r2 = (half2)((half)(0.125f + 0.0625f * (float)((gid + 2u) & 15u)), (half)(0.1875f + 0.0625f * (float)((gid + 2u) & 15u)));
    half2 r3 = (half2)((half)(0.125f + 0.0625f * (float)((gid + 3u) & 15u)), (half)(0.1875f + 0.0625f * (float)((gid + 3u) & 15u)));
    half2 r4 = (half2)((half)(0.125f + 0.0625f * (float)((gid + 4u) & 15u)), (half)(0.1875f + 0.0625f * (float)((gid + 4u) & 15u)));
    half2 r5 = (half2)((half)(0.125f + 0.0625f * (float)((gid + 5u) & 15u)), (half)(0.1875f + 0.0625f * (float)((gid + 5u) & 15u)));
    half2 r6 = (half2)((half)(0.125f + 0.0625f * (float)((gid + 6u) & 15u)), (half)(0.1875f + 0.0625f * (float)((gid + 6u) & 15u)));
    half2 r7 = (half2)((half)(0.125f + 0.0625f * (float)((gid + 7u) & 15u)), (half)(0.1875f + 0.0625f * (float)((gid + 7u) & 15u)));

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

    const float s0 = (float)r0.s0 + (float)r0.s1;
    const float s1 = (float)r1.s0 + (float)r1.s1;
    const float s2 = (float)r2.s0 + (float)r2.s1;
    const float s3 = (float)r3.s0 + (float)r3.s1;
    const float s4 = (float)r4.s0 + (float)r4.s1;
    const float s5 = (float)r5.s0 + (float)r5.s1;
    const float s6 = (float)r6.s0 + (float)r6.s1;
    const float s7 = (float)r7.s0 + (float)r7.s1;

    const int ok = isfinite(s0) && isfinite(s1) && isfinite(s2) && isfinite(s3) &&
                   isfinite(s4) && isfinite(s5) && isfinite(s6) && isfinite(s7);
    const float sum = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    out_buf[gid] = (ok && isfinite(sum)) ? sum : as_float(0x7FC00000u);
}
