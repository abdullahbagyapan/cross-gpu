/*
 * cross-gpu kernel: compute_bf16 (hip)
 * Entry point(s): kernel_compute_bf16
 * ABI: (float* out_buf, uint32_t num_elements, uint32_t iterations, float seed_a, float seed_b)
 * Arch / Extension guard: CDNA2/3/4 or RDNA3/4 native __bf16 ALU / dot2_f32_bf16
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 4 = 512 (8 bf16x2 accs x 16 FMAs x 2 lanes x 2 FLOPs)
 */

#if !defined(__gfx90a__) && !defined(__gfx940__) && !defined(__gfx941__) && \
    !defined(__gfx942__) && !defined(__gfx950__) && !defined(__gfx1100__) && \
    !defined(__gfx1101__) && !defined(__gfx1102__) && !defined(__gfx1103__) && \
    !defined(__gfx1150__) && !defined(__gfx1151__) && !defined(__gfx1200__) && !defined(__gfx1201__)
#error "UNSUPPORTED: HIP compute_bf16 requires CDNA2+ (gfx90a+) or RDNA3+ (gfx1100+) native BF16 support"
#endif

typedef unsigned int u32;
typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));

extern "C" __global__ void kernel_compute_bf16(
    float* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    float seed_a,
    float seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) {
        return;
    }

    const __bf16 a_b = (__bf16)fmaf(0.25f, seed_a - floorf(seed_a), 0.5f);
    const __bf16 bp_b = (__bf16)fmaf(0.25f, seed_b - floorf(seed_b), 0.25f);
    const __bf16 bn_b = (__bf16)(-0.5f * (float)bp_b);

    const bf16x2_t a = {a_b, a_b};
    const bf16x2_t b_pos = {bp_b, bp_b};
    const bf16x2_t b_neg = {bn_b, bn_b};

    bf16x2_t r0 = {(__bf16)(0.125f + 0.0625f * (float)((gid + 0u) & 15u)), (__bf16)(0.1875f + 0.0625f * (float)((gid + 0u) & 15u))};
    bf16x2_t r1 = {(__bf16)(0.125f + 0.0625f * (float)((gid + 1u) & 15u)), (__bf16)(0.1875f + 0.0625f * (float)((gid + 1u) & 15u))};
    bf16x2_t r2 = {(__bf16)(0.125f + 0.0625f * (float)((gid + 2u) & 15u)), (__bf16)(0.1875f + 0.0625f * (float)((gid + 2u) & 15u))};
    bf16x2_t r3 = {(__bf16)(0.125f + 0.0625f * (float)((gid + 3u) & 15u)), (__bf16)(0.1875f + 0.0625f * (float)((gid + 3u) & 15u))};
    bf16x2_t r4 = {(__bf16)(0.125f + 0.0625f * (float)((gid + 4u) & 15u)), (__bf16)(0.1875f + 0.0625f * (float)((gid + 4u) & 15u))};
    bf16x2_t r5 = {(__bf16)(0.125f + 0.0625f * (float)((gid + 5u) & 15u)), (__bf16)(0.1875f + 0.0625f * (float)((gid + 5u) & 15u))};
    bf16x2_t r6 = {(__bf16)(0.125f + 0.0625f * (float)((gid + 6u) & 15u)), (__bf16)(0.1875f + 0.0625f * (float)((gid + 6u) & 15u))};
    bf16x2_t r7 = {(__bf16)(0.125f + 0.0625f * (float)((gid + 7u) & 15u)), (__bf16)(0.1875f + 0.0625f * (float)((gid + 7u) & 15u))};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 8u; ++pair) {
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

    const float s0 = (float)r0[0] + (float)r0[1];
    const float s1 = (float)r1[0] + (float)r1[1];
    const float s2 = (float)r2[0] + (float)r2[1];
    const float s3 = (float)r3[0] + (float)r3[1];
    const float s4 = (float)r4[0] + (float)r4[1];
    const float s5 = (float)r5[0] + (float)r5[1];
    const float s6 = (float)r6[0] + (float)r6[1];
    const float s7 = (float)r7[0] + (float)r7[1];

    const bool ok = isfinite(s0) && isfinite(s1) && isfinite(s2) && isfinite(s3) &&
                    isfinite(s4) && isfinite(s5) && isfinite(s6) && isfinite(s7);
    const float sum = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    out_buf[gid] = (ok && isfinite(sum)) ? sum : __uint_as_float(0x7FC00000u);
}
