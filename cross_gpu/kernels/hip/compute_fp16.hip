/*
 * cross-gpu kernel: compute_fp16 (hip)
 * Entry point(s): kernel_compute_fp16
 * ABI: (float* out_buf, uint32_t num_elements, uint32_t iterations, float seed_a, float seed_b)
 * Arch / Extension guard: gfx900+ / CDNA / RDNA (native _Float16 packed v_pk_fma_f16)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 4 = 512 (8 half2 accs x 16 FMAs x 2 lanes x 2 FLOPs)
 */

#if defined(__gfx803__)
#error "UNSUPPORTED: HIP compute_fp16 requires gfx900+ for native packed FP16 FMA"
#endif

typedef unsigned int u32;
typedef _Float16 fp16x2_t __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ fp16x2_t fma_fp16x2(fp16x2_t r, fp16x2_t a, fp16x2_t b) {
    return fp16x2_t{__builtin_fmaf16(r[0], a[0], b[0]), __builtin_fmaf16(r[1], a[1], b[1])};
}

extern "C" __global__ void kernel_compute_fp16(
    float* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    float seed_a,
    float seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) {
        return;
    }

    const _Float16 a_h = (_Float16)fmaf(0.25f, seed_a - floorf(seed_a), 0.5f);
    const _Float16 bp_h = (_Float16)fmaf(0.25f, seed_b - floorf(seed_b), 0.25f);
    const _Float16 bn_h = (_Float16)(-0.5f * (float)bp_h);

    const fp16x2_t a = {a_h, a_h};
    const fp16x2_t b_pos = {bp_h, bp_h};
    const fp16x2_t b_neg = {bn_h, bn_h};

    fp16x2_t r0 = {(_Float16)(0.125f + 0.0625f * (float)((gid + 0u) & 15u)), (_Float16)(0.1875f + 0.0625f * (float)((gid + 0u) & 15u))};
    fp16x2_t r1 = {(_Float16)(0.125f + 0.0625f * (float)((gid + 1u) & 15u)), (_Float16)(0.1875f + 0.0625f * (float)((gid + 1u) & 15u))};
    fp16x2_t r2 = {(_Float16)(0.125f + 0.0625f * (float)((gid + 2u) & 15u)), (_Float16)(0.1875f + 0.0625f * (float)((gid + 2u) & 15u))};
    fp16x2_t r3 = {(_Float16)(0.125f + 0.0625f * (float)((gid + 3u) & 15u)), (_Float16)(0.1875f + 0.0625f * (float)((gid + 3u) & 15u))};
    fp16x2_t r4 = {(_Float16)(0.125f + 0.0625f * (float)((gid + 4u) & 15u)), (_Float16)(0.1875f + 0.0625f * (float)((gid + 4u) & 15u))};
    fp16x2_t r5 = {(_Float16)(0.125f + 0.0625f * (float)((gid + 5u) & 15u)), (_Float16)(0.1875f + 0.0625f * (float)((gid + 5u) & 15u))};
    fp16x2_t r6 = {(_Float16)(0.125f + 0.0625f * (float)((gid + 6u) & 15u)), (_Float16)(0.1875f + 0.0625f * (float)((gid + 6u) & 15u))};
    fp16x2_t r7 = {(_Float16)(0.125f + 0.0625f * (float)((gid + 7u) & 15u)), (_Float16)(0.1875f + 0.0625f * (float)((gid + 7u) & 15u))};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 8u; ++pair) {
            r0 = fma_fp16x2(r0, a, b_pos);
            r1 = fma_fp16x2(r1, a, b_pos);
            r2 = fma_fp16x2(r2, a, b_pos);
            r3 = fma_fp16x2(r3, a, b_pos);
            r4 = fma_fp16x2(r4, a, b_pos);
            r5 = fma_fp16x2(r5, a, b_pos);
            r6 = fma_fp16x2(r6, a, b_pos);
            r7 = fma_fp16x2(r7, a, b_pos);

            r0 = fma_fp16x2(r0, a, b_neg);
            r1 = fma_fp16x2(r1, a, b_neg);
            r2 = fma_fp16x2(r2, a, b_neg);
            r3 = fma_fp16x2(r3, a, b_neg);
            r4 = fma_fp16x2(r4, a, b_neg);
            r5 = fma_fp16x2(r5, a, b_neg);
            r6 = fma_fp16x2(r6, a, b_neg);
            r7 = fma_fp16x2(r7, a, b_neg);
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
