/*
 * cross-gpu kernel: compute_fp8 (hip)
 * Entry point(s): kernel_compute_fp8
 * ABI: (float* out_buf, uint32_t num_elements, uint32_t iterations, float seed_a, float seed_b)
 * Arch / Extension guard: CDNA3 (gfx94x/gfx950) MFMA FP8 or RDNA4 (gfx12xx) WMMA FP8
 * OPS_PER_THREAD_PER_ITER: 8192 on CDNA3 (32 x 256 FLOPs/lane); 4096 on RDNA4 (16 x 256 FLOPs/lane)
 */

#if !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__) && \
    !defined(__gfx950__) && !defined(__gfx1200__) && !defined(__gfx1201__)
#error "UNSUPPORTED: HIP compute_fp8 requires CDNA3 (gfx94x) or RDNA4 (gfx12xx) hardware FP8 instructions"
#endif

typedef unsigned int u32;
typedef long long i64;
typedef float fp32x4_t __attribute__((ext_vector_type(4)));
typedef float fp32x8_t __attribute__((ext_vector_type(8)));
typedef int i32x2_t __attribute__((ext_vector_type(2)));

extern "C" __global__ void kernel_compute_fp8(
    float* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    float seed_a,
    float seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    const float base = 0.5f + 0.25f * (seed_a - floorf(seed_a)) +
                       0.25f * (seed_b - floorf(seed_b)) + 0.0625f * (float)(gid & 15u);

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
    /* CDNA3 FP8 (E4M3FNUZ): 0x40 = +1.0, 0x38 = +0.5, 0xB8 = -0.5. */
    const i64 a_val = 0x4040404040404040LL;
    const i64 b_pos = 0x3838383838383838LL;
    const i64 b_neg = (i64)0xB8B8B8B8B8B8B8B8ULL;

    fp32x4_t c0 = {base, base + 0.125f, base + 0.25f, base + 0.375f};
    fp32x4_t c1 = {base + 0.5f, base + 0.625f, base + 0.75f, base + 0.875f};
    fp32x4_t c2 = {base + 1.0f, base + 1.125f, base + 1.25f, base + 1.375f};
    fp32x4_t c3 = {base + 1.5f, base + 1.625f, base + 1.75f, base + 1.875f};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            c0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_val, b_pos, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_val, b_pos, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_val, b_pos, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_val, b_pos, c3, 0, 0, 0);

            c0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_val, b_neg, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_val, b_neg, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_val, b_neg, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_val, b_neg, c3, 0, 0, 0);
        }
    }

    if (gid < num_elements) {
        const float sum = (c0[0] + c0[1] + c0[2] + c0[3]) + (c1[0] + c1[1] + c1[2] + c1[3]) +
                          (c2[0] + c2[1] + c2[2] + c2[3]) + (c3[0] + c3[1] + c3[2] + c3[3]);
        out_buf[gid] = isfinite(sum) ? sum : __uint_as_float(0x7FC00000u);
    }
#else
    /* RDNA4 (gfx12xx) FP8 E4M3FN: 0x38 = +1.0, 0x30 = +0.5, 0xB0 = -0.5. */
    const i32x2_t a_val = {0x38383838, 0x38383838};
    const i32x2_t b_pos = {0x30303030, 0x30303030};
    const i32x2_t b_neg = {(int)0xB0B0B0B0u, (int)0xB0B0B0B0u};

    fp32x8_t c0 = {base, base + 0.125f, base + 0.25f, base + 0.375f,
                   base + 0.5f, base + 0.625f, base + 0.75f, base + 0.875f};
    fp32x8_t c1 = {base + 1.0f, base + 1.125f, base + 1.25f, base + 1.375f,
                   base + 1.5f, base + 1.625f, base + 1.75f, base + 1.875f};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            c0 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_val, b_pos, c0);
            c1 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_val, b_pos, c1);
            c0 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_val, b_neg, c0);
            c1 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_val, b_neg, c1);
        }
    }

    if (gid < num_elements) {
        const float sum = (c0[0] + c0[1] + c0[2] + c0[3] + c0[4] + c0[5] + c0[6] + c0[7]) +
                          (c1[0] + c1[1] + c1[2] + c1[3] + c1[4] + c1[5] + c1[6] + c1[7]);
        out_buf[gid] = isfinite(sum) ? sum : __uint_as_float(0x7FC00000u);
    }
#endif
}
