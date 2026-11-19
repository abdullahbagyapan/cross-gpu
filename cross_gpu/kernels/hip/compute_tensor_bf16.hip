/*
 * cross-gpu kernel: compute_tensor_bf16 (hip)
 * Entry point(s): kernel_compute_tensor_bf16
 * ABI: (float* out_buf, uint32_t num_elements, uint32_t iterations, float seed_a, float seed_b)
 * Arch / Extension guard: CDNA (mfma_f32_16x16x16bf16_1k / 8bf16) or RDNA3/4 (wmma_f32_16x16x16_bf16)
 * OPS_PER_THREAD_PER_ITER: 4096 on CDNA2+/RDNA3/4; 2048 on CDNA1 (gfx908)
 */

#if !defined(__gfx908__) && !defined(__gfx90a__) && !defined(__gfx940__) && \
    !defined(__gfx941__) && !defined(__gfx942__) && !defined(__gfx950__) && \
    !defined(__gfx1100__) && !defined(__gfx1101__) && !defined(__gfx1102__) && \
    !defined(__gfx1103__) && !defined(__gfx1150__) && !defined(__gfx1151__) && \
    !defined(__gfx1200__) && !defined(__gfx1201__)
#error "UNSUPPORTED: HIP compute_tensor_bf16 requires CDNA (gfx908+) or RDNA3+ (gfx1100+) BF16 matrix cores"
#endif

typedef unsigned int u32;
typedef float fp32x4_t __attribute__((ext_vector_type(4)));
typedef float fp32x8_t __attribute__((ext_vector_type(8)));
typedef short i16x2_t __attribute__((ext_vector_type(2)));
typedef short i16x4_t __attribute__((ext_vector_type(4)));
typedef short i16x8_t __attribute__((ext_vector_type(8)));
typedef short i16x16_t __attribute__((ext_vector_type(16)));

extern "C" __global__ void kernel_compute_tensor_bf16(
    float* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    float seed_a,
    float seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    const float base = 0.5f + 0.25f * (seed_a - floorf(seed_a)) +
                       0.25f * (seed_b - floorf(seed_b)) + 0.0625f * (float)(gid & 15u);

    /* Exact BF16 bit encodings: 0x3F80 = +1.0, 0x3F00 = +0.5, 0xBF00 = -0.5. */
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || \
    defined(__gfx942__) || defined(__gfx950__)
    const i16x4_t a_val = {0x3F80, 0x3F80, 0x3F80, 0x3F80};
    const i16x4_t b_pos = {0x3F00, 0x3F00, 0x3F00, 0x3F00};
    const i16x4_t b_neg = {(short)0xBF00, (short)0xBF00, (short)0xBF00, (short)0xBF00};

    fp32x4_t c0 = {base, base + 0.125f, base + 0.25f, base + 0.375f};
    fp32x4_t c1 = {base + 0.5f, base + 0.625f, base + 0.75f, base + 0.875f};
    fp32x4_t c2 = {base + 1.0f, base + 1.125f, base + 1.25f, base + 1.375f};
    fp32x4_t c3 = {base + 1.5f, base + 1.625f, base + 1.75f, base + 1.875f};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            c0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_val, b_pos, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_val, b_pos, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_val, b_pos, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_val, b_pos, c3, 0, 0, 0);

            c0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_val, b_neg, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_val, b_neg, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_val, b_neg, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_val, b_neg, c3, 0, 0, 0);
        }
    }

    if (gid < num_elements) {
        const float sum = (c0[0] + c0[1] + c0[2] + c0[3]) + (c1[0] + c1[1] + c1[2] + c1[3]) +
                          (c2[0] + c2[1] + c2[2] + c2[3]) + (c3[0] + c3[1] + c3[2] + c3[3]);
        out_buf[gid] = isfinite(sum) ? sum : __uint_as_float(0x7FC00000u);
    }
#elif defined(__gfx908__)
    const i16x2_t a_val = {0x3F80, 0x3F80};
    const i16x2_t b_pos = {0x3F00, 0x3F00};
    const i16x2_t b_neg = {(short)0xBF00, (short)0xBF00};

    fp32x4_t c0 = {base, base + 0.125f, base + 0.25f, base + 0.375f};
    fp32x4_t c1 = {base + 0.5f, base + 0.625f, base + 0.75f, base + 0.875f};
    fp32x4_t c2 = {base + 1.0f, base + 1.125f, base + 1.25f, base + 1.375f};
    fp32x4_t c3 = {base + 1.5f, base + 1.625f, base + 1.75f, base + 1.875f};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            c0 = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_val, b_pos, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_val, b_pos, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_val, b_pos, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_val, b_pos, c3, 0, 0, 0);

            c0 = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_val, b_neg, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_val, b_neg, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_val, b_neg, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_val, b_neg, c3, 0, 0, 0);
        }
    }

    if (gid < num_elements) {
        const float sum = (c0[0] + c0[1] + c0[2] + c0[3]) + (c1[0] + c1[1] + c1[2] + c1[3]) +
                          (c2[0] + c2[1] + c2[2] + c2[3]) + (c3[0] + c3[1] + c3[2] + c3[3]);
        out_buf[gid] = isfinite(sum) ? sum : __uint_as_float(0x7FC00000u);
    }
#elif defined(__gfx1200__) || defined(__gfx1201__)
    const i16x8_t a_val = {0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80};
    const i16x8_t b_pos = {0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00};
    const i16x8_t b_neg = {(short)0xBF00, (short)0xBF00, (short)0xBF00, (short)0xBF00,
                           (short)0xBF00, (short)0xBF00, (short)0xBF00, (short)0xBF00};

    fp32x8_t c0 = {base, base + 0.125f, base + 0.25f, base + 0.375f,
                   base + 0.5f, base + 0.625f, base + 0.75f, base + 0.875f};
    fp32x8_t c1 = {base + 1.0f, base + 1.125f, base + 1.25f, base + 1.375f,
                   base + 1.5f, base + 1.625f, base + 1.75f, base + 1.875f};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            c0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a_val, b_pos, c0);
            c1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a_val, b_pos, c1);
            c0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a_val, b_neg, c0);
            c1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a_val, b_neg, c1);
        }
    }

    if (gid < num_elements) {
        const float sum = (c0[0] + c0[1] + c0[2] + c0[3] + c0[4] + c0[5] + c0[6] + c0[7]) +
                          (c1[0] + c1[1] + c1[2] + c1[3] + c1[4] + c1[5] + c1[6] + c1[7]);
        out_buf[gid] = isfinite(sum) ? sum : __uint_as_float(0x7FC00000u);
    }
#else
    const i16x16_t a_val = {0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
                            0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80};
    const i16x16_t b_pos = {0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00,
                            0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00, 0x3F00};
    const i16x16_t b_neg = {(short)0xBF00, (short)0xBF00, (short)0xBF00, (short)0xBF00,
                            (short)0xBF00, (short)0xBF00, (short)0xBF00, (short)0xBF00,
                            (short)0xBF00, (short)0xBF00, (short)0xBF00, (short)0xBF00,
                            (short)0xBF00, (short)0xBF00, (short)0xBF00, (short)0xBF00};

    fp32x8_t c0 = {base, base + 0.125f, base + 0.25f, base + 0.375f,
                   base + 0.5f, base + 0.625f, base + 0.75f, base + 0.875f};
    fp32x8_t c1 = {base + 1.0f, base + 1.125f, base + 1.25f, base + 1.375f,
                   base + 1.5f, base + 1.625f, base + 1.75f, base + 1.875f};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            c0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_val, b_pos, c0);
            c1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_val, b_pos, c1);
            c0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_val, b_neg, c0);
            c1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_val, b_neg, c1);
        }
    }

    if (gid < num_elements) {
        const float sum = (c0[0] + c0[1] + c0[2] + c0[3] + c0[4] + c0[5] + c0[6] + c0[7]) +
                          (c1[0] + c1[1] + c1[2] + c1[3] + c1[4] + c1[5] + c1[6] + c1[7]);
        out_buf[gid] = isfinite(sum) ? sum : __uint_as_float(0x7FC00000u);
    }
#endif
}
