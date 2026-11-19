// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_fp4 (hip)
 * Entry point(s): kernel_compute_fp4
 * ABI: (float* out_buf, uint32_t num_elements, uint32_t iterations, float seed_a, float seed_b)
 * Arch / Extension guard: CDNA4 (gfx950+) hardware FP4 MFMA (__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
 * OPS_PER_THREAD_PER_ITER: 4 * 8 * (2 * 16 * 16 * 128 / 64) = 32768 (32 FP4 MFMAs x 1024 FLOPs/lane)
 */

#if !defined(__gfx950__)
#error "UNSUPPORTED: HIP compute_fp4 requires CDNA4 (gfx950+) hardware FP4 MFMA instructions"
#endif

typedef unsigned int u32;
typedef float fp32x4_t __attribute__((ext_vector_type(4)));
typedef int i32x8_t __attribute__((ext_vector_type(8)));

extern "C" __global__ void kernel_compute_fp4(
    float* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    float seed_a,
    float seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    const float base = 0.5f + 0.25f * (seed_a - floorf(seed_a)) +
                       0.25f * (seed_b - floorf(seed_b)) + 0.0625f * (float)(gid & 15u);

    /* Packed e2m1 FP4 nibbles: 0x2 = +1.0, 0x1 = +0.5, 0x9 = -0.5; E8M0 scale 127 = 1.0. */
    const i32x8_t a_val = {0x22222222, 0x22222222, 0x22222222, 0x22222222, 0, 0, 0, 0};
    const i32x8_t b_pos = {0x11111111, 0x11111111, 0x11111111, 0x11111111, 0, 0, 0, 0};
    const i32x8_t b_neg = {(int)0x99999999u, (int)0x99999999u, (int)0x99999999u, (int)0x99999999u, 0, 0, 0, 0};
    const u32 unit_scale = 0x7F7F7F7Fu;

    fp32x4_t c0 = {base, base + 0.125f, base + 0.25f, base + 0.375f};
    fp32x4_t c1 = {base + 0.5f, base + 0.625f, base + 0.75f, base + 0.875f};
    fp32x4_t c2 = {base + 1.0f, base + 1.125f, base + 1.25f, base + 1.375f};
    fp32x4_t c3 = {base + 1.5f, base + 1.625f, base + 1.75f, base + 1.875f};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            c0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_val, b_pos, c0, 4, 4, 0, unit_scale, 0, unit_scale);
            c1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_val, b_pos, c1, 4, 4, 0, unit_scale, 0, unit_scale);
            c2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_val, b_pos, c2, 4, 4, 0, unit_scale, 0, unit_scale);
            c3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_val, b_pos, c3, 4, 4, 0, unit_scale, 0, unit_scale);

            c0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_val, b_neg, c0, 4, 4, 0, unit_scale, 0, unit_scale);
            c1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_val, b_neg, c1, 4, 4, 0, unit_scale, 0, unit_scale);
            c2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_val, b_neg, c2, 4, 4, 0, unit_scale, 0, unit_scale);
            c3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_val, b_neg, c3, 4, 4, 0, unit_scale, 0, unit_scale);
        }
    }

    if (gid < num_elements) {
        const float sum = (c0[0] + c0[1] + c0[2] + c0[3]) + (c1[0] + c1[1] + c1[2] + c1[3]) +
                          (c2[0] + c2[1] + c2[2] + c2[3]) + (c3[0] + c3[1] + c3[2] + c3[3]);
        out_buf[gid] = isfinite(sum) ? sum : __uint_as_float(0x7FC00000u);
    }
}
