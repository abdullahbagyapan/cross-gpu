/*
 * cross-gpu kernel: compute_tensor_fp32 (hip)
 * Entry point(s): kernel_compute_tensor_fp32
 * ABI: (float* out_buf, uint32_t num_elements, uint32_t iterations, float seed_a, float seed_b)
 * Arch / Extension guard: CDNA (gfx908/gfx90a/gfx94x/gfx950) __builtin_amdgcn_mfma_f32_16x16x4f32
 * OPS_PER_THREAD_PER_ITER: 4 * 8 * (2 * 16 * 16 * 4 / 64) = 1024 (32 MFMAs x 32 FLOPs/lane)
 */

#if !defined(__gfx908__) && !defined(__gfx90a__) && !defined(__gfx940__) && \
    !defined(__gfx941__) && !defined(__gfx942__) && !defined(__gfx950__)
#error "UNSUPPORTED: HIP compute_tensor_fp32 requires CDNA (gfx908+) MFMA FP32 hardware"
#endif

typedef unsigned int u32;
typedef float fp32x4_t __attribute__((ext_vector_type(4)));

extern "C" __global__ void kernel_compute_tensor_fp32(
    float* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    float seed_a,
    float seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;

    const float a_val = 1.0f;
    const float b_pos = 0.5f;
    const float b_neg = -0.5f;
    const float base = 0.5f + 0.25f * (seed_a - floorf(seed_a)) +
                       0.25f * (seed_b - floorf(seed_b)) + 0.0625f * (float)(gid & 15u);

    fp32x4_t c0 = {base, base + 0.125f, base + 0.25f, base + 0.375f};
    fp32x4_t c1 = {base + 0.5f, base + 0.625f, base + 0.75f, base + 0.875f};
    fp32x4_t c2 = {base + 1.0f, base + 1.125f, base + 1.25f, base + 1.375f};
    fp32x4_t c3 = {base + 1.5f, base + 1.625f, base + 1.75f, base + 1.875f};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            c0 = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_pos, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_pos, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_pos, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_pos, c3, 0, 0, 0);

            c0 = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_neg, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_neg, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_neg, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_f32_16x16x4f32(a_val, b_neg, c3, 0, 0, 0);
        }
    }

    if (gid < num_elements) {
        const float sum = (c0[0] + c0[1] + c0[2] + c0[3]) + (c1[0] + c1[1] + c1[2] + c1[3]) +
                          (c2[0] + c2[1] + c2[2] + c2[3]) + (c3[0] + c3[1] + c3[2] + c3[3]);
        out_buf[gid] = isfinite(sum) ? sum : __uint_as_float(0x7FC00000u);
    }
}
