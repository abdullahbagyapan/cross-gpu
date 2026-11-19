// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_tensor_bf16 (cuda)
 * Entry point(s): kernel_compute_tensor_bf16
 * ABI: (float* out_buf, uint32_t num_elements, uint32_t iterations, float seed_a, float seed_b)
 * Arch / Extension guard: __CUDA_ARCH__ >= 800 (BF16 mma.sync.aligned.m16n8k16)
 * OPS_PER_THREAD_PER_ITER: 4 * 8 * (2 * 16 * 8 * 16 / 32) = 4096 (32 BF16 MMAs x 128 FLOPs/lane)
 */

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 800)
#error "UNSUPPORTED: CUDA compute_tensor_bf16 requires SM 8.0+ for native BF16 MMA"
#endif

typedef unsigned int u32;

static __device__ __forceinline__ void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    u32 a_val, u32 b_val,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a_val), "r"(a_val), "r"(a_val), "r"(a_val),
          "r"(b_val), "r"(b_val),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

extern "C" __global__ void kernel_compute_tensor_bf16(
    float* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    float seed_a,
    float seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;

    /* Exact BF16 pairs: 0x3F80 = +1.0, 0x3F00 = +0.5, 0xBF00 = -0.5. */
    const u32 a_val = 0x3F803F80u;
    const u32 b_pos = 0x3F003F00u;
    const u32 b_neg = 0xBF00BF00u;

    const float base = 0.5f + 0.25f * (seed_a - floorf(seed_a)) +
                       0.25f * (seed_b - floorf(seed_b)) + 0.0625f * (float)(gid & 15u);

    float c00 = base, c01 = base + 0.125f, c02 = base + 0.25f, c03 = base + 0.375f;
    float c10 = base + 0.5f, c11 = base + 0.625f, c12 = base + 0.75f, c13 = base + 0.875f;
    float c20 = base + 1.0f, c21 = base + 1.125f, c22 = base + 1.25f, c23 = base + 1.375f;
    float c30 = base + 1.5f, c31 = base + 1.625f, c32 = base + 1.75f, c33 = base + 1.875f;

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            mma_m16n8k16_bf16(c00, c01, c02, c03, a_val, b_pos, c00, c01, c02, c03);
            mma_m16n8k16_bf16(c10, c11, c12, c13, a_val, b_pos, c10, c11, c12, c13);
            mma_m16n8k16_bf16(c20, c21, c22, c23, a_val, b_pos, c20, c21, c22, c23);
            mma_m16n8k16_bf16(c30, c31, c32, c33, a_val, b_pos, c30, c31, c32, c33);

            mma_m16n8k16_bf16(c00, c01, c02, c03, a_val, b_neg, c00, c01, c02, c03);
            mma_m16n8k16_bf16(c10, c11, c12, c13, a_val, b_neg, c10, c11, c12, c13);
            mma_m16n8k16_bf16(c20, c21, c22, c23, a_val, b_neg, c20, c21, c22, c23);
            mma_m16n8k16_bf16(c30, c31, c32, c33, a_val, b_neg, c30, c31, c32, c33);
        }
    }

    if (gid < num_elements) {
        const float sum = (c00 + c01 + c02 + c03) + (c10 + c11 + c12 + c13) +
                          (c20 + c21 + c22 + c23) + (c30 + c31 + c32 + c33);
        out_buf[gid] = isfinite(sum) ? sum : __uint_as_float(0x7FC00000u);
    }
}
