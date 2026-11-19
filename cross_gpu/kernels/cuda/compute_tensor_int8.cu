// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_tensor_int8 (cuda)
 * Entry point(s): kernel_compute_tensor_int8
 * ABI: (uint32_t* out_buf, uint32_t num_elements, uint32_t iterations, uint32_t seed_a, uint32_t seed_b)
 * Arch / Extension guard: __CUDA_ARCH__ >= 800 (INT8 mma.sync.aligned.m16n8k32)
 * OPS_PER_THREAD_PER_ITER: 4 * 8 * (2 * 16 * 8 * 32 / 32) = 8192 (32 s8 MMAs x 256 ops/lane)
 */

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 800)
#error "UNSUPPORTED: CUDA compute_tensor_int8 requires SM 8.0+ for m16n8k32 s8 MMA"
#endif

typedef unsigned int u32;
typedef int i32;

static __device__ __forceinline__ void mma_m16n8k32_s8(
    i32& d0, i32& d1, i32& d2, i32& d3,
    u32 a_val, u32 b_val,
    i32 c0, i32 c1, i32 c2, i32 c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a_val), "r"(a_val), "r"(a_val), "r"(a_val),
          "r"(b_val), "r"(b_val),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));
}

extern "C" __global__ void kernel_compute_tensor_int8(
    u32* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    u32 seed_a,
    u32 seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;

    /* Packed 4x signed int8 bytes: +1 = 0x01, -1 = 0xFF (paired +32/-32 net zero). */
    const u32 a_val = 0x01010101u;
    const u32 b_pos = 0x01010101u;
    const u32 b_neg = 0xFFFFFFFFu;

    const i32 base = (i32)((gid ^ seed_a ^ seed_b) & 0xFFFFu);
    i32 c00 = base + 1, c01 = base + 2, c02 = base + 3, c03 = base + 4;
    i32 c10 = base + 5, c11 = base + 6, c12 = base + 7, c13 = base + 8;
    i32 c20 = base + 9, c21 = base + 10, c22 = base + 11, c23 = base + 12;
    i32 c30 = base + 13, c31 = base + 14, c32 = base + 15, c33 = base + 16;

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            mma_m16n8k32_s8(c00, c01, c02, c03, a_val, b_pos, c00, c01, c02, c03);
            mma_m16n8k32_s8(c10, c11, c12, c13, a_val, b_pos, c10, c11, c12, c13);
            mma_m16n8k32_s8(c20, c21, c22, c23, a_val, b_pos, c20, c21, c22, c23);
            mma_m16n8k32_s8(c30, c31, c32, c33, a_val, b_pos, c30, c31, c32, c33);

            mma_m16n8k32_s8(c00, c01, c02, c03, a_val, b_neg, c00, c01, c02, c03);
            mma_m16n8k32_s8(c10, c11, c12, c13, a_val, b_neg, c10, c11, c12, c13);
            mma_m16n8k32_s8(c20, c21, c22, c23, a_val, b_neg, c20, c21, c22, c23);
            mma_m16n8k32_s8(c30, c31, c32, c33, a_val, b_neg, c30, c31, c32, c33);
        }
    }

    if (gid < num_elements) {
        const u32 sum = (u32)c00 + (u32)c01 + (u32)c02 + (u32)c03 +
                        (u32)c10 + (u32)c11 + (u32)c12 + (u32)c13 +
                        (u32)c20 + (u32)c21 + (u32)c22 + (u32)c23 +
                        (u32)c30 + (u32)c31 + (u32)c32 + (u32)c33;
        out_buf[gid] = sum;
    }
}
