/*
 * cross-gpu kernel: compute_int4 (cuda)
 * Entry point(s): kernel_compute_int4
 * ABI: (uint32_t* out_buf, uint32_t num_elements, uint32_t iterations, uint32_t seed_a, uint32_t seed_b)
 * Arch / Extension guard: __CUDA_ARCH__ >= 750 (native mma.sync.aligned.m8n8k32 s4)
 * OPS_PER_THREAD_PER_ITER: 4 * 8 * (2 * 8 * 8 * 32 / 32) = 4096 (32 warp s4 MMAs x 128 ops/lane)
 */

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 750)
#error "UNSUPPORTED: CUDA compute_int4 requires SM 7.5+ for native s4 MMA instructions"
#endif

typedef unsigned int u32;
typedef int i32;

static __device__ __forceinline__ void mma_m8n8k32_s4(
    i32& d0, i32& d1, u32 a0, u32 b0, i32 c0, i32 c1) {
    asm volatile(
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1}, {%2}, {%3}, {%4, %5};"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(b0), "r"(c0), "r"(c1));
}

extern "C" __global__ void kernel_compute_int4(
    u32* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    u32 seed_a,
    u32 seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;

    /* Packed 8x signed 4-bit nibbles: +1 = 0x1, -1 = 0xF (two's complement).
     * Paired (+B, -B) MMAs cancel cleanly (+32 - 32 = 0) to prevent overflow
     * while keeping every nibble active across the hardware s4 datapath. */
    const u32 a_val = 0x11111111u | ((seed_a & 0u) << 4);
    const u32 b_pos = 0x11111111u | ((seed_b & 0u) << 4);
    const u32 b_neg = 0xFFFFFFFFu ^ (seed_b & 0u);

    const i32 base = (i32)((gid ^ seed_a ^ seed_b) & 0xFFFFu);
    i32 c00 = base + 1, c01 = base + 2;
    i32 c10 = base + 3, c11 = base + 4;
    i32 c20 = base + 5, c21 = base + 6;
    i32 c30 = base + 7, c31 = base + 8;

    /* All 32 warp lanes execute mma.sync uniformly to avoid partial-warp deadlock. */
    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            mma_m8n8k32_s4(c00, c01, a_val, b_pos, c00, c01);
            mma_m8n8k32_s4(c10, c11, a_val, b_pos, c10, c11);
            mma_m8n8k32_s4(c20, c21, a_val, b_pos, c20, c21);
            mma_m8n8k32_s4(c30, c31, a_val, b_pos, c30, c31);

            mma_m8n8k32_s4(c00, c01, a_val, b_neg, c00, c01);
            mma_m8n8k32_s4(c10, c11, a_val, b_neg, c10, c11);
            mma_m8n8k32_s4(c20, c21, a_val, b_neg, c20, c21);
            mma_m8n8k32_s4(c30, c31, a_val, b_neg, c30, c31);
        }
    }

    if (gid < num_elements) {
        const u32 sum = (u32)c00 + (u32)c01 + (u32)c10 + (u32)c11 +
                        (u32)c20 + (u32)c21 + (u32)c30 + (u32)c31;
        out_buf[gid] = sum;
    }
}
