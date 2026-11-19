// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_int16 (cuda)
 * Entry point(s): kernel_compute_int16
 * ABI: (uint32_t* out_buf, uint32_t num_elements, uint32_t iterations, uint32_t seed_a, uint32_t seed_b)
 * Arch / Extension guard: all CUDA architectures (native 16-bit unsigned MAD)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 2 = 256 (8 accumulators x 16 unrolled MADs x 2 ops)
 */

typedef unsigned int u32;
typedef unsigned short u16;

extern "C" __global__ void kernel_compute_int16(
    u32* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    u32 seed_a,
    u32 seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) {
        return;
    }

    const u16 mul = (u16)((seed_a & 0xFFFFu) | 1u);
    const u16 add = (u16)((seed_b & 0xFFFFu) | 1u);

    u16 r0 = (u16)(((gid + 1u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    u16 r1 = (u16)(((gid + 2u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    u16 r2 = (u16)(((gid + 3u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    u16 r3 = (u16)(((gid + 4u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    u16 r4 = (u16)(((gid + 5u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    u16 r5 = (u16)(((gid + 6u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    u16 r6 = (u16)(((gid + 7u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    u16 r7 = (u16)(((gid + 8u * 0x9E37u) ^ seed_a) & 0xFFFFu);

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 step = 0; step < 16u; ++step) {
            r0 = (u16)(r0 * mul + add);
            r1 = (u16)(r1 * mul + add);
            r2 = (u16)(r2 * mul + add);
            r3 = (u16)(r3 * mul + add);
            r4 = (u16)(r4 * mul + add);
            r5 = (u16)(r5 * mul + add);
            r6 = (u16)(r6 * mul + add);
            r7 = (u16)(r7 * mul + add);
        }
    }

    const u16 acc = (u16)((u16)(r0 ^ r1) + (u16)(r2 ^ r3) + (u16)(r4 ^ r5) + (u16)(r6 ^ r7));
    out_buf[gid] = (u32)acc;
}
