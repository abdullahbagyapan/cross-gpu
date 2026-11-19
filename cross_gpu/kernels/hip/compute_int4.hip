// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_int4 (hip)
 * Entry point(s): kernel_compute_int4
 * ABI: (uint32_t* out_buf, uint32_t num_elements, uint32_t iterations, uint32_t seed_a, uint32_t seed_b)
 * Arch / Extension guard: gfx906/gfx908/gfx90a/gfx103x (__builtin_amdgcn_sdot8)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 16 = 2048 (8 accumulators x 16 unrolled sdot8 x 16 ops)
 */

#if !defined(__gfx906__) && !defined(__gfx908__) && !defined(__gfx90a__) && \
    !defined(__gfx1030__) && !defined(__gfx1031__) && !defined(__gfx1032__) && \
    !defined(__gfx1034__) && !defined(__gfx1035__) && !defined(__gfx1036__)
#error "UNSUPPORTED: HIP compute_int4 requires hardware sdot8 support (gfx906/CDNA1-2/RDNA2)"
#endif

typedef unsigned int u32;
typedef int i32;

extern "C" __global__ void kernel_compute_int4(
    u32* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    u32 seed_a,
    u32 seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) {
        return;
    }

    /* Packed 8x signed 4-bit nibbles in [1, 7] via bitmasking keep sdot8 non-zero. */
    const i32 mul_pack = (i32)((seed_a & 0x77777777u) | 0x11111111u);
    const u32 mix = seed_a ^ seed_b;

    i32 r0 = (i32)(((gid + 1u * 0x9E3779B9u) ^ mix) & 0x77777777u);
    i32 r1 = (i32)(((gid + 2u * 0x9E3779B9u) ^ mix) & 0x77777777u);
    i32 r2 = (i32)(((gid + 3u * 0x9E3779B9u) ^ mix) & 0x77777777u);
    i32 r3 = (i32)(((gid + 4u * 0x9E3779B9u) ^ mix) & 0x77777777u);
    i32 r4 = (i32)(((gid + 5u * 0x9E3779B9u) ^ mix) & 0x77777777u);
    i32 r5 = (i32)(((gid + 6u * 0x9E3779B9u) ^ mix) & 0x77777777u);
    i32 r6 = (i32)(((gid + 7u * 0x9E3779B9u) ^ mix) & 0x77777777u);
    i32 r7 = (i32)(((gid + 8u * 0x9E3779B9u) ^ mix) & 0x77777777u);

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 step = 0; step < 16u; ++step) {
            r0 = __builtin_amdgcn_sdot8(r0, mul_pack, r0, false);
            r1 = __builtin_amdgcn_sdot8(r1, mul_pack, r1, false);
            r2 = __builtin_amdgcn_sdot8(r2, mul_pack, r2, false);
            r3 = __builtin_amdgcn_sdot8(r3, mul_pack, r3, false);
            r4 = __builtin_amdgcn_sdot8(r4, mul_pack, r4, false);
            r5 = __builtin_amdgcn_sdot8(r5, mul_pack, r5, false);
            r6 = __builtin_amdgcn_sdot8(r6, mul_pack, r6, false);
            r7 = __builtin_amdgcn_sdot8(r7, mul_pack, r7, false);
        }
    }

    out_buf[gid] = ((u32)r0 ^ (u32)r1) + ((u32)r2 ^ (u32)r3) +
                   ((u32)r4 ^ (u32)r5) + ((u32)r6 ^ (u32)r7);
}
