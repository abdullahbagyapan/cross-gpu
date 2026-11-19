// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_int8 (hip)
 * Entry point(s): kernel_compute_int8
 * ABI: (uint32_t* out_buf, uint32_t num_elements, uint32_t iterations, uint32_t seed_a, uint32_t seed_b)
 * Arch / Extension guard: gfx906+ / CDNA / RDNA (__builtin_amdgcn_udot4)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 8 = 1024 (8 accumulators x 16 unrolled udot4 x 8 ops)
 */

#if !defined(__gfx906__) && !defined(__gfx908__) && !defined(__gfx90a__) && \
    !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__) && !defined(__gfx950__) && \
    !defined(__gfx1011__) && !defined(__gfx1012__) && !defined(__gfx1030__) && \
    !defined(__gfx1031__) && !defined(__gfx1032__) && !defined(__gfx1034__) && \
    !defined(__gfx1035__) && !defined(__gfx1036__) && !defined(__gfx1100__) && \
    !defined(__gfx1101__) && !defined(__gfx1102__) && !defined(__gfx1103__) && \
    !defined(__gfx1150__) && !defined(__gfx1151__) && !defined(__gfx1200__) && !defined(__gfx1201__)
#error "UNSUPPORTED: HIP compute_int8 requires gfx906+ / RDNA+ hardware dot4 support"
#endif

typedef unsigned int u32;

extern "C" __global__ void kernel_compute_int8(
    u32* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    u32 seed_a,
    u32 seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) {
        return;
    }

    const u32 mul_pack = (seed_a & 0x7F7F7F7Fu) | 0x01010101u;
    const u32 mix = seed_a ^ seed_b;

    u32 r0 = ((gid + 1u * 0x9E3779B9u) ^ mix) | 1u;
    u32 r1 = ((gid + 2u * 0x9E3779B9u) ^ mix) | 1u;
    u32 r2 = ((gid + 3u * 0x9E3779B9u) ^ mix) | 1u;
    u32 r3 = ((gid + 4u * 0x9E3779B9u) ^ mix) | 1u;
    u32 r4 = ((gid + 5u * 0x9E3779B9u) ^ mix) | 1u;
    u32 r5 = ((gid + 6u * 0x9E3779B9u) ^ mix) | 1u;
    u32 r6 = ((gid + 7u * 0x9E3779B9u) ^ mix) | 1u;
    u32 r7 = ((gid + 8u * 0x9E3779B9u) ^ mix) | 1u;

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 step = 0; step < 16u; ++step) {
            r0 = __builtin_amdgcn_udot4(r0, mul_pack, r0, false);
            r1 = __builtin_amdgcn_udot4(r1, mul_pack, r1, false);
            r2 = __builtin_amdgcn_udot4(r2, mul_pack, r2, false);
            r3 = __builtin_amdgcn_udot4(r3, mul_pack, r3, false);
            r4 = __builtin_amdgcn_udot4(r4, mul_pack, r4, false);
            r5 = __builtin_amdgcn_udot4(r5, mul_pack, r5, false);
            r6 = __builtin_amdgcn_udot4(r6, mul_pack, r6, false);
            r7 = __builtin_amdgcn_udot4(r7, mul_pack, r7, false);
        }
    }

    out_buf[gid] = (r0 ^ r1) + (r2 ^ r3) + (r4 ^ r5) + (r6 ^ r7);
}
