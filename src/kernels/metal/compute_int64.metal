// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_int64 (metal)
 * Entry point(s): kernel_compute_int64
 * ABI: (device ulong* out_buf, constant uint& num_elements, constant uint& iterations, constant ulong& seed_a, constant ulong& seed_b)
 * Arch / Extension guard: Metal 2.3+ native 64-bit ulong MAD
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 2 = 256 (8 accumulators x 16 unrolled MADs x 2 ops)
 */

#include <metal_stdlib>
using namespace metal;

kernel void kernel_compute_int64(
    device ulong* out_buf [[buffer(0)]],
    constant uint& num_elements [[buffer(1)]],
    constant uint& iterations [[buffer(2)]],
    constant ulong& seed_a [[buffer(3)]],
    constant ulong& seed_b [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= num_elements) {
        return;
    }

    const ulong mul = seed_a | 1UL;
    const ulong add = seed_b | 1UL;
    const ulong base = ulong(gid);

    ulong r0 = (base + 1UL * 0x9E3779B97F4A7C15UL) ^ seed_a;
    ulong r1 = (base + 2UL * 0x9E3779B97F4A7C15UL) ^ seed_a;
    ulong r2 = (base + 3UL * 0x9E3779B97F4A7C15UL) ^ seed_a;
    ulong r3 = (base + 4UL * 0x9E3779B97F4A7C15UL) ^ seed_a;
    ulong r4 = (base + 5UL * 0x9E3779B97F4A7C15UL) ^ seed_a;
    ulong r5 = (base + 6UL * 0x9E3779B97F4A7C15UL) ^ seed_a;
    ulong r6 = (base + 7UL * 0x9E3779B97F4A7C15UL) ^ seed_a;
    ulong r7 = (base + 8UL * 0x9E3779B97F4A7C15UL) ^ seed_a;

    for (uint it = 0u; it < iterations; ++it) {
        #pragma unroll
        for (uint step = 0u; step < 16u; ++step) {
            r0 = r0 * mul + add;
            r1 = r1 * mul + add;
            r2 = r2 * mul + add;
            r3 = r3 * mul + add;
            r4 = r4 * mul + add;
            r5 = r5 * mul + add;
            r6 = r6 * mul + add;
            r7 = r7 * mul + add;
        }
    }

    out_buf[gid] = (r0 ^ r1) + (r2 ^ r3) + (r4 ^ r5) + (r6 ^ r7);
}
