/*
 * cross-gpu kernel: compute_int32 (metal)
 * Entry point(s): kernel_compute_int32
 * ABI: (device uint* out_buf, constant uint& num_elements, constant uint& iterations, constant uint& seed_a, constant uint& seed_b)
 * Arch / Extension guard: all Metal devices (native 32-bit uint MAD)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 2 = 256 (8 accumulators x 16 unrolled MADs x 2 ops)
 */

#include <metal_stdlib>
using namespace metal;

kernel void kernel_compute_int32(
    device uint* out_buf [[buffer(0)]],
    constant uint& num_elements [[buffer(1)]],
    constant uint& iterations [[buffer(2)]],
    constant uint& seed_a [[buffer(3)]],
    constant uint& seed_b [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= num_elements) {
        return;
    }

    const uint mul = seed_a | 1u;
    const uint add = seed_b | 1u;

    uint r0 = (gid + 1u * 0x9E3779B9u) ^ seed_a;
    uint r1 = (gid + 2u * 0x9E3779B9u) ^ seed_a;
    uint r2 = (gid + 3u * 0x9E3779B9u) ^ seed_a;
    uint r3 = (gid + 4u * 0x9E3779B9u) ^ seed_a;
    uint r4 = (gid + 5u * 0x9E3779B9u) ^ seed_a;
    uint r5 = (gid + 6u * 0x9E3779B9u) ^ seed_a;
    uint r6 = (gid + 7u * 0x9E3779B9u) ^ seed_a;
    uint r7 = (gid + 8u * 0x9E3779B9u) ^ seed_a;

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
