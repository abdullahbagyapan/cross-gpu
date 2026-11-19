// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_int16 (levelzero)
 * Entry point(s): kernel_compute_int16
 * ABI: (__global uint* out_buf, uint num_elements, uint iterations, uint seed_a, uint seed_b)
 * Arch / Extension guard: OpenCL C / SPIR-V native 16-bit unsigned MAD
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 2 = 256 (8 accumulators x 16 unrolled MADs x 2 ops)
 */

__kernel void kernel_compute_int16(
    __global uint* restrict out_buf,
    const uint num_elements,
    const uint iterations,
    const uint seed_a,
    const uint seed_b) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= num_elements) {
        return;
    }

    const ushort mul = (ushort)((seed_a & 0xFFFFu) | 1u);
    const ushort add = (ushort)((seed_b & 0xFFFFu) | 1u);

    ushort r0 = (ushort)(((gid + 1u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    ushort r1 = (ushort)(((gid + 2u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    ushort r2 = (ushort)(((gid + 3u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    ushort r3 = (ushort)(((gid + 4u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    ushort r4 = (ushort)(((gid + 5u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    ushort r5 = (ushort)(((gid + 6u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    ushort r6 = (ushort)(((gid + 7u * 0x9E37u) ^ seed_a) & 0xFFFFu);
    ushort r7 = (ushort)(((gid + 8u * 0x9E37u) ^ seed_a) & 0xFFFFu);

    for (uint it = 0u; it < iterations; ++it) {
        #pragma unroll
        for (uint step = 0u; step < 16u; ++step) {
            r0 = (ushort)(r0 * mul + add);
            r1 = (ushort)(r1 * mul + add);
            r2 = (ushort)(r2 * mul + add);
            r3 = (ushort)(r3 * mul + add);
            r4 = (ushort)(r4 * mul + add);
            r5 = (ushort)(r5 * mul + add);
            r6 = (ushort)(r6 * mul + add);
            r7 = (ushort)(r7 * mul + add);
        }
    }

    const ushort acc = (ushort)((ushort)(r0 ^ r1) + (ushort)(r2 ^ r3) +
                                (ushort)(r4 ^ r5) + (ushort)(r6 ^ r7));
    out_buf[gid] = (uint)acc;
}
