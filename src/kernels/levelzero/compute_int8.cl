// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_int8 (levelzero)
 * Entry point(s): kernel_compute_int8
 * ABI: (__global uint* out_buf, uint num_elements, uint iterations, uint seed_a, uint seed_b)
 * Arch / Extension guard: OpenCL C / SPIR-V native uchar4 8-bit vector MAD
 * OPS_PER_THREAD_PER_ITER: 8 * 4 * 16 * 2 = 1024 (8 uchar4 accs x 4 lanes x 16 MADs x 2 ops)
 */

static inline uchar4 init_u8x4(uint gid, uint k, uint seed_a) {
    const uint base = gid + (k + 1u) * 37u;
    return (uchar4)(
        (uchar)(((base + 0u * 17u) ^ (seed_a >> 0)) & 0xFFu),
        (uchar)(((base + 1u * 17u) ^ (seed_a >> 8)) & 0xFFu),
        (uchar)(((base + 2u * 17u) ^ (seed_a >> 16)) & 0xFFu),
        (uchar)(((base + 3u * 17u) ^ (seed_a >> 24)) & 0xFFu)
    );
}

__kernel void kernel_compute_int8(
    __global uint* restrict out_buf,
    const uint num_elements,
    const uint iterations,
    const uint seed_a,
    const uint seed_b) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= num_elements) {
        return;
    }

    const uchar4 mul = (uchar4)(
        (uchar)((seed_a & 0xFFu) | 1u),
        (uchar)(((seed_a >> 8) & 0xFFu) | 1u),
        (uchar)(((seed_a >> 16) & 0xFFu) | 1u),
        (uchar)(((seed_a >> 24) & 0xFFu) | 1u)
    );
    const uchar4 add = (uchar4)(
        (uchar)((seed_b & 0xFFu) | 1u),
        (uchar)(((seed_b >> 8) & 0xFFu) | 1u),
        (uchar)(((seed_b >> 16) & 0xFFu) | 1u),
        (uchar)(((seed_b >> 24) & 0xFFu) | 1u)
    );

    uchar4 r0 = init_u8x4(gid, 0u, seed_a);
    uchar4 r1 = init_u8x4(gid, 1u, seed_a);
    uchar4 r2 = init_u8x4(gid, 2u, seed_a);
    uchar4 r3 = init_u8x4(gid, 3u, seed_a);
    uchar4 r4 = init_u8x4(gid, 4u, seed_a);
    uchar4 r5 = init_u8x4(gid, 5u, seed_a);
    uchar4 r6 = init_u8x4(gid, 6u, seed_a);
    uchar4 r7 = init_u8x4(gid, 7u, seed_a);

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

    const uchar4 acc = (r0 ^ r1) + (r2 ^ r3) + (r4 ^ r5) + (r6 ^ r7);
    out_buf[gid] = as_uint(acc);
}
