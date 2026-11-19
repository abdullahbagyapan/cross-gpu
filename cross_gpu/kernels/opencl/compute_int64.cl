/*
 * cross-gpu kernel: compute_int64 (opencl)
 * Entry point(s): kernel_compute_int64
 * ABI: (__global ulong* out_buf, uint num_elements, uint iterations, ulong seed_a, ulong seed_b)
 * Arch / Extension guard: OpenCL C 1.2/3.0 FULL_PROFILE (native 64-bit ulong MAD)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 2 = 256 (8 accumulators x 16 unrolled MADs x 2 ops)
 */

__kernel void kernel_compute_int64(
    __global ulong* restrict out_buf,
    const uint num_elements,
    const uint iterations,
    const ulong seed_a,
    const ulong seed_b) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= num_elements) {
        return;
    }

    const ulong mul = seed_a | 1UL;
    const ulong add = seed_b | 1UL;
    const ulong base = (ulong)gid;

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
