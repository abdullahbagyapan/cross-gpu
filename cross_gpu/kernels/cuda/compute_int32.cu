/*
 * cross-gpu kernel: compute_int32 (cuda)
 * Entry point(s): kernel_compute_int32
 * ABI: (uint32_t* out_buf, uint32_t num_elements, uint32_t iterations, uint32_t seed_a, uint32_t seed_b)
 * Arch / Extension guard: all CUDA architectures (native 32-bit unsigned MAD)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 2 = 256 (8 accumulators x 16 unrolled MADs x 2 ops)
 */

typedef unsigned int u32;

extern "C" __global__ void kernel_compute_int32(
    u32* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    u32 seed_a,
    u32 seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) {
        return;
    }

    const u32 mul = seed_a | 1u;
    const u32 add = seed_b | 1u;

    u32 r0 = (gid + 1u * 0x9E3779B9u) ^ seed_a;
    u32 r1 = (gid + 2u * 0x9E3779B9u) ^ seed_a;
    u32 r2 = (gid + 3u * 0x9E3779B9u) ^ seed_a;
    u32 r3 = (gid + 4u * 0x9E3779B9u) ^ seed_a;
    u32 r4 = (gid + 5u * 0x9E3779B9u) ^ seed_a;
    u32 r5 = (gid + 6u * 0x9E3779B9u) ^ seed_a;
    u32 r6 = (gid + 7u * 0x9E3779B9u) ^ seed_a;
    u32 r7 = (gid + 8u * 0x9E3779B9u) ^ seed_a;

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 step = 0; step < 16u; ++step) {
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
