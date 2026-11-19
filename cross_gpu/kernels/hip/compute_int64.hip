/*
 * cross-gpu kernel: compute_int64 (hip)
 * Entry point(s): kernel_compute_int64
 * ABI: (uint64_t* out_buf, uint32_t num_elements, uint32_t iterations, uint64_t seed_a, uint64_t seed_b)
 * Arch / Extension guard: all HIP/AMDGCN architectures (native 64-bit unsigned MAD)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 2 = 256 (8 accumulators x 16 unrolled MADs x 2 ops)
 */

typedef unsigned long long u64;
typedef unsigned int u32;

extern "C" __global__ void kernel_compute_int64(
    u64* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    u64 seed_a,
    u64 seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) {
        return;
    }

    const u64 mul = seed_a | 1ULL;
    const u64 add = seed_b | 1ULL;
    const u64 base = (u64)gid;

    u64 r0 = (base + 1ULL * 0x9E3779B97F4A7C15ULL) ^ seed_a;
    u64 r1 = (base + 2ULL * 0x9E3779B97F4A7C15ULL) ^ seed_a;
    u64 r2 = (base + 3ULL * 0x9E3779B97F4A7C15ULL) ^ seed_a;
    u64 r3 = (base + 4ULL * 0x9E3779B97F4A7C15ULL) ^ seed_a;
    u64 r4 = (base + 5ULL * 0x9E3779B97F4A7C15ULL) ^ seed_a;
    u64 r5 = (base + 6ULL * 0x9E3779B97F4A7C15ULL) ^ seed_a;
    u64 r6 = (base + 7ULL * 0x9E3779B97F4A7C15ULL) ^ seed_a;
    u64 r7 = (base + 8ULL * 0x9E3779B97F4A7C15ULL) ^ seed_a;

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
