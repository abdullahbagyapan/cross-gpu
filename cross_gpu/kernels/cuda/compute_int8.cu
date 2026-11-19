/*
 * cross-gpu kernel: compute_int8 (cuda)
 * Entry point(s): kernel_compute_int8
 * ABI: (uint32_t* out_buf, uint32_t num_elements, uint32_t iterations, uint32_t seed_a, uint32_t seed_b)
 * Arch / Extension guard: __CUDA_ARCH__ >= 610 (native dp4a.u32.u32)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 8 = 1024 (8 accumulators x 16 unrolled dp4a x 8 ops)
 */

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 610)
#error "UNSUPPORTED: CUDA compute_int8 requires SM 6.1+ for native dp4a.u32.u32"
#endif

typedef unsigned int u32;

static __device__ __forceinline__ u32 dp4a_u8(u32 a, u32 b, u32 c) {
    u32 d;
    asm volatile("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

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

    /* Non-zero packed unsigned bytes in [1, 127] keep dot4 active every step. */
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
            r0 = dp4a_u8(r0, mul_pack, r0);
            r1 = dp4a_u8(r1, mul_pack, r1);
            r2 = dp4a_u8(r2, mul_pack, r2);
            r3 = dp4a_u8(r3, mul_pack, r3);
            r4 = dp4a_u8(r4, mul_pack, r4);
            r5 = dp4a_u8(r5, mul_pack, r5);
            r6 = dp4a_u8(r6, mul_pack, r6);
            r7 = dp4a_u8(r7, mul_pack, r7);
        }
    }

    out_buf[gid] = (r0 ^ r1) + (r2 ^ r3) + (r4 ^ r5) + (r6 ^ r7);
}
