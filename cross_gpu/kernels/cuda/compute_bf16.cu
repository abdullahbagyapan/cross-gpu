// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_bf16 (cuda)
 * Entry point(s): kernel_compute_bf16
 * ABI: (float* out_buf, uint32_t num_elements, uint32_t iterations, float seed_a, float seed_b)
 * Arch / Extension guard: __CUDA_ARCH__ >= 800 (native fma.rn.bf16x2)
 * OPS_PER_THREAD_PER_ITER: 8 * 16 * 4 = 512 (8 bf16x2 accs x 16 FMAs x 2 lanes x 2 FLOPs)
 */

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 800)
#error "UNSUPPORTED: CUDA compute_bf16 requires SM 8.0+ for native fma.rn.bf16x2"
#endif

typedef unsigned int u32;

static __device__ __forceinline__ u32 pack_bf16x2(float lo, float hi) {
    u32 out;
    /* PTX cvt.rn.bf16x2.f32 packs operand 1 into upper 16 bits, operand 2 into lower 16 bits. */
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(hi), "f"(lo));
    return out;
}

static __device__ __forceinline__ u32 fma_bf16x2(u32 a, u32 b, u32 c) {
    u32 d;
    asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

static __device__ __forceinline__ float unpack_bf16x2_sum(u32 packed) {
    const float lo = __uint_as_float((packed & 0xFFFFu) << 16);
    const float hi = __uint_as_float(packed & 0xFFFF0000u);
    return lo + hi;
}

extern "C" __global__ void kernel_compute_bf16(
    float* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    float seed_a,
    float seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) {
        return;
    }

    const float a_f = fmaf(0.25f, seed_a - floorf(seed_a), 0.5f);
    const float bp_f = fmaf(0.25f, seed_b - floorf(seed_b), 0.25f);
    const u32 a = pack_bf16x2(a_f, a_f);
    const u32 b_pos = pack_bf16x2(bp_f, bp_f);
    const u32 b_neg = pack_bf16x2(-0.5f * bp_f, -0.5f * bp_f);

    u32 r0 = pack_bf16x2(0.125f + 0.0625f * (float)((gid + 0u) & 15u), 0.1875f + 0.0625f * (float)((gid + 0u) & 15u));
    u32 r1 = pack_bf16x2(0.125f + 0.0625f * (float)((gid + 1u) & 15u), 0.1875f + 0.0625f * (float)((gid + 1u) & 15u));
    u32 r2 = pack_bf16x2(0.125f + 0.0625f * (float)((gid + 2u) & 15u), 0.1875f + 0.0625f * (float)((gid + 2u) & 15u));
    u32 r3 = pack_bf16x2(0.125f + 0.0625f * (float)((gid + 3u) & 15u), 0.1875f + 0.0625f * (float)((gid + 3u) & 15u));
    u32 r4 = pack_bf16x2(0.125f + 0.0625f * (float)((gid + 4u) & 15u), 0.1875f + 0.0625f * (float)((gid + 4u) & 15u));
    u32 r5 = pack_bf16x2(0.125f + 0.0625f * (float)((gid + 5u) & 15u), 0.1875f + 0.0625f * (float)((gid + 5u) & 15u));
    u32 r6 = pack_bf16x2(0.125f + 0.0625f * (float)((gid + 6u) & 15u), 0.1875f + 0.0625f * (float)((gid + 6u) & 15u));
    u32 r7 = pack_bf16x2(0.125f + 0.0625f * (float)((gid + 7u) & 15u), 0.1875f + 0.0625f * (float)((gid + 7u) & 15u));

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 8u; ++pair) {
            r0 = fma_bf16x2(r0, a, b_pos);
            r1 = fma_bf16x2(r1, a, b_pos);
            r2 = fma_bf16x2(r2, a, b_pos);
            r3 = fma_bf16x2(r3, a, b_pos);
            r4 = fma_bf16x2(r4, a, b_pos);
            r5 = fma_bf16x2(r5, a, b_pos);
            r6 = fma_bf16x2(r6, a, b_pos);
            r7 = fma_bf16x2(r7, a, b_pos);

            r0 = fma_bf16x2(r0, a, b_neg);
            r1 = fma_bf16x2(r1, a, b_neg);
            r2 = fma_bf16x2(r2, a, b_neg);
            r3 = fma_bf16x2(r3, a, b_neg);
            r4 = fma_bf16x2(r4, a, b_neg);
            r5 = fma_bf16x2(r5, a, b_neg);
            r6 = fma_bf16x2(r6, a, b_neg);
            r7 = fma_bf16x2(r7, a, b_neg);
        }
    }

    const float s0 = unpack_bf16x2_sum(r0);
    const float s1 = unpack_bf16x2_sum(r1);
    const float s2 = unpack_bf16x2_sum(r2);
    const float s3 = unpack_bf16x2_sum(r3);
    const float s4 = unpack_bf16x2_sum(r4);
    const float s5 = unpack_bf16x2_sum(r5);
    const float s6 = unpack_bf16x2_sum(r6);
    const float s7 = unpack_bf16x2_sum(r7);

    const bool ok = isfinite(s0) && isfinite(s1) && isfinite(s2) && isfinite(s3) &&
                    isfinite(s4) && isfinite(s5) && isfinite(s6) && isfinite(s7);
    const float sum = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    out_buf[gid] = (ok && isfinite(sum)) ? sum : __uint_as_float(0x7FC00000u);
}
