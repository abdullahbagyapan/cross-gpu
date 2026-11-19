/*
 * cross-gpu kernel: compute_tensor_int8 (hip)
 * Entry point(s): kernel_compute_tensor_int8
 * ABI: (uint32_t* out_buf, uint32_t num_elements, uint32_t iterations, uint32_t seed_a, uint32_t seed_b)
 * Arch / Extension guard: CDNA (mfma_i32_16x16x16i8) or RDNA3/4 (wmma_i32_16x16x16_iu8)
 * OPS_PER_THREAD_PER_ITER: 4096 (CDNA: 32 x 128 ops/lane; RDNA3/4 wave32: 16 x 256 ops/lane)
 */

#if !defined(__gfx908__) && !defined(__gfx90a__) && !defined(__gfx940__) && \
    !defined(__gfx941__) && !defined(__gfx942__) && !defined(__gfx950__) && \
    !defined(__gfx1100__) && !defined(__gfx1101__) && !defined(__gfx1102__) && \
    !defined(__gfx1103__) && !defined(__gfx1150__) && !defined(__gfx1151__) && \
    !defined(__gfx1200__) && !defined(__gfx1201__)
#error "UNSUPPORTED: HIP compute_tensor_int8 requires CDNA (gfx908+) or RDNA3+ (gfx1100+) INT8 matrix cores"
#endif

typedef unsigned int u32;
typedef int i32;
typedef int i32x2_t __attribute__((ext_vector_type(2)));
typedef int i32x4_t __attribute__((ext_vector_type(4)));
typedef int i32x8_t __attribute__((ext_vector_type(8)));

extern "C" __global__ void kernel_compute_tensor_int8(
    u32* __restrict__ out_buf,
    u32 num_elements,
    u32 iterations,
    u32 seed_a,
    u32 seed_b) {
    const u32 gid = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 base = (i32)((gid ^ seed_a ^ seed_b) & 0xFFFFu);

#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || \
    defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
    const i32 a_val = 0x01010101;
    const i32 b_pos = 0x01010101;
    const i32 b_neg = (i32)0xFFFFFFFFu;

    i32x4_t c0 = {base + 1, base + 2, base + 3, base + 4};
    i32x4_t c1 = {base + 5, base + 6, base + 7, base + 8};
    i32x4_t c2 = {base + 9, base + 10, base + 11, base + 12};
    i32x4_t c3 = {base + 13, base + 14, base + 15, base + 16};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            c0 = __builtin_amdgcn_mfma_i32_16x16x16i8(a_val, b_pos, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_i32_16x16x16i8(a_val, b_pos, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_i32_16x16x16i8(a_val, b_pos, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_i32_16x16x16i8(a_val, b_pos, c3, 0, 0, 0);

            c0 = __builtin_amdgcn_mfma_i32_16x16x16i8(a_val, b_neg, c0, 0, 0, 0);
            c1 = __builtin_amdgcn_mfma_i32_16x16x16i8(a_val, b_neg, c1, 0, 0, 0);
            c2 = __builtin_amdgcn_mfma_i32_16x16x16i8(a_val, b_neg, c2, 0, 0, 0);
            c3 = __builtin_amdgcn_mfma_i32_16x16x16i8(a_val, b_neg, c3, 0, 0, 0);
        }
    }

    if (gid < num_elements) {
        const u32 sum = (u32)c0[0] + (u32)c0[1] + (u32)c0[2] + (u32)c0[3] +
                        (u32)c1[0] + (u32)c1[1] + (u32)c1[2] + (u32)c1[3] +
                        (u32)c2[0] + (u32)c2[1] + (u32)c2[2] + (u32)c2[3] +
                        (u32)c3[0] + (u32)c3[1] + (u32)c3[2] + (u32)c3[3];
        out_buf[gid] = sum;
    }
#elif defined(__gfx1200__) || defined(__gfx1201__)
    const i32x2_t a_val = {0x01010101, 0x01010101};
    const i32x2_t b_pos = {0x01010101, 0x01010101};
    const i32x2_t b_neg = {(i32)0xFFFFFFFFu, (i32)0xFFFFFFFFu};

    i32x8_t c0 = {base + 1, base + 2, base + 3, base + 4,
                  base + 5, base + 6, base + 7, base + 8};
    i32x8_t c1 = {base + 9, base + 10, base + 11, base + 12,
                  base + 13, base + 14, base + 15, base + 16};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_val, true, b_pos, c0, false);
            c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_val, true, b_pos, c1, false);
            c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_val, true, b_neg, c0, false);
            c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_val, true, b_neg, c1, false);
        }
    }

    if (gid < num_elements) {
        const u32 sum = (u32)c0[0] + (u32)c0[1] + (u32)c0[2] + (u32)c0[3] +
                        (u32)c0[4] + (u32)c0[5] + (u32)c0[6] + (u32)c0[7] +
                        (u32)c1[0] + (u32)c1[1] + (u32)c1[2] + (u32)c1[3] +
                        (u32)c1[4] + (u32)c1[5] + (u32)c1[6] + (u32)c1[7];
        out_buf[gid] = sum;
    }
#else
    const i32x4_t a_val = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
    const i32x4_t b_pos = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
    const i32x4_t b_neg = {(i32)0xFFFFFFFFu, (i32)0xFFFFFFFFu, (i32)0xFFFFFFFFu, (i32)0xFFFFFFFFu};

    i32x8_t c0 = {base + 1, base + 2, base + 3, base + 4,
                  base + 5, base + 6, base + 7, base + 8};
    i32x8_t c1 = {base + 9, base + 10, base + 11, base + 12,
                  base + 13, base + 14, base + 15, base + 16};

    for (u32 it = 0; it < iterations; ++it) {
        #pragma unroll
        for (u32 pair = 0; pair < 4u; ++pair) {
            c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_val, true, b_pos, c0, false);
            c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_val, true, b_pos, c1, false);
            c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_val, true, b_neg, c0, false);
            c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_val, true, b_neg, c1, false);
        }
    }

    if (gid < num_elements) {
        const u32 sum = (u32)c0[0] + (u32)c0[1] + (u32)c0[2] + (u32)c0[3] +
                        (u32)c0[4] + (u32)c0[5] + (u32)c0[6] + (u32)c0[7] +
                        (u32)c1[0] + (u32)c1[1] + (u32)c1[2] + (u32)c1[3] +
                        (u32)c1[4] + (u32)c1[5] + (u32)c1[6] + (u32)c1[7];
        out_buf[gid] = sum;
    }
#endif
}
