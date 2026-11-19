// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_tensor_fp16 (levelzero)
 * Entry point(s): kernel_compute_tensor_fp16
 * ABI: (__global float* out_buf, uint num_elements, uint iterations, float seed_a, float seed_b)
 * Arch / Extension guard: cl_intel_subgroup_matrix_multiply_accumulate (Intel XMX / DPAS f16 k16)
 * OPS_PER_THREAD_PER_ITER: 2 * 8 * (2 * 8 * 16) = 4096 (16 DPAS f16 k16 ops x 256 FLOPs/lane at M=8)
 */

#ifndef cl_intel_subgroup_matrix_multiply_accumulate
#error "UNSUPPORTED: Level Zero compute_tensor_fp16 requires cl_intel_subgroup_matrix_multiply_accumulate"
#endif

__kernel void kernel_compute_tensor_fp16(
    __global float* restrict out_buf,
    const uint num_elements,
    const uint iterations,
    const float seed_a,
    const float seed_b) {
    const uint gid = (uint)get_global_id(0);

    /* Packed IEEE-754 binary16 constants: 0x3C00 = +1.0, 0x3800 = +0.5, 0xB800 = -0.5. */
    const short8 a_val = (short8)(0x3C00);
    const int8 b_pos = (int8)(0x38003800);
    const int8 b_neg = (int8)((int)0xB800B800u);

    const float base = 0.5f + 0.25f * (seed_a - floor(seed_a)) +
                       0.25f * (seed_b - floor(seed_b)) + 0.0625f * (float)(gid & 15u);

    float8 c0 = (float8)(base, base + 0.125f, base + 0.25f, base + 0.375f,
                         base + 0.5f, base + 0.625f, base + 0.75f, base + 0.875f);
    float8 c1 = (float8)(base + 1.0f, base + 1.125f, base + 1.25f, base + 1.375f,
                         base + 1.5f, base + 1.625f, base + 1.75f, base + 1.875f);

    for (uint it = 0u; it < iterations; ++it) {
        #pragma unroll
        for (uint pair = 0u; pair < 4u; ++pair) {
            c0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_pos, c0);
            c1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_pos, c1);
            c0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_neg, c0);
            c1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_neg, c1);
        }
    }

    if (gid < num_elements) {
        const float sum = (c0.s0 + c0.s1 + c0.s2 + c0.s3 + c0.s4 + c0.s5 + c0.s6 + c0.s7) +
                          (c1.s0 + c1.s1 + c1.s2 + c1.s3 + c1.s4 + c1.s5 + c1.s6 + c1.s7);
        out_buf[gid] = isfinite(sum) ? sum : as_float(0x7FC00000u);
    }
}
