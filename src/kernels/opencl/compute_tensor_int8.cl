// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_tensor_int8 (opencl)
 * Entry point(s): kernel_compute_tensor_int8
 * ABI: (__global uint* out_buf, uint num_elements, uint iterations, uint seed_a, uint seed_b)
 * Arch / Extension guard: cl_intel_subgroup_matrix_multiply_accumulate
 * OPS_PER_THREAD_PER_ITER: 2 * 8 * (2 * 8 * 32) = 8192 (16 DPAS i8 k32 ops x 512 ops/lane at M=8)
 */

#ifndef cl_intel_subgroup_matrix_multiply_accumulate
#error "UNSUPPORTED: OpenCL compute_tensor_int8 requires cl_intel_subgroup_matrix_multiply_accumulate"
#endif

__kernel void kernel_compute_tensor_int8(
    __global uint* restrict out_buf,
    const uint num_elements,
    const uint iterations,
    const uint seed_a,
    const uint seed_b) {
    const uint gid = (uint)get_global_id(0);

    const short8 a_val = (short8)(0x0101);
    const int8 b_pos = (int8)(0x01010101);
    const int8 b_neg = (int8)((int)0xFFFFFFFFu);

    const int base = (int)((gid ^ seed_a ^ seed_b) & 0xFFFFu);
    int8 c0 = (int8)(base + 1, base + 2, base + 3, base + 4,
                     base + 5, base + 6, base + 7, base + 8);
    int8 c1 = (int8)(base + 9, base + 10, base + 11, base + 12,
                     base + 13, base + 14, base + 15, base + 16);

    for (uint it = 0u; it < iterations; ++it) {
        #pragma unroll
        for (uint pair = 0u; pair < 4u; ++pair) {
            c0 = intel_sub_group_i8_i8_matrix_mad_k32(a_val, b_pos, c0);
            c1 = intel_sub_group_i8_i8_matrix_mad_k32(a_val, b_pos, c1);
            c0 = intel_sub_group_i8_i8_matrix_mad_k32(a_val, b_neg, c0);
            c1 = intel_sub_group_i8_i8_matrix_mad_k32(a_val, b_neg, c1);
        }
    }

    if (gid < num_elements) {
        const uint sum = (uint)c0.s0 + (uint)c0.s1 + (uint)c0.s2 + (uint)c0.s3 +
                         (uint)c0.s4 + (uint)c0.s5 + (uint)c0.s6 + (uint)c0.s7 +
                         (uint)c1.s0 + (uint)c1.s1 + (uint)c1.s2 + (uint)c1.s3 +
                         (uint)c1.s4 + (uint)c1.s5 + (uint)c1.s6 + (uint)c1.s7;
        out_buf[gid] = sum;
    }
}
