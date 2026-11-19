// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_tensor_fp32 (opencl)
 * Entry point(s): kernel_compute_tensor_fp32
 * ABI: (__global float* out_buf, uint num_elements, uint iterations, float seed_a, float seed_b)
 * Arch / Extension guard: cl_intel_subgroup_matrix_multiply_accumulate_tf32
 * OPS_PER_THREAD_PER_ITER: 2 * 8 * (2 * 8 * 8) = 2048 (16 DPAS TF32 k8 ops x 128 FLOPs/lane at M=8)
 */

#if !defined(cl_intel_subgroup_matrix_multiply_accumulate) || \
    !defined(cl_intel_subgroup_matrix_multiply_accumulate_tf32)
#error "UNSUPPORTED: OpenCL compute_tensor_fp32 requires cl_intel_subgroup_matrix_multiply_accumulate_tf32"
#endif

__kernel void kernel_compute_tensor_fp32(
    __global float* restrict out_buf,
    const uint num_elements,
    const uint iterations,
    const float seed_a,
    const float seed_b) {
    const uint gid = (uint)get_global_id(0);

    const float4 a_val = (float4)(1.0f);
    const float8 b_pos = (float8)(0.5f);
    const float8 b_neg = (float8)(-0.5f);

    const float base = 0.5f + 0.25f * (seed_a - floor(seed_a)) +
                       0.25f * (seed_b - floor(seed_b)) + 0.0625f * (float)(gid & 15u);

    float8 c0 = (float8)(base, base + 0.125f, base + 0.25f, base + 0.375f,
                         base + 0.5f, base + 0.625f, base + 0.75f, base + 0.875f);
    float8 c1 = (float8)(base + 1.0f, base + 1.125f, base + 1.25f, base + 1.375f,
                         base + 1.5f, base + 1.625f, base + 1.75f, base + 1.875f);

    for (uint it = 0u; it < iterations; ++it) {
        #pragma unroll
        for (uint pair = 0u; pair < 4u; ++pair) {
            c0 = intel_sub_group_tf32_tf32_matrix_mad_k8(a_val, b_pos, c0);
            c1 = intel_sub_group_tf32_tf32_matrix_mad_k8(a_val, b_pos, c1);
            c0 = intel_sub_group_tf32_tf32_matrix_mad_k8(a_val, b_neg, c0);
            c1 = intel_sub_group_tf32_tf32_matrix_mad_k8(a_val, b_neg, c1);
        }
    }

    if (gid < num_elements) {
        const float sum = (c0.s0 + c0.s1 + c0.s2 + c0.s3 + c0.s4 + c0.s5 + c0.s6 + c0.s7) +
                          (c1.s0 + c1.s1 + c1.s2 + c1.s3 + c1.s4 + c1.s5 + c1.s6 + c1.s7);
        out_buf[gid] = isfinite(sum) ? sum : as_float(0x7FC00000u);
    }
}
