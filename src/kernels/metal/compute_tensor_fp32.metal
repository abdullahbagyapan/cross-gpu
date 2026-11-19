// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: compute_tensor_fp32 (metal)
 * Entry point(s): kernel_compute_tensor_fp32
 * ABI: (device float* out_buf, constant uint& num_elements, constant uint& iterations, constant float& seed_a, constant float& seed_b)
 * Arch / Extension guard: Metal Apple7+ (<metal_simdgroup_matrix> simdgroup_float8x8)
 * OPS_PER_THREAD_PER_ITER: 4 * 8 * (2 * 8 * 8 * 8 / 32) = 1024 (32 8x8x8 MMAs x 32 FLOPs/lane)
 */

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void kernel_compute_tensor_fp32(
    device float* out_buf [[buffer(0)]],
    constant uint& num_elements [[buffer(1)]],
    constant uint& iterations [[buffer(2)]],
    constant float& seed_a [[buffer(3)]],
    constant float& seed_b [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint sgl [[thread_index_in_simdgroup]]) {
    const float base = 0.5f + 0.25f * (seed_a - floor(seed_a)) +
                       0.25f * (seed_b - floor(seed_b));

    const simdgroup_float8x8 a = make_filled_simdgroup_matrix<float, 8, 8>(1.0f);
    const simdgroup_float8x8 b_pos = make_filled_simdgroup_matrix<float, 8, 8>(0.5f);
    const simdgroup_float8x8 b_neg = make_filled_simdgroup_matrix<float, 8, 8>(-0.5f);

    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(base);
    simdgroup_float8x8 c1 = make_filled_simdgroup_matrix<float, 8, 8>(base + 0.25f);
    simdgroup_float8x8 c2 = make_filled_simdgroup_matrix<float, 8, 8>(base + 0.5f);
    simdgroup_float8x8 c3 = make_filled_simdgroup_matrix<float, 8, 8>(base + 0.75f);

    for (uint it = 0u; it < iterations; ++it) {
        #pragma unroll
        for (uint pair = 0u; pair < 4u; ++pair) {
            simdgroup_multiply_accumulate(c0, a, b_pos, c0);
            simdgroup_multiply_accumulate(c1, a, b_pos, c1);
            simdgroup_multiply_accumulate(c2, a, b_pos, c2);
            simdgroup_multiply_accumulate(c3, a, b_pos, c3);

            simdgroup_multiply_accumulate(c0, a, b_neg, c0);
            simdgroup_multiply_accumulate(c1, a, b_neg, c1);
            simdgroup_multiply_accumulate(c2, a, b_neg, c2);
            simdgroup_multiply_accumulate(c3, a, b_neg, c3);
        }
    }

    /* Per-simdgroup scratch (up to 32 simdgroups = 1024 threads/threadgroup). */
    threadgroup float scratch[32 * 64];
    threadgroup float* tile = scratch + (sgid & 31u) * 64u;

    float lane_sum = 0.0f;
    /* Threadgroup barriers ensure simdgroup_store visibility and prevent WAR hazards on reused tile scratch. */
    simdgroup_store(c0, tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lane_sum += tile[sgl] + tile[sgl + 32u];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_store(c1, tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lane_sum += tile[sgl] + tile[sgl + 32u];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_store(c2, tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lane_sum += tile[sgl] + tile[sgl + 32u];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_store(c3, tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lane_sum += tile[sgl] + tile[sgl + 32u];

    if (gid < num_elements) {
        out_buf[gid] = isfinite(lane_sum) ? lane_sum : as_type<float>(0x7FC00000u);
    }
}
