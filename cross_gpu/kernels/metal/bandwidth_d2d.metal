/*
 * cross-gpu kernel: bandwidth_d2d (metal)
 * Entry point(s): kernel_bandwidth_d2d
 * ABI: (device uint4* dst, device const uint4* src, constant ulong& num_vec4)
 * Arch / Extension guard: all Metal devices (coalesced 128-bit uint4 streaming copy)
 * BYTES_PER_ELEMENT: 32 (16-byte uint4 read + 16-byte uint4 write)
 */

#include <metal_stdlib>
using namespace metal;

kernel void kernel_bandwidth_d2d(
    device uint4* dst [[buffer(0)]],
    device const uint4* src [[buffer(1)]],
    constant ulong& num_vec4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
    for (ulong idx = ulong(gid); idx < num_vec4; idx += ulong(grid_size)) {
        dst[idx] = src[idx];
    }
}
