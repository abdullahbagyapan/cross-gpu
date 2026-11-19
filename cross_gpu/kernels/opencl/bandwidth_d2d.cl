/*
 * cross-gpu kernel: bandwidth_d2d (opencl)
 * Entry point(s): kernel_bandwidth_d2d
 * ABI: (__global uint4* dst, __global const uint4* src, ulong num_vec4)
 * Arch / Extension guard: OpenCL C 1.2/3.0 (coalesced 128-bit streaming copy)
 * BYTES_PER_ELEMENT: 32 (16-byte uint4 read + 16-byte uint4 write)
 */

__kernel void kernel_bandwidth_d2d(
    __global uint4* restrict dst,
    __global const uint4* restrict src,
    const ulong num_vec4) {
    const ulong gid = (ulong)get_global_id(0);
    const ulong stride = (ulong)get_global_size(0);
    for (ulong idx = gid; idx < num_vec4; idx += stride) {
        dst[idx] = src[idx];
    }
}
