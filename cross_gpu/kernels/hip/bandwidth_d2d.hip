/*
 * cross-gpu kernel: bandwidth_d2d (hip)
 * Entry point(s): kernel_bandwidth_d2d
 * ABI: (uint4* dst, const uint4* src, uint64_t num_vec4)
 * Arch / Extension guard: all HIP/AMDGCN architectures (coalesced 128-bit streaming copy)
 * BYTES_PER_ELEMENT: 32 (16-byte uint4 read + 16-byte uint4 write)
 */

typedef unsigned long long u64;

extern "C" __global__ void kernel_bandwidth_d2d(
    uint4* __restrict__ dst,
    const uint4* __restrict__ src,
    u64 num_vec4) {
    const u64 gid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    const u64 stride = (u64)gridDim.x * (u64)blockDim.x;
    for (u64 idx = gid; idx < num_vec4; idx += stride) {
        dst[idx] = src[idx];
    }
}
