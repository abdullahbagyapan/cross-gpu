// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: memory_sweep (cuda)
 * Entry point(s): kernel_memory_sweep_write, kernel_memory_sweep_verify
 * ABI write:  (uint64_t* buf, uint64_t num_qwords, uint64_t seed)
 * ABI verify: (const uint64_t* buf, uint64_t num_qwords, uint64_t seed, uint32_t* error_count)
 * Arch / Extension guard: all CUDA architectures
 * BYTES_PER_ELEMENT: 8 (1 x uint64_t qword per write or verify pass)
 */

typedef unsigned long long u64;
typedef unsigned int u32;

static __device__ __forceinline__ u64 splitmix64(u64 x) {
    u64 z = x + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

extern "C" __global__ void kernel_memory_sweep_write(
    u64* __restrict__ buf,
    u64 num_qwords,
    u64 seed) {
    const u64 gid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    const u64 stride = (u64)gridDim.x * (u64)blockDim.x;
    for (u64 idx = gid; idx < num_qwords; idx += stride) {
        buf[idx] = splitmix64(idx ^ seed);
    }
}

extern "C" __global__ void kernel_memory_sweep_verify(
    const u64* __restrict__ buf,
    u64 num_qwords,
    u64 seed,
    u32* __restrict__ error_count) {
    const u64 gid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    const u64 stride = (u64)gridDim.x * (u64)blockDim.x;
    u32 local_errors = 0u;
    for (u64 idx = gid; idx < num_qwords; idx += stride) {
        if (buf[idx] != splitmix64(idx ^ seed)) {
            ++local_errors;
        }
    }
    if (local_errors > 0u) {
        atomicAdd(error_count, local_errors);
    }
}
