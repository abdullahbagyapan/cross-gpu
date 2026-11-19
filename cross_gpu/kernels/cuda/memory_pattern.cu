// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: memory_pattern (cuda)
 * Entry point(s): kernel_memory_pattern_write, kernel_memory_pattern_verify
 * ABI write:  (uint32_t* buf, uint64_t num_words, uint32_t pattern_mode, uint64_t seed)
 * ABI verify: (const uint32_t* buf, uint64_t num_words, uint32_t pattern_mode, uint64_t seed, uint32_t* error_count)
 * Arch / Extension guard: all CUDA architectures
 * BYTES_PER_ELEMENT: 4 (1 x uint32_t word per write or verify pass)
 */

typedef unsigned long long u64;
typedef unsigned int u32;

static __device__ __forceinline__ u32 pattern_word(u64 idx, u32 mode, u64 seed) {
    if (mode == 0u) return 0x00000000u;
    if (mode == 1u) return 0xFFFFFFFFu;
    if (mode == 2u) return 0x55555555u;
    if (mode == 3u) return 0xAAAAAAAAu;
    if (mode == 4u) return 1u << (u32)(idx & 31ULL);
    if (mode == 5u) return ~(1u << (u32)(idx & 31ULL));

    /* Mode 6: deterministic SplitMix32/Murmur3 finalizer keyed by (idx, seed). */
    u32 z = (u32)(idx ^ seed ^ (seed >> 32)) + 0x9E3779B9u;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

extern "C" __global__ void kernel_memory_pattern_write(
    u32* __restrict__ buf,
    u64 num_words,
    u32 pattern_mode,
    u64 seed) {
    const u64 gid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    const u64 stride = (u64)gridDim.x * (u64)blockDim.x;
    for (u64 idx = gid; idx < num_words; idx += stride) {
        buf[idx] = pattern_word(idx, pattern_mode, seed);
    }
}

extern "C" __global__ void kernel_memory_pattern_verify(
    const u32* __restrict__ buf,
    u64 num_words,
    u32 pattern_mode,
    u64 seed,
    u32* __restrict__ error_count) {
    const u64 gid = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    const u64 stride = (u64)gridDim.x * (u64)blockDim.x;
    u32 local_errors = 0u;
    for (u64 idx = gid; idx < num_words; idx += stride) {
        if (buf[idx] != pattern_word(idx, pattern_mode, seed)) {
            ++local_errors;
        }
    }
    if (local_errors > 0u) {
        atomicAdd(error_count, local_errors);
    }
}
