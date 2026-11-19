/*
 * cross-gpu kernel: memory_pattern (metal)
 * Entry point(s): kernel_memory_pattern_write, kernel_memory_pattern_verify
 * ABI write:  (device uint* buf, constant ulong& num_words, constant uint& pattern_mode, constant ulong& seed)
 * ABI verify: (device const uint* buf, constant ulong& num_words, constant uint& pattern_mode, constant ulong& seed, device atomic_uint* error_count)
 * Arch / Extension guard: all Metal devices
 * BYTES_PER_ELEMENT: 4 (1 x uint32 word per write or verify pass)
 */

#include <metal_stdlib>
using namespace metal;

static inline uint pattern_word(ulong idx, uint mode, ulong seed) {
    if (mode == 0u) return 0x00000000u;
    if (mode == 1u) return 0xFFFFFFFFu;
    if (mode == 2u) return 0x55555555u;
    if (mode == 3u) return 0xAAAAAAAAu;
    if (mode == 4u) return 1u << uint(idx & 31UL);
    if (mode == 5u) return ~(1u << uint(idx & 31UL));

    uint z = uint(idx ^ seed ^ (seed >> 32)) + 0x9E3779B9u;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

kernel void kernel_memory_pattern_write(
    device uint* buf [[buffer(0)]],
    constant ulong& num_words [[buffer(1)]],
    constant uint& pattern_mode [[buffer(2)]],
    constant ulong& seed [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
    for (ulong idx = ulong(gid); idx < num_words; idx += ulong(grid_size)) {
        buf[idx] = pattern_word(idx, pattern_mode, seed);
    }
}

kernel void kernel_memory_pattern_verify(
    device const uint* buf [[buffer(0)]],
    constant ulong& num_words [[buffer(1)]],
    constant uint& pattern_mode [[buffer(2)]],
    constant ulong& seed [[buffer(3)]],
    device atomic_uint* error_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
    uint local_errors = 0u;
    for (ulong idx = ulong(gid); idx < num_words; idx += ulong(grid_size)) {
        if (buf[idx] != pattern_word(idx, pattern_mode, seed)) {
            ++local_errors;
        }
    }
    if (local_errors > 0u) {
        atomic_fetch_add_explicit(error_count, local_errors, memory_order_relaxed);
    }
}
