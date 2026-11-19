/*
 * cross-gpu kernel: memory_pattern (opencl)
 * Entry point(s): kernel_memory_pattern_write, kernel_memory_pattern_verify
 * ABI write:  (__global uint* buf, ulong num_words, uint pattern_mode, ulong seed)
 * ABI verify: (__global const uint* buf, ulong num_words, uint pattern_mode, ulong seed, __global volatile uint* error_count)
 * Arch / Extension guard: OpenCL C 1.2/3.0
 * BYTES_PER_ELEMENT: 4 (1 x uint32 word per write or verify pass)
 */

static inline uint pattern_word(ulong idx, uint mode, ulong seed) {
    if (mode == 0u) return 0x00000000u;
    if (mode == 1u) return 0xFFFFFFFFu;
    if (mode == 2u) return 0x55555555u;
    if (mode == 3u) return 0xAAAAAAAAu;
    if (mode == 4u) return 1u << (uint)(idx & 31UL);
    if (mode == 5u) return ~(1u << (uint)(idx & 31UL));

    uint z = (uint)(idx ^ seed ^ (seed >> 32)) + 0x9E3779B9u;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

__kernel void kernel_memory_pattern_write(
    __global uint* restrict buf,
    const ulong num_words,
    const uint pattern_mode,
    const ulong seed) {
    const ulong gid = (ulong)get_global_id(0);
    const ulong stride = (ulong)get_global_size(0);
    for (ulong idx = gid; idx < num_words; idx += stride) {
        buf[idx] = pattern_word(idx, pattern_mode, seed);
    }
}

__kernel void kernel_memory_pattern_verify(
    __global const uint* restrict buf,
    const ulong num_words,
    const uint pattern_mode,
    const ulong seed,
    __global volatile uint* error_count) {
    const ulong gid = (ulong)get_global_id(0);
    const ulong stride = (ulong)get_global_size(0);
    uint local_errors = 0u;
    for (ulong idx = gid; idx < num_words; idx += stride) {
        if (buf[idx] != pattern_word(idx, pattern_mode, seed)) {
            ++local_errors;
        }
    }
    if (local_errors > 0u) {
        atomic_add(error_count, local_errors);
    }
}
