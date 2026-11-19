/*
 * cross-gpu kernel: memory_sweep (metal)
 * Entry point(s): kernel_memory_sweep_write, kernel_memory_sweep_verify
 * ABI write:  (device ulong* buf, constant ulong& num_qwords, constant ulong& seed)
 * ABI verify: (device const ulong* buf, constant ulong& num_qwords, constant ulong& seed, device atomic_uint* error_count)
 * Arch / Extension guard: Metal 2.3+ (native 64-bit ulong)
 * BYTES_PER_ELEMENT: 8 (1 x uint64 qword per write or verify pass)
 */

#include <metal_stdlib>
using namespace metal;

static inline ulong splitmix64(ulong x) {
    ulong z = x + 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

kernel void kernel_memory_sweep_write(
    device ulong* buf [[buffer(0)]],
    constant ulong& num_qwords [[buffer(1)]],
    constant ulong& seed [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
    for (ulong idx = ulong(gid); idx < num_qwords; idx += ulong(grid_size)) {
        buf[idx] = splitmix64(idx ^ seed);
    }
}

kernel void kernel_memory_sweep_verify(
    device const ulong* buf [[buffer(0)]],
    constant ulong& num_qwords [[buffer(1)]],
    constant ulong& seed [[buffer(2)]],
    device atomic_uint* error_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
    uint local_errors = 0u;
    for (ulong idx = ulong(gid); idx < num_qwords; idx += ulong(grid_size)) {
        if (buf[idx] != splitmix64(idx ^ seed)) {
            ++local_errors;
        }
    }
    if (local_errors > 0u) {
        atomic_fetch_add_explicit(error_count, local_errors, memory_order_relaxed);
    }
}
