// SPDX-License-Identifier: MIT
/*
 * cross-gpu kernel: memory_sweep (levelzero)
 * Entry point(s): kernel_memory_sweep_write, kernel_memory_sweep_verify
 * ABI write:  (__global ulong* buf, ulong num_qwords, ulong seed)
 * ABI verify: (__global const ulong* buf, ulong num_qwords, ulong seed, __global volatile uint* error_count)
 * Arch / Extension guard: OpenCL C / SPIR-V (optional cl_khr_int64_base_atomics)
 * BYTES_PER_ELEMENT: 8 (1 x uint64 qword per write or verify pass)
 */

#ifdef cl_khr_int64_base_atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

static inline ulong splitmix64(ulong x) {
    ulong z = x + 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

__kernel void kernel_memory_sweep_write(
    __global ulong* restrict buf,
    const ulong num_qwords,
    const ulong seed) {
    const ulong gid = (ulong)get_global_id(0);
    const ulong stride = (ulong)get_global_size(0);
    for (ulong idx = gid; idx < num_qwords; idx += stride) {
        buf[idx] = splitmix64(idx ^ seed);
    }
}

__kernel void kernel_memory_sweep_verify(
    __global const ulong* restrict buf,
    const ulong num_qwords,
    const ulong seed,
    __global volatile uint* error_count) {
    const ulong gid = (ulong)get_global_id(0);
    const ulong stride = (ulong)get_global_size(0);
    uint local_errors = 0u;
    for (ulong idx = gid; idx < num_qwords; idx += stride) {
        if (buf[idx] != splitmix64(idx ^ seed)) {
            ++local_errors;
        }
    }
    if (local_errors > 0u) {
        atomic_add(error_count, local_errors);
    }
}
