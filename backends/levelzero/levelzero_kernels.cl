/*
 * CrossGPU Level Zero kernels.
 *
 * This source is compiled to generic SPIR-V with ocloc at build time. The
 * Level Zero driver then builds the SPIR-V module for each selected device.
 */

#define CG_COMPUTE_INNER_ITERATIONS 256u

__kernel void compute_kernel(__global float *data,
                             volatile __global uint *errors,
                             uint elements)
{
    const size_t index = get_global_id(0);
    if (index < (size_t)elements) {
        float value = data[index];
        for (uint i = 0; i < CG_COMPUTE_INNER_ITERATIONS; ++i)
            value = fma(value, 0.99999f, 0.00001f);
        data[index] = value;
        if (!(value >= 0.0f && value <= 1.01f))
            atomic_inc(errors);
    }
}

__kernel void pattern_write_kernel(__global uint *data,
                                   uint elements,
                                   uint seed)
{
    const size_t index = get_global_id(0);
    if (index < (size_t)elements)
        data[index] = ((uint)index * 1664525u + seed * 1013904223u) ^ 0xA5A55A5Au;
}

__kernel void pattern_verify_kernel(__global const uint *data,
                                    volatile __global uint *errors,
                                    uint elements,
                                    uint seed)
{
    const size_t index = get_global_id(0);
    if (index < (size_t)elements) {
        const uint expected = ((uint)index * 1664525u + seed * 1013904223u) ^ 0xA5A55A5Au;
        if (data[index] != expected)
            atomic_inc(errors);
    }
}
