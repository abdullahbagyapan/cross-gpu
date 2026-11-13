#include <cuda_runtime.h>
#include "crossgpu_backend.h"
#include <math.h>
#include <stdio.h>

#define CG_BACKEND_NAME "cuda"
#define CG_BACKEND_KIND CROSSGPU_BACKEND_CUDA
#define CG_VENDOR_ID CROSSGPU_VENDOR_NVIDIA
#define CG_VENDOR_NAME "NVIDIA"
#define CG_ERROR_T cudaError_t
#define CG_ERROR_SUCCESS cudaSuccess
#define CG_GET_ERROR_STRING cudaGetErrorString
#define CG_GET_LAST_ERROR cudaGetLastError
#define CG_GET_DEVICE_COUNT cudaGetDeviceCount
#define CG_GET_DEVICE_PROPERTIES cudaGetDeviceProperties
#define CG_GET_PCI_BUS_ID cudaDeviceGetPCIBusId
#define CG_SET_DEVICE cudaSetDevice
#define CG_STREAM_T cudaStream_t
#define CG_STREAM_CREATE(stream) cudaStreamCreateWithFlags((stream), cudaStreamNonBlocking)
#define CG_STREAM_DESTROY cudaStreamDestroy
#define CG_STREAM_SYNCHRONIZE cudaStreamSynchronize
#define CG_MALLOC cudaMalloc
#define CG_FREE cudaFree
#define CG_MEMSET_ASYNC cudaMemsetAsync
#define CG_MEMCPY_ASYNC cudaMemcpyAsync
#define CG_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define CG_MEMCPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice
#define CG_GET_RUNTIME_VERSION cudaRuntimeGetVersion
#define CG_GET_DRIVER_VERSION cudaDriverGetVersion
#define CG_DEVICE_PROPS_T cudaDeviceProp

__global__ void cg_initialize_float_kernel(float *data, uint32_t n);
__global__ void cg_compute_kernel(float *data, uint32_t *errors, uint32_t n);
__global__ void cg_pattern_write_kernel(uint32_t *data, uint32_t n, uint32_t seed);
__global__ void cg_pattern_verify_kernel(const uint32_t *data, uint32_t *errors, uint32_t n, uint32_t seed);
__global__ void cg_copy_kernel(const uint32_t *source, uint32_t *destination, uint32_t n);

#include "../common/runtime_backend_impl.hpp"

__global__ void cg_initialize_float_kernel(float *data, uint32_t n)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        data[index] = 0.5f;
}

__global__ void cg_compute_kernel(float *data, uint32_t *errors, uint32_t n)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        float value = data[index];
        for (uint32_t i = 0; i < CG_COMPUTE_INNER_ITERATIONS; i++)
            value = fmaf(value, 0.99999f, 0.00001f);
        data[index] = value;
        if (!(value >= 0.0f && value <= 1.01f))
            atomicAdd(errors, 1u);
    }
}

__global__ void cg_pattern_write_kernel(uint32_t *data, uint32_t n, uint32_t seed)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        data[index] = (index * 1664525u + seed * 1013904223u) ^ 0xA5A55A5Au;
}

__global__ void cg_pattern_verify_kernel(const uint32_t *data, uint32_t *errors,
                                         uint32_t n, uint32_t seed)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        uint32_t expected = (index * 1664525u + seed * 1013904223u) ^ 0xA5A55A5Au;
        if (data[index] != expected)
            atomicAdd(errors, 1u);
    }
}

__global__ void cg_copy_kernel(const uint32_t *source, uint32_t *destination, uint32_t n)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        destination[index] = source[index];
}
