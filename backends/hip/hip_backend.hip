#include <hip/hip_runtime.h>
#include "crossgpu_backend.h"
#include <math.h>
#include <stdio.h>

#define CG_BACKEND_NAME "hip"
#define CG_BACKEND_KIND CROSSGPU_BACKEND_HIP
#define CG_VENDOR_ID CROSSGPU_VENDOR_AMD
#define CG_VENDOR_NAME "AMD"
#define CG_ERROR_T hipError_t
#define CG_ERROR_SUCCESS hipSuccess
#define CG_GET_ERROR_STRING hipGetErrorString
#define CG_GET_LAST_ERROR hipGetLastError
#define CG_GET_DEVICE_COUNT hipGetDeviceCount
#define CG_GET_DEVICE_PROPERTIES hipGetDeviceProperties
#define CG_GET_PCI_BUS_ID hipDeviceGetPCIBusId
#define CG_SET_DEVICE hipSetDevice
#define CG_STREAM_T hipStream_t
#define CG_STREAM_CREATE(stream) hipStreamCreateWithFlags((stream), hipStreamNonBlocking)
#define CG_STREAM_DESTROY hipStreamDestroy
#define CG_STREAM_SYNCHRONIZE hipStreamSynchronize
#define CG_MALLOC hipMalloc
#define CG_FREE hipFree
#define CG_MEMSET_ASYNC hipMemsetAsync
#define CG_MEMCPY_ASYNC hipMemcpyAsync
#define CG_MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define CG_MEMCPY_DEVICE_TO_DEVICE hipMemcpyDeviceToDevice
#define CG_GET_RUNTIME_VERSION hipRuntimeGetVersion
#define CG_GET_DRIVER_VERSION hipDriverGetVersion
#define CG_DEVICE_PROPS_T hipDeviceProp_t

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
