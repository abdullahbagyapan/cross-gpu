#ifndef CROSSGPU_OPENCL_MINIMAL_H
#define CROSSGPU_OPENCL_MINIMAL_H

/*
 * Minimal OpenCL 1.2 ABI declarations used by the dynamically-loaded OpenCL
 * backend. This avoids requiring OpenCL development headers at build time.
 * The declarations follow the Khronos OpenCL C API ABI.
 */
#include <stddef.h>
#include <stdint.h>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef intptr_t cl_context_properties;
typedef uint32_t cl_bool;
typedef uint32_t cl_device_info;
typedef uint32_t cl_program_build_info;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_event *cl_event;

#define CL_SUCCESS                         0
#define CL_DEVICE_NOT_FOUND               -1
#define CL_BUILD_PROGRAM_FAILURE         -11
#define CL_PLATFORM_NOT_FOUND_KHR      -1001

#define CL_FALSE                           0
#define CL_TRUE                            1
#define CL_DEVICE_TYPE_GPU       (1ULL << 2)
#define CL_MEM_READ_WRITE        (1ULL << 0)

#define CL_CONTEXT_PLATFORM              0x1084
#define CL_DEVICE_VENDOR_ID              0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS      0x1002
#define CL_DEVICE_MAX_CLOCK_FREQUENCY    0x100C
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE    0x1010
#define CL_DEVICE_GLOBAL_MEM_SIZE       0x101F
#define CL_DEVICE_AVAILABLE             0x1027
#define CL_DEVICE_NAME                  0x102B
#define CL_DEVICE_VENDOR                0x102C
#define CL_DRIVER_VERSION               0x102D
#define CL_DEVICE_VERSION               0x102F
#define CL_DEVICE_EXTENSIONS            0x1030
#define CL_DEVICE_OPENCL_C_VERSION      0x103D
#define CL_DEVICE_PCI_BUS_INFO_KHR       0x410F
#define CL_PROGRAM_BUILD_LOG            0x1183

#endif
