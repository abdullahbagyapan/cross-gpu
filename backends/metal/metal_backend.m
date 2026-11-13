#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "crossgpu_backend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/sysctl.h>

#define METAL_BACKEND_NAME "metal"
#define METAL_MIN_WORKING_SET (UINT64_C(4) * UINT64_C(1024) * UINT64_C(1024))
#define METAL_MAX_WORKING_SET (UINT64_C(128) * UINT64_C(1024) * UINT64_C(1024))
#define METAL_COMPUTE_INNER_ITERATIONS 256u

static NSString *const metal_kernel_source =
@"#include <metal_stdlib>\n"
 @"using namespace metal;\n"
 @"kernel void cg_compute(device float *data [[buffer(0)]], device atomic_uint *errors [[buffer(1)]],\n"
 @"                       constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
 @"    if (gid >= n) return;\n"
 @"    float value = data[gid];\n"
 @"    for (uint i = 0; i < 256; ++i) value = fma(value, 0.99999f, 0.00001f);\n"
 @"    data[gid] = value;\n"
 @"    if (!(value >= 0.0f && value <= 1.01f)) atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);\n"
 @"}\n"
 @"kernel void cg_pattern_write(device uint *data [[buffer(0)]], constant uint &n [[buffer(1)]],\n"
 @"                             constant uint &seed [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
 @"    if (gid >= n) return;\n"
 @"    data[gid] = (gid * 1664525u + seed * 1013904223u) ^ 0xA5A55A5Au;\n"
 @"}\n"
 @"kernel void cg_pattern_verify(device const uint *data [[buffer(0)]], device atomic_uint *errors [[buffer(1)]],\n"
 @"                              constant uint &n [[buffer(2)]], constant uint &seed [[buffer(3)]],\n"
 @"                              uint gid [[thread_position_in_grid]]) {\n"
 @"    if (gid >= n) return;\n"
 @"    uint expected = (gid * 1664525u + seed * 1013904223u) ^ 0xA5A55A5Au;\n"
 @"    if (data[gid] != expected) atomic_fetch_add_explicit(errors, 1u, memory_order_relaxed);\n"
 @"}\n"
 @"kernel void cg_copy(device const uint *src [[buffer(0)]], device uint *dst [[buffer(1)]],\n"
 @"                    constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
 @"    if (gid < n) dst[gid] = src[gid];\n"
 @"}\n";

typedef struct metal_device_record {
    __strong id<MTLDevice> device;
    crossgpu_device_info_t info;
} metal_device_record_t;

typedef struct metal_device_context {
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    uint32_t device_index;
    __strong id<MTLComputePipelineState> compute_pipeline;
    __strong id<MTLComputePipelineState> pattern_write_pipeline;
    __strong id<MTLComputePipelineState> pattern_verify_pipeline;
    __strong id<MTLComputePipelineState> copy_pipeline;
} metal_device_context_t;

typedef struct metal_workload_context {
    __strong id<MTLBuffer> buffer_a;
    __strong id<MTLBuffer> buffer_b;
    __strong id<MTLBuffer> error_buffer;
    size_t bytes;
    uint32_t elements;
    uint32_t seed;
} metal_workload_context_t;

static __strong NSArray<id<MTLDevice>> *g_metal_devices;
static metal_device_record_t *g_metal_records;
static size_t g_metal_device_count;

static void metal_set_error(char *error, size_t error_size, const char *message)
{
    if (error != NULL && error_size > 0)
        (void)snprintf(error, error_size, "%s", message != NULL ? message : "unknown Metal error");
}

static void metal_set_nserror(char *error, size_t error_size, NSString *operation, NSError *ns_error)
{
    NSString *message = ns_error != nil
        ? [NSString stringWithFormat:@"%@ failed: %@", operation, ns_error.localizedDescription]
        : [NSString stringWithFormat:@"%@ failed", operation];
    metal_set_error(error, error_size, message.UTF8String);
}

static crossgpu_vendor_id_t metal_guess_vendor(NSString *name)
{
    NSString *lower = [name lowercaseString];
    if ([lower containsString:@"amd"] || [lower containsString:@"radeon"])
        return CROSSGPU_VENDOR_AMD;
    if ([lower containsString:@"intel"])
        return CROSSGPU_VENDOR_INTEL;
    if ([lower containsString:@"nvidia"] || [lower containsString:@"geforce"])
        return CROSSGPU_VENDOR_NVIDIA;
    if ([lower containsString:@"apple"])
        return CROSSGPU_VENDOR_APPLE;
    return CROSSGPU_VENDOR_UNKNOWN;
}

static int metal_initialize(char *error, size_t error_size)
{
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    NSUInteger count = devices.count;
    NSUInteger i;

    if (count == 0) {
        metal_set_error(error, error_size, "no Metal-compatible GPU devices found");
        return -1;
    }
    if ((uint64_t)count > SIZE_MAX / sizeof(*g_metal_records)) {
        metal_set_error(error, error_size, "Metal device count is too large");
        return -1;
    }

    metal_device_record_t *records = (metal_device_record_t *)calloc((size_t)count, sizeof(*records));
    if (records == NULL) {
        metal_set_error(error, error_size, "out of memory allocating Metal device registry");
        return -1;
    }

    for (i = 0; i < count; i++) {
        id<MTLDevice> device = devices[i];
        crossgpu_device_info_t *info = &records[i].info;
        NSString *name = device.name != nil ? device.name : @"Unknown Metal GPU";
        uint64_t registry_id = device.registryID;

        records[i].device = device;
        memset(info, 0, sizeof(*info));
        info->struct_size = (uint32_t)sizeof(*info);
        info->backend_device_index = (uint32_t)i;
        info->vendor_id = metal_guess_vendor(name);
        info->working_set_bytes = (uint64_t)device.recommendedMaxWorkingSetSize;
        info->max_allocation_bytes = (uint64_t)device.maxBufferLength;
        if (device.hasUnifiedMemory) {
            uint64_t physical_memory = 0;
            size_t physical_memory_size = sizeof(physical_memory);
            if (sysctlbyname("hw.memsize", &physical_memory,
                             &physical_memory_size, NULL, 0) == 0)
                info->physical_memory_bytes = physical_memory;
        }
        (void)snprintf(info->name, sizeof(info->name), "%s", name.UTF8String);
        (void)snprintf(info->vendor, sizeof(info->vendor), "%s",
                       info->vendor_id == CROSSGPU_VENDOR_AMD ? "AMD" :
                       info->vendor_id == CROSSGPU_VENDOR_INTEL ? "Intel" :
                       info->vendor_id == CROSSGPU_VENDOR_NVIDIA ? "NVIDIA" :
                       info->vendor_id == CROSSGPU_VENDOR_APPLE ? "Apple" : "Unknown");
        (void)snprintf(info->driver_version, sizeof(info->driver_version), "%s", "Metal system driver");
        (void)snprintf(info->api_version, sizeof(info->api_version), "%s", "Metal" );
        /* registryID is an OS-level device identity, not a product-name guess.
         * Other APIs may merge with it only if they can report this same
         * IORegistry identity in the same namespace. */
        if (registry_id != 0) {
            info->identity_kind = CROSSGPU_IDENTITY_OS_DEVICE;
            (void)snprintf(info->identity_namespace,
                           sizeof(info->identity_namespace), "%s",
                           CROSSGPU_IDENTITY_NAMESPACE_APPLE_IOREGISTRY);
            (void)snprintf(info->identity_value, sizeof(info->identity_value),
                           "%016llx", (unsigned long long)registry_id);
        }
        (void)snprintf(info->backend_device_id, sizeof(info->backend_device_id),
                       "metal:%016llx", (unsigned long long)registry_id);
    }

    g_metal_devices = devices;
    if (g_metal_records != NULL) {
        for (size_t old_index = 0; old_index < g_metal_device_count; old_index++)
            g_metal_records[old_index].device = nil;
        free(g_metal_records);
    }
    g_metal_records = records;
    g_metal_device_count = (size_t)count;
    return 0;
}

static void metal_shutdown(void)
{
    g_metal_devices = nil;
    if (g_metal_records != NULL) {
        for (size_t i = 0; i < g_metal_device_count; i++)
            g_metal_records[i].device = nil;
        free(g_metal_records);
    }
    g_metal_records = NULL;
    g_metal_device_count = 0;
}

static int metal_enumerate_devices(crossgpu_device_info_t *devices,
                                   size_t capacity,
                                   size_t *device_count,
                                   char *error,
                                   size_t error_size)
{
    size_t copy_count;
    if (device_count == NULL || (capacity > 0 && devices == NULL)) {
        metal_set_error(error, error_size, "invalid Metal device enumeration arguments");
        return -1;
    }
    if (g_metal_records == NULL && metal_initialize(error, error_size) != 0)
        return -1;
    *device_count = g_metal_device_count;
    copy_count = g_metal_device_count < capacity ? g_metal_device_count : capacity;
    for (size_t i = 0; i < copy_count; i++)
        devices[i] = g_metal_records[i].info;
    return 0;
}

static id<MTLComputePipelineState> metal_make_pipeline(id<MTLDevice> device,
                                                       id<MTLLibrary> library,
                                                       NSString *function_name,
                                                       char *error,
                                                       size_t error_size)
{
    NSError *ns_error = nil;
    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (function == nil) {
        NSString *message = [NSString stringWithFormat:@"Metal function %@ is missing", function_name];
        metal_set_error(error, error_size, message.UTF8String);
        return nil;
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&ns_error];
    if (pipeline == nil)
        metal_set_nserror(error, error_size, @"create Metal compute pipeline", ns_error);
    return pipeline;
}

static void metal_device_context_free(metal_device_context_t *context)
{
    if (context == NULL)
        return;
    context->copy_pipeline = nil;
    context->pattern_verify_pipeline = nil;
    context->pattern_write_pipeline = nil;
    context->compute_pipeline = nil;
    context->queue = nil;
    context->device = nil;
    free(context);
}

static int metal_device_open(uint32_t index, void **device_context,
                             char *error, size_t error_size)
{
    metal_device_context_t *context;
    NSError *ns_error = nil;
    id<MTLLibrary> library;

    if (device_context == NULL || index >= g_metal_device_count) {
        metal_set_error(error, error_size, "invalid Metal device index");
        return -1;
    }
    *device_context = NULL;
    context = (metal_device_context_t *)calloc(1, sizeof(*context));
    if (context == NULL) {
        metal_set_error(error, error_size, "out of memory allocating Metal device context");
        return -1;
    }
    context->device = g_metal_records[index].device;
    context->device_index = index;
    context->queue = [context->device newCommandQueue];
    if (context->queue == nil) {
        metal_set_error(error, error_size, "create Metal command queue failed");
        metal_device_context_free(context);
        return -1;
    }
    library = [context->device newLibraryWithSource:metal_kernel_source options:nil error:&ns_error];
    if (library == nil) {
        metal_set_nserror(error, error_size, @"compile embedded Metal kernels", ns_error);
        metal_device_context_free(context);
        return -1;
    }

    context->compute_pipeline = metal_make_pipeline(context->device, library, @"cg_compute", error, error_size);
    context->pattern_write_pipeline = metal_make_pipeline(context->device, library, @"cg_pattern_write", error, error_size);
    context->pattern_verify_pipeline = metal_make_pipeline(context->device, library, @"cg_pattern_verify", error, error_size);
    context->copy_pipeline = metal_make_pipeline(context->device, library, @"cg_copy", error, error_size);
    if (context->compute_pipeline == nil || context->pattern_write_pipeline == nil ||
        context->pattern_verify_pipeline == nil ||
        context->copy_pipeline == nil) {
        metal_device_context_free(context);
        return -1;
    }

    *device_context = context;
    return 0;
}

static void metal_device_close(void *device_context)
{
    metal_device_context_free((metal_device_context_t *)device_context);
}

static size_t metal_choose_working_set(const crossgpu_device_info_t *info, uint64_t requested)
{
    uint64_t bytes = requested;
    uint64_t maximum = info->max_allocation_bytes != 0 ? info->max_allocation_bytes : info->working_set_bytes;

    if (bytes == 0) {
        bytes = info->working_set_bytes > 0 ? info->working_set_bytes / 16 : METAL_MAX_WORKING_SET;
        if (bytes < METAL_MIN_WORKING_SET) bytes = METAL_MIN_WORKING_SET;
        if (bytes > METAL_MAX_WORKING_SET) bytes = METAL_MAX_WORKING_SET;
    }
    if (maximum > 0 && bytes > maximum) bytes = maximum;
    if (bytes > SIZE_MAX) bytes = SIZE_MAX;
    bytes &= ~(uint64_t)(sizeof(uint32_t) - 1);
    return (size_t)bytes;
}

static int metal_workload_begin(void *device_context,
                                crossgpu_test_type_t test,
                                const crossgpu_workload_config_t *config,
                                void **workload_context,
                                char *error,
                                size_t error_size)
{
    metal_device_context_t *device = (metal_device_context_t *)device_context;
    metal_workload_context_t *workload;
    size_t bytes;

    if (device == NULL || config == NULL || workload_context == NULL) {
        metal_set_error(error, error_size, "invalid Metal workload initialization arguments");
        return -1;
    }
    *workload_context = NULL;
    if (test != CROSSGPU_TEST_COMPUTE && test != CROSSGPU_TEST_MEMORY && test != CROSSGPU_TEST_BANDWIDTH) {
        metal_set_error(error, error_size, "unsupported Metal workload type");
        return -1;
    }
    bytes = metal_choose_working_set(&g_metal_records[device->device_index].info,
                                    config->memory_bytes);
    if (bytes < 4096 || bytes / sizeof(uint32_t) > UINT32_MAX) {
        metal_set_error(error, error_size, "Metal device cannot provide a usable working set");
        return -1;
    }

    workload = (metal_workload_context_t *)calloc(1, sizeof(*workload));
    if (workload == NULL) {
        metal_set_error(error, error_size, "out of memory allocating Metal workload context");
        return -1;
    }
    workload->bytes = bytes;
    workload->elements = (uint32_t)(bytes / sizeof(uint32_t));
    workload->seed = 1;
    workload->buffer_a = [device->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    workload->error_buffer = [device->device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if (test == CROSSGPU_TEST_BANDWIDTH)
        workload->buffer_b = [device->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (workload->buffer_a == nil || workload->error_buffer == nil ||
        (test == CROSSGPU_TEST_BANDWIDTH && workload->buffer_b == nil)) {
        metal_set_error(error, error_size, "allocate Metal workload buffers failed");
        workload->buffer_a = nil;
        workload->buffer_b = nil;
        workload->error_buffer = nil;
        free(workload);
        return -1;
    }

    memset(workload->buffer_a.contents, 0, bytes);
    memset(workload->error_buffer.contents, 0, sizeof(uint32_t));
    if (workload->buffer_b != nil)
        memset(workload->buffer_b.contents, 0, bytes);
    if (test == CROSSGPU_TEST_COMPUTE) {
        float *values = (float *)workload->buffer_a.contents;
        for (uint32_t i = 0; i < workload->elements; i++)
            values[i] = 0.5f;
    }
    *workload_context = workload;
    return 0;
}

static int metal_run_kernel(metal_device_context_t *device,
                            id<MTLComputePipelineState> pipeline,
                            metal_workload_context_t *workload,
                            crossgpu_test_type_t test,
                            int phase,
                            char *error,
                            size_t error_size)
{
    id<MTLCommandBuffer> command_buffer = [device->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder;
    NSUInteger threads;
    NSUInteger groups;

    if (command_buffer == nil) {
        metal_set_error(error, error_size, "create Metal command buffer failed");
        return -1;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        metal_set_error(error, error_size, "create Metal compute encoder failed");
        return -1;
    }
    [encoder setComputePipelineState:pipeline];

    switch (test) {
    case CROSSGPU_TEST_COMPUTE:
        [encoder setBuffer:workload->buffer_a offset:0 atIndex:0];
        [encoder setBuffer:workload->error_buffer offset:0 atIndex:1];
        {
            uint32_t count = workload->elements;
            [encoder setBytes:&count length:sizeof(count) atIndex:2];
        }
        break;
    case CROSSGPU_TEST_MEMORY:
        if (phase == 0) {
            [encoder setBuffer:workload->buffer_a offset:0 atIndex:0];
            {
                uint32_t count = workload->elements;
                [encoder setBytes:&count length:sizeof(count) atIndex:1];
            }
            [encoder setBytes:&workload->seed length:sizeof(workload->seed) atIndex:2];
        } else {
            [encoder setBuffer:workload->buffer_a offset:0 atIndex:0];
            [encoder setBuffer:workload->error_buffer offset:0 atIndex:1];
            {
                uint32_t count = workload->elements;
                [encoder setBytes:&count length:sizeof(count) atIndex:2];
            }
            [encoder setBytes:&workload->seed length:sizeof(workload->seed) atIndex:3];
        }
        break;
    case CROSSGPU_TEST_BANDWIDTH:
        [encoder setBuffer:workload->buffer_a offset:0 atIndex:0];
        [encoder setBuffer:workload->buffer_b offset:0 atIndex:1];
        {
            uint32_t count = workload->elements;
            [encoder setBytes:&count length:sizeof(count) atIndex:2];
        }
        break;
    default:
        [encoder endEncoding];
        metal_set_error(error, error_size, "unsupported Metal kernel operation");
        return -1;
    }

    threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (threads == 0 || threads > 256)
        threads = 256;
    groups = ((NSUInteger)workload->elements + threads - 1) / threads;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        metal_set_nserror(error, error_size, @"execute Metal compute workload", command_buffer.error);
        return -1;
    }
    return 0;
}

static int metal_workload_step(void *device_context,
                               crossgpu_test_type_t test,
                               void *workload_context,
                               crossgpu_metrics_t *metrics,
                               char *error,
                               size_t error_size)
{
    metal_device_context_t *device = (metal_device_context_t *)device_context;
    metal_workload_context_t *workload = (metal_workload_context_t *)workload_context;
    uint64_t bytes;
    uint32_t errors = 0;

    if (device == NULL || workload == NULL || metrics == NULL) {
        metal_set_error(error, error_size, "invalid Metal workload step arguments");
        return -1;
    }
    memset(metrics, 0, sizeof(*metrics));
    bytes = (uint64_t)workload->bytes;
    memset(workload->error_buffer.contents, 0, sizeof(uint32_t));

    switch (test) {
    case CROSSGPU_TEST_COMPUTE:
        if (metal_run_kernel(device, device->compute_pipeline, workload, test, 0, error, error_size) != 0)
            return -1;
        errors = *(uint32_t *)workload->error_buffer.contents;
        metrics->operations = (uint64_t)workload->elements * METAL_COMPUTE_INNER_ITERATIONS * 2;
        metrics->bytes_read = bytes;
        metrics->bytes_written = bytes;
        break;
    case CROSSGPU_TEST_MEMORY:
        workload->seed++;
        if (metal_run_kernel(device, device->pattern_write_pipeline, workload, test, 0, error, error_size) != 0)
            return -1;
        if (metal_run_kernel(device, device->pattern_verify_pipeline, workload, test, 1, error, error_size) != 0)
            return -1;
        errors = *(uint32_t *)workload->error_buffer.contents;
        metrics->bytes_read = bytes;
        metrics->bytes_written = bytes;
        break;
    case CROSSGPU_TEST_BANDWIDTH:
        if (metal_run_kernel(device, device->copy_pipeline, workload, test, 0, error, error_size) != 0)
            return -1;
        {
            id<MTLBuffer> temporary = workload->buffer_a;
            workload->buffer_a = workload->buffer_b;
            workload->buffer_b = temporary;
        }
        metrics->bytes_read = bytes;
        metrics->bytes_written = bytes;
        break;
    default:
        metal_set_error(error, error_size, "unsupported Metal workload type");
        return -1;
    }

    metrics->errors = errors;
    metrics->iterations = 1;
    return 0;
}

static void metal_workload_end(void *device_context,
                               crossgpu_test_type_t test,
                               void *workload_context)
{
    metal_workload_context_t *workload = (metal_workload_context_t *)workload_context;
    (void)device_context;
    (void)test;
    if (workload == NULL)
        return;
    workload->buffer_a = nil;
    workload->buffer_b = nil;
    workload->error_buffer = nil;
    free(workload);
}

static const crossgpu_backend_plugin_t metal_plugin = {
    .abi_version = CROSSGPU_BACKEND_ABI_VERSION,
    .struct_size = sizeof(crossgpu_backend_plugin_t),
    .kind = CROSSGPU_BACKEND_METAL,
    .name = METAL_BACKEND_NAME,
    .initialize = metal_initialize,
    .shutdown = metal_shutdown,
    .enumerate_devices = metal_enumerate_devices,
    .device_open = metal_device_open,
    .device_close = metal_device_close,
    .workload_begin = metal_workload_begin,
    .workload_step = metal_workload_step,
    .workload_end = metal_workload_end
};

CROSSGPU_BACKEND_EXPORT
const crossgpu_backend_plugin_t *crossgpu_backend_get_api(uint32_t host_abi_version)
{
    return host_abi_version == CROSSGPU_BACKEND_ABI_VERSION ? &metal_plugin : NULL;
}
