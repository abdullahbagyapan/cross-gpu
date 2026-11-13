#include "crossgpu_backend.h"
#include "opencl_minimal.h"
#include "../../src/platform.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if CROSSGPU_PLATFORM_APPLE
/* OpenCL and CGL are deprecated on macOS, but remain available for the
 * compatibility backend. Suppress deprecation diagnostics for this bridge. */
#ifndef GL_SILENCE_DEPRECATION
#define GL_SILENCE_DEPRECATION 1
#endif
#include <OpenGL/OpenGL.h>
#include <OpenGL/CGLRenderers.h>
#endif

#define DEFAULT_WORKING_SET (UINT64_C(64) * 1024 * 1024)
#define MAX_WORKING_SET (UINT64_C(128) * 1024 * 1024)
#define MIN_WORKING_SET (UINT64_C(4) * 1024 * 1024)
#define COMPUTE_INNER_ITERATIONS 256u
#define WORKGROUP_SIZE 64u

struct opencl_api {
    cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
    cl_int (*GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    cl_int (*GetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);
    cl_context (*CreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
    cl_int (*ReleaseContext)(cl_context);
    cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
    cl_int (*ReleaseCommandQueue)(cl_command_queue);
    cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
    cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *,
                           void (*)(cl_program, void *), void *);
    cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
    cl_int (*ReleaseProgram)(cl_program);
    cl_kernel (*CreateKernel)(cl_program, const char *, cl_int *);
    cl_int (*ReleaseKernel)(cl_kernel);
    cl_mem (*CreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
    cl_int (*ReleaseMemObject)(cl_mem);
    cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
    cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
                                   const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
    cl_int (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                 const void *, cl_uint, const cl_event *, cl_event *);
    cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                void *, cl_uint, const cl_event *, cl_event *);
    cl_int (*Finish)(cl_command_queue);
    cl_int (*EnqueueFillBuffer)(cl_command_queue, cl_mem, const void *, size_t,
                                size_t, size_t, cl_uint, const cl_event *, cl_event *);
};

typedef struct cl_device_pci_bus_info_khr {
    cl_uint pci_domain;
    cl_uint pci_bus;
    cl_uint pci_device;
    cl_uint pci_function;
} cl_device_pci_bus_info_khr;

typedef struct opencl_device_record {
    cl_platform_id platform;
    cl_device_id device;
    crossgpu_device_info_t info;
} opencl_device_record_t;

typedef struct opencl_device_context {
    opencl_device_record_t *record;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel compute_kernel;
    cl_kernel pattern_write_kernel;
    cl_kernel pattern_verify_kernel;
    cl_kernel copy_kernel;
    size_t max_work_group_size;
} opencl_device_context_t;

typedef struct opencl_workload_context {
    cl_mem buffer_a;
    cl_mem buffer_b;
    cl_mem error_buffer;
    size_t bytes;
    cl_uint elements;
    cl_uint seed;
} opencl_workload_context_t;

static void *g_library;
static struct opencl_api g_cl;
static opencl_device_record_t *g_devices;
static size_t g_device_count;
static size_t g_device_capacity;
static int g_devices_enumerated;
static char g_init_error[CROSSGPU_ERROR_MAX];

static const char opencl_kernel_source[] =
"__kernel void compute_stress(__global float *data, volatile __global uint *errors, const uint n) {\n"
"    const uint gid = get_global_id(0);\n"
"    if (gid >= n) return;\n"
"    float value = data[gid];\n"
"    for (uint i = 0; i < 256; i++)\n"
"        value = fma(value, 0.99999f, 0.00001f);\n"
"    data[gid] = value;\n"
"    if (!isfinite(value) || value < 0.0f || value > 1.01f) atomic_inc(errors);\n"
"}\n"
"__kernel void pattern_write(__global uint *data, const uint n, const uint seed) {\n"
"    const uint gid = get_global_id(0);\n"
"    if (gid >= n) return;\n"
"    data[gid] = (gid * 1664525u + seed * 1013904223u) ^ 0xA5A55A5Au;\n"
"}\n"
"__kernel void pattern_verify(__global const uint *data, volatile __global uint *errors,\n"
"                             const uint n, const uint seed) {\n"
"    const uint gid = get_global_id(0);\n"
"    if (gid >= n) return;\n"
"    const uint expected = (gid * 1664525u + seed * 1013904223u) ^ 0xA5A55A5Au;\n"
"    if (data[gid] != expected) atomic_inc(errors);\n"
"}\n"
"__kernel void copy_buffer(__global const uint *src, __global uint *dst, const uint n) {\n"
"    const uint gid = get_global_id(0);\n"
"    if (gid < n) dst[gid] = src[gid];\n"
"}\n";

static void set_error(char *error, size_t error_size, const char *message)
{
    if (error != NULL && error_size > 0)
        (void)snprintf(error, error_size, "%s", message != NULL ? message : "unknown OpenCL error");
}

static const char *cl_error_name(cl_int error)
{
    switch (error) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_PLATFORM_NOT_FOUND_KHR: return "CL_PLATFORM_NOT_FOUND_KHR";
    default: return "OpenCL error";
    }
}

static int load_symbol(void *library, const char *name, void *destination, size_t destination_size)
{
    void *symbol = crossgpu_platform_library_symbol(library, name);
    if (symbol == NULL || destination_size != sizeof(symbol))
        return -1;
    memcpy(destination, &symbol, sizeof(symbol));
    return 0;
}

#define LOAD_REQUIRED(symbol) \
    do { \
        if (load_symbol(g_library, "cl" #symbol, &g_cl.symbol, sizeof(g_cl.symbol)) != 0) { \
            (void)snprintf(g_init_error, sizeof(g_init_error), "OpenCL loader is missing cl%s", #symbol); \
            goto fail_load; \
        } \
    } while (0)

static int opencl_initialize(char *error, size_t error_size)
{
    const char *override = getenv("CROSSGPU_OPENCL_LIBRARY");
#if CROSSGPU_PLATFORM_APPLE
    const char *libraries[] = { "/System/Library/Frameworks/OpenCL.framework/OpenCL", NULL };
#elif CROSSGPU_PLATFORM_WINDOWS
    const char *libraries[] = { "OpenCL.dll", NULL };
#else
    const char *libraries[] = { "libOpenCL.so.1", "libOpenCL.so", NULL };
#endif
    size_t i;

    if (g_library != NULL)
        return 0;

    if (override != NULL && *override != '\0')
        g_library = crossgpu_platform_library_open(override);
    else {
        for (i = 0; libraries[i] != NULL && g_library == NULL; i++)
            g_library = crossgpu_platform_library_open(libraries[i]);
    }
    if (g_library == NULL) {
        const char *detail = crossgpu_platform_library_error();
        (void)snprintf(g_init_error, sizeof(g_init_error), "cannot load OpenCL runtime: %s",
                       detail != NULL ? detail : "unknown loader error");
        set_error(error, error_size, g_init_error);
        return -1;
    }

    memset(&g_cl, 0, sizeof(g_cl));
    LOAD_REQUIRED(GetPlatformIDs);
    LOAD_REQUIRED(GetDeviceIDs);
    LOAD_REQUIRED(GetDeviceInfo);
    LOAD_REQUIRED(CreateContext);
    LOAD_REQUIRED(ReleaseContext);
    LOAD_REQUIRED(CreateCommandQueue);
    LOAD_REQUIRED(ReleaseCommandQueue);
    LOAD_REQUIRED(CreateProgramWithSource);
    LOAD_REQUIRED(BuildProgram);
    LOAD_REQUIRED(GetProgramBuildInfo);
    LOAD_REQUIRED(ReleaseProgram);
    LOAD_REQUIRED(CreateKernel);
    LOAD_REQUIRED(ReleaseKernel);
    LOAD_REQUIRED(CreateBuffer);
    LOAD_REQUIRED(ReleaseMemObject);
    LOAD_REQUIRED(SetKernelArg);
    LOAD_REQUIRED(EnqueueNDRangeKernel);
    LOAD_REQUIRED(EnqueueWriteBuffer);
    LOAD_REQUIRED(EnqueueReadBuffer);
    LOAD_REQUIRED(Finish);
    /* OpenCL 1.2 fill is optional; a host-write fallback is provided. */
    (void)load_symbol(g_library, "clEnqueueFillBuffer", &g_cl.EnqueueFillBuffer,
                      sizeof(g_cl.EnqueueFillBuffer));

    g_init_error[0] = '\0';
    return 0;

fail_load:
    if (g_library != NULL)
        crossgpu_platform_library_close(g_library);
    g_library = NULL;
    memset(&g_cl, 0, sizeof(g_cl));
    set_error(error, error_size, g_init_error);
    return -1;
}

static void opencl_shutdown(void)
{
    if (g_library != NULL)
        crossgpu_platform_library_close(g_library);
    g_library = NULL;
    memset(&g_cl, 0, sizeof(g_cl));
    free(g_devices);
    g_devices = NULL;
    g_device_count = 0;
    g_device_capacity = 0;
    g_devices_enumerated = 0;
}

#if CROSSGPU_PLATFORM_APPLE
/* These CGL renderer properties were added in macOS 10.13. */
#define CROSSGPU_CGL_RP_REGISTRY_ID_LOW  kCGLRPRegistryIDLow
#define CROSSGPU_CGL_RP_REGISTRY_ID_HIGH kCGLRPRegistryIDHigh

/*
 * Apple Technical Note TN2335 documents the CGL renderer-ID -> cl_device_id
 * conversion: clear the low byte of kCGLRPRendererID and reinterpret the
 * result as cl_device_id. CGL's registry-ID properties (140/141) provide the
 * same IORegistry ID returned by MTLDevice.registryID.
 *
 * This is an exact API-level association; it deliberately does not use GPU
 * names, vendor strings, indices, or product IDs. If the CGL/OpenCL mapping is
 * unavailable or ambiguous, the OpenCL device remains without a physical ID.
 */
static void opencl_set_apple_registry_identity(cl_device_id device,
                                                crossgpu_device_info_t *info)
{
    CGLRendererInfoObj renderer_info = NULL;
    GLint renderer_count = 0;
    uint64_t matched_registry_id = 0;
    unsigned int match_count = 0;
    int ambiguous = 0;
    GLint renderer_index;

    if (device == NULL || info == NULL)
        return;

    if (CGLQueryRendererInfo((GLuint)0xffffffffu, &renderer_info,
                             &renderer_count) != kCGLNoError ||
        renderer_info == NULL || renderer_count <= 0) {
        if (renderer_info != NULL)
            CGLDestroyRendererInfo(renderer_info);
        return;
    }

    for (renderer_index = 0; renderer_index < renderer_count; renderer_index++) {
        GLint supports_opencl = 0;
        GLint renderer_id = 0;
        GLint registry_low = 0;
        GLint registry_high = 0;
        uintptr_t opencl_handle;
        uint64_t registry_id;

        if (CGLDescribeRenderer(renderer_info, renderer_index,
                                kCGLRPAcceleratedCompute,
                                &supports_opencl) != kCGLNoError ||
            supports_opencl == 0)
            continue;

        if (CGLDescribeRenderer(renderer_info, renderer_index,
                                kCGLRPRendererID, &renderer_id) != kCGLNoError)
            continue;

        /* Match Apple's TN2335 conversion exactly: rendererID & ~0xff,
         * converted through intptr_t to preserve the documented pointer cast. */
        opencl_handle = (uintptr_t)(intptr_t)(renderer_id & ~0xff);
        if ((cl_device_id)opencl_handle != device)
            continue;

        if (CGLDescribeRenderer(renderer_info, renderer_index,
                                CROSSGPU_CGL_RP_REGISTRY_ID_LOW,
                                &registry_low) != kCGLNoError ||
            CGLDescribeRenderer(renderer_info, renderer_index,
                                CROSSGPU_CGL_RP_REGISTRY_ID_HIGH,
                                &registry_high) != kCGLNoError)
            continue;

        registry_id = ((uint64_t)(uint32_t)registry_high << 32) |
                      (uint64_t)(uint32_t)registry_low;
        if (registry_id == 0)
            continue;

        if (match_count == 0)
            matched_registry_id = registry_id;
        else if (matched_registry_id != registry_id)
            ambiguous = 1;
        match_count++;
    }

    CGLDestroyRendererInfo(renderer_info);

    /* More than one CGL renderer resolving to the same OpenCL handle is not
     * a safe identity claim, so refuse to merge in that case. */
    if (match_count == 0 || ambiguous || matched_registry_id == 0)
        return;

    info->identity_kind = CROSSGPU_IDENTITY_OS_DEVICE;
    (void)snprintf(info->identity_namespace,
                   sizeof(info->identity_namespace), "%s",
                   CROSSGPU_IDENTITY_NAMESPACE_APPLE_IOREGISTRY);
    (void)snprintf(info->identity_value, sizeof(info->identity_value),
                   "%016llx", (unsigned long long)matched_registry_id);
}
#endif

/* Apple uses a Khronos-assigned OpenCL vendor ID rather than its PCI ID.
 * CrossGPU's vendor enum represents PCI vendor IDs, so normalize this known
 * OpenCL value instead of leaking it into the PCI-ID field. */
#define CL_KHRONOS_VENDOR_ID_APPLE UINT32_C(0x01027f00)

static crossgpu_vendor_id_t normalize_opencl_vendor_id(cl_uint vendor_id)
{
    if (vendor_id == CL_KHRONOS_VENDOR_ID_APPLE)
        return CROSSGPU_VENDOR_APPLE;

    /* OpenCL specifies that PCI vendor IDs occupy the low 16 bits. IDs in
     * the Khronos vendor-ID namespace are not PCI IDs; leave unknown ones as
     * UNKNOWN rather than misclassifying them as hardware vendors. */
    if (vendor_id <= UINT16_MAX)
        return (crossgpu_vendor_id_t)vendor_id;

    return CROSSGPU_VENDOR_UNKNOWN;
}

static int get_device_value(cl_device_id device, cl_device_info key,
                            size_t size, void *value)
{
    return g_cl.GetDeviceInfo(device, key, size, value, NULL) == CL_SUCCESS ? 0 : -1;
}

static void get_device_string(cl_device_id device, cl_device_info key,
                              char *buffer, size_t size)
{
    size_t actual = 0;
    if (size == 0)
        return;
    buffer[0] = '\0';
    if (g_cl.GetDeviceInfo(device, key, size, buffer, &actual) != CL_SUCCESS) {
        buffer[0] = '\0';
        return;
    }
    buffer[size - 1] = '\0';
}

static int opencl_grow_devices(size_t needed)
{
    size_t capacity;
    opencl_device_record_t *grown;

    if (needed <= g_device_capacity)
        return 0;
    capacity = g_device_capacity ? g_device_capacity : 8;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(*grown))
        return -1;
    grown = realloc(g_devices, capacity * sizeof(*grown));
    if (grown == NULL)
        return -1;
    g_devices = grown;
    g_device_capacity = capacity;
    return 0;
}

static int opencl_refresh_devices(char *error, size_t error_size)
{
    cl_platform_id *platforms = NULL;
    cl_uint platform_count = 0;
    cl_int status;
    cl_uint p;

    free(g_devices);
    g_devices = NULL;
    g_device_count = 0;
    g_device_capacity = 0;
    g_devices_enumerated = 0;

    status = g_cl.GetPlatformIDs(0, NULL, &platform_count);
    if (status == CL_PLATFORM_NOT_FOUND_KHR || platform_count == 0) {
        g_devices_enumerated = 1;
        return 0;
    }
    if (status != CL_SUCCESS) {
        char message[128];
        (void)snprintf(message, sizeof(message), "clGetPlatformIDs failed: %s (%d)",
                       cl_error_name(status), status);
        set_error(error, error_size, message);
        return -1;
    }
    platforms = calloc((size_t)platform_count, sizeof(*platforms));
    if (platforms == NULL) {
        set_error(error, error_size, "out of memory allocating OpenCL platform list");
        return -1;
    }
    status = g_cl.GetPlatformIDs(platform_count, platforms, NULL);
    if (status != CL_SUCCESS) {
        char message[128];
        (void)snprintf(message, sizeof(message), "clGetPlatformIDs(list) failed: %s (%d)",
                       cl_error_name(status), status);
        set_error(error, error_size, message);
        free(platforms);
        return -1;
    }

    for (p = 0; p < platform_count; p++) {
        cl_uint count = 0, d;
        cl_device_id *found = NULL;

        status = g_cl.GetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &count);
        if (status == CL_DEVICE_NOT_FOUND || count == 0)
            continue;
        if (status != CL_SUCCESS)
            continue;
        found = calloc((size_t)count, sizeof(*found));
        if (found == NULL) {
            set_error(error, error_size, "out of memory allocating OpenCL device list");
            free(platforms);
            return -1;
        }
        status = g_cl.GetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, count, found, NULL);
        if (status != CL_SUCCESS) {
            free(found);
            continue;
        }

        for (d = 0; d < count; d++) {
            opencl_device_record_t record;
            crossgpu_device_info_t *info = &record.info;
            cl_bool available = CL_TRUE;
            cl_ulong global_mem = 0, max_alloc = 0;
            cl_uint vendor_id = 0, compute_units = 0, clock_mhz = 0;
            char extensions[2048] = { 0 };

            if (get_device_value(found[d], CL_DEVICE_AVAILABLE, sizeof(available), &available) != 0 || !available)
                continue;

            memset(&record, 0, sizeof(record));
            memset(info, 0, sizeof(*info));
            info->struct_size = sizeof(*info);
            info->backend_device_index = (uint32_t)g_device_count;
            (void)get_device_value(found[d], CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id);
            (void)get_device_value(found[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units);
            (void)get_device_value(found[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_mhz), &clock_mhz);
            (void)get_device_value(found[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem);
            (void)get_device_value(found[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc);
            info->vendor_id = normalize_opencl_vendor_id(vendor_id);
            info->compute_units = compute_units;
            info->clock_mhz = clock_mhz;
            info->physical_memory_bytes = global_mem;
            info->max_allocation_bytes = max_alloc;
            get_device_string(found[d], CL_DEVICE_NAME, info->name, sizeof(info->name));
            get_device_string(found[d], CL_DEVICE_VENDOR, info->vendor, sizeof(info->vendor));
            get_device_string(found[d], CL_DRIVER_VERSION, info->driver_version, sizeof(info->driver_version));
            get_device_string(found[d], CL_DEVICE_VERSION, info->api_version, sizeof(info->api_version));
            get_device_string(found[d], CL_DEVICE_EXTENSIONS, extensions, sizeof(extensions));
            if (strstr(extensions, "cl_khr_pci_bus_info") != NULL) {
                cl_device_pci_bus_info_khr pci_info;
                if (get_device_value(found[d], CL_DEVICE_PCI_BUS_INFO_KHR,
                                     sizeof(pci_info), &pci_info) == 0) {
                    info->identity_kind = CROSSGPU_IDENTITY_PCI_BDF;
                    (void)snprintf(info->identity_namespace,
                                   sizeof(info->identity_namespace), "%s",
                                   CROSSGPU_IDENTITY_NAMESPACE_PCI_BDF);
                    (void)snprintf(info->identity_value,
                                   sizeof(info->identity_value),
                                   "%04x:%02x:%02x.%x",
                                   pci_info.pci_domain, pci_info.pci_bus,
                                   pci_info.pci_device, pci_info.pci_function);
                }
            }
#if CROSSGPU_PLATFORM_APPLE
            /* CL_DEVICE_VENDOR_ID is not necessarily a PCI vendor ID. Apple's
             * OpenCL runtime reports a Khronos-allocated value (0x01027f00),
             * not CROSSGPU_VENDOR_APPLE (0x106b). Therefore do not gate the
             * CGL renderer-to-cl_device_id bridge on info->vendor_id. On macOS,
             * attempt the exact CGL handle mapping for every OpenCL GPU that
             * did not already provide a comparable PCI identity. The bridge
             * only assigns an identity when the CGL renderer handle matches
             * this exact cl_device_id. */
            if (info->identity_kind == CROSSGPU_IDENTITY_NONE)
                opencl_set_apple_registry_identity(found[d], info);
#endif
            (void)snprintf(info->backend_device_id,
                           sizeof(info->backend_device_id), "opencl:%u:%u", p, d);

            record.platform = platforms[p];
            record.device = found[d];
            if (opencl_grow_devices(g_device_count + 1) != 0) {
                set_error(error, error_size, "out of memory growing OpenCL device registry");
                free(found);
                free(platforms);
                return -1;
            }
            g_devices[g_device_count++] = record;
        }
        free(found);
    }

    free(platforms);
    g_devices_enumerated = 1;
    return 0;
}

static int opencl_enumerate_devices(crossgpu_device_info_t *devices,
                                    size_t capacity,
                                    size_t *device_count,
                                    char *error,
                                    size_t error_size)
{
    size_t i, copy_count;

    if (device_count == NULL || (capacity > 0 && devices == NULL)) {
        set_error(error, error_size, "invalid device enumeration arguments");
        return -1;
    }
    if (!g_devices_enumerated && opencl_refresh_devices(error, error_size) != 0)
        return -1;

    *device_count = g_device_count;
    copy_count = g_device_count < capacity ? g_device_count : capacity;
    for (i = 0; i < copy_count; i++)
        devices[i] = g_devices[i].info;
    return 0;
}

static void release_device_context(opencl_device_context_t *device)
{
    if (device == NULL)
        return;
    if (device->compute_kernel != NULL) g_cl.ReleaseKernel(device->compute_kernel);
    if (device->pattern_write_kernel != NULL) g_cl.ReleaseKernel(device->pattern_write_kernel);
    if (device->pattern_verify_kernel != NULL) g_cl.ReleaseKernel(device->pattern_verify_kernel);
    if (device->copy_kernel != NULL) g_cl.ReleaseKernel(device->copy_kernel);
    if (device->program != NULL) g_cl.ReleaseProgram(device->program);
    if (device->queue != NULL) g_cl.ReleaseCommandQueue(device->queue);
    if (device->context != NULL) g_cl.ReleaseContext(device->context);
    free(device);
}

static int opencl_device_open(uint32_t index, void **device_context,
                              char *error, size_t error_size)
{
    opencl_device_context_t *device;
    cl_context_properties properties[3];
    cl_int status;
    const char *source = opencl_kernel_source;
    size_t source_length = sizeof(opencl_kernel_source) - 1;
    char build_log[CROSSGPU_ERROR_MAX] = { 0 };

    if (device_context == NULL || index >= g_device_count) {
        set_error(error, error_size, "invalid OpenCL device index");
        return -1;
    }
    *device_context = NULL;
    device = calloc(1, sizeof(*device));
    if (device == NULL) {
        set_error(error, error_size, "out of memory creating OpenCL device context");
        return -1;
    }
    device->record = &g_devices[index];
    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties)device->record->platform;
    properties[2] = 0;

    device->context = g_cl.CreateContext(properties, 1, &device->record->device, NULL, NULL, &status);
    if (status != CL_SUCCESS || device->context == NULL)
        goto fail_context;
    device->queue = g_cl.CreateCommandQueue(device->context, device->record->device, 0, &status);
    if (status != CL_SUCCESS || device->queue == NULL)
        goto fail_queue;
    device->program = g_cl.CreateProgramWithSource(device->context, 1, &source, &source_length, &status);
    if (status != CL_SUCCESS || device->program == NULL)
        goto fail_program;
    status = g_cl.BuildProgram(device->program, 1, &device->record->device,
                               "-cl-std=CL1.2", NULL, NULL);
    /* Retry with the implementation's default OpenCL C language version for
     * older 1.1-era compilers that do not accept the CL1.2 build option. */
    if (status != CL_SUCCESS)
        status = g_cl.BuildProgram(device->program, 1, &device->record->device,
                                   NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        (void)g_cl.GetProgramBuildInfo(device->program, device->record->device,
                                       CL_PROGRAM_BUILD_LOG, sizeof(build_log) - 1,
                                       build_log, &log_size);
        if (build_log[0] == '\0')
            (void)snprintf(build_log, sizeof(build_log), "OpenCL program build failed with %d", status);
        set_error(error, error_size, build_log);
        release_device_context(device);
        return -1;
    }

    device->compute_kernel = g_cl.CreateKernel(device->program, "compute_stress", &status);
    if (status != CL_SUCCESS || device->compute_kernel == NULL) goto fail_kernel;
    device->pattern_write_kernel = g_cl.CreateKernel(device->program, "pattern_write", &status);
    if (status != CL_SUCCESS || device->pattern_write_kernel == NULL) goto fail_kernel;
    device->pattern_verify_kernel = g_cl.CreateKernel(device->program, "pattern_verify", &status);
    if (status != CL_SUCCESS || device->pattern_verify_kernel == NULL) goto fail_kernel;
    device->copy_kernel = g_cl.CreateKernel(device->program, "copy_buffer", &status);
    if (status != CL_SUCCESS || device->copy_kernel == NULL) goto fail_kernel;

    *device_context = device;
    return 0;

fail_kernel:
    (void)snprintf(build_log, sizeof(build_log), "could not create OpenCL kernel: %d", status);
    set_error(error, error_size, build_log);
    release_device_context(device);
    return -1;
fail_program:
    set_error(error, error_size, "clCreateProgramWithSource failed");
    release_device_context(device);
    return -1;
fail_queue:
    set_error(error, error_size, "clCreateCommandQueue failed");
    release_device_context(device);
    return -1;
fail_context:
    set_error(error, error_size, "clCreateContext failed");
    release_device_context(device);
    return -1;
}

static void opencl_device_close(void *device_context)
{
    release_device_context((opencl_device_context_t *)device_context);
}

static int fill_buffer(opencl_device_context_t *device, cl_mem buffer,
                       size_t bytes, const void *pattern, size_t pattern_size)
{
    cl_int status;
    if (g_cl.EnqueueFillBuffer != NULL) {
        status = g_cl.EnqueueFillBuffer(device->queue, buffer, pattern, pattern_size,
                                        0, bytes, 0, NULL, NULL);
        if (status == CL_SUCCESS)
            return 0;
    }

    {
        unsigned char *host = malloc(bytes);
        size_t offset;
        if (host == NULL)
            return -1;
        for (offset = 0; offset < bytes; offset += pattern_size) {
            size_t copy = pattern_size;
            if (offset + copy > bytes)
                copy = bytes - offset;
            memcpy(host + offset, pattern, copy);
        }
        status = g_cl.EnqueueWriteBuffer(device->queue, buffer, CL_TRUE, 0, bytes,
                                         host, 0, NULL, NULL);
        free(host);
    }
    return status == CL_SUCCESS ? 0 : -1;
}

static size_t choose_working_set(const crossgpu_device_info_t *info, uint64_t requested)
{
    uint64_t bytes = requested;
    if (bytes == 0) {
        if (info->working_set_bytes > 0)
            bytes = info->working_set_bytes / 16;
        else if (info->physical_memory_bytes > 0)
            bytes = info->physical_memory_bytes / 16;
        else
            bytes = DEFAULT_WORKING_SET;
        if (bytes > MAX_WORKING_SET)
            bytes = MAX_WORKING_SET;
        if (bytes < MIN_WORKING_SET)
            bytes = MIN_WORKING_SET;
    }
    if (info->max_allocation_bytes > 0 && bytes > info->max_allocation_bytes)
        bytes = info->max_allocation_bytes;
    bytes -= bytes % sizeof(uint32_t);
    if (bytes < sizeof(uint32_t))
        return 0;
    return (size_t)bytes;
}

typedef int (*opencl_workload_prepare_fn)(opencl_device_context_t *device,
                                          opencl_workload_context_t *workload,
                                          char *error,
                                          size_t error_size);
typedef int (*opencl_workload_step_fn)(opencl_device_context_t *device,
                                       opencl_workload_context_t *workload,
                                       crossgpu_metrics_t *metrics,
                                       char *error,
                                       size_t error_size);

typedef struct opencl_test_operations {
    crossgpu_test_type_t type;
    opencl_workload_prepare_fn prepare;
    opencl_workload_step_fn step;
} opencl_test_operations_t;

static int opencl_prepare_compute(opencl_device_context_t *device,
                                  opencl_workload_context_t *workload,
                                  char *error, size_t error_size);
static int opencl_prepare_memory(opencl_device_context_t *device,
                                 opencl_workload_context_t *workload,
                                 char *error, size_t error_size);
static int opencl_prepare_bandwidth(opencl_device_context_t *device,
                                    opencl_workload_context_t *workload,
                                    char *error, size_t error_size);
static int opencl_compute_step(opencl_device_context_t *device,
                               opencl_workload_context_t *workload,
                               crossgpu_metrics_t *metrics,
                               char *error, size_t error_size);
static int opencl_memory_step(opencl_device_context_t *device,
                              opencl_workload_context_t *workload,
                              crossgpu_metrics_t *metrics,
                              char *error, size_t error_size);
static int opencl_bandwidth_step(opencl_device_context_t *device,
                                 opencl_workload_context_t *workload,
                                 crossgpu_metrics_t *metrics,
                                 char *error, size_t error_size);

/* Add new OpenCL workloads here instead of extending central if/else chains. */
static const opencl_test_operations_t opencl_test_operations[] = {
    { CROSSGPU_TEST_COMPUTE,   opencl_prepare_compute,   opencl_compute_step },
    { CROSSGPU_TEST_MEMORY,    opencl_prepare_memory,    opencl_memory_step },
    { CROSSGPU_TEST_BANDWIDTH, opencl_prepare_bandwidth, opencl_bandwidth_step }
};

static const opencl_test_operations_t *opencl_find_test_operations(crossgpu_test_type_t test)
{
    size_t i;
    for (i = 0; i < sizeof(opencl_test_operations) / sizeof(opencl_test_operations[0]); i++)
        if (opencl_test_operations[i].type == test)
            return &opencl_test_operations[i];
    return NULL;
}

static int opencl_prepare_compute(opencl_device_context_t *device,
                                  opencl_workload_context_t *workload,
                                  char *error, size_t error_size)
{
    cl_int status;
    uint32_t zero = 0;
    const float initial_float = 0.5f;

    if (fill_buffer(device, workload->buffer_a, workload->bytes,
                    &initial_float, sizeof(initial_float)) != 0) {
        set_error(error, error_size, "failed to initialize compute working buffer");
        return -1;
    }
    workload->error_buffer = g_cl.CreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                sizeof(uint32_t), NULL, &status);
    if (status != CL_SUCCESS || workload->error_buffer == NULL) {
        set_error(error, error_size, "failed to allocate compute validation buffer");
        return -1;
    }
    status = g_cl.EnqueueWriteBuffer(device->queue, workload->error_buffer, CL_TRUE,
                                     0, sizeof(zero), &zero, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        set_error(error, error_size, "failed to initialize compute validation buffer");
        return -1;
    }
    return 0;
}

static int opencl_prepare_memory(opencl_device_context_t *device,
                                 opencl_workload_context_t *workload,
                                 char *error, size_t error_size)
{
    cl_int status;
    uint32_t zero = 0;

    workload->error_buffer = g_cl.CreateBuffer(device->context, CL_MEM_READ_WRITE,
                                                sizeof(uint32_t), NULL, &status);
    if (status != CL_SUCCESS || workload->error_buffer == NULL) {
        set_error(error, error_size, "failed to allocate memory validation buffer");
        return -1;
    }
    status = g_cl.EnqueueWriteBuffer(device->queue, workload->error_buffer, CL_TRUE,
                                     0, sizeof(zero), &zero, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        set_error(error, error_size, "failed to initialize memory validation buffer");
        return -1;
    }
    return 0;
}

static int opencl_prepare_bandwidth(opencl_device_context_t *device,
                                    opencl_workload_context_t *workload,
                                    char *error, size_t error_size)
{
    cl_int status;
    const uint32_t initial_uint = UINT32_C(0x12345678);

    workload->buffer_b = g_cl.CreateBuffer(device->context, CL_MEM_READ_WRITE,
                                            workload->bytes, NULL, &status);
    if (status != CL_SUCCESS || workload->buffer_b == NULL) {
        set_error(error, error_size, "failed to allocate second bandwidth buffer");
        return -1;
    }
    if (fill_buffer(device, workload->buffer_a, workload->bytes,
                    &initial_uint, sizeof(initial_uint)) != 0) {
        set_error(error, error_size, "failed to initialize bandwidth buffer");
        return -1;
    }
    return 0;
}

static int opencl_workload_begin(void *device_context,
                                 crossgpu_test_type_t test,
                                 const crossgpu_workload_config_t *config,
                                 void **workload_context,
                                 char *error,
                                 size_t error_size)
{
    opencl_device_context_t *device = (opencl_device_context_t *)device_context;
    opencl_workload_context_t *workload;
    const opencl_test_operations_t *operations;
    cl_int status;

    if (device == NULL || config == NULL || workload_context == NULL) {
        set_error(error, error_size, "invalid workload arguments");
        return -1;
    }
    *workload_context = NULL;
    operations = opencl_find_test_operations(test);
    if (operations == NULL) {
        set_error(error, error_size, "unsupported OpenCL workload type");
        return -1;
    }

    workload = calloc(1, sizeof(*workload));
    if (workload == NULL) {
        set_error(error, error_size, "out of memory creating workload");
        return -1;
    }
    workload->bytes = choose_working_set(&device->record->info, config->memory_bytes);
    if (workload->bytes == 0 || workload->bytes / sizeof(uint32_t) > UINT32_MAX) {
        set_error(error, error_size, "device cannot provide a usable working set");
        free(workload);
        return -1;
    }
    workload->elements = (cl_uint)(workload->bytes / sizeof(uint32_t));
    workload->seed = 1;
    workload->buffer_a = g_cl.CreateBuffer(device->context, CL_MEM_READ_WRITE,
                                            workload->bytes, NULL, &status);
    if (status != CL_SUCCESS || workload->buffer_a == NULL) {
        set_error(error, error_size, "failed to allocate primary OpenCL working buffer");
        free(workload);
        return -1;
    }

    if (operations->prepare(device, workload, error, error_size) != 0) {
        if (workload->buffer_a != NULL) g_cl.ReleaseMemObject(workload->buffer_a);
        if (workload->buffer_b != NULL) g_cl.ReleaseMemObject(workload->buffer_b);
        if (workload->error_buffer != NULL) g_cl.ReleaseMemObject(workload->error_buffer);
        free(workload);
        return -1;
    }

    *workload_context = workload;
    return 0;
}

static cl_int enqueue_kernel(opencl_device_context_t *device, cl_kernel kernel, size_t elements)
{
    size_t global_size = ((elements + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) * WORKGROUP_SIZE;
    size_t local_size = WORKGROUP_SIZE;
    return g_cl.EnqueueNDRangeKernel(device->queue, kernel, 1, NULL,
                                     &global_size, &local_size, 0, NULL, NULL);
}

static int opencl_report_step_error(cl_int status, char *error, size_t error_size)
{
    char message[128];
    (void)snprintf(message, sizeof(message), "OpenCL command failed: %s (%d)",
                   cl_error_name(status), status);
    set_error(error, error_size, message);
    return -1;
}

static int opencl_compute_step(opencl_device_context_t *device,
                               opencl_workload_context_t *workload,
                               crossgpu_metrics_t *metrics,
                               char *error, size_t error_size)
{
    uint32_t zero = 0, error_count = 0;
    cl_uint n = workload->elements;
    cl_int status;

    status = g_cl.EnqueueWriteBuffer(device->queue, workload->error_buffer, CL_TRUE,
                                     0, sizeof(zero), &zero, 0, NULL, NULL);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = g_cl.SetKernelArg(device->compute_kernel, 0, sizeof(workload->buffer_a), &workload->buffer_a);
    if (status == CL_SUCCESS) status = g_cl.SetKernelArg(device->compute_kernel, 1, sizeof(workload->error_buffer), &workload->error_buffer);
    if (status == CL_SUCCESS) status = g_cl.SetKernelArg(device->compute_kernel, 2, sizeof(n), &n);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = enqueue_kernel(device, device->compute_kernel, n);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = g_cl.Finish(device->queue);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = g_cl.EnqueueReadBuffer(device->queue, workload->error_buffer, CL_TRUE,
                                    0, sizeof(error_count), &error_count, 0, NULL, NULL);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);

    metrics->errors = error_count;
    metrics->operations = (uint64_t)n * COMPUTE_INNER_ITERATIONS * 2;
    metrics->bytes_read = (uint64_t)workload->bytes;
    metrics->bytes_written = (uint64_t)workload->bytes;
    metrics->iterations = 1;
    return 0;
}

static int opencl_memory_step(opencl_device_context_t *device,
                              opencl_workload_context_t *workload,
                              crossgpu_metrics_t *metrics,
                              char *error, size_t error_size)
{
    uint32_t zero = 0, error_count = 0;
    cl_uint n = workload->elements;
    cl_int status;
    uint64_t bytes = (uint64_t)workload->bytes;

    workload->seed++;
    status = g_cl.EnqueueWriteBuffer(device->queue, workload->error_buffer, CL_TRUE,
                                     0, sizeof(zero), &zero, 0, NULL, NULL);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = g_cl.SetKernelArg(device->pattern_write_kernel, 0, sizeof(workload->buffer_a), &workload->buffer_a);
    if (status == CL_SUCCESS) status = g_cl.SetKernelArg(device->pattern_write_kernel, 1, sizeof(n), &n);
    if (status == CL_SUCCESS) status = g_cl.SetKernelArg(device->pattern_write_kernel, 2, sizeof(workload->seed), &workload->seed);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = enqueue_kernel(device, device->pattern_write_kernel, n);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = g_cl.SetKernelArg(device->pattern_verify_kernel, 0, sizeof(workload->buffer_a), &workload->buffer_a);
    if (status == CL_SUCCESS) status = g_cl.SetKernelArg(device->pattern_verify_kernel, 1, sizeof(workload->error_buffer), &workload->error_buffer);
    if (status == CL_SUCCESS) status = g_cl.SetKernelArg(device->pattern_verify_kernel, 2, sizeof(n), &n);
    if (status == CL_SUCCESS) status = g_cl.SetKernelArg(device->pattern_verify_kernel, 3, sizeof(workload->seed), &workload->seed);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = enqueue_kernel(device, device->pattern_verify_kernel, n);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = g_cl.Finish(device->queue);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = g_cl.EnqueueReadBuffer(device->queue, workload->error_buffer, CL_TRUE,
                                    0, sizeof(error_count), &error_count, 0, NULL, NULL);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);

    metrics->errors = error_count;
    metrics->bytes_written = bytes;
    metrics->bytes_read = bytes;
    metrics->iterations = 1;
    return 0;
}

static int opencl_bandwidth_step(opencl_device_context_t *device,
                                 opencl_workload_context_t *workload,
                                 crossgpu_metrics_t *metrics,
                                 char *error, size_t error_size)
{
    cl_mem temporary;
    cl_uint n = workload->elements;
    cl_int status;
    uint64_t bytes = (uint64_t)workload->bytes;

    status = g_cl.SetKernelArg(device->copy_kernel, 0, sizeof(workload->buffer_a), &workload->buffer_a);
    if (status == CL_SUCCESS) status = g_cl.SetKernelArg(device->copy_kernel, 1, sizeof(workload->buffer_b), &workload->buffer_b);
    if (status == CL_SUCCESS) status = g_cl.SetKernelArg(device->copy_kernel, 2, sizeof(n), &n);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = enqueue_kernel(device, device->copy_kernel, n);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);
    status = g_cl.Finish(device->queue);
    if (status != CL_SUCCESS) return opencl_report_step_error(status, error, error_size);

    temporary = workload->buffer_a;
    workload->buffer_a = workload->buffer_b;
    workload->buffer_b = temporary;
    metrics->bytes_read = bytes;
    metrics->bytes_written = bytes;
    metrics->iterations = 1;
    return 0;
}

static int opencl_workload_step(void *device_context,
                                crossgpu_test_type_t test,
                                void *workload_context,
                                crossgpu_metrics_t *metrics,
                                char *error,
                                size_t error_size)
{
    opencl_device_context_t *device = (opencl_device_context_t *)device_context;
    opencl_workload_context_t *workload = (opencl_workload_context_t *)workload_context;
    const opencl_test_operations_t *operations;

    if (device == NULL || workload == NULL || metrics == NULL) {
        set_error(error, error_size, "invalid OpenCL workload step arguments");
        return -1;
    }
    memset(metrics, 0, sizeof(*metrics));
    operations = opencl_find_test_operations(test);
    if (operations == NULL) {
        set_error(error, error_size, "unsupported OpenCL workload type");
        return -1;
    }
    return operations->step(device, workload, metrics, error, error_size);
}

static void opencl_workload_end(void *device_context,
                                crossgpu_test_type_t test,
                                void *workload_context)
{
    opencl_workload_context_t *workload = (opencl_workload_context_t *)workload_context;
    (void)device_context;
    (void)test;
    if (workload == NULL)
        return;
    if (workload->buffer_a != NULL) g_cl.ReleaseMemObject(workload->buffer_a);
    if (workload->buffer_b != NULL) g_cl.ReleaseMemObject(workload->buffer_b);
    if (workload->error_buffer != NULL) g_cl.ReleaseMemObject(workload->error_buffer);
    free(workload);
}

static const crossgpu_backend_plugin_t opencl_plugin = {
    .abi_version = CROSSGPU_BACKEND_ABI_VERSION,
    .struct_size = sizeof(crossgpu_backend_plugin_t),
    .kind = CROSSGPU_BACKEND_OPENCL,
    .name = "opencl",
    .initialize = opencl_initialize,
    .shutdown = opencl_shutdown,
    .enumerate_devices = opencl_enumerate_devices,
    .device_open = opencl_device_open,
    .device_close = opencl_device_close,
    .workload_begin = opencl_workload_begin,
    .workload_step = opencl_workload_step,
    .workload_end = opencl_workload_end
};

CROSSGPU_BACKEND_EXPORT const crossgpu_backend_plugin_t *crossgpu_backend_get_api(uint32_t host_abi_version)
{
    if (host_abi_version != CROSSGPU_BACKEND_ABI_VERSION)
        return NULL;
    return &opencl_plugin;
}
