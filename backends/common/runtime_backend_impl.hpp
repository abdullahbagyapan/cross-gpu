/*
 * Shared CUDA/HIP implementation.
 *
 * This file is included by the CUDA and HIP plugin translation units after
 * defining the CG_* runtime abstraction macros.  Each translation unit is
 * compiled independently into its own plugin shared library.
 */

#include "crossgpu_backend.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CG_BACKEND_NAME
#error "CG_BACKEND_NAME must be defined"
#endif
#ifndef CG_BACKEND_KIND
#error "CG_BACKEND_KIND must be defined"
#endif
#ifndef CG_VENDOR_ID
#error "CG_VENDOR_ID must be defined"
#endif

enum {
    CG_THREADS_PER_BLOCK = 256,
    CG_COMPUTE_INNER_ITERATIONS = 256
};

#define CG_MIN_WORKING_SET (UINT64_C(4) * UINT64_C(1024) * UINT64_C(1024))
#define CG_MAX_WORKING_SET (UINT64_C(128) * UINT64_C(1024) * UINT64_C(1024))

struct cg_device_record {
    CG_DEVICE_PROPS_T properties;
    crossgpu_device_info_t info;
};

typedef struct cg_device_context {
    int device_index;
    CG_STREAM_T stream;
} cg_device_context_t;

typedef struct cg_workload_context {
    void *buffer_a;
    void *buffer_b;
    void *error_buffer;
    size_t bytes;
    uint32_t elements;
    uint32_t seed;
} cg_workload_context_t;

static struct cg_device_record *g_cg_devices;
static size_t g_cg_device_count;
static int g_cg_initialized;

static void cg_set_error(char *error, size_t error_size, const char *message)
{
    if (error != NULL && error_size > 0)
        (void)snprintf(error, error_size, "%s", message != NULL ? message : "unknown runtime error");
}

static int cg_check(CG_ERROR_T status, const char *operation,
                    char *error, size_t error_size)
{
    char message[CROSSGPU_ERROR_MAX];

    if (status == CG_ERROR_SUCCESS)
        return 0;
    (void)snprintf(message, sizeof(message), "%s failed: %s (%d)",
                   operation, CG_GET_ERROR_STRING(status), (int)status);
    cg_set_error(error, error_size, message);
    return -1;
}

static void cg_fill_versions(char *driver, size_t driver_size,
                             char *api, size_t api_size)
{
    int runtime_version = 0;
    int driver_version = 0;
    (void)CG_GET_RUNTIME_VERSION(&runtime_version);
    (void)CG_GET_DRIVER_VERSION(&driver_version);
    if (driver_size > 0)
        (void)snprintf(driver, driver_size, "%d", driver_version);
    if (api_size > 0)
        (void)snprintf(api, api_size, "runtime %d", runtime_version);
}

static int cg_refresh_devices(char *error, size_t error_size)
{
    int count = 0;
    int index;
    struct cg_device_record *records = NULL;

    if (cg_check(CG_GET_DEVICE_COUNT(&count), "device enumeration", error, error_size) != 0)
        return -1;
    if (count < 0) {
        cg_set_error(error, error_size, "runtime returned a negative device count");
        return -1;
    }
    if (count > 0) {
        if ((size_t)count > SIZE_MAX / sizeof(*records)) {
            cg_set_error(error, error_size, "device count is too large");
            return -1;
        }
        records = (struct cg_device_record *)calloc((size_t)count, sizeof(*records));
        if (records == NULL) {
            cg_set_error(error, error_size, "out of memory allocating device registry");
            return -1;
        }
    }

    for (index = 0; index < count; index++) {
        struct cg_device_record *record = &records[index];
        crossgpu_device_info_t *info = &record->info;
        char pci_bus_id[64] = { 0 };
        CG_ERROR_T pci_status;

        if (cg_check(CG_GET_DEVICE_PROPERTIES(&record->properties, index),
                     "query device properties", error, error_size) != 0) {
            free(records);
            return -1;
        }

        memset(info, 0, sizeof(*info));
        info->struct_size = (uint32_t)sizeof(*info);
        info->backend_device_index = (uint32_t)index;
        info->vendor_id = CG_VENDOR_ID;
        info->device_id = (uint32_t)record->properties.pciDeviceID;
        info->compute_units = (uint32_t)record->properties.multiProcessorCount;
        info->clock_mhz = (uint32_t)(record->properties.clockRate / 1000);
        info->physical_memory_bytes = (uint64_t)record->properties.totalGlobalMem;
        info->max_allocation_bytes = info->physical_memory_bytes;
        (void)snprintf(info->name, sizeof(info->name), "%s", record->properties.name);
        (void)snprintf(info->vendor, sizeof(info->vendor), "%s", CG_VENDOR_NAME);
        cg_fill_versions(info->driver_version, sizeof(info->driver_version),
                         info->api_version, sizeof(info->api_version));

        pci_status = CG_GET_PCI_BUS_ID(pci_bus_id, (int)sizeof(pci_bus_id), index);
        if (pci_status == CG_ERROR_SUCCESS && pci_bus_id[0] != '\0') {
            info->identity_kind = CROSSGPU_IDENTITY_PCI_BDF;
            (void)snprintf(info->identity_namespace,
                           sizeof(info->identity_namespace), "%s",
                           CROSSGPU_IDENTITY_NAMESPACE_PCI_BDF);
            (void)snprintf(info->identity_value, sizeof(info->identity_value),
                           "%s", pci_bus_id);
        } else {
            /* A product/device ID is not a physical PCI address. Without a
             * shared physical identity, this device remains backend-local. */
        }
        (void)snprintf(info->backend_device_id, sizeof(info->backend_device_id),
                       "%s:index:%d", CG_BACKEND_NAME, index);
    }

    free(g_cg_devices);
    g_cg_devices = records;
    g_cg_device_count = (size_t)count;
    return 0;
}

static int cg_initialize(char *error, size_t error_size)
{
    if (cg_refresh_devices(error, error_size) != 0)
        return -1;
    if (g_cg_device_count == 0) {
        cg_set_error(error, error_size, "no compatible GPU devices found");
        return -1;
    }
    g_cg_initialized = 1;
    return 0;
}

static void cg_shutdown(void)
{
    free(g_cg_devices);
    g_cg_devices = NULL;
    g_cg_device_count = 0;
    g_cg_initialized = 0;
}

static int cg_enumerate_devices(crossgpu_device_info_t *devices,
                                size_t capacity,
                                size_t *device_count,
                                char *error,
                                size_t error_size)
{
    size_t copy_count;

    if (device_count == NULL || (capacity > 0 && devices == NULL)) {
        cg_set_error(error, error_size, "invalid device enumeration arguments");
        return -1;
    }
    if (!g_cg_initialized && cg_initialize(error, error_size) != 0)
        return -1;
    *device_count = g_cg_device_count;
    copy_count = g_cg_device_count < capacity ? g_cg_device_count : capacity;
    for (size_t i = 0; i < copy_count; i++)
        devices[i] = g_cg_devices[i].info;
    return 0;
}

static int cg_device_open(uint32_t index, void **device_context,
                          char *error, size_t error_size)
{
    cg_device_context_t *context;

    if (device_context == NULL || index >= g_cg_device_count) {
        cg_set_error(error, error_size, "invalid device index");
        return -1;
    }
    *device_context = NULL;
    if (cg_check(CG_SET_DEVICE((int)index), "select GPU", error, error_size) != 0)
        return -1;

    context = (cg_device_context_t *)calloc(1, sizeof(*context));
    if (context == NULL) {
        cg_set_error(error, error_size, "out of memory allocating device context");
        return -1;
    }
    context->device_index = (int)index;
    if (cg_check(CG_STREAM_CREATE(&context->stream), "create GPU command stream", error, error_size) != 0) {
        free(context);
        return -1;
    }
    *device_context = context;
    return 0;
}

static void cg_device_close(void *device_context)
{
    cg_device_context_t *context = (cg_device_context_t *)device_context;
    if (context == NULL)
        return;
    (void)CG_SET_DEVICE(context->device_index);
    if (context->stream != (CG_STREAM_T)0) {
        (void)CG_STREAM_SYNCHRONIZE(context->stream);
        (void)CG_STREAM_DESTROY(context->stream);
    }
    free(context);
}

static uint32_t cg_blocks_for_elements(uint32_t elements)
{
    return elements / CG_THREADS_PER_BLOCK +
           (elements % CG_THREADS_PER_BLOCK != 0u ? 1u : 0u);
}

static size_t cg_choose_working_set(const crossgpu_device_info_t *info, uint64_t requested)
{
    uint64_t bytes;
    uint64_t maximum = info->max_allocation_bytes != 0 ? info->max_allocation_bytes : info->physical_memory_bytes;

    if (requested != 0)
        bytes = requested;
    else {
        bytes = info->physical_memory_bytes / 16;
        if (bytes < CG_MIN_WORKING_SET)
            bytes = CG_MIN_WORKING_SET;
        if (bytes > CG_MAX_WORKING_SET)
            bytes = CG_MAX_WORKING_SET;
    }
    if (maximum != 0 && bytes > maximum)
        bytes = maximum;
    if (bytes > SIZE_MAX)
        bytes = SIZE_MAX;
    bytes &= ~(uint64_t)(sizeof(uint32_t) - 1);
    return (size_t)bytes;
}

static int cg_release_workload(cg_device_context_t *device, cg_workload_context_t *workload)
{
    int failed = 0;
    if (device != NULL)
        (void)CG_SET_DEVICE(device->device_index);
    if (workload == NULL)
        return 0;
    if (device != NULL && device->stream != (CG_STREAM_T)0)
        (void)CG_STREAM_SYNCHRONIZE(device->stream);
    if (workload->buffer_a != NULL && CG_FREE(workload->buffer_a) != CG_ERROR_SUCCESS)
        failed = 1;
    if (workload->buffer_b != NULL && CG_FREE(workload->buffer_b) != CG_ERROR_SUCCESS)
        failed = 1;
    if (workload->error_buffer != NULL && CG_FREE(workload->error_buffer) != CG_ERROR_SUCCESS)
        failed = 1;
    free(workload);
    return failed ? -1 : 0;
}

static int cg_workload_begin(void *device_context,
                             crossgpu_test_type_t test,
                             const crossgpu_workload_config_t *config,
                             void **workload_context,
                             char *error,
                             size_t error_size)
{
    cg_device_context_t *device = (cg_device_context_t *)device_context;
    cg_workload_context_t *workload;
    size_t bytes;
    CG_ERROR_T status;

    if (device == NULL || config == NULL || workload_context == NULL) {
        cg_set_error(error, error_size, "invalid workload initialization arguments");
        return -1;
    }
    *workload_context = NULL;
    if (test != CROSSGPU_TEST_COMPUTE && test != CROSSGPU_TEST_MEMORY && test != CROSSGPU_TEST_BANDWIDTH) {
        cg_set_error(error, error_size, "unsupported workload type");
        return -1;
    }
    if (cg_check(CG_SET_DEVICE(device->device_index), "select GPU", error, error_size) != 0)
        return -1;

    bytes = cg_choose_working_set(&g_cg_devices[device->device_index].info, config->memory_bytes);
    if (bytes < 4096 || bytes / sizeof(uint32_t) > UINT32_MAX) {
        cg_set_error(error, error_size, "GPU cannot provide a usable working set");
        return -1;
    }

    workload = (cg_workload_context_t *)calloc(1, sizeof(*workload));
    if (workload == NULL) {
        cg_set_error(error, error_size, "out of memory allocating workload context");
        return -1;
    }
    workload->bytes = bytes;
    workload->elements = (uint32_t)(bytes / sizeof(uint32_t));
    workload->seed = 1;

    status = CG_MALLOC(&workload->buffer_a, bytes);
    if (cg_check(status, "allocate primary GPU buffer", error, error_size) != 0)
        goto fail;
    status = CG_MALLOC(&workload->error_buffer, sizeof(uint32_t));
    if (cg_check(status, "allocate GPU validation counter", error, error_size) != 0)
        goto fail;
    if (test == CROSSGPU_TEST_BANDWIDTH) {
        status = CG_MALLOC(&workload->buffer_b, bytes);
        if (cg_check(status, "allocate secondary GPU buffer", error, error_size) != 0)
            goto fail;
    }
    if (cg_check(CG_MEMSET_ASYNC(workload->buffer_a, 0, bytes, device->stream),
                 "initialize primary GPU buffer", error, error_size) != 0)
        goto fail;
    if (workload->buffer_b != NULL &&
        cg_check(CG_MEMSET_ASYNC(workload->buffer_b, 0, bytes, device->stream),
                 "initialize secondary GPU buffer", error, error_size) != 0)
        goto fail;
    if (cg_check(CG_MEMSET_ASYNC(workload->error_buffer, 0, sizeof(uint32_t), device->stream),
                 "initialize validation counter", error, error_size) != 0)
        goto fail;

    if (test == CROSSGPU_TEST_COMPUTE) {
        uint32_t blocks = cg_blocks_for_elements(workload->elements);
        cg_initialize_float_kernel<<<blocks, CG_THREADS_PER_BLOCK, 0, device->stream>>>(
            (float *)workload->buffer_a, workload->elements);
        if (cg_check(CG_GET_LAST_ERROR(), "launch compute initialization kernel", error, error_size) != 0)
            goto fail;
    }
    if (cg_check(CG_STREAM_SYNCHRONIZE(device->stream), "synchronize workload initialization", error, error_size) != 0)
        goto fail;

    *workload_context = workload;
    return 0;

fail:
    (void)cg_release_workload(device, workload);
    return -1;
}

static int cg_copy_validation_count(cg_device_context_t *device,
                                     cg_workload_context_t *workload,
                                     uint32_t *count,
                                     char *error,
                                     size_t error_size)
{
    if (cg_check(CG_MEMCPY_ASYNC(count, workload->error_buffer, sizeof(*count),
                                 CG_MEMCPY_DEVICE_TO_HOST, device->stream),
                 "read GPU validation counter", error, error_size) != 0)
        return -1;
    return cg_check(CG_STREAM_SYNCHRONIZE(device->stream), "synchronize GPU workload", error, error_size);
}

static int cg_workload_step(void *device_context,
                            crossgpu_test_type_t test,
                            void *workload_context,
                            crossgpu_metrics_t *metrics,
                            char *error,
                            size_t error_size)
{
    cg_device_context_t *device = (cg_device_context_t *)device_context;
    cg_workload_context_t *workload = (cg_workload_context_t *)workload_context;
    uint32_t errors = 0;
    uint32_t blocks;
    uint64_t bytes;

    if (device == NULL || workload == NULL || metrics == NULL) {
        cg_set_error(error, error_size, "invalid workload step arguments");
        return -1;
    }
    memset(metrics, 0, sizeof(*metrics));
    if (cg_check(CG_SET_DEVICE(device->device_index), "select GPU", error, error_size) != 0)
        return -1;
    if (cg_check(CG_MEMSET_ASYNC(workload->error_buffer, 0, sizeof(uint32_t), device->stream),
                 "reset GPU validation counter", error, error_size) != 0)
        return -1;

    blocks = cg_blocks_for_elements(workload->elements);
    bytes = (uint64_t)workload->bytes;

    switch (test) {
    case CROSSGPU_TEST_COMPUTE:
        cg_compute_kernel<<<blocks, CG_THREADS_PER_BLOCK, 0, device->stream>>>(
            (float *)workload->buffer_a, (uint32_t *)workload->error_buffer, workload->elements);
        if (cg_check(CG_GET_LAST_ERROR(), "launch compute kernel", error, error_size) != 0)
            return -1;
        if (cg_copy_validation_count(device, workload, &errors, error, error_size) != 0)
            return -1;
        metrics->operations = (uint64_t)workload->elements * CG_COMPUTE_INNER_ITERATIONS * 2;
        metrics->bytes_read = bytes;
        metrics->bytes_written = bytes;
        break;

    case CROSSGPU_TEST_MEMORY:
        workload->seed++;
        cg_pattern_write_kernel<<<blocks, CG_THREADS_PER_BLOCK, 0, device->stream>>>(
            (uint32_t *)workload->buffer_a, workload->elements, workload->seed);
        if (cg_check(CG_GET_LAST_ERROR(), "launch memory pattern write kernel", error, error_size) != 0)
            return -1;
        cg_pattern_verify_kernel<<<blocks, CG_THREADS_PER_BLOCK, 0, device->stream>>>(
            (const uint32_t *)workload->buffer_a, (uint32_t *)workload->error_buffer,
            workload->elements, workload->seed);
        if (cg_check(CG_GET_LAST_ERROR(), "launch memory pattern verification kernel", error, error_size) != 0)
            return -1;
        if (cg_copy_validation_count(device, workload, &errors, error, error_size) != 0)
            return -1;
        metrics->bytes_read = bytes;
        metrics->bytes_written = bytes;
        break;

    case CROSSGPU_TEST_BANDWIDTH:
        if (workload->buffer_b == NULL) {
            cg_set_error(error, error_size, "bandwidth workload has no secondary GPU buffer");
            return -1;
        }
        cg_copy_kernel<<<blocks, CG_THREADS_PER_BLOCK, 0, device->stream>>>(
            (const uint32_t *)workload->buffer_a, (uint32_t *)workload->buffer_b,
            workload->elements);
        if (cg_check(CG_GET_LAST_ERROR(), "launch bandwidth copy kernel", error, error_size) != 0)
            return -1;
        if (cg_check(CG_STREAM_SYNCHRONIZE(device->stream), "synchronize GPU copy", error, error_size) != 0)
            return -1;
        {
            void *temporary = workload->buffer_a;
            workload->buffer_a = workload->buffer_b;
            workload->buffer_b = temporary;
        }
        metrics->bytes_read = bytes;
        metrics->bytes_written = bytes;
        break;

    default:
        cg_set_error(error, error_size, "unsupported workload type");
        return -1;
    }

    metrics->errors = errors;
    metrics->iterations = 1;
    return 0;
}

static void cg_workload_end(void *device_context,
                            crossgpu_test_type_t test,
                            void *workload_context)
{
    (void)test;
    (void)cg_release_workload((cg_device_context_t *)device_context,
                              (cg_workload_context_t *)workload_context);
}

/* Positional initialization keeps this shared implementation valid under
 * the C++14 baseline used by the CUDA and HIP compiler drivers. */
static const crossgpu_backend_plugin_t cg_plugin = {
    CROSSGPU_BACKEND_ABI_VERSION,
    sizeof(crossgpu_backend_plugin_t),
    CG_BACKEND_KIND,
    CG_BACKEND_NAME,
    cg_initialize,
    cg_shutdown,
    cg_enumerate_devices,
    cg_device_open,
    cg_device_close,
    cg_workload_begin,
    cg_workload_step,
    cg_workload_end
};

extern "C" CROSSGPU_BACKEND_EXPORT
const crossgpu_backend_plugin_t *crossgpu_backend_get_api(uint32_t host_abi_version)
{
    return host_abi_version == CROSSGPU_BACKEND_ABI_VERSION ? &cg_plugin : NULL;
}
