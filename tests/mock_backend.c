#include "crossgpu_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MOCK_BACKEND_KIND
#define MOCK_BACKEND_KIND CROSSGPU_BACKEND_OPENCL
#endif
#ifndef MOCK_BACKEND_NAME
#define MOCK_BACKEND_NAME "opencl-mock-test"
#endif
#ifndef MOCK_VENDOR_ID
#define MOCK_VENDOR_ID CROSSGPU_VENDOR_NVIDIA
#endif
#ifndef MOCK_FAIL_OPEN
#define MOCK_FAIL_OPEN 0
#endif
#ifndef MOCK_IDENTITY_KIND
#define MOCK_IDENTITY_KIND CROSSGPU_IDENTITY_PCI_BDF
#endif
#ifndef MOCK_IDENTITY_NAMESPACE
#define MOCK_IDENTITY_NAMESPACE CROSSGPU_IDENTITY_NAMESPACE_PCI_BDF
#endif
#ifndef MOCK_IDENTITY_VALUE
#define MOCK_IDENTITY_VALUE "0000:01:00.0"
#endif

typedef struct mock_device_context {
    unsigned int marker;
} mock_device_context_t;

typedef struct mock_workload_context {
    crossgpu_test_type_t test;
} mock_workload_context_t;

static int mock_initialize(char *error, size_t error_size)
{
    (void)error;
    (void)error_size;
    return 0;
}

static int mock_enumerate(crossgpu_device_info_t *devices, size_t capacity,
                          size_t *count, char *error, size_t error_size)
{
    (void)error;
    (void)error_size;
    if (count == NULL || (capacity > 0 && devices == NULL))
        return -1;
    *count = 1;
    if (capacity == 0)
        return 0;
    if (devices == NULL)
        return -1;
    memset(&devices[0], 0, sizeof(devices[0]));
    devices[0].struct_size = sizeof(devices[0]);
    devices[0].backend_device_index = 0;
    devices[0].vendor_id = (crossgpu_vendor_id_t)MOCK_VENDOR_ID;
    devices[0].device_id = 0x5678;
    devices[0].compute_units = 8;
    devices[0].physical_memory_bytes = UINT64_C(2) * 1024 * 1024 * 1024;
    devices[0].max_allocation_bytes = UINT64_C(512) * 1024 * 1024;
    devices[0].identity_kind = MOCK_IDENTITY_KIND;
    (void)snprintf(devices[0].identity_namespace,
                   sizeof(devices[0].identity_namespace), "%s",
                   MOCK_IDENTITY_NAMESPACE);
    (void)snprintf(devices[0].identity_value,
                   sizeof(devices[0].identity_value), "%s", MOCK_IDENTITY_VALUE);
    (void)snprintf(devices[0].backend_device_id,
                   sizeof(devices[0].backend_device_id), "%s:index:0", MOCK_BACKEND_NAME);
    (void)snprintf(devices[0].name, sizeof(devices[0].name), "Mock GPU (test backend)");
    (void)snprintf(devices[0].vendor, sizeof(devices[0].vendor), "%s",
                   MOCK_VENDOR_ID == CROSSGPU_VENDOR_NVIDIA ? "NVIDIA (mock test)" :
                   MOCK_VENDOR_ID == CROSSGPU_VENDOR_INTEL ? "Intel (mock test)" :
                   "CrossGPU Mock Vendor");
    (void)snprintf(devices[0].driver_version, sizeof(devices[0].driver_version), "mock-1.0");
    (void)snprintf(devices[0].api_version, sizeof(devices[0].api_version), "mock-api-1");
    return 0;
}

static int mock_device_open(uint32_t index, void **context, char *error, size_t error_size)
{
    mock_device_context_t *device;
    (void)error;
    (void)error_size;
    if (index != 0 || context == NULL)
        return -1;
#if MOCK_FAIL_OPEN
    if (error != NULL && error_size > 0)
        (void)snprintf(error, error_size, "mock preferred backend unavailable");
    return -1;
#else
    (void)error;
    (void)error_size;
#endif
    device = calloc(1, sizeof(*device));
    if (device == NULL)
        return -1;
    device->marker = 1;
    *context = device;
    return 0;
}

static void mock_device_close(void *context)
{
    free(context);
}

static int mock_workload_begin(void *device, crossgpu_test_type_t test,
                               const crossgpu_workload_config_t *config,
                               void **context, char *error, size_t error_size)
{
    mock_workload_context_t *workload;
    (void)device;
    (void)config;
    (void)error;
    (void)error_size;
    workload = calloc(1, sizeof(*workload));
    if (workload == NULL)
        return -1;
    workload->test = test;
    *context = workload;
    return 0;
}

static int mock_workload_step(void *device, crossgpu_test_type_t test, void *context,
                              crossgpu_metrics_t *metrics, char *error, size_t error_size)
{
    (void)device;
    (void)context;
    (void)error;
    (void)error_size;
    memset(metrics, 0, sizeof(*metrics));
    metrics->iterations = 1;
    if (test == CROSSGPU_TEST_COMPUTE)
        metrics->operations = 1000000;
    else {
        metrics->bytes_read = 1024 * 1024;
        metrics->bytes_written = 1024 * 1024;
    }
    return 0;
}

static void mock_workload_end(void *device, crossgpu_test_type_t test, void *context)
{
    (void)device;
    (void)test;
    free(context);
}

static const crossgpu_backend_plugin_t mock_plugin = {
    .abi_version = CROSSGPU_BACKEND_ABI_VERSION,
    .struct_size = sizeof(crossgpu_backend_plugin_t),
    .kind = MOCK_BACKEND_KIND,
    .name = MOCK_BACKEND_NAME,
    .initialize = mock_initialize,
    .shutdown = NULL,
    .enumerate_devices = mock_enumerate,
    .device_open = mock_device_open,
    .device_close = mock_device_close,
    .workload_begin = mock_workload_begin,
    .workload_step = mock_workload_step,
    .workload_end = mock_workload_end
};

CROSSGPU_BACKEND_EXPORT const crossgpu_backend_plugin_t *crossgpu_backend_get_api(uint32_t host_abi_version)
{
    return host_abi_version == CROSSGPU_BACKEND_ABI_VERSION ? &mock_plugin : NULL;
}
