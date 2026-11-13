#ifndef CROSSGPU_BACKEND_H
#define CROSSGPU_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include "crossgpu_platform_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initial public plugin ABI. Incompatible changes require a version bump. */
#define CROSSGPU_BACKEND_ABI_VERSION 1u
#define CROSSGPU_BACKEND_ENTRYPOINT "crossgpu_backend_get_api"

#define CROSSGPU_NAME_MAX       128
#define CROSSGPU_VENDOR_MAX      64
#define CROSSGPU_VERSION_MAX    128
#define CROSSGPU_DEVICE_ID_MAX  128
#define CROSSGPU_IDENTITY_NAMESPACE_MAX 64
#define CROSSGPU_IDENTITY_VALUE_MAX 128
#define CROSSGPU_ERROR_MAX      512

#if CROSSGPU_PLATFORM_WINDOWS
#define CROSSGPU_BACKEND_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define CROSSGPU_BACKEND_EXPORT __attribute__((visibility("default")))
#else
#define CROSSGPU_BACKEND_EXPORT
#endif

typedef enum crossgpu_backend_kind {
    CROSSGPU_BACKEND_AUTO = 0,
    CROSSGPU_BACKEND_OPENCL = 1,
    CROSSGPU_BACKEND_CUDA = 2,
    CROSSGPU_BACKEND_HIP = 3,
    CROSSGPU_BACKEND_METAL = 4,
    CROSSGPU_BACKEND_LEVEL_ZERO = 5,
    /* Reserved for independently developed plugins with their own name. */
    CROSSGPU_BACKEND_CUSTOM = 255
} crossgpu_backend_kind_t;

/* PCI vendor IDs are represented as named enum constants throughout the ABI. */
typedef enum crossgpu_vendor_id {
    CROSSGPU_VENDOR_UNKNOWN = 0x0000,
    CROSSGPU_VENDOR_NVIDIA = 0x10de,
    CROSSGPU_VENDOR_AMD = 0x1002,
    CROSSGPU_VENDOR_AMD_LEGACY = 0x1022,
    CROSSGPU_VENDOR_INTEL = 0x8086,
    CROSSGPU_VENDOR_APPLE = 0x106b,
    CROSSGPU_VENDOR_QUALCOMM = 0x5143,
    CROSSGPU_VENDOR_ARM = 0x13b5,
    CROSSGPU_VENDOR_BROADCOM = 0x14e4,
    CROSSGPU_VENDOR_IMAGINATION = 0x1010
} crossgpu_vendor_id_t;

/*
 * Physical identity categories. A non-NONE identity is a claim that the
 * identity can be compared with values reported by other backend APIs.
 * The manager does not infer semantics from string prefixes.
 */
typedef enum crossgpu_identity_kind {
    CROSSGPU_IDENTITY_NONE = 0,
    CROSSGPU_IDENTITY_PCI_BDF = 1,
    CROSSGPU_IDENTITY_DEVICE_UUID = 2,
    CROSSGPU_IDENTITY_OS_DEVICE = 3,
    CROSSGPU_IDENTITY_PLATFORM_DEVICE = 4
} crossgpu_identity_kind_t;

/* Canonical namespace names shared by plugins that expose the same ID space. */
#define CROSSGPU_IDENTITY_NAMESPACE_PCI_BDF "pci-bdf"
#define CROSSGPU_IDENTITY_NAMESPACE_APPLE_IOREGISTRY "apple.iokit.registry-id"

#if defined(__cplusplus)
static_assert(sizeof(crossgpu_vendor_id_t) == sizeof(uint32_t),
              "crossgpu_vendor_id_t must remain 32-bit for the plugin ABI");
static_assert(sizeof(crossgpu_identity_kind_t) == sizeof(uint32_t),
              "crossgpu_identity_kind_t must remain 32-bit for the plugin ABI");
#else
_Static_assert(sizeof(crossgpu_vendor_id_t) == sizeof(uint32_t),
               "crossgpu_vendor_id_t must remain 32-bit for the plugin ABI");
_Static_assert(sizeof(crossgpu_identity_kind_t) == sizeof(uint32_t),
               "crossgpu_identity_kind_t must remain 32-bit for the plugin ABI");
#endif

typedef enum crossgpu_test_type {
    CROSSGPU_TEST_COMPUTE = 1,
    CROSSGPU_TEST_MEMORY = 2,
    CROSSGPU_TEST_BANDWIDTH = 3
} crossgpu_test_type_t;

typedef struct crossgpu_device_info {
    uint32_t struct_size;
    uint32_t backend_device_index;
    crossgpu_vendor_id_t vendor_id;
    uint32_t device_id;
    uint32_t compute_units;
    uint32_t clock_mhz;
    uint64_t physical_memory_bytes;   /* 0 when not exposed by the backend */
    uint64_t working_set_bytes;      /* recommended/usable working set; 0 if unknown */
    uint64_t max_allocation_bytes;   /* maximum single allocation; 0 if unknown */
    crossgpu_identity_kind_t identity_kind;
    char identity_namespace[CROSSGPU_IDENTITY_NAMESPACE_MAX];
    char identity_value[CROSSGPU_IDENTITY_VALUE_MAX];
    /* Backend-local ID for diagnostics; never used for cross-backend merging. */
    char backend_device_id[CROSSGPU_DEVICE_ID_MAX];
    char name[CROSSGPU_NAME_MAX];
    char vendor[CROSSGPU_VENDOR_MAX];
    char driver_version[CROSSGPU_VERSION_MAX];
    char api_version[CROSSGPU_VERSION_MAX];
} crossgpu_device_info_t;

typedef struct crossgpu_workload_config {
    uint32_t struct_size;
    uint64_t memory_bytes;       /* 0 means backend-selected conservative default */
    uint32_t batch_iterations;   /* reserved for workload tuning */
} crossgpu_workload_config_t;

typedef struct crossgpu_metrics {
    uint64_t operations;         /* floating-point operations for compute workload */
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t errors;
    uint64_t iterations;
} crossgpu_metrics_t;

typedef struct crossgpu_backend_plugin {
    uint32_t abi_version;
    uint32_t struct_size;
    crossgpu_backend_kind_t kind;
    const char *name;

    int  (*initialize)(char *error, size_t error_size);
    void (*shutdown)(void);

    /*
     * Device enumeration supports a size query: call with devices == NULL and
     * capacity == 0 to obtain device_count, then call again with enough storage.
     */
    int (*enumerate_devices)(crossgpu_device_info_t *devices,
                             size_t capacity,
                             size_t *device_count,
                             char *error,
                             size_t error_size);

    int (*device_open)(uint32_t backend_device_index,
                       void **device_context,
                       char *error,
                       size_t error_size);
    void (*device_close)(void *device_context);

    int (*workload_begin)(void *device_context,
                          crossgpu_test_type_t test,
                          const crossgpu_workload_config_t *config,
                          void **workload_context,
                          char *error,
                          size_t error_size);
    int (*workload_step)(void *device_context,
                         crossgpu_test_type_t test,
                         void *workload_context,
                         crossgpu_metrics_t *metrics,
                         char *error,
                         size_t error_size);
    void (*workload_end)(void *device_context,
                         crossgpu_test_type_t test,
                         void *workload_context);
} crossgpu_backend_plugin_t;

typedef const crossgpu_backend_plugin_t *(*crossgpu_backend_get_api_fn)(uint32_t host_abi_version);

const char *crossgpu_backend_kind_name(crossgpu_backend_kind_t kind);
const char *crossgpu_test_type_name(crossgpu_test_type_t test);
const char *crossgpu_vendor_id_name(crossgpu_vendor_id_t vendor_id);

#ifdef __cplusplus
}
#endif

#endif
