#include "registry.h"

#include <string.h>

static const crossgpu_backend_descriptor_t backend_descriptors[] = {
    { CROSSGPU_BACKEND_OPENCL, "opencl", "OpenCL" },
    { CROSSGPU_BACKEND_CUDA,   "cuda",   "CUDA" },
    { CROSSGPU_BACKEND_HIP,    "hip",    "HIP" },
    { CROSSGPU_BACKEND_METAL,  "metal",  "Metal" },
    { CROSSGPU_BACKEND_LEVEL_ZERO, "level-zero", "Intel oneAPI Level Zero" },
    { CROSSGPU_BACKEND_CUSTOM, "custom", "Custom" }
};

static const crossgpu_test_descriptor_t test_descriptors[] = {
    { CROSSGPU_TEST_COMPUTE,   "compute",   "Floating-point compute stress", CROSSGPU_TEST_FLAG_DEFAULT_ALL | CROSSGPU_TEST_FLAG_COMPUTE_RATE },
    { CROSSGPU_TEST_MEMORY,    "memory",    "VRAM pattern and integrity stress", CROSSGPU_TEST_FLAG_DEFAULT_ALL | CROSSGPU_TEST_FLAG_BANDWIDTH_RATE },
    { CROSSGPU_TEST_BANDWIDTH, "bandwidth", "Device-memory bandwidth stress", CROSSGPU_TEST_FLAG_DEFAULT_ALL | CROSSGPU_TEST_FLAG_BANDWIDTH_RATE }
};

static const crossgpu_vendor_descriptor_t vendor_descriptors[] = {
    { CROSSGPU_VENDOR_NVIDIA,      "NVIDIA" },
    { CROSSGPU_VENDOR_AMD,         "AMD" },
    { CROSSGPU_VENDOR_AMD_LEGACY,  "AMD" },
    { CROSSGPU_VENDOR_INTEL,       "Intel" },
    { CROSSGPU_VENDOR_APPLE,       "Apple" },
    { CROSSGPU_VENDOR_QUALCOMM,    "Qualcomm" },
    { CROSSGPU_VENDOR_ARM,         "Arm" },
    { CROSSGPU_VENDOR_BROADCOM,    "Broadcom" },
    { CROSSGPU_VENDOR_IMAGINATION, "Imagination" }
};

/*
 * Preference rows are keyed by vendor, expected ABI kind, and plugin runtime
 * name. The runtime name lets future CUSTOM plugins (for example Vulkan or
 * oneAPI) participate without adding backend-manager conditionals or a new
 * CLI parser branch. The kind prevents an unrelated plugin from borrowing a
 * built-in name's preference.
 */
static const crossgpu_backend_preference_descriptor_t backend_preferences[] = {
    /* Native APIs are opt-in for their matching vendor only. */
    { CROSSGPU_VENDOR_NVIDIA,      CROSSGPU_BACKEND_CUDA,      "cuda",       100 },
    { CROSSGPU_VENDOR_AMD,         CROSSGPU_BACKEND_HIP,       "hip",        100 },
    { CROSSGPU_VENDOR_AMD_LEGACY,  CROSSGPU_BACKEND_HIP,       "hip",        100 },
    { CROSSGPU_VENDOR_INTEL,       CROSSGPU_BACKEND_LEVEL_ZERO,"level-zero", 100 },
    { CROSSGPU_VENDOR_APPLE,       CROSSGPU_BACKEND_METAL,     "metal",      100 },

    /* Vendor-specific OpenCL scores may make it the preferred backend. */
    { CROSSGPU_VENDOR_QUALCOMM,    CROSSGPU_BACKEND_OPENCL,    "opencl",     100 },
    { CROSSGPU_VENDOR_ARM,         CROSSGPU_BACKEND_OPENCL,    "opencl",     100 },
    { CROSSGPU_VENDOR_BROADCOM,    CROSSGPU_BACKEND_OPENCL,    "opencl",     100 },
    { CROSSGPU_VENDOR_IMAGINATION, CROSSGPU_BACKEND_OPENCL,    "opencl",     100 },

    /* OpenCL is the generic compatibility backend for every vendor not
     * assigned a higher exact-vendor preference above. */
    { CROSSGPU_VENDOR_ANY,         CROSSGPU_BACKEND_OPENCL,    "opencl",      50 }
};

const crossgpu_backend_descriptor_t *crossgpu_backend_registry(size_t *count)
{
    if (count != NULL)
        *count = sizeof(backend_descriptors) / sizeof(backend_descriptors[0]);
    return backend_descriptors;
}

const crossgpu_test_descriptor_t *crossgpu_test_registry(size_t *count)
{
    if (count != NULL)
        *count = sizeof(test_descriptors) / sizeof(test_descriptors[0]);
    return test_descriptors;
}

const crossgpu_vendor_descriptor_t *crossgpu_vendor_registry(size_t *count)
{
    if (count != NULL)
        *count = sizeof(vendor_descriptors) / sizeof(vendor_descriptors[0]);
    return vendor_descriptors;
}

const crossgpu_backend_preference_descriptor_t *crossgpu_backend_preference_registry(size_t *count)
{
    if (count != NULL)
        *count = sizeof(backend_preferences) / sizeof(backend_preferences[0]);
    return backend_preferences;
}

int crossgpu_backend_kind_parse(const char *name, crossgpu_backend_kind_t *kind)
{
    size_t i, count;
    const crossgpu_backend_descriptor_t *items;

    if (name == NULL || kind == NULL)
        return -1;
    if (strcmp(name, "auto") == 0) {
        *kind = CROSSGPU_BACKEND_AUTO;
        return 0;
    }
    items = crossgpu_backend_registry(&count);
    for (i = 0; i < count; i++) {
        if (strcmp(name, items[i].name) == 0) {
            *kind = items[i].kind;
            return 0;
        }
    }
    return -1;
}

int crossgpu_test_type_parse(const char *name, crossgpu_test_type_t *test)
{
    size_t i, count;
    const crossgpu_test_descriptor_t *items;

    if (name == NULL || test == NULL)
        return -1;
    items = crossgpu_test_registry(&count);
    for (i = 0; i < count; i++) {
        if (strcmp(name, items[i].name) == 0) {
            *test = items[i].type;
            return 0;
        }
    }
    return -1;
}

const crossgpu_test_descriptor_t *crossgpu_test_descriptor(crossgpu_test_type_t test)
{
    size_t i, count;
    const crossgpu_test_descriptor_t *items = crossgpu_test_registry(&count);

    for (i = 0; i < count; i++) {
        if (items[i].type == test)
            return &items[i];
    }
    return NULL;
}

size_t crossgpu_test_default_count(void)
{
    size_t i, count, result = 0;
    const crossgpu_test_descriptor_t *items = crossgpu_test_registry(&count);

    for (i = 0; i < count; i++)
        if ((items[i].flags & CROSSGPU_TEST_FLAG_DEFAULT_ALL) != 0)
            result++;
    return result;
}

size_t crossgpu_test_copy_default(crossgpu_test_type_t *tests, size_t capacity)
{
    size_t i, count, output = 0;
    const crossgpu_test_descriptor_t *items = crossgpu_test_registry(&count);

    for (i = 0; i < count; i++) {
        if ((items[i].flags & CROSSGPU_TEST_FLAG_DEFAULT_ALL) == 0)
            continue;
        if (tests != NULL && output < capacity)
            tests[output] = items[i].type;
        output++;
    }
    return output;
}

const char *crossgpu_backend_kind_name(crossgpu_backend_kind_t kind)
{
    size_t i, count;
    const crossgpu_backend_descriptor_t *items = crossgpu_backend_registry(&count);

    if (kind == CROSSGPU_BACKEND_AUTO)
        return "auto";
    for (i = 0; i < count; i++)
        if (items[i].kind == kind)
            return items[i].display_name;
    return "unknown";
}

const char *crossgpu_test_type_name(crossgpu_test_type_t test)
{
    const crossgpu_test_descriptor_t *descriptor = crossgpu_test_descriptor(test);
    return descriptor != NULL ? descriptor->name : "unknown";
}

const char *crossgpu_vendor_id_name(crossgpu_vendor_id_t vendor_id)
{
    size_t i, count;
    const crossgpu_vendor_descriptor_t *items = crossgpu_vendor_registry(&count);

    for (i = 0; i < count; i++)
        if (items[i].vendor_id == vendor_id)
            return items[i].name;
    return "Unknown";
}

const char *crossgpu_identity_kind_name(crossgpu_identity_kind_t kind)
{
    switch (kind) {
    case CROSSGPU_IDENTITY_NONE:
        return "none";
    case CROSSGPU_IDENTITY_PCI_BDF:
        return "pci-bdf";
    case CROSSGPU_IDENTITY_DEVICE_UUID:
        return "device-uuid";
    case CROSSGPU_IDENTITY_OS_DEVICE:
        return "os-device";
    case CROSSGPU_IDENTITY_PLATFORM_DEVICE:
        return "platform-device";
    default:
        return "plugin-defined";
    }
}

int crossgpu_vendor_backend_score(crossgpu_vendor_id_t vendor_id,
                                  crossgpu_backend_kind_t backend_kind,
                                  const char *backend_name)
{
    size_t i, count;
    int exact_score = 0;
    int wildcard_score = 0;
    int exact_match = 0;
    int wildcard_match = 0;
    const crossgpu_backend_preference_descriptor_t *items =
        crossgpu_backend_preference_registry(&count);

    /* A preference must match both the declared API kind and runtime name.
     * An absent entry means the plugin is not approved for this vendor.
     * Exact-vendor entries override wildcard entries, including score 0
     * entries which can explicitly deny a backend for one vendor. */
    if (backend_name == NULL || backend_name[0] == '\0')
        return 0;

    for (i = 0; i < count; i++) {
        if (items[i].backend_kind != backend_kind || items[i].backend_name == NULL ||
            strcmp(items[i].backend_name, backend_name) != 0)
            continue;

        if (items[i].vendor_id == (uint32_t)vendor_id) {
            if (!exact_match || items[i].score > exact_score)
                exact_score = items[i].score;
            exact_match = 1;
        } else if (items[i].vendor_id == CROSSGPU_VENDOR_ANY) {
            if (!wildcard_match || items[i].score > wildcard_score)
                wildcard_score = items[i].score;
            wildcard_match = 1;
        }
    }

    if (exact_match)
        return exact_score;
    return wildcard_match ? wildcard_score : 0;
}

const char *crossgpu_test_rate_unit(crossgpu_test_type_t test)
{
    const crossgpu_test_descriptor_t *descriptor = crossgpu_test_descriptor(test);
    if (descriptor == NULL)
        return "";
    if ((descriptor->flags & CROSSGPU_TEST_FLAG_COMPUTE_RATE) != 0)
        return "GFLOP/s";
    if ((descriptor->flags & CROSSGPU_TEST_FLAG_BANDWIDTH_RATE) != 0)
        return "GB/s";
    return "";
}
