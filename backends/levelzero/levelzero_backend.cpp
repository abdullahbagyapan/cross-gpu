#include "crossgpu_backend.h"

#include <level_zero/ze_api.h>

#include <algorithm>
#include <cstddef>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

extern "C" {
extern const unsigned char crossgpu_levelzero_kernels_spv[];
extern const size_t crossgpu_levelzero_kernels_spv_size;
}

namespace {

constexpr uint32_t kIntelPciVendorId = 0x8086u;
constexpr uint32_t kWorkgroupSize = 64u;
constexpr uint32_t kComputeInnerIterations = 256u;
constexpr uint64_t kMinWorkingSet = UINT64_C(4) * 1024u * 1024u;
constexpr uint64_t kDefaultWorkingSet = UINT64_C(64) * 1024u * 1024u;
constexpr uint64_t kMaxWorkingSet = UINT64_C(128) * 1024u * 1024u;

struct device_record {
    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    ze_device_properties_t properties{};
    crossgpu_device_info_t info{};
    uint32_t memory_ordinal = 0;
};

struct device_context {
    const device_record *record = nullptr;
    ze_context_handle_t context = nullptr;
    ze_command_queue_handle_t queue = nullptr;
    ze_command_list_handle_t command_list = nullptr;
    ze_module_handle_t module = nullptr;
    ze_kernel_handle_t compute_kernel = nullptr;
    ze_kernel_handle_t pattern_write_kernel = nullptr;
    ze_kernel_handle_t pattern_verify_kernel = nullptr;
    uint32_t queue_group_ordinal = 0;
};

struct workload_context {
    void *buffer_a = nullptr;
    void *buffer_b = nullptr;
    void *error_buffer = nullptr;
    size_t bytes = 0;
    uint32_t elements = 0;
    uint32_t seed = 1;
    uint32_t host_errors = 0;
};

using pci_properties_fn = ze_result_t (ZE_APICALL *)(
    ze_device_handle_t, ze_pci_ext_properties_t *);

std::vector<device_record> g_devices;
bool g_initialized = false;

void set_error(char *error, size_t error_size, const char *message)
{
    if (error != nullptr && error_size > 0)
        (void)std::snprintf(error, error_size, "%s", message != nullptr ? message : "unknown Level Zero error");
}

void set_result_error(char *error, size_t error_size,
                      const char *operation, ze_result_t result)
{
    char message[CROSSGPU_ERROR_MAX];
    (void)std::snprintf(message, sizeof(message), "%s failed (Level Zero result 0x%08x)",
                        operation, static_cast<unsigned int>(result));
    set_error(error, error_size, message);
}

bool check_result(ze_result_t result, const char *operation,
                  char *error, size_t error_size)
{
    if (result == ZE_RESULT_SUCCESS)
        return true;
    set_result_error(error, error_size, operation, result);
    return false;
}

bool uuid_is_zero(const ze_device_uuid_t &uuid)
{
    for (uint8_t byte : uuid.id)
        if (byte != 0)
            return false;
    return true;
}

void assign_physical_identity(device_record &record)
{
    void *function_address = nullptr;
    pci_properties_fn get_pci_properties = nullptr;

    if (zeDriverGetExtensionFunctionAddress(
            record.driver, "zeDevicePciGetPropertiesExt", &function_address) == ZE_RESULT_SUCCESS &&
        function_address != nullptr) {
        static_assert(sizeof(get_pci_properties) == sizeof(function_address),
                      "function pointer size is unsupported on this platform");
        std::memcpy(&get_pci_properties, &function_address, sizeof(get_pci_properties));

        ze_pci_ext_properties_t pci{};
        pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
        pci.pNext = nullptr;
        if (get_pci_properties(record.device, &pci) == ZE_RESULT_SUCCESS) {
            const ze_pci_address_ext_t &address = pci.address;
            record.info.identity_kind = CROSSGPU_IDENTITY_PCI_BDF;
            (void)std::snprintf(record.info.identity_namespace,
                                sizeof(record.info.identity_namespace), "%s",
                                CROSSGPU_IDENTITY_NAMESPACE_PCI_BDF);
            (void)std::snprintf(record.info.identity_value,
                                sizeof(record.info.identity_value), "%04x:%02x:%02x.%x",
                                address.domain, address.bus, address.device, address.function);
            return;
        }
    }

    /* A Level Zero UUID is a useful Level Zero-local identity. It is deliberately
     * namespaced separately from UUIDs reported by other APIs unless a backend
     * explicitly documents that the UUID namespaces are equivalent. */
    if (!uuid_is_zero(record.properties.uuid)) {
        char uuid_text[ZE_MAX_DEVICE_UUID_SIZE * 2 + 1];
        for (size_t i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; ++i)
            (void)std::snprintf(uuid_text + i * 2, sizeof(uuid_text) - i * 2,
                                "%02x", record.properties.uuid.id[i]);
        uuid_text[sizeof(uuid_text) - 1] = '\0';
        record.info.identity_kind = CROSSGPU_IDENTITY_DEVICE_UUID;
        (void)std::snprintf(record.info.identity_namespace,
                            sizeof(record.info.identity_namespace), "%s",
                            "level-zero.device-uuid");
        (void)std::snprintf(record.info.identity_value,
                            sizeof(record.info.identity_value), "%s", uuid_text);
    }
}

uint64_t query_device_memory(ze_device_handle_t device, uint32_t *preferred_ordinal)
{
    uint32_t count = 0;
    if (preferred_ordinal != nullptr)
        *preferred_ordinal = 0;
    if (zeDeviceGetMemoryProperties(device, &count, nullptr) != ZE_RESULT_SUCCESS || count == 0)
        return 0;

    std::vector<ze_device_memory_properties_t> properties(count);
    for (auto &property : properties) {
        std::memset(&property, 0, sizeof(property));
        property.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    }
    if (zeDeviceGetMemoryProperties(device, &count, properties.data()) != ZE_RESULT_SUCCESS)
        return 0;

    uint64_t total = 0;
    uint64_t largest_bank = 0;
    uint32_t largest_bank_ordinal = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (properties[i].totalSize > std::numeric_limits<uint64_t>::max() - total)
            return 0;
        total += properties[i].totalSize;
        if (properties[i].totalSize > largest_bank) {
            largest_bank = properties[i].totalSize;
            largest_bank_ordinal = i;
        }
    }

    /* zeMemAllocDevice's ordinal is the index in this exact properties list.
     * Allocate from the largest local-memory bank instead of assuming ordinal 0. */
    if (preferred_ordinal != nullptr)
        *preferred_ordinal = largest_bank_ordinal;
    return total;
}

uint64_t default_working_set(const crossgpu_device_info_t &info)
{
    uint64_t bytes = info.physical_memory_bytes != 0
                         ? info.physical_memory_bytes / 16
                         : kDefaultWorkingSet;
    bytes = std::max(bytes, kMinWorkingSet);
    bytes = std::min(bytes, kMaxWorkingSet);
    if (info.max_allocation_bytes != 0)
        bytes = std::min(bytes, info.max_allocation_bytes);
    return bytes & ~UINT64_C(3);
}

void fill_device_info(device_record &record, uint32_t index)
{
    ze_driver_properties_t driver_properties{};
    ze_api_version_t api_version = 0;

    record.info = {};
    record.info.struct_size = static_cast<uint32_t>(sizeof(record.info));
    record.info.backend_device_index = index;
    record.info.vendor_id = CROSSGPU_VENDOR_INTEL;
    record.info.device_id = record.properties.deviceId;
    record.info.clock_mhz = record.properties.coreClockRate;
    {
        const uint64_t eus = static_cast<uint64_t>(record.properties.numSlices) *
                             record.properties.numSubslicesPerSlice *
                             record.properties.numEUsPerSubslice;
        record.info.compute_units = eus > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(eus);
    }
    record.info.physical_memory_bytes = query_device_memory(record.device,
                                                            &record.memory_ordinal);
    record.info.max_allocation_bytes = record.properties.maxMemAllocSize;
    record.info.working_set_bytes = default_working_set(record.info);
    (void)std::snprintf(record.info.name, sizeof(record.info.name), "%s",
                        record.properties.name[0] != '\0' ? record.properties.name : "Intel GPU");
    (void)std::snprintf(record.info.vendor, sizeof(record.info.vendor), "Intel");

    driver_properties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
    if (zeDriverGetProperties(record.driver, &driver_properties) == ZE_RESULT_SUCCESS)
        (void)std::snprintf(record.info.driver_version, sizeof(record.info.driver_version),
                            "%u", driver_properties.driverVersion);
    if (zeDriverGetApiVersion(record.driver, &api_version) == ZE_RESULT_SUCCESS) {
        (void)std::snprintf(record.info.api_version, sizeof(record.info.api_version),
                            "Level Zero %u.%u", ZE_MAJOR_VERSION(api_version), ZE_MINOR_VERSION(api_version));
    } else {
        (void)std::snprintf(record.info.api_version, sizeof(record.info.api_version), "Level Zero");
    }

    assign_physical_identity(record);
    (void)std::snprintf(record.info.backend_device_id,
                        sizeof(record.info.backend_device_id), "level-zero:index:%u", index);
}

int refresh_devices(char *error, size_t error_size)
{
    ze_init_driver_type_desc_t init_desc{};
    init_desc.stype = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC;
    init_desc.pNext = nullptr;
    init_desc.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;

    uint32_t driver_count = 0;
    ze_result_t result = zeInitDrivers(&driver_count, nullptr, &init_desc);
    if (!check_result(result, "enumerate Level Zero GPU drivers", error, error_size))
        return -1;
    if (driver_count == 0) {
        set_error(error, error_size, "Level Zero reports no GPU drivers");
        return -1;
    }

    std::vector<ze_driver_handle_t> drivers(driver_count);
    result = zeInitDrivers(&driver_count, drivers.data(), &init_desc);
    if (!check_result(result, "retrieve Level Zero GPU drivers", error, error_size))
        return -1;

    std::vector<device_record> found;
    for (ze_driver_handle_t driver : drivers) {
        uint32_t device_count = 0;
        result = zeDeviceGet(driver, &device_count, nullptr);
        if (result != ZE_RESULT_SUCCESS || device_count == 0)
            continue;

        std::vector<ze_device_handle_t> devices(device_count);
        result = zeDeviceGet(driver, &device_count, devices.data());
        if (result != ZE_RESULT_SUCCESS)
            continue;

        for (ze_device_handle_t device : devices) {
            device_record record{};
            record.driver = driver;
            record.device = device;
            record.properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            result = zeDeviceGetProperties(device, &record.properties);
            if (result != ZE_RESULT_SUCCESS)
                continue;

            /* Level Zero can expose sub-devices/tiles as separate handles.
             * CrossGPU lists the root physical GPU once, not each tile. */
            if (record.properties.type != ZE_DEVICE_TYPE_GPU ||
                record.properties.vendorId != kIntelPciVendorId ||
                (record.properties.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) != 0)
                continue;

            fill_device_info(record, static_cast<uint32_t>(found.size()));
            found.push_back(record);
        }
    }

    if (found.empty()) {
        set_error(error, error_size, "no Intel GPU devices found through Level Zero");
        return -1;
    }

    g_devices.swap(found);
    return 0;
}

int initialize(char *error, size_t error_size)
{
    if (g_initialized)
        return 0;
    if (refresh_devices(error, error_size) != 0)
        return -1;
    g_initialized = true;
    return 0;
}

void shutdown()
{
    g_devices.clear();
    g_initialized = false;
}

int enumerate_devices(crossgpu_device_info_t *devices, size_t capacity,
                      size_t *device_count, char *error, size_t error_size)
{
    if (device_count == nullptr || (capacity > 0 && devices == nullptr)) {
        set_error(error, error_size, "invalid Level Zero enumeration arguments");
        return -1;
    }
    if (!g_initialized && initialize(error, error_size) != 0)
        return -1;

    *device_count = g_devices.size();
    const size_t copy_count = std::min(capacity, g_devices.size());
    for (size_t i = 0; i < copy_count; ++i)
        devices[i] = g_devices[i].info;
    return 0;
}

void close_device_resources(device_context *context)
{
    if (context == nullptr)
        return;
    if (context->queue != nullptr)
        (void)zeCommandQueueSynchronize(context->queue, UINT64_MAX);
    if (context->compute_kernel != nullptr)
        (void)zeKernelDestroy(context->compute_kernel);
    if (context->pattern_write_kernel != nullptr)
        (void)zeKernelDestroy(context->pattern_write_kernel);
    if (context->pattern_verify_kernel != nullptr)
        (void)zeKernelDestroy(context->pattern_verify_kernel);
    if (context->module != nullptr)
        (void)zeModuleDestroy(context->module);
    if (context->command_list != nullptr)
        (void)zeCommandListDestroy(context->command_list);
    if (context->queue != nullptr)
        (void)zeCommandQueueDestroy(context->queue);
    if (context->context != nullptr)
        (void)zeContextDestroy(context->context);
    delete context;
}

bool create_kernel(ze_module_handle_t module, const char *name,
                   ze_kernel_handle_t *kernel,
                   char *error, size_t error_size)
{
    ze_kernel_desc_t descriptor{};
    descriptor.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    descriptor.pKernelName = name;
    if (!check_result(zeKernelCreate(module, &descriptor, kernel), name, error, error_size))
        return false;
    if (!check_result(zeKernelSetGroupSize(*kernel, kWorkgroupSize, 1, 1),
                      "set Level Zero kernel workgroup size", error, error_size))
        return false;
    return true;
}

int device_open(uint32_t index, void **device_context_out,
                char *error, size_t error_size)
{
    if (device_context_out == nullptr || index >= g_devices.size()) {
        set_error(error, error_size, "invalid Level Zero device index");
        return -1;
    }
    *device_context_out = nullptr;

    device_context *context = new (std::nothrow) device_context();
    if (context == nullptr) {
        set_error(error, error_size, "out of memory allocating Level Zero device context");
        return -1;
    }
    context->record = &g_devices[index];

    ze_context_desc_t context_desc{};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    if (!check_result(zeContextCreate(context->record->driver, &context_desc, &context->context),
                      "create Level Zero context", error, error_size))
        goto fail;

    {
        uint32_t group_count = 0;
        if (!check_result(zeDeviceGetCommandQueueGroupProperties(
                              context->record->device, &group_count, nullptr),
                          "query Level Zero command queue groups", error, error_size))
            goto fail;
        if (group_count == 0) {
            set_error(error, error_size, "Intel GPU exposes no command queue groups");
            goto fail;
        }
        std::vector<ze_command_queue_group_properties_t> groups(group_count);
        for (auto &group : groups) {
            std::memset(&group, 0, sizeof(group));
            group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        }
        if (!check_result(zeDeviceGetCommandQueueGroupProperties(
                              context->record->device, &group_count, groups.data()),
                          "retrieve Level Zero command queue groups", error, error_size))
            goto fail;
        bool has_compute_group = false;
        for (uint32_t i = 0; i < group_count; ++i) {
            if ((groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0 &&
                groups[i].numQueues > 0) {
                context->queue_group_ordinal = i;
                has_compute_group = true;
                break;
            }
        }
        if (!has_compute_group) {
            set_error(error, error_size, "Intel GPU has no compute-capable Level Zero queue group");
            goto fail;
        }
    }

    {
        ze_command_queue_desc_t queue_desc{};
        queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        queue_desc.ordinal = context->queue_group_ordinal;
        queue_desc.index = 0;
        queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
        queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        if (!check_result(zeCommandQueueCreate(context->context, context->record->device,
                                              &queue_desc, &context->queue),
                          "create Level Zero command queue", error, error_size))
            goto fail;
    }

    {
        ze_command_list_desc_t list_desc{};
        list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
        list_desc.commandQueueGroupOrdinal = context->queue_group_ordinal;
        if (!check_result(zeCommandListCreate(context->context, context->record->device,
                                              &list_desc, &context->command_list),
                          "create Level Zero command list", error, error_size))
            goto fail;
    }

    {
        ze_module_desc_t module_desc{};
        module_desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
        module_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
        module_desc.inputSize = crossgpu_levelzero_kernels_spv_size;
        module_desc.pInputModule = crossgpu_levelzero_kernels_spv;
        if (!check_result(zeModuleCreate(context->context, context->record->device,
                                         &module_desc, &context->module, nullptr),
                          "build Level Zero SPIR-V module (check ocloc/driver compatibility)",
                          error, error_size))
            goto fail;
    }

    if (!create_kernel(context->module, "compute_kernel", &context->compute_kernel,
                       error, error_size) ||
        !create_kernel(context->module, "pattern_write_kernel", &context->pattern_write_kernel,
                       error, error_size) ||
        !create_kernel(context->module, "pattern_verify_kernel", &context->pattern_verify_kernel,
                       error, error_size))
        goto fail;

    *device_context_out = context;
    return 0;

fail:
    close_device_resources(context);
    return -1;
}

void device_close(void *device_context_in)
{
    close_device_resources(static_cast<device_context *>(device_context_in));
}

uint64_t choose_working_set(const crossgpu_device_info_t &info, uint64_t requested)
{
    uint64_t bytes = requested;
    const uint64_t maximum = info.max_allocation_bytes != 0
                                 ? info.max_allocation_bytes
                                 : info.physical_memory_bytes;
    if (bytes == 0)
        bytes = info.working_set_bytes != 0 ? info.working_set_bytes
                                            : default_working_set(info);
    if (maximum != 0)
        bytes = std::min(bytes, maximum);
    bytes = std::min<uint64_t>(bytes, static_cast<uint64_t>(SIZE_MAX));
    bytes &= ~UINT64_C(3);
    return bytes;
}

bool execute_and_wait(device_context *device, char *error, size_t error_size)
{
    if (!check_result(zeCommandListClose(device->command_list),
                      "close Level Zero command list", error, error_size))
        return false;
    ze_command_list_handle_t list = device->command_list;
    if (!check_result(zeCommandQueueExecuteCommandLists(device->queue, 1, &list, nullptr),
                      "submit Level Zero command list", error, error_size))
        return false;
    if (!check_result(zeCommandQueueSynchronize(device->queue, UINT64_MAX),
                      "synchronize Level Zero command queue", error, error_size))
        return false;
    return check_result(zeCommandListReset(device->command_list),
                        "reset Level Zero command list", error, error_size);
}

bool append_fill(device_context *device, void *destination,
                 const void *pattern, size_t pattern_size, size_t bytes,
                 char *error, size_t error_size)
{
    return check_result(zeCommandListAppendMemoryFill(
                            device->command_list, destination, pattern, pattern_size,
                            bytes, nullptr, 0, nullptr),
                        "append Level Zero memory fill", error, error_size);
}

bool append_copy(device_context *device, void *destination, const void *source,
                 size_t bytes, char *error, size_t error_size)
{
    return check_result(zeCommandListAppendMemoryCopy(
                            device->command_list, destination, source, bytes,
                            nullptr, 0, nullptr),
                        "append Level Zero memory copy", error, error_size);
}

bool set_argument(ze_kernel_handle_t kernel, uint32_t index,
                  size_t size, const void *value,
                  char *error, size_t error_size)
{
    return check_result(zeKernelSetArgumentValue(kernel, index, size, value),
                        "set Level Zero kernel argument", error, error_size);
}

bool append_kernel(device_context *device, ze_kernel_handle_t kernel,
                   uint32_t elements, char *error, size_t error_size)
{
    ze_group_count_t groups{};
    groups.groupCountX = elements / kWorkgroupSize +
                         (elements % kWorkgroupSize != 0 ? 1u : 0u);
    groups.groupCountY = 1;
    groups.groupCountZ = 1;
    return check_result(zeCommandListAppendLaunchKernel(
                            device->command_list, kernel, &groups, nullptr, 0, nullptr),
                        "append Level Zero kernel launch", error, error_size);
}

int workload_begin(void *device_context_in, crossgpu_test_type_t test,
                   const crossgpu_workload_config_t *config,
                   void **workload_context_out, char *error, size_t error_size)
{
    device_context *device = static_cast<device_context *>(device_context_in);
    if (device == nullptr || config == nullptr || workload_context_out == nullptr) {
        set_error(error, error_size, "invalid Level Zero workload initialization arguments");
        return -1;
    }
    *workload_context_out = nullptr;
    if (test != CROSSGPU_TEST_COMPUTE && test != CROSSGPU_TEST_MEMORY &&
        test != CROSSGPU_TEST_BANDWIDTH) {
        set_error(error, error_size, "unsupported Level Zero workload type");
        return -1;
    }

    const uint64_t bytes64 = choose_working_set(device->record->info, config->memory_bytes);
    if (bytes64 < 4096 || bytes64 / sizeof(uint32_t) > UINT32_MAX) {
        set_error(error, error_size, "Intel GPU cannot provide the requested working set");
        return -1;
    }

    workload_context *workload = new (std::nothrow) workload_context();
    if (workload == nullptr) {
        set_error(error, error_size, "out of memory allocating Level Zero workload context");
        return -1;
    }
    workload->bytes = static_cast<size_t>(bytes64);
    workload->elements = static_cast<uint32_t>(bytes64 / sizeof(uint32_t));

    ze_device_mem_alloc_desc_t allocation_desc{};
    allocation_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    allocation_desc.ordinal = device->record->memory_ordinal;
    ze_result_t result = zeMemAllocDevice(device->context, &allocation_desc,
                                          workload->bytes, 64,
                                          device->record->device, &workload->buffer_a);
    if (!check_result(result, "allocate primary Level Zero device buffer", error, error_size))
        goto fail;

    result = zeMemAllocDevice(device->context, &allocation_desc, sizeof(uint32_t), 64,
                              device->record->device, &workload->error_buffer);
    if (!check_result(result, "allocate Level Zero validation counter", error, error_size))
        goto fail;

    if (test == CROSSGPU_TEST_BANDWIDTH) {
        result = zeMemAllocDevice(device->context, &allocation_desc,
                                  workload->bytes, 64,
                                  device->record->device, &workload->buffer_b);
        if (!check_result(result, "allocate secondary Level Zero device buffer", error, error_size))
            goto fail;
    }

    {
        const uint32_t zero = 0;
        if (!append_fill(device, workload->buffer_a, &zero, sizeof(zero), workload->bytes,
                         error, error_size) ||
            !append_fill(device, workload->error_buffer, &zero, sizeof(zero), sizeof(zero),
                         error, error_size))
            goto fail;
        if (workload->buffer_b != nullptr &&
            !append_fill(device, workload->buffer_b, &zero, sizeof(zero), workload->bytes,
                         error, error_size))
            goto fail;
    }
    if (test == CROSSGPU_TEST_COMPUTE) {
        const float initial_value = 0.5f;
        if (!append_fill(device, workload->buffer_a, &initial_value, sizeof(initial_value),
                         workload->bytes, error, error_size))
            goto fail;
    }
    if (!execute_and_wait(device, error, error_size))
        goto fail;

    *workload_context_out = workload;
    return 0;

fail:
    if (workload->buffer_a != nullptr) (void)zeMemFree(device->context, workload->buffer_a);
    if (workload->buffer_b != nullptr) (void)zeMemFree(device->context, workload->buffer_b);
    if (workload->error_buffer != nullptr) (void)zeMemFree(device->context, workload->error_buffer);
    delete workload;
    return -1;
}

int workload_step(void *device_context_in, crossgpu_test_type_t test,
                  void *workload_context_in, crossgpu_metrics_t *metrics,
                  char *error, size_t error_size)
{
    device_context *device = static_cast<device_context *>(device_context_in);
    workload_context *workload = static_cast<workload_context *>(workload_context_in);
    if (device == nullptr || workload == nullptr || metrics == nullptr) {
        set_error(error, error_size, "invalid Level Zero workload step arguments");
        return -1;
    }
    std::memset(metrics, 0, sizeof(*metrics));

    const uint32_t zero = 0;
    if (test != CROSSGPU_TEST_BANDWIDTH &&
        !append_fill(device, workload->error_buffer, &zero, sizeof(zero), sizeof(zero),
                     error, error_size))
        return -1;

    const uint32_t n = workload->elements;
    const uint64_t bytes = static_cast<uint64_t>(workload->bytes);
    workload->host_errors = 0;

    switch (test) {
    case CROSSGPU_TEST_COMPUTE: {
        void *buffer = workload->buffer_a;
        if (!set_argument(device->compute_kernel, 0, sizeof(buffer), &buffer, error, error_size) ||
            !set_argument(device->compute_kernel, 1, sizeof(workload->error_buffer),
                          &workload->error_buffer, error, error_size) ||
            !set_argument(device->compute_kernel, 2, sizeof(n), &n, error, error_size) ||
            !append_kernel(device, device->compute_kernel, n, error, error_size) ||
            !append_copy(device, &workload->host_errors, workload->error_buffer,
                         sizeof(workload->host_errors), error, error_size) ||
            !execute_and_wait(device, error, error_size))
            return -1;
        metrics->operations = static_cast<uint64_t>(n) * kComputeInnerIterations * 2u;
        metrics->bytes_read = bytes;
        metrics->bytes_written = bytes;
        metrics->errors = workload->host_errors;
        break;
    }

    case CROSSGPU_TEST_MEMORY:
        ++workload->seed;
        {
            void *buffer = workload->buffer_a;
            if (!set_argument(device->pattern_write_kernel, 0, sizeof(buffer), &buffer, error, error_size) ||
                !set_argument(device->pattern_write_kernel, 1, sizeof(n), &n, error, error_size) ||
                !set_argument(device->pattern_write_kernel, 2, sizeof(workload->seed),
                              &workload->seed, error, error_size) ||
                !append_kernel(device, device->pattern_write_kernel, n, error, error_size))
                return -1;
        }
        {
            const void *buffer = workload->buffer_a;
            if (!set_argument(device->pattern_verify_kernel, 0, sizeof(buffer), &buffer, error, error_size) ||
                !set_argument(device->pattern_verify_kernel, 1, sizeof(workload->error_buffer),
                              &workload->error_buffer, error, error_size) ||
                !set_argument(device->pattern_verify_kernel, 2, sizeof(n), &n, error, error_size) ||
                !set_argument(device->pattern_verify_kernel, 3, sizeof(workload->seed),
                              &workload->seed, error, error_size) ||
                !append_kernel(device, device->pattern_verify_kernel, n, error, error_size) ||
                !append_copy(device, &workload->host_errors, workload->error_buffer,
                             sizeof(workload->host_errors), error, error_size) ||
                !execute_and_wait(device, error, error_size))
                return -1;
        }
        metrics->errors = workload->host_errors;
        metrics->bytes_read = bytes;
        metrics->bytes_written = bytes;
        break;

    case CROSSGPU_TEST_BANDWIDTH:
        if (workload->buffer_b == nullptr) {
            set_error(error, error_size, "Level Zero bandwidth workload has no secondary buffer");
            return -1;
        }
        if (!append_copy(device, workload->buffer_b, workload->buffer_a,
                         workload->bytes, error, error_size) ||
            !execute_and_wait(device, error, error_size))
            return -1;
        std::swap(workload->buffer_a, workload->buffer_b);
        metrics->bytes_read = bytes;
        metrics->bytes_written = bytes;
        break;

    default:
        set_error(error, error_size, "unsupported Level Zero workload type");
        return -1;
    }

    metrics->iterations = 1;
    return 0;
}

void workload_end(void *device_context_in, crossgpu_test_type_t test,
                  void *workload_context_in)
{
    (void)test;
    device_context *device = static_cast<device_context *>(device_context_in);
    workload_context *workload = static_cast<workload_context *>(workload_context_in);
    if (device == nullptr || workload == nullptr)
        return;
    (void)zeCommandQueueSynchronize(device->queue, UINT64_MAX);
    if (workload->buffer_a != nullptr) (void)zeMemFree(device->context, workload->buffer_a);
    if (workload->buffer_b != nullptr) (void)zeMemFree(device->context, workload->buffer_b);
    if (workload->error_buffer != nullptr) (void)zeMemFree(device->context, workload->error_buffer);
    delete workload;
}

const crossgpu_backend_plugin_t levelzero_plugin = {
    CROSSGPU_BACKEND_ABI_VERSION,
    sizeof(crossgpu_backend_plugin_t),
    CROSSGPU_BACKEND_LEVEL_ZERO,
    "level-zero",
    initialize,
    shutdown,
    enumerate_devices,
    device_open,
    device_close,
    workload_begin,
    workload_step,
    workload_end
};

} // namespace

extern "C" CROSSGPU_BACKEND_EXPORT
const crossgpu_backend_plugin_t *crossgpu_backend_get_api(uint32_t host_abi_version)
{
    return host_abi_version == CROSSGPU_BACKEND_ABI_VERSION ? &levelzero_plugin : nullptr;
}
