// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "shim_common.h"

typedef int32_t ze_result_t;
typedef void *ze_driver_handle_t;
typedef void *ze_device_handle_t;
typedef void *ze_context_handle_t;
typedef void *ze_command_queue_handle_t;
typedef void *ze_command_list_handle_t;
typedef void *ze_module_handle_t;
typedef void *ze_module_build_log_handle_t;
typedef void *ze_kernel_handle_t;

#define ZE_RESULT_SUCCESS 0
#define ZE_RESULT_ERROR_DEVICE_LOST 0x70000001
#define ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY 0x70000003
#define ZE_DEVICE_TYPE_GPU 1
#define ZE_DEVICE_MODULE_FLAG_FP16 (1u << 0)
#define ZE_DEVICE_MODULE_FLAG_FP64 (1u << 1)
#define ZE_DEVICE_MODULE_FLAG_DP4A (1u << 3)

typedef struct {
    uint32_t stype;
    void *pNext;
    uint8_t uuid[16];
    uint32_t driverVersion;
} ze_driver_properties_t;

typedef struct {
    uint32_t stype;
    void *pNext;
    uint32_t type;
    uint32_t vendorId;
    uint32_t deviceId;
    uint32_t flags;
    uint32_t subdeviceId;
    uint32_t coreClockRate;
    uint64_t maxMemAllocSize;
    uint32_t maxHardwareContexts;
    uint32_t maxCommandQueuePriority;
    uint32_t numThreadsPerEU;
    uint32_t physicalEUSimdWidth;
    uint32_t numEUsPerSubslice;
    uint32_t numSubslicesPerSlice;
    uint32_t numSlices;
    uint64_t timerResolution;
    uint32_t timestampValidBits;
    uint32_t kernelTimestampValidBits;
    uint8_t uuid[16];
    char name[256];
} ze_device_properties_t;

typedef struct {
    uint32_t stype;
    void *pNext;
    uint32_t spirvVersionSupported;
    uint32_t flags;
    uint32_t fp16flags;
    uint32_t fp32flags;
    uint32_t fp64flags;
    uint32_t maxArgumentsSize;
    uint32_t printfBufferSize;
    uint8_t nativeKernelSupported[16];
} ze_device_module_properties_t;

typedef struct {
    uint32_t stype;
    void *pNext;
    uint32_t domain;
    uint32_t bus;
    uint32_t device;
    uint32_t function;
    int32_t genVersion;
    int32_t width;
    int64_t maxBandwidth;
} ze_pci_ext_properties_t;

typedef struct {
    uint32_t stype;
    void *pNext;
    uint32_t flags;
    uint32_t maxClockRate;
    uint32_t maxBusWidth;
    uint64_t totalSize;
    char name[256];
} ze_device_memory_properties_t;

typedef struct {
    int dev_idx;
} fake_ze_handle_t;

static _Atomic int g_active_allocs = 0;
static _Atomic int g_active_handles = 0;
static _Atomic uintptr_t g_dev_ptrs[256];

static int l0_count(void) { return shim_env_int("FAKE_L0_COUNT", 1, 0, SHIM_MAX_GPUS); }
static int l0_bus(int dev) { return shim_env_int("FAKE_L0_BUS_BASE", 4, 1, 200) + dev; }

static int l0_handle_to_idx(ze_device_handle_t h) {
    uintptr_t v = (uintptr_t)h;
    return v >= 0x2000u ? (int)(v - 0x2000u) : 0;
}

static void track_dev_ptr(void *p) {
    uintptr_t pv = (uintptr_t)p;
    for (int i = 0; i < 256; ++i) {
        uintptr_t expected = 0;
        if (atomic_compare_exchange_strong(&g_dev_ptrs[i], &expected, pv)) return;
    }
}

static void untrack_dev_ptr(void *p) {
    uintptr_t pv = (uintptr_t)p;
    for (int i = 0; i < 256; ++i) {
        uintptr_t expected = pv;
        if (atomic_compare_exchange_strong(&g_dev_ptrs[i], &expected, 0)) return;
    }
}

static int is_dev_ptr(const void *p) {
    uintptr_t pv = (uintptr_t)p;
    if (!pv) return 0;
    for (int i = 0; i < 256; ++i) {
        if (atomic_load(&g_dev_ptrs[i]) == pv) return 1;
    }
    return 0;
}

__attribute__((destructor))
static void check_l0_leaks(void) {
    if (!shim_leak_check_enabled()) return;
    int allocs = atomic_load(&g_active_allocs);
    int handles = atomic_load(&g_active_handles);
    if (allocs != 0 || handles != 0) {
        fprintf(stderr, "[fake_levelzero] LEAK: allocs=%d handles=%d\n", allocs, handles);
        _exit(99);
    }
}

ze_result_t zeInit(uint32_t flags) { (void)flags; return ZE_RESULT_SUCCESS; }

ze_result_t zeDriverGet(uint32_t *pCount, ze_driver_handle_t *phDrivers) {
    if (!pCount) return ZE_RESULT_SUCCESS;
    uint32_t n = l0_count() > 0 ? 1u : 0u;
    if (!phDrivers) { *pCount = n; return ZE_RESULT_SUCCESS; }
    if (*pCount > n) *pCount = n;
    if (*pCount == 1) phDrivers[0] = (ze_driver_handle_t)(uintptr_t)0x1000;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeDriverGetProperties(ze_driver_handle_t hDriver, ze_driver_properties_t *p) {
    (void)hDriver;
    if (!p) return ZE_RESULT_SUCCESS;
    for (int i = 0; i < 16; ++i) p->uuid[i] = (uint8_t)(0xD0 + i);
    p->driverVersion = (1u << 16) | 9u;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeDriverGetApiVersion(ze_driver_handle_t hDriver, uint32_t *version) {
    (void)hDriver;
    if (version) *version = (1u << 16) | 9u;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeDeviceGet(ze_driver_handle_t hDriver, uint32_t *pCount, ze_device_handle_t *phDevices) {
    (void)hDriver;
    uint32_t n = (uint32_t)l0_count();
    if (!pCount) return ZE_RESULT_SUCCESS;
    if (!phDevices) { *pCount = n; return ZE_RESULT_SUCCESS; }
    if (*pCount > n) *pCount = n;
    for (uint32_t i = 0; i < *pCount; ++i) {
        phDevices[i] = (ze_device_handle_t)(uintptr_t)(0x2000u + i);
    }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeDeviceGetProperties(ze_device_handle_t hDevice, ze_device_properties_t *p) {
    if (!p) return ZE_RESULT_SUCCESS;
    int idx = l0_handle_to_idx(hDevice);
    size_t vram = shim_vram_bytes(256);
    p->type = ZE_DEVICE_TYPE_GPU;
    p->vendorId = 0x8086;
    p->deviceId = 0x56A0u + (uint32_t)idx;
    p->flags = 0;
    p->subdeviceId = 0;
    p->coreClockRate = 2100;
    p->maxMemAllocSize = (uint64_t)vram;
    p->maxHardwareContexts = 64;
    p->maxCommandQueuePriority = 0;
    p->numThreadsPerEU = 8;
    p->physicalEUSimdWidth = 8;
    p->numEUsPerSubslice = 16;
    p->numSubslicesPerSlice = 4;
    p->numSlices = 4;
    p->timerResolution = 83;
    p->timestampValidBits = 36;
    p->kernelTimestampValidBits = 32;
    for (int i = 0; i < 15; ++i) p->uuid[i] = (uint8_t)(0xE0 + i);
    p->uuid[15] = (uint8_t)idx;
    snprintf(p->name, sizeof(p->name), "Fake Intel Level Zero GPU %d", idx);
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeDeviceGetModuleProperties(ze_device_handle_t hDevice, ze_device_module_properties_t *p) {
    (void)hDevice;
    if (!p) return ZE_RESULT_SUCCESS;
    p->spirvVersionSupported = (1u << 16) | 5u;
    p->flags = ZE_DEVICE_MODULE_FLAG_FP16 | ZE_DEVICE_MODULE_FLAG_FP64 | ZE_DEVICE_MODULE_FLAG_DP4A;
    p->fp16flags = 1;
    p->fp32flags = 1;
    p->fp64flags = 1;
    p->maxArgumentsSize = 1024;
    p->printfBufferSize = 1048576;
    memset(p->nativeKernelSupported, 0, sizeof(p->nativeKernelSupported));
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeDevicePciGetPropertiesExt(ze_device_handle_t hDevice, ze_pci_ext_properties_t *p) {
    if (!p) return ZE_RESULT_SUCCESS;
    int idx = l0_handle_to_idx(hDevice);
    p->domain = 0;
    p->bus = (uint32_t)l0_bus(idx);
    p->device = 0;
    p->function = 0;
    p->genVersion = 4;
    p->width = 16;
    p->maxBandwidth = 31500000000LL;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeDeviceGetMemoryProperties(ze_device_handle_t hDevice, uint32_t *pCount,
                                        ze_device_memory_properties_t *p) {
    (void)hDevice;
    if (!pCount) return ZE_RESULT_SUCCESS;
    if (!p) { *pCount = 1; return ZE_RESULT_SUCCESS; }
    if (*pCount >= 1) {
        p[0].flags = 0;
        p[0].maxClockRate = 2000;
        p[0].maxBusWidth = 256;
        p[0].totalSize = (uint64_t)shim_vram_bytes(256);
        snprintf(p[0].name, sizeof(p[0].name), "VRAM");
        *pCount = 1;
    }
    return ZE_RESULT_SUCCESS;
}

static fake_ze_handle_t *alloc_ze_handle(int dev_idx) {
    fake_ze_handle_t *h = (fake_ze_handle_t *)malloc(sizeof(fake_ze_handle_t));
    if (!h) return NULL;
    h->dev_idx = dev_idx;
    atomic_fetch_add(&g_active_handles, 1);
    return h;
}

static void free_ze_handle(void *h) {
    if (h) {
        free(h);
        atomic_fetch_sub(&g_active_handles, 1);
    }
}

ze_result_t zeContextCreate(ze_driver_handle_t d, const void *desc, ze_context_handle_t *out) {
    (void)d; (void)desc;
    fake_ze_handle_t *h = alloc_ze_handle(0);
    if (!h) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    if (out) *out = h;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeContextDestroy(ze_context_handle_t h) {
    free_ze_handle(h);
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeMemAllocDevice(ze_context_handle_t c, const void *desc, size_t sz, size_t align,
                             ze_device_handle_t d, void **pptr) {
    (void)c; (void)desc; (void)align; (void)d;
    void *buf = calloc(1, sz ? sz : 1);
    if (!buf) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    atomic_fetch_add(&g_active_allocs, 1);
    track_dev_ptr(buf);
    if (pptr) *pptr = buf;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeMemAllocHost(ze_context_handle_t c, const void *desc, size_t sz, size_t align, void **pptr) {
    (void)c; (void)desc; (void)align;
    void *buf = calloc(1, sz ? sz : 1);
    if (!buf) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    atomic_fetch_add(&g_active_allocs, 1);
    if (pptr) *pptr = buf;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeMemFree(ze_context_handle_t c, void *ptr) {
    (void)c;
    if (ptr) {
        untrack_dev_ptr(ptr);
        free(ptr);
        atomic_fetch_sub(&g_active_allocs, 1);
    }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeCommandQueueCreate(ze_context_handle_t c, ze_device_handle_t d, const void *desc,
                                 ze_command_queue_handle_t *out) {
    (void)c; (void)desc;
    fake_ze_handle_t *h = alloc_ze_handle(l0_handle_to_idx(d));
    if (!h) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    if (out) *out = h;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeCommandQueueDestroy(ze_command_queue_handle_t h) {
    free_ze_handle(h);
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeCommandQueueExecuteCommandLists(ze_command_queue_handle_t q, uint32_t n,
                                              ze_command_list_handle_t *lists, void *fence) {
    (void)n; (void)lists; (void)fence;
    int dev_idx = q ? ((fake_ze_handle_t *)q)->dev_idx : 0;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("levelzero", dev_idx)) {
        return ZE_RESULT_ERROR_DEVICE_LOST;
    }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeCommandQueueSynchronize(ze_command_queue_handle_t q, uint64_t timeout) {
    (void)timeout;
    int dev_idx = q ? ((fake_ze_handle_t *)q)->dev_idx : 0;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("levelzero", dev_idx)) {
        return ZE_RESULT_ERROR_DEVICE_LOST;
    }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeCommandListCreate(ze_context_handle_t c, ze_device_handle_t d, const void *desc,
                                ze_command_list_handle_t *out) {
    return zeCommandQueueCreate(c, d, desc, (ze_command_queue_handle_t *)out);
}

ze_result_t zeCommandListCreateImmediate(ze_context_handle_t c, ze_device_handle_t d, const void *desc,
                                         ze_command_list_handle_t *out) {
    return zeCommandQueueCreate(c, d, desc, (ze_command_queue_handle_t *)out);
}

ze_result_t zeCommandListDestroy(ze_command_list_handle_t h) {
    free_ze_handle(h);
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeCommandListClose(ze_command_list_handle_t h) { (void)h; return ZE_RESULT_SUCCESS; }
ze_result_t zeCommandListReset(ze_command_list_handle_t h) { (void)h; return ZE_RESULT_SUCCESS; }

ze_result_t zeCommandListAppendMemoryCopy(ze_command_list_handle_t cl, void *dst, const void *src,
                                          size_t sz, void *sig, uint32_t nw, void *wait) {
    (void)sig; (void)nw; (void)wait;
    int dev_idx = cl ? ((fake_ze_handle_t *)cl)->dev_idx : 0;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("levelzero", dev_idx)) {
        return ZE_RESULT_ERROR_DEVICE_LOST;
    }
    if (dst && src && sz) {
        memcpy(dst, src, sz);
        /* Flip bit on D2H copy (src is device ptr, dst is host ptr) */
        if (is_dev_ptr(src) && !is_dev_ptr(dst) && shim_fault_is("bitflip") &&
            shim_fault_target_matches("levelzero", dev_idx)) {
            ((unsigned char *)dst)[0] ^= 0x01;
        }
    }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeCommandListAppendLaunchKernel(ze_command_list_handle_t cl, ze_kernel_handle_t kernel,
                                            const void *group_count, void *sig,
                                            uint32_t nw, void *wait) {
    (void)kernel; (void)group_count; (void)sig; (void)nw; (void)wait;
    int dev_idx = cl ? ((fake_ze_handle_t *)cl)->dev_idx : 0;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("levelzero", dev_idx)) {
        return ZE_RESULT_ERROR_DEVICE_LOST;
    }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeModuleCreate(ze_context_handle_t c, ze_device_handle_t d, const void *desc,
                           ze_module_handle_t *out_mod, ze_module_build_log_handle_t *out_log) {
    (void)c; (void)desc;
    fake_ze_handle_t *h = alloc_ze_handle(l0_handle_to_idx(d));
    if (!h) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    if (out_mod) *out_mod = h;
    if (out_log) *out_log = NULL;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeModuleDestroy(ze_module_handle_t h) {
    free_ze_handle(h);
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeModuleBuildLogGetString(ze_module_build_log_handle_t log, size_t *pSize, char *pLog) {
    (void)log;
    if (pSize) *pSize = 1;
    if (pLog) pLog[0] = '\0';
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeModuleBuildLogDestroy(ze_module_build_log_handle_t log) {
    (void)log;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeKernelCreate(ze_module_handle_t mod, const void *desc, ze_kernel_handle_t *out_kernel) {
    (void)desc;
    int dev_idx = mod ? ((fake_ze_handle_t *)mod)->dev_idx : 0;
    fake_ze_handle_t *h = alloc_ze_handle(dev_idx);
    if (!h) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    if (out_kernel) *out_kernel = h;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeKernelDestroy(ze_kernel_handle_t h) {
    free_ze_handle(h);
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeKernelSetGroupSize(ze_kernel_handle_t k, uint32_t gx, uint32_t gy, uint32_t gz) {
    (void)k; (void)gx; (void)gy; (void)gz;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeKernelSetArgumentValue(ze_kernel_handle_t k, uint32_t idx, size_t sz, const void *val) {
    (void)k; (void)idx; (void)sz; (void)val;
    return ZE_RESULT_SUCCESS;
}
