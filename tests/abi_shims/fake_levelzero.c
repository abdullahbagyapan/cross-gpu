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
typedef void *ze_kernel_handle_t;

#define ZE_RESULT_SUCCESS 0
#define ZE_RESULT_ERROR_DEVICE_LOST 0x70000001
#define ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY 0x70000003
#define ZE_DEVICE_TYPE_GPU 1

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
    uint32_t timerResolution;
    uint32_t timestampValidBits;
    uint32_t kernelTimestampValidBits;
    uint8_t uuid[16];
    char name[256];
} ze_device_properties_t;

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

static int g_active_allocs = 0;
static int g_active_handles = 0;
static int g_bitflip_injected = 0;
static void *g_dev_ptrs[256];

static int l0_count(void) { return shim_env_int("FAKE_L0_COUNT", 1, 0, SHIM_MAX_GPUS); }
static int l0_bus(int dev) { return shim_env_int("FAKE_L0_BUS_BASE", 4, 1, 200) + dev; }

static void track_dev_ptr(void *p) {
    for (int i = 0; i < 256; ++i) {
        if (!g_dev_ptrs[i]) { g_dev_ptrs[i] = p; return; }
    }
}

static int untrack_dev_ptr(void *p) {
    for (int i = 0; i < 256; ++i) {
        if (g_dev_ptrs[i] == p) { g_dev_ptrs[i] = NULL; return 1; }
    }
    return 0;
}

static int is_dev_ptr(const void *p) {
    for (int i = 0; i < 256; ++i) {
        if (g_dev_ptrs[i] == p) return 1;
    }
    return 0;
}

__attribute__((destructor))
static void check_l0_leaks(void) {
    if (!shim_leak_check_enabled()) return;
    if (g_active_allocs != 0 || g_active_handles != 0) {
        fprintf(stderr, "[fake_levelzero] LEAK: allocs=%d handles=%d\n", g_active_allocs, g_active_handles);
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

ze_result_t zeDriverGetApiVersion(ze_driver_handle_t hDriver, uint32_t *version) {
    (void)hDriver;
    if (version) *version = (1u << 16) | 9u; /* 1.9 */
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

static int l0_handle_to_idx(ze_device_handle_t h) {
    uintptr_t v = (uintptr_t)h;
    return v >= 0x2000u ? (int)(v - 0x2000u) : 0;
}

ze_result_t zeDeviceGetProperties(ze_device_handle_t hDevice, ze_device_properties_t *p) {
    if (!p) return ZE_RESULT_SUCCESS;
    int idx = l0_handle_to_idx(hDevice);
    p->type = ZE_DEVICE_TYPE_GPU;
    p->vendorId = 0x8086;
    p->deviceId = 0x56A0u + (uint32_t)idx;
    p->coreClockRate = 2100;
    p->maxMemAllocSize = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    p->numSlices = 4;
    p->numSubslicesPerSlice = 4;
    p->numEUsPerSubslice = 16;
    p->numThreadsPerEU = 8;
    p->physicalEUSimdWidth = 8;
    for (int i = 0; i < 15; ++i) p->uuid[i] = (uint8_t)(0xE0 + i);
    p->uuid[15] = (uint8_t)idx;
    snprintf(p->name, sizeof(p->name), "Fake Intel Level Zero GPU %d", idx);
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
        p[0].totalSize = 12ULL * 1024ULL * 1024ULL * 1024ULL;
        snprintf(p[0].name, sizeof(p[0].name), "VRAM");
        *pCount = 1;
    }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeContextCreate(ze_driver_handle_t d, const void *desc, ze_context_handle_t *out) {
    (void)d; (void)desc;
    void *h = malloc(16);
    if (!h) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    g_active_handles++;
    if (out) *out = h;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeContextDestroy(ze_context_handle_t h) {
    if (h) { free(h); g_active_handles--; }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeMemAllocDevice(ze_context_handle_t c, const void *desc, size_t sz, size_t align,
                             ze_device_handle_t d, void **pptr) {
    (void)c; (void)desc; (void)align; (void)d;
    void *buf = calloc(1, sz ? sz : 1);
    if (!buf) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    g_active_allocs++;
    track_dev_ptr(buf);
    if (pptr) *pptr = buf;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeMemAllocHost(ze_context_handle_t c, const void *desc, size_t sz, size_t align, void **pptr) {
    (void)c; (void)desc; (void)align;
    void *buf = calloc(1, sz ? sz : 1);
    if (!buf) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    g_active_allocs++;
    if (pptr) *pptr = buf;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeMemFree(ze_context_handle_t c, void *ptr) {
    (void)c;
    if (ptr) { untrack_dev_ptr(ptr); free(ptr); g_active_allocs--; }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeCommandQueueCreate(ze_context_handle_t c, ze_device_handle_t d, const void *desc,
                                 ze_command_queue_handle_t *out) {
    (void)c; (void)d; (void)desc;
    void *h = malloc(16);
    if (!h) return ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    g_active_handles++;
    if (out) *out = h;
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeCommandQueueDestroy(ze_command_queue_handle_t h) {
    if (h) { free(h); g_active_handles--; }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeCommandQueueExecuteCommandLists(ze_command_queue_handle_t q, uint32_t n,
                                              ze_command_list_handle_t *lists, void *fence) {
    (void)q; (void)n; (void)lists; (void)fence;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("levelzero", 0)) {
        return ZE_RESULT_ERROR_DEVICE_LOST;
    }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeCommandQueueSynchronize(ze_command_queue_handle_t q, uint64_t timeout) {
    (void)q; (void)timeout;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("levelzero", 0)) {
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

ze_result_t zeCommandListDestroy(ze_command_list_handle_t h) { return zeCommandQueueDestroy(h); }
ze_result_t zeCommandListClose(ze_command_list_handle_t h) { (void)h; return ZE_RESULT_SUCCESS; }
ze_result_t zeCommandListReset(ze_command_list_handle_t h) { (void)h; return ZE_RESULT_SUCCESS; }

ze_result_t zeCommandListAppendMemoryCopy(ze_command_list_handle_t cl, void *dst, const void *src,
                                          size_t sz, void *sig, uint32_t nw, void *wait) {
    (void)cl; (void)sig; (void)nw; (void)wait;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("levelzero", 0)) {
        return ZE_RESULT_ERROR_DEVICE_LOST;
    }
    if (dst && src && sz) {
        memcpy(dst, src, sz);
        /* Flip bit on D2H copy (src is device ptr, dst is host ptr) */
        if (is_dev_ptr(src) && !is_dev_ptr(dst) && shim_fault_is("bitflip") &&
            shim_fault_target_matches("levelzero", 0) && !g_bitflip_injected) {
            ((unsigned char *)dst)[0] ^= 0x01;
            g_bitflip_injected = 1;
        }
    }
    return ZE_RESULT_SUCCESS;
}