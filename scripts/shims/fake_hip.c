// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "shim_common.h"

typedef int hipError_t;
typedef void *hipModule_t;
typedef void *hipFunction_t;
typedef void *hipStream_t;

#define HIP_SUCCESS 0
#define HIP_ERROR_OUT_OF_MEMORY 2
#define HIP_ERROR_INVALID_DEVICE 101
#define HIP_ERROR_ECC_NOT_CORRECTABLE 214

static _Atomic int g_active_allocs = 0;
static _Atomic int g_active_modules = 0;
static _Thread_local int g_current_dev = 0;

static int hip_count(void) {
    return shim_env_int("FAKE_HIP_COUNT", 1, 0, SHIM_MAX_GPUS);
}

static int hip_bus(int dev) {
    return shim_env_int("FAKE_HIP_BUS_BASE", 3, 1, 200) + dev;
}

__attribute__((destructor))
static void check_hip_leaks(void) {
    if (!shim_leak_check_enabled()) return;
    int allocs = atomic_load(&g_active_allocs);
    int modules = atomic_load(&g_active_modules);
    if (allocs != 0 || modules != 0) {
        fprintf(stderr, "[fake_hip] LEAK: allocs=%d modules=%d\n", allocs, modules);
        _exit(99);
    }
}

hipError_t hipInit(unsigned int flags) { (void)flags; return HIP_SUCCESS; }
hipError_t hipDriverGetVersion(int *v) { if (v) *v = 60241133; return HIP_SUCCESS; }
hipError_t hipRuntimeGetVersion(int *v) { if (v) *v = 60241133; return HIP_SUCCESS; }
hipError_t hipGetDeviceCount(int *c) { if (c) *c = hip_count(); return HIP_SUCCESS; }

hipError_t hipSetDevice(int dev) {
    if (dev < 0 || dev >= hip_count()) return HIP_ERROR_INVALID_DEVICE;
    g_current_dev = dev;
    return HIP_SUCCESS;
}

hipError_t hipDeviceGetName(char *name, int len, int dev) {
    if (name && len > 0) snprintf(name, (size_t)len, "Fake AMD HIP GPU %d", dev);
    return HIP_SUCCESS;
}

hipError_t hipDeviceGetPCIBusId(char *pci, int len, int dev) {
    if (pci && len > 0) snprintf(pci, (size_t)len, "0000:%02x:00.0", hip_bus(dev));
    return HIP_SUCCESS;
}

hipError_t hipDeviceGetUuid(unsigned char uuid[16], int dev) {
    if (uuid) {
        for (int i = 0; i < 15; ++i) uuid[i] = (unsigned char)(0xA0 + i);
        uuid[15] = (unsigned char)dev;
    }
    return HIP_SUCCESS;
}

hipError_t hipDeviceTotalMem(size_t *bytes, int dev) {
    (void)dev;
    if (bytes) *bytes = shim_vram_bytes(256);
    return HIP_SUCCESS;
}

hipError_t hipMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
    size_t total = shim_vram_bytes(256);
    if (total_bytes) *total_bytes = total;
    if (free_bytes) *free_bytes = (total * 7) / 8;
    return HIP_SUCCESS;
}

hipError_t hipGetDevicePropertiesR0600(void *prop_buf, int dev) {
    if (dev < 0 || dev >= hip_count()) return HIP_ERROR_INVALID_DEVICE;
    if (prop_buf) {
        /* Write a null-terminated gfx* token inside the opaque buffer for HipBackend's scanner. */
        char *dst = (char *)prop_buf;
        snprintf(dst, 256, "Fake AMD HIP GPU %d", dev);
        snprintf(dst + 512, 64, "gfx1100:xnack-");
    }
    return HIP_SUCCESS;
}

hipError_t hipGetDeviceProperties(void *prop_buf, int dev) {
    return hipGetDevicePropertiesR0600(prop_buf, dev);
}

hipError_t hipDeviceGetAttribute(int *pi, int attr, int dev) {
    if (!pi) return HIP_SUCCESS;
    switch (attr) {
        case 16: *pi = 0; break;            /* HIP_DEVICE_ATTRIBUTE_INTEGRATED */
        case 23: *pi = 11; break;           /* HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */
        case 56: *pi = 1024; break;         /* HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK */
        case 61: *pi = 0; break;            /* HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR */
        case 63: *pi = 60; break;           /* HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */
        case 67: *pi = hip_bus(dev); break; /* HIP_DEVICE_ATTRIBUTE_PCI_BUS_ID */
        case 68: *pi = 0; break;            /* HIP_DEVICE_ATTRIBUTE_PCI_DEVICE_ID */
        case 69: *pi = 0; break;            /* HIP_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID */
        case 87: *pi = 32; break;           /* HIP_DEVICE_ATTRIBUTE_WARP_SIZE */
        default: *pi = 0; break;
    }
    return HIP_SUCCESS;
}

hipError_t hipMalloc(void **ptr, size_t size) {
    void *buf = calloc(1, size ? size : 1);
    if (!buf) return HIP_ERROR_OUT_OF_MEMORY;
    atomic_fetch_add(&g_active_allocs, 1);
    if (ptr) *ptr = buf;
    return HIP_SUCCESS;
}

hipError_t hipFree(void *ptr) {
    if (ptr) {
        free(ptr);
        atomic_fetch_sub(&g_active_allocs, 1);
    }
    return HIP_SUCCESS;
}

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
    (void)flags;
    return hipMalloc(ptr, size);
}

hipError_t hipHostFree(void *ptr) { return hipFree(ptr); }

hipError_t hipMemcpy(void *dst, const void *src, size_t size, int kind) {
    if (shim_fault_is("ecc_ue") && shim_fault_target_matches("hip", g_current_dev)) {
        return HIP_ERROR_ECC_NOT_CORRECTABLE;
    }
    if (dst && src && size) {
        memcpy(dst, src, size);
        /* kind == 2 is hipMemcpyDeviceToHost */
        if (kind == 2 && shim_fault_is("bitflip") &&
            shim_fault_target_matches("hip", g_current_dev)) {
            ((unsigned char *)dst)[0] ^= 0x01;
        }
    }
    return HIP_SUCCESS;
}

hipError_t hipMemcpyHtoD(void *dst, const void *src, size_t n) { return hipMemcpy(dst, src, n, 1); }
hipError_t hipMemcpyDtoH(void *dst, const void *src, size_t n) { return hipMemcpy(dst, src, n, 2); }
hipError_t hipMemcpyDtoD(void *dst, const void *src, size_t n) { return hipMemcpy(dst, src, n, 3); }

hipError_t hipDeviceSynchronize(void) {
    if (shim_fault_is("ecc_ue") && shim_fault_target_matches("hip", g_current_dev)) {
        return HIP_ERROR_ECC_NOT_CORRECTABLE;
    }
    return HIP_SUCCESS;
}

hipError_t hipModuleLoadData(hipModule_t *mod, const void *img) {
    (void)img;
    void *m = malloc(16);
    if (!m) return HIP_ERROR_OUT_OF_MEMORY;
    atomic_fetch_add(&g_active_modules, 1);
    if (mod) *mod = m;
    return HIP_SUCCESS;
}

hipError_t hipModuleUnload(hipModule_t mod) {
    if (mod) {
        free(mod);
        atomic_fetch_sub(&g_active_modules, 1);
    }
    return HIP_SUCCESS;
}

hipError_t hipModuleGetFunction(hipFunction_t *fn, hipModule_t mod, const char *name) {
    (void)mod; (void)name;
    if (fn) *fn = (hipFunction_t)(uintptr_t)0x1;
    return HIP_SUCCESS;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gx, unsigned int gy, unsigned int gz,
                                 unsigned int bx, unsigned int by, unsigned int bz,
                                 unsigned int smem, hipStream_t s, void **params, void **extra) {
    (void)f; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
    (void)smem; (void)s; (void)params; (void)extra;
    if (shim_fault_is("ecc_ue") && shim_fault_target_matches("hip", g_current_dev)) {
        return HIP_ERROR_ECC_NOT_CORRECTABLE;
    }
    return HIP_SUCCESS;
}

const char *hipGetErrorName(hipError_t err) {
    return (err == HIP_ERROR_ECC_NOT_CORRECTABLE) ? "hipErrorECCNotCorrectable" : "hipErrorUnknown";
}

const char *hipGetErrorString(hipError_t err) { return hipGetErrorName(err); }
