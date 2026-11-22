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

static int g_active_allocs = 0;
static int g_active_modules = 0;
static int g_current_dev = 0;
static int g_bitflip_injected = 0;

static int hip_count(void) {
    return shim_env_int("FAKE_HIP_COUNT", 1, 0, SHIM_MAX_GPUS);
}

static int hip_bus(int dev) {
    return shim_env_int("FAKE_HIP_BUS_BASE", 3, 1, 200) + dev;
}

__attribute__((destructor))
static void check_hip_leaks(void) {
    if (!shim_leak_check_enabled()) return;
    if (g_active_allocs != 0 || g_active_modules != 0) {
        fprintf(stderr, "[fake_hip] LEAK: allocs=%d modules=%d\n", g_active_allocs, g_active_modules);
        _exit(99);
    }
}

hipError_t hipInit(unsigned int flags) { (void)flags; return HIP_SUCCESS; }
hipError_t hipDriverGetVersion(int *v) { if (v) *v = 60200; return HIP_SUCCESS; }
hipError_t hipRuntimeGetVersion(int *v) { if (v) *v = 60200; return HIP_SUCCESS; }
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

hipError_t hipDeviceTotalMem(size_t *bytes, int dev) {
    (void)dev;
    if (bytes) *bytes = (size_t)16ULL * 1024ULL * 1024ULL * 1024ULL;
    return HIP_SUCCESS;
}

hipError_t hipDeviceGetAttribute(int *pi, int attr, int dev) {
    (void)dev;
    if (!pi) return HIP_SUCCESS;
    switch (attr) {
        case 20: *pi = 11; break;   /* ComputeCapabilityMajor (gfx11xx) */
        case 21: *pi = 0; break;    /* ComputeCapabilityMinor */
        case 63: *pi = 60; break;   /* MultiprocessorCount */
        case 87: *pi = 32; break;   /* WarpSize */
        case 56: *pi = 1024; break; /* MaxThreadsPerBlock */
        default: *pi = 1; break;
    }
    return HIP_SUCCESS;
}

hipError_t hipMalloc(void **ptr, size_t size) {
    void *buf = calloc(1, size ? size : 1);
    if (!buf) return HIP_ERROR_OUT_OF_MEMORY;
    g_active_allocs++;
    if (ptr) *ptr = buf;
    return HIP_SUCCESS;
}

hipError_t hipFree(void *ptr) {
    if (ptr) { free(ptr); g_active_allocs--; }
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
            shim_fault_target_matches("hip", g_current_dev) && !g_bitflip_injected) {
            ((unsigned char *)dst)[0] ^= 0x01;
            g_bitflip_injected = 1;
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
    g_active_modules++;
    if (mod) *mod = m;
    return HIP_SUCCESS;
}

hipError_t hipModuleUnload(hipModule_t mod) {
    if (mod) { free(mod); g_active_modules--; }
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