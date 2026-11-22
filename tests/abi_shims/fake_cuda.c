// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "shim_common.h"

typedef int CUresult;
typedef int CUdevice;
typedef uint64_t CUdeviceptr;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_DEVICE 101
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_ECC_UNCORRECTABLE 214

static int g_active_allocs = 0;
static int g_active_contexts = 0;
static int g_active_modules = 0;
static int g_current_dev = 0;
static int g_bitflip_injected = 0;

static int cuda_count(void) {
    return shim_env_int("FAKE_CUDA_COUNT", 2, 0, SHIM_MAX_GPUS);
}

static int cuda_bus(int dev) {
    return shim_env_int("FAKE_CUDA_BUS_BASE", 1, 1, 200) + dev;
}

__attribute__((destructor))
static void check_cuda_leaks(void) {
    if (!shim_leak_check_enabled()) return;
    if (g_active_allocs != 0 || g_active_contexts != 0 || g_active_modules != 0) {
        fprintf(stderr, "[fake_cuda] LEAK: allocs=%d contexts=%d modules=%d\n",
                g_active_allocs, g_active_contexts, g_active_modules);
        _exit(99);
    }
}

CUresult cuInit(unsigned int flags) { (void)flags; return CUDA_SUCCESS; }
CUresult cuDriverGetVersion(int *v) { if (v) *v = 12060; return CUDA_SUCCESS; }
CUresult cuDeviceGetCount(int *c) { if (c) *c = cuda_count(); return CUDA_SUCCESS; }

CUresult cuDeviceGet(CUdevice *dev, int ordinal) {
    if (ordinal < 0 || ordinal >= cuda_count()) return CUDA_ERROR_INVALID_DEVICE;
    if (dev) *dev = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (name && len > 0) snprintf(name, (size_t)len, "Fake CUDA GPU %d", dev);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetPCIBusId(char *pci, int len, CUdevice dev) {
    if (pci && len > 0) snprintf(pci, (size_t)len, "0000:%02x:00.0", cuda_bus(dev));
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid(unsigned char uuid[16], CUdevice dev) {
    if (uuid) {
        for (int i = 0; i < 15; ++i) uuid[i] = (unsigned char)(0xC0 + i);
        uuid[15] = (unsigned char)dev;
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid_v2(unsigned char uuid[16], CUdevice dev) {
    return cuDeviceGetUuid(uuid, dev);
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    (void)dev;
    if (bytes) *bytes = (size_t)8ULL * 1024ULL * 1024ULL * 1024ULL;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev) {
    (void)dev;
    if (!pi) return CUDA_SUCCESS;
    switch (attrib) {
        case 75: *pi = 8; break;
        case 76: *pi = 9; break;
        case 16: *pi = 64; break;
        case 1:  *pi = 1024; break;
        case 10: *pi = 32; break;
        default: *pi = 1; break;
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    (void)flags;
    int *ctx = (int *)malloc(sizeof(int));
    if (!ctx) return CUDA_ERROR_OUT_OF_MEMORY;
    *ctx = dev;
    g_current_dev = dev;
    g_active_contexts++;
    if (pctx) *pctx = ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    if (ctx) g_current_dev = *(int *)ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
    if (ctx) { free(ctx); g_active_contexts--; }
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
    if (shim_fault_is("ecc_ue") && shim_fault_target_matches("cuda", g_current_dev)) {
        return CUDA_ERROR_ECC_UNCORRECTABLE;
    }
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    void *buf = calloc(1, bytesize ? bytesize : 1);
    if (!buf) return CUDA_ERROR_OUT_OF_MEMORY;
    g_active_allocs++;
    if (dptr) *dptr = (CUdeviceptr)(uintptr_t)buf;
    return CUDA_SUCCESS;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    if (dptr) { free((void *)(uintptr_t)dptr); g_active_allocs--; }
    return CUDA_SUCCESS;
}

CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
    void *buf = calloc(1, bytesize ? bytesize : 1);
    if (!buf) return CUDA_ERROR_OUT_OF_MEMORY;
    g_active_allocs++;
    if (pp) *pp = buf;
    return CUDA_SUCCESS;
}

CUresult cuMemFreeHost(void *p) {
    if (p) { free(p); g_active_allocs--; }
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n) {
    if (shim_fault_is("ecc_ue") && shim_fault_target_matches("cuda", g_current_dev)) {
        return CUDA_ERROR_ECC_UNCORRECTABLE;
    }
    if (dst && src && n) memcpy((void *)(uintptr_t)dst, src, n);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t n) {
    if (shim_fault_is("ecc_ue") && shim_fault_target_matches("cuda", g_current_dev)) {
        return CUDA_ERROR_ECC_UNCORRECTABLE;
    }
    if (dst && src && n) {
        memcpy(dst, (const void *)(uintptr_t)src, n);
        if (shim_fault_is("bitflip") && shim_fault_target_matches("cuda", g_current_dev) && !g_bitflip_injected) {
            ((unsigned char *)dst)[0] ^= 0x01;
            g_bitflip_injected = 1;
        }
    }
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t n) {
    if (shim_fault_is("ecc_ue") && shim_fault_target_matches("cuda", g_current_dev)) {
        return CUDA_ERROR_ECC_UNCORRECTABLE;
    }
    if (dst && src && n) memcpy((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, n);
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *mod, const void *img) {
    (void)img;
    void *m = malloc(16);
    if (!m) return CUDA_ERROR_OUT_OF_MEMORY;
    g_active_modules++;
    if (mod) *mod = m;
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadDataEx(CUmodule *mod, const void *img, unsigned int n, void *o, void **v) {
    (void)n; (void)o; (void)v;
    return cuModuleLoadData(mod, img);
}

CUresult cuModuleUnload(CUmodule mod) {
    if (mod) { free(mod); g_active_modules--; }
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *fn, CUmodule mod, const char *name) {
    (void)mod; (void)name;
    if (fn) *fn = (CUfunction)(uintptr_t)0x1;
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz,
                        unsigned int smem, CUstream s, void **params, void **extra) {
    (void)f; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
    (void)smem; (void)s; (void)params; (void)extra;
    if (shim_fault_is("ecc_ue") && shim_fault_target_matches("cuda", g_current_dev)) {
        return CUDA_ERROR_ECC_UNCORRECTABLE;
    }
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult err, const char **pstr) {
    if (pstr) *pstr = (err == CUDA_ERROR_ECC_UNCORRECTABLE) ? "CUDA_ERROR_ECC_UNCORRECTABLE" : "CUDA_ERROR_UNKNOWN";
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult err, const char **pstr) { return cuGetErrorName(err, pstr); }