// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "shim_common.h"

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef void *cl_platform_id;
typedef void *cl_device_id;
typedef void *cl_context;
typedef void *cl_command_queue;
typedef void *cl_mem;

#define CL_SUCCESS 0
#define CL_DEVICE_NOT_FOUND -1
#define CL_OUT_OF_HOST_MEMORY -6
#define CL_INVALID_VALUE -30
#define CL_DEVICE_TYPE_GPU (1ULL << 2)

#define CL_PLATFORM_NAME 0x0902
#define CL_PLATFORM_VENDOR 0x0903
#define CL_PLATFORM_VERSION 0x0901

#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_VENDOR_ID 0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DRIVER_VERSION 0x102D
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_EXTENSIONS 0x1030
#define CL_DEVICE_PCI_BUS_ID_NV 0x4008
#define CL_DEVICE_PCI_SLOT_ID_NV 0x4009
#define CL_DEVICE_PCI_DOMAIN_ID_NV 0x400A
#define CL_DEVICE_TOPOLOGY_AMD 0x4037
#define CL_DEVICE_PCI_BUS_INFO_KHR 0x410F

typedef struct {
    uint32_t pci_domain;
    uint32_t pci_bus;
    uint32_t pci_device;
    uint32_t pci_function;
} cl_pci_bus_info_khr_t;

typedef struct {
    uint32_t type;
    char unused[17];
    uint8_t bus;
    uint8_t device;
    uint8_t function;
} cl_topology_amd_pcie_t;

typedef struct {
    void *buf;
    size_t size;
} fake_cl_mem_t;

static int g_active_mems = 0;
static int g_active_handles = 0;
static int g_bitflip_injected = 0;

static int cl_count(void) { return shim_env_int("FAKE_CL_COUNT", 5, 0, SHIM_MAX_GPUS); }
static int cl_bus(int dev) { return shim_env_int("FAKE_CL_BUS_BASE", 1, 1, 200) + dev; }
static int cl_handle_to_idx(cl_device_id d) {
    uintptr_t v = (uintptr_t)d;
    return v >= 0x3000u ? (int)(v - 0x3000u) : 0;
}

__attribute__((destructor))
static void check_cl_leaks(void) {
    if (!shim_leak_check_enabled()) return;
    if (g_active_mems != 0 || g_active_handles != 0) {
        fprintf(stderr, "[fake_opencl] LEAK: mems=%d handles=%d\n", g_active_mems, g_active_handles);
        _exit(99);
    }
}

static cl_int write_info(const void *src, size_t src_sz, size_t param_sz, void *param_val, size_t *ret_sz) {
    if (ret_sz) *ret_sz = src_sz;
    if (param_val) {
        if (param_sz < src_sz) return CL_INVALID_VALUE;
        memcpy(param_val, src, src_sz);
    }
    return CL_SUCCESS;
}

static cl_int write_str(const char *s, size_t param_sz, void *param_val, size_t *ret_sz) {
    return write_info(s, strlen(s) + 1, param_sz, param_val, ret_sz);
}

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
    cl_uint n = cl_count() > 0 ? 1u : 0u;
    if (num_platforms) *num_platforms = n;
    if (platforms && num_entries > 0 && n > 0) platforms[0] = (cl_platform_id)(uintptr_t)0x300;
    return CL_SUCCESS;
}

cl_int clGetPlatformInfo(cl_platform_id p, cl_uint param, size_t sz, void *val, size_t *ret) {
    (void)p;
    switch (param) {
        case CL_PLATFORM_NAME:    return write_str("Fake Multi-GPU OpenCL ICD", sz, val, ret);
        case CL_PLATFORM_VENDOR:  return write_str("CrossGPU ABI Shim", sz, val, ret);
        case CL_PLATFORM_VERSION: return write_str("OpenCL 3.0 Shim", sz, val, ret);
        default:                  return write_str("", sz, val, ret);
    }
}

cl_int clGetDeviceIDs(cl_platform_id p, cl_bitfield type, cl_uint num_entries,
                      cl_device_id *devices, cl_uint *num_devices) {
    (void)p;
    if ((type & CL_DEVICE_TYPE_GPU) == 0) {
        if (num_devices) *num_devices = 0;
        return CL_DEVICE_NOT_FOUND;
    }
    cl_uint n = (cl_uint)cl_count();
    if (num_devices) *num_devices = n;
    if (devices && num_entries > 0) {
        cl_uint m = num_entries < n ? num_entries : n;
        for (cl_uint i = 0; i < m; ++i) devices[i] = (cl_device_id)(uintptr_t)(0x3000u + i);
    }
    return n > 0 ? CL_SUCCESS : CL_DEVICE_NOT_FOUND;
}

cl_int clGetDeviceInfo(cl_device_id d, cl_uint param, size_t sz, void *val, size_t *ret) {
    int idx = cl_handle_to_idx(d);
    uint32_t bus = (uint32_t)cl_bus(idx);
    char buf[128];

    switch (param) {
        case CL_DEVICE_TYPE: {
            cl_bitfield t = CL_DEVICE_TYPE_GPU;
            return write_info(&t, sizeof(t), sz, val, ret);
        }
        case CL_DEVICE_VENDOR_ID: {
            cl_uint vid = (idx < 2) ? 0x10DEu : (idx == 2 ? 0x1002u : (idx == 3 ? 0x8086u : 0x1AF4u));
            return write_info(&vid, sizeof(vid), sz, val, ret);
        }
        case CL_DEVICE_MAX_COMPUTE_UNITS: {
            cl_uint cu = 32;
            return write_info(&cu, sizeof(cu), sz, val, ret);
        }
        case CL_DEVICE_MAX_WORK_GROUP_SIZE: {
            size_t wg = 256;
            return write_info(&wg, sizeof(wg), sz, val, ret);
        }
        case CL_DEVICE_GLOBAL_MEM_SIZE: {
            cl_ulong mem = 8ULL * 1024ULL * 1024ULL * 1024ULL;
            return write_info(&mem, sizeof(mem), sz, val, ret);
        }
        case CL_DEVICE_NAME:
            snprintf(buf, sizeof(buf), "Fake OpenCL GPU %d", idx);
            return write_str(buf, sz, val, ret);
        case CL_DEVICE_VENDOR:
            return write_str("CrossGPU Shim Vendor", sz, val, ret);
        case CL_DRIVER_VERSION:
            return write_str("1.0.0-shim", sz, val, ret);
        case CL_DEVICE_VERSION:
            return write_str("OpenCL 3.0", sz, val, ret);
        case CL_DEVICE_EXTENSIONS:
            return write_str("cl_khr_pci_bus_info cl_khr_fp64", sz, val, ret);
        case CL_DEVICE_PCI_BUS_INFO_KHR: {
            cl_pci_bus_info_khr_t info = { 0, bus, 0, 0 };
            return write_info(&info, sizeof(info), sz, val, ret);
        }
        case CL_DEVICE_TOPOLOGY_AMD: {
            cl_topology_amd_pcie_t top;
            memset(&top, 0, sizeof(top));
            top.type = 1;
            top.bus = (uint8_t)bus;
            return write_info(&top, sizeof(top), sz, val, ret);
        }
        case CL_DEVICE_PCI_DOMAIN_ID_NV: {
            cl_uint dom = 0;
            return write_info(&dom, sizeof(dom), sz, val, ret);
        }
        case CL_DEVICE_PCI_BUS_ID_NV:
            return write_info(&bus, sizeof(bus), sz, val, ret);
        case CL_DEVICE_PCI_SLOT_ID_NV: {
            cl_uint slot = 0;
            return write_info(&slot, sizeof(slot), sz, val, ret);
        }
        default:
            return CL_INVALID_VALUE;
    }
}

cl_context clCreateContext(const void *props, cl_uint nd, const cl_device_id *devs,
                           void *cb, void *ud, cl_int *err) {
    (void)props; (void)nd; (void)devs; (void)cb; (void)ud;
    void *h = malloc(16);
    if (!h) { if (err) *err = CL_OUT_OF_HOST_MEMORY; return NULL; }
    g_active_handles++;
    if (err) *err = CL_SUCCESS;
    return h;
}

cl_int clReleaseContext(cl_context ctx) {
    if (ctx) { free(ctx); g_active_handles--; }
    return CL_SUCCESS;
}

cl_command_queue clCreateCommandQueueWithProperties(cl_context c, cl_device_id d, const void *p, cl_int *err) {
    (void)c; (void)d; (void)p;
    void *h = malloc(16);
    if (!h) { if (err) *err = CL_OUT_OF_HOST_MEMORY; return NULL; }
    g_active_handles++;
    if (err) *err = CL_SUCCESS;
    return h;
}

cl_command_queue clCreateCommandQueue(cl_context c, cl_device_id d, cl_bitfield p, cl_int *err) {
    (void)p;
    return clCreateCommandQueueWithProperties(c, d, NULL, err);
}

cl_int clReleaseCommandQueue(cl_command_queue q) {
    if (q) { free(q); g_active_handles--; }
    return CL_SUCCESS;
}

cl_mem clCreateBuffer(cl_context c, cl_bitfield flags, size_t size, void *host_ptr, cl_int *err) {
    (void)c; (void)flags;
    fake_cl_mem_t *m = (fake_cl_mem_t *)malloc(sizeof(fake_cl_mem_t));
    if (!m) { if (err) *err = CL_OUT_OF_HOST_MEMORY; return NULL; }
    m->buf = calloc(1, size ? size : 1);
    m->size = size;
    if (!m->buf) { free(m); if (err) *err = CL_OUT_OF_HOST_MEMORY; return NULL; }
    if (host_ptr && size) memcpy(m->buf, host_ptr, size);
    g_active_mems++;
    if (err) *err = CL_SUCCESS;
    return m;
}

cl_int clReleaseMemObject(cl_mem mem) {
    if (mem) {
        fake_cl_mem_t *m = (fake_cl_mem_t *)mem;
        free(m->buf);
        free(m);
        g_active_mems--;
    }
    return CL_SUCCESS;
}

cl_int clEnqueueWriteBuffer(cl_command_queue q, cl_mem mem, cl_uint blk, size_t off, size_t sz,
                            const void *ptr, cl_uint nw, const void *wl, void *ev) {
    (void)q; (void)blk; (void)nw; (void)wl; (void)ev;
    fake_cl_mem_t *m = (fake_cl_mem_t *)mem;
    if (m && ptr && off + sz <= m->size) memcpy((unsigned char *)m->buf + off, ptr, sz);
    return CL_SUCCESS;
}

cl_int clEnqueueReadBuffer(cl_command_queue q, cl_mem mem, cl_uint blk, size_t off, size_t sz,
                           void *ptr, cl_uint nw, const void *wl, void *ev) {
    (void)q; (void)blk; (void)nw; (void)wl; (void)ev;
    fake_cl_mem_t *m = (fake_cl_mem_t *)mem;
    if (m && ptr && off + sz <= m->size) {
        memcpy(ptr, (const unsigned char *)m->buf + off, sz);
        if (shim_fault_is("bitflip") && shim_fault_target_matches("opencl", 4) && !g_bitflip_injected) {
            ((unsigned char *)ptr)[0] ^= 0x01;
            g_bitflip_injected = 1;
        }
    }
    return CL_SUCCESS;
}

cl_int clEnqueueCopyBuffer(cl_command_queue q, cl_mem src, cl_mem dst, size_t so, size_t doff,
                           size_t sz, cl_uint nw, const void *wl, void *ev) {
    (void)q; (void)nw; (void)wl; (void)ev;
    fake_cl_mem_t *s = (fake_cl_mem_t *)src;
    fake_cl_mem_t *d = (fake_cl_mem_t *)dst;
    if (s && d && so + sz <= s->size && doff + sz <= d->size) {
        memcpy((unsigned char *)d->buf + doff, (const unsigned char *)s->buf + so, sz);
    }
    return CL_SUCCESS;
}

cl_int clFinish(cl_command_queue q) { (void)q; return CL_SUCCESS; }