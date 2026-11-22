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
typedef void *cl_program;
typedef void *cl_kernel;

#define CL_SUCCESS 0
#define CL_DEVICE_NOT_FOUND -1
#define CL_DEVICE_NOT_AVAILABLE -2
#define CL_OUT_OF_HOST_MEMORY -6
#define CL_INVALID_VALUE -30
#define CL_DEVICE_TYPE_GPU (1ULL << 2)

#define CL_PLATFORM_VERSION 0x0901
#define CL_PLATFORM_NAME 0x0902
#define CL_PLATFORM_VENDOR 0x0903

#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_VENDOR_ID 0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DRIVER_VERSION 0x102D
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_EXTENSIONS 0x1030
#define CL_DEVICE_DOUBLE_FP_CONFIG 0x1032
#define CL_DEVICE_HALF_FP_CONFIG 0x1033
#define CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035
#define CL_DEVICE_PCI_BUS_ID_NV 0x4008
#define CL_DEVICE_PCI_SLOT_ID_NV 0x4009
#define CL_DEVICE_PCI_DOMAIN_ID_NV 0x400A
#define CL_DEVICE_TOPOLOGY_AMD 0x4037
#define CL_DEVICE_PCI_BUS_INFO_KHR 0x410F

#define CL_PROGRAM_BINARY_SIZES 0x1165
#define CL_PROGRAM_BINARIES 0x1166
#define CL_PROGRAM_BUILD_LOG 0x1183

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

typedef struct {
    int dev_idx;
} fake_cl_handle_t;

static _Atomic int g_active_mems = 0;
static _Atomic int g_active_handles = 0;
static const unsigned char FAKE_CL_BIN[] = { 'C', 'L', 'B', 'I', 'N', 0, 1, 0 };

static int cl_count(void) { return shim_env_int("FAKE_CL_COUNT", 5, 0, SHIM_MAX_GPUS); }
static int cl_bus(int dev) { return shim_env_int("FAKE_CL_BUS_BASE", 1, 1, 200) + dev; }

static int cl_handle_to_idx(cl_device_id d) {
    uintptr_t v = (uintptr_t)d;
    return v >= 0x3000u ? (int)(v - 0x3000u) : 0;
}

__attribute__((destructor))
static void check_cl_leaks(void) {
    if (!shim_leak_check_enabled()) return;
    int mems = atomic_load(&g_active_mems);
    int handles = atomic_load(&g_active_handles);
    if (mems != 0 || handles != 0) {
        fprintf(stderr, "[fake_opencl] LEAK: mems=%d handles=%d\n", mems, handles);
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

static fake_cl_handle_t *alloc_cl_handle(int dev_idx) {
    fake_cl_handle_t *h = (fake_cl_handle_t *)malloc(sizeof(fake_cl_handle_t));
    if (!h) return NULL;
    h->dev_idx = dev_idx;
    atomic_fetch_add(&g_active_handles, 1);
    return h;
}

static void free_cl_handle(void *h) {
    if (h) {
        free(h);
        atomic_fetch_sub(&g_active_handles, 1);
    }
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
    cl_ulong vram = (cl_ulong)shim_vram_bytes(256);

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
        case CL_DEVICE_MAX_CLOCK_FREQUENCY: {
            cl_uint mhz = 1800;
            return write_info(&mhz, sizeof(mhz), sz, val, ret);
        }
        case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
        case CL_DEVICE_GLOBAL_MEM_SIZE:
            return write_info(&vram, sizeof(vram), sz, val, ret);
        case CL_DEVICE_DOUBLE_FP_CONFIG: {
            cl_ulong fp64 = 0x3Fu;
            return write_info(&fp64, sizeof(fp64), sz, val, ret);
        }
        case CL_DEVICE_HALF_FP_CONFIG: {
            cl_ulong fp16 = 0;
            return write_info(&fp16, sizeof(fp16), sz, val, ret);
        }
        case CL_DEVICE_HOST_UNIFIED_MEMORY: {
            cl_uint unified = 0;
            return write_info(&unified, sizeof(unified), sz, val, ret);
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
    (void)props; (void)cb; (void)ud;
    int idx = (nd > 0 && devs) ? cl_handle_to_idx(devs[0]) : 0;
    fake_cl_handle_t *h = alloc_cl_handle(idx);
    if (!h) { if (err) *err = CL_OUT_OF_HOST_MEMORY; return NULL; }
    if (err) *err = CL_SUCCESS;
    return h;
}

cl_int clReleaseContext(cl_context ctx) {
    free_cl_handle(ctx);
    return CL_SUCCESS;
}

cl_command_queue clCreateCommandQueueWithProperties(cl_context c, cl_device_id d, const void *p, cl_int *err) {
    (void)c; (void)p;
    fake_cl_handle_t *h = alloc_cl_handle(cl_handle_to_idx(d));
    if (!h) { if (err) *err = CL_OUT_OF_HOST_MEMORY; return NULL; }
    if (err) *err = CL_SUCCESS;
    return h;
}

cl_command_queue clCreateCommandQueue(cl_context c, cl_device_id d, cl_bitfield p, cl_int *err) {
    (void)p;
    return clCreateCommandQueueWithProperties(c, d, NULL, err);
}

cl_int clReleaseCommandQueue(cl_command_queue q) {
    free_cl_handle(q);
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
    atomic_fetch_add(&g_active_mems, 1);
    if (err) *err = CL_SUCCESS;
    return m;
}

cl_int clReleaseMemObject(cl_mem mem) {
    if (mem) {
        fake_cl_mem_t *m = (fake_cl_mem_t *)mem;
        free(m->buf);
        free(m);
        atomic_fetch_sub(&g_active_mems, 1);
    }
    return CL_SUCCESS;
}

void *clEnqueueMapBuffer(cl_command_queue q, cl_mem mem, cl_uint blk, cl_bitfield flags,
                         size_t off, size_t sz, cl_uint nw, const void *wl, void *ev, cl_int *err) {
    (void)q; (void)blk; (void)flags; (void)nw; (void)wl; (void)ev;
    fake_cl_mem_t *m = (fake_cl_mem_t *)mem;
    if (!m || off + sz > m->size) {
        if (err) *err = CL_INVALID_VALUE;
        return NULL;
    }
    if (err) *err = CL_SUCCESS;
    return (unsigned char *)m->buf + off;
}

cl_int clEnqueueUnmapMemObject(cl_command_queue q, cl_mem mem, void *mapped_ptr,
                               cl_uint nw, const void *wl, void *ev) {
    (void)q; (void)mem; (void)mapped_ptr; (void)nw; (void)wl; (void)ev;
    return CL_SUCCESS;
}

cl_int clEnqueueWriteBuffer(cl_command_queue q, cl_mem mem, cl_uint blk, size_t off, size_t sz,
                            const void *ptr, cl_uint nw, const void *wl, void *ev) {
    (void)blk; (void)nw; (void)wl; (void)ev;
    int dev_idx = q ? ((fake_cl_handle_t *)q)->dev_idx : 0;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("opencl", dev_idx)) {
        return CL_DEVICE_NOT_AVAILABLE;
    }
    fake_cl_mem_t *m = (fake_cl_mem_t *)mem;
    if (m && ptr && off + sz <= m->size) memcpy((unsigned char *)m->buf + off, ptr, sz);
    return CL_SUCCESS;
}

cl_int clEnqueueReadBuffer(cl_command_queue q, cl_mem mem, cl_uint blk, size_t off, size_t sz,
                           void *ptr, cl_uint nw, const void *wl, void *ev) {
    (void)blk; (void)nw; (void)wl; (void)ev;
    int dev_idx = q ? ((fake_cl_handle_t *)q)->dev_idx : 0;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("opencl", dev_idx)) {
        return CL_DEVICE_NOT_AVAILABLE;
    }
    fake_cl_mem_t *m = (fake_cl_mem_t *)mem;
    if (m && ptr && off + sz <= m->size) {
        memcpy(ptr, (const unsigned char *)m->buf + off, sz);
        if (shim_fault_is("bitflip") && shim_fault_target_matches("opencl", dev_idx)) {
            ((unsigned char *)ptr)[0] ^= 0x01;
        }
    }
    return CL_SUCCESS;
}

cl_int clEnqueueCopyBuffer(cl_command_queue q, cl_mem src, cl_mem dst, size_t so, size_t doff,
                           size_t sz, cl_uint nw, const void *wl, void *ev) {
    (void)nw; (void)wl; (void)ev;
    int dev_idx = q ? ((fake_cl_handle_t *)q)->dev_idx : 0;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("opencl", dev_idx)) {
        return CL_DEVICE_NOT_AVAILABLE;
    }
    fake_cl_mem_t *s = (fake_cl_mem_t *)src;
    fake_cl_mem_t *d = (fake_cl_mem_t *)dst;
    if (s && d && so + sz <= s->size && doff + sz <= d->size) {
        memcpy((unsigned char *)d->buf + doff, (const unsigned char *)s->buf + so, sz);
    }
    return CL_SUCCESS;
}

cl_int clFinish(cl_command_queue q) {
    int dev_idx = q ? ((fake_cl_handle_t *)q)->dev_idx : 0;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("opencl", dev_idx)) {
        return CL_DEVICE_NOT_AVAILABLE;
    }
    return CL_SUCCESS;
}

cl_program clCreateProgramWithSource(cl_context ctx, cl_uint count, const char **strings,
                                     const size_t *lengths, cl_int *err) {
    (void)count; (void)strings; (void)lengths;
    int dev_idx = ctx ? ((fake_cl_handle_t *)ctx)->dev_idx : 0;
    fake_cl_handle_t *h = alloc_cl_handle(dev_idx);
    if (!h) { if (err) *err = CL_OUT_OF_HOST_MEMORY; return NULL; }
    if (err) *err = CL_SUCCESS;
    return h;
}

cl_program clCreateProgramWithBinary(cl_context ctx, cl_uint nd, const cl_device_id *devs,
                                     const size_t *lengths, const unsigned char **bins,
                                     cl_int *bin_status, cl_int *err) {
    (void)nd; (void)devs; (void)lengths; (void)bins;
    if (bin_status) bin_status[0] = CL_SUCCESS;
    int dev_idx = ctx ? ((fake_cl_handle_t *)ctx)->dev_idx : 0;
    fake_cl_handle_t *h = alloc_cl_handle(dev_idx);
    if (!h) { if (err) *err = CL_OUT_OF_HOST_MEMORY; return NULL; }
    if (err) *err = CL_SUCCESS;
    return h;
}

cl_int clBuildProgram(cl_program p, cl_uint nd, const cl_device_id *devs, const char *opts,
                      void *cb, void *ud) {
    (void)p; (void)nd; (void)devs; (void)opts; (void)cb; (void)ud;
    return CL_SUCCESS;
}

cl_int clGetProgramBuildInfo(cl_program p, cl_device_id d, cl_uint param, size_t sz, void *val, size_t *ret) {
    (void)p; (void)d;
    if (param == CL_PROGRAM_BUILD_LOG) return write_str("", sz, val, ret);
    return CL_INVALID_VALUE;
}

cl_int clGetProgramInfo(cl_program p, cl_uint param, size_t sz, void *val, size_t *ret) {
    (void)p;
    if (param == CL_PROGRAM_BINARY_SIZES) {
        size_t bsz = sizeof(FAKE_CL_BIN);
        return write_info(&bsz, sizeof(bsz), sz, val, ret);
    }
    if (param == CL_PROGRAM_BINARIES) {
        if (ret) *ret = sizeof(unsigned char *);
        if (val && sz >= sizeof(unsigned char *)) {
            unsigned char **ptrs = (unsigned char **)val;
            if (ptrs[0]) memcpy(ptrs[0], FAKE_CL_BIN, sizeof(FAKE_CL_BIN));
        }
        return CL_SUCCESS;
    }
    return CL_INVALID_VALUE;
}

cl_int clReleaseProgram(cl_program p) {
    free_cl_handle(p);
    return CL_SUCCESS;
}

cl_kernel clCreateKernel(cl_program p, const char *name, cl_int *err) {
    (void)name;
    int dev_idx = p ? ((fake_cl_handle_t *)p)->dev_idx : 0;
    fake_cl_handle_t *h = alloc_cl_handle(dev_idx);
    if (!h) { if (err) *err = CL_OUT_OF_HOST_MEMORY; return NULL; }
    if (err) *err = CL_SUCCESS;
    return h;
}

cl_int clReleaseKernel(cl_kernel k) {
    free_cl_handle(k);
    return CL_SUCCESS;
}

cl_int clSetKernelArg(cl_kernel k, cl_uint idx, size_t sz, const void *val) {
    (void)k; (void)idx; (void)sz; (void)val;
    return CL_SUCCESS;
}

cl_int clEnqueueNDRangeKernel(cl_command_queue q, cl_kernel k, cl_uint work_dim,
                              const size_t *gwo, const size_t *gws, const size_t *lws,
                              cl_uint nw, const void *wl, void *ev) {
    (void)k; (void)work_dim; (void)gwo; (void)gws; (void)lws; (void)nw; (void)wl; (void)ev;
    int dev_idx = q ? ((fake_cl_handle_t *)q)->dev_idx : 0;
    if (shim_fault_is("device_lost") && shim_fault_target_matches("opencl", dev_idx)) {
        return CL_DEVICE_NOT_AVAILABLE;
    }
    return CL_SUCCESS;
}
