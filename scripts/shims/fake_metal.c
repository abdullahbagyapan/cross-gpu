// SPDX-License-Identifier: MIT
/*
 * Portable C11 Apple Metal / Objective-C Runtime ABI Shim for Docker Emulation.
 * Exports libobjc (objc_msgSend, objc_getClass, sel_registerName, retain/release,
 * autorelease pools), Metal.framework (MTLCopyAllDevices), and libSystem
 * (dispatch_data_create/dispatch_release) so CrossGPU's MetalBackend ctypes
 * bridge can be exercised end-to-end inside Linux Docker containers.
 */
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "shim_common.h"

typedef struct {
    uint64_t width;
    uint64_t height;
    uint64_t depth;
} FakeMTLSize;

#define MTL_RESOURCE_STORAGE_MODE_PRIVATE 0x20ULL
#define MTL_COMMAND_BUFFER_STATUS_COMPLETED 4ULL
#define MTL_COMMAND_BUFFER_STATUS_ERROR 5ULL
#define MTL_COMMAND_BUFFER_ERROR_TIMEOUT 2LL
#define MTL_GPU_FAMILY_APPLE7 1007LL
#define MTL_GPU_FAMILY_APPLE9 1009LL

enum FakeObjKind {
    OBJ_CLASS_NSSTRING = 1,
    OBJ_ARRAY,
    OBJ_STRING,
    OBJ_ERROR,
    OBJ_ARCH,
    OBJ_DEVICE,
    OBJ_QUEUE,
    OBJ_BUFFER,
    OBJ_LIBRARY,
    OBJ_FUNCTION,
    OBJ_PIPELINE,
    OBJ_CMD_BUF,
    OBJ_BLIT_ENC,
    OBJ_COMPUTE_ENC,
};

typedef struct FakeObj {
    enum FakeObjKind kind;
    int immortal;
    _Atomic int refcount;
    int dev_idx;
    union {
        struct {
            struct FakeObj **items;
            uint64_t count;
        } arr;
        struct {
            char *cstr;
        } str;
        struct {
            int64_t code;
            struct FakeObj *desc;
        } err;
        struct {
            struct FakeObj *name_str;
            struct FakeObj *arch_obj;
        } dev;
        struct {
            void *bytes;
            size_t length;
            int is_private;
        } buf;
        struct {
            uint64_t status;
            struct FakeObj *err_obj;
        } cmd;
        struct {
            int ended;
        } enc;
    } u;
} FakeObj;

/* Canonical selector table (pointer identity via static strings). */
static const char *const SEL_NAMES[] = {
    "count",
    "objectAtIndex:",
    "UTF8String",
    "stringWithUTF8String:",
    "respondsToSelector:",
    "name",
    "registryID",
    "architecture",
    "recommendedMaxWorkingSetSize",
    "currentAllocatedSize",
    "maxBufferLength",
    "hasUnifiedMemory",
    "isLowPower",
    "supportsFamily:",
    "supportsBfloat16",
    "newCommandQueue",
    "newBufferWithLength:options:",
    "contents",
    "newLibraryWithData:error:",
    "newLibraryWithSource:options:error:",
    "newFunctionWithName:",
    "newComputePipelineStateWithFunction:error:",
    "commandBuffer",
    "computeCommandEncoder",
    "blitCommandEncoder",
    "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
    "setComputePipelineState:",
    "setBuffer:offset:atIndex:",
    "setBytes:length:atIndex:",
    "setThreadgroupMemoryLength:atIndex:",
    "dispatchThreadgroups:threadsPerThreadgroup:",
    "endEncoding",
    "commit",
    "waitUntilCompleted",
    "status",
    "error",
    "code",
    "localizedDescription",
    "retain",
    "release",
};

#define NUM_SELS (sizeof(SEL_NAMES) / sizeof(SEL_NAMES[0]))

static _Atomic int g_active_arrays = 0;
static _Atomic int g_active_queues = 0;
static _Atomic int g_active_buffers = 0;
static _Atomic int g_active_libraries = 0;
static _Atomic int g_active_functions = 0;
static _Atomic int g_active_pipelines = 0;
static _Atomic int g_active_cmd_bufs = 0;
static _Atomic int g_active_encoders = 0;
static _Atomic int g_unended_encoders = 0;
static _Atomic int g_active_dispatch_data = 0;
static _Atomic int g_active_pools = 0;
static _Atomic uint64_t g_dispatch_calls = 0;

/* Per-thread autorelease pool stack (supports concurrent multi-GPU threads). */
#define ARPOOL_MAX_OBJS 1024
#define ARPOOL_MAX_DEPTH 64
static _Thread_local FakeObj *t_arpool_objs[ARPOOL_MAX_OBJS];
static _Thread_local size_t t_arpool_top = 0;
static _Thread_local size_t t_arpool_marks[ARPOOL_MAX_DEPTH];
static _Thread_local size_t t_arpool_depth = 0;

static FakeObj g_cls_nsstring = { .kind = OBJ_CLASS_NSSTRING, .immortal = 1, .refcount = 1 };
static FakeObj g_singleton_devices[SHIM_MAX_GPUS];
static FakeObj g_singleton_dev_names[SHIM_MAX_GPUS];
static char g_singleton_dev_name_bufs[SHIM_MAX_GPUS][64];
static FakeObj g_singleton_arch_str = { .kind = OBJ_STRING, .immortal = 1, .refcount = 1 };
static FakeObj g_singleton_arch = { .kind = OBJ_ARCH, .immortal = 1, .refcount = 1 };
static _Atomic int g_singletons_ready = 0;

static int metal_count(void) {
    return shim_env_int("FAKE_METAL_COUNT", 2, 0, SHIM_MAX_GPUS);
}

static void init_singletons(void) {
    int expected = 0;
    if (atomic_compare_exchange_strong(&g_singletons_ready, &expected, 1)) {
        g_singleton_arch_str.u.str.cstr = "applegpu_g15g";
        g_singleton_arch.u.dev.name_str = &g_singleton_arch_str;
        for (int i = 0; i < SHIM_MAX_GPUS; ++i) {
            snprintf(g_singleton_dev_name_bufs[i], sizeof(g_singleton_dev_name_bufs[i]),
                     "Fake Apple Metal GPU %d", i);
            g_singleton_dev_names[i].kind = OBJ_STRING;
            g_singleton_dev_names[i].immortal = 1;
            atomic_store(&g_singleton_dev_names[i].refcount, 1);
            g_singleton_dev_names[i].u.str.cstr = g_singleton_dev_name_bufs[i];

            g_singleton_devices[i].kind = OBJ_DEVICE;
            g_singleton_devices[i].immortal = 1;
            atomic_store(&g_singleton_devices[i].refcount, 1);
            g_singleton_devices[i].dev_idx = i;
            g_singleton_devices[i].u.dev.name_str = &g_singleton_dev_names[i];
            g_singleton_devices[i].u.dev.arch_obj = &g_singleton_arch;
        }
        atomic_store(&g_singletons_ready, 2);
    } else {
        while (atomic_load(&g_singletons_ready) != 2) {
            /* Spin briefly until initialization finishes. */
        }
    }
}

__attribute__((destructor))
static void check_metal_leaks(void) {
    if (!shim_leak_check_enabled()) return;
    int arrays = atomic_load(&g_active_arrays);
    int queues = atomic_load(&g_active_queues);
    int bufs = atomic_load(&g_active_buffers);
    int libs = atomic_load(&g_active_libraries);
    int fns = atomic_load(&g_active_functions);
    int psos = atomic_load(&g_active_pipelines);
    int cmds = atomic_load(&g_active_cmd_bufs);
    int encs = atomic_load(&g_active_encoders);
    int unended = atomic_load(&g_unended_encoders);
    int ddata = atomic_load(&g_active_dispatch_data);
    int pools = atomic_load(&g_active_pools);
    if (arrays != 0 || queues != 0 || bufs != 0 || libs != 0 || fns != 0 ||
        psos != 0 || cmds != 0 || encs != 0 || unended != 0 || ddata != 0 || pools != 0) {
        fprintf(stderr,
                "[fake_metal] LEAK: arrays=%d queues=%d buffers=%d libs=%d fns=%d "
                "psos=%d cmd_bufs=%d encoders=%d unended=%d ddata=%d pools=%d\n",
                arrays, queues, bufs, libs, fns, psos, cmds, encs, unended, ddata, pools);
        _exit(99);
    }
}

static FakeObj *alloc_obj(enum FakeObjKind kind, int dev_idx) {
    FakeObj *o = (FakeObj *)calloc(1, sizeof(FakeObj));
    if (!o) return NULL;
    o->kind = kind;
    o->immortal = 0;
    atomic_store(&o->refcount, 1);
    o->dev_idx = dev_idx;
    return o;
}

void *objc_retain(void *ptr) {
    FakeObj *o = (FakeObj *)ptr;
    if (!o || o->immortal) return ptr;
    atomic_fetch_add(&o->refcount, 1);
    return ptr;
}

void objc_release(void *ptr) {
    FakeObj *o = (FakeObj *)ptr;
    if (!o || o->immortal) return;
    if (atomic_fetch_sub(&o->refcount, 1) != 1) return;

    switch (o->kind) {
        case OBJ_ARRAY:
            free(o->u.arr.items);
            atomic_fetch_sub(&g_active_arrays, 1);
            break;
        case OBJ_STRING:
            free(o->u.str.cstr);
            break;
        case OBJ_ERROR:
            objc_release(o->u.err.desc);
            break;
        case OBJ_QUEUE:
            atomic_fetch_sub(&g_active_queues, 1);
            break;
        case OBJ_BUFFER:
            free(o->u.buf.bytes);
            atomic_fetch_sub(&g_active_buffers, 1);
            break;
        case OBJ_LIBRARY:
            atomic_fetch_sub(&g_active_libraries, 1);
            break;
        case OBJ_FUNCTION:
            atomic_fetch_sub(&g_active_functions, 1);
            break;
        case OBJ_PIPELINE:
            atomic_fetch_sub(&g_active_pipelines, 1);
            break;
        case OBJ_CMD_BUF:
            objc_release(o->u.cmd.err_obj);
            atomic_fetch_sub(&g_active_cmd_bufs, 1);
            break;
        case OBJ_BLIT_ENC:
        case OBJ_COMPUTE_ENC:
            atomic_fetch_sub(&g_active_encoders, 1);
            break;
        default:
            break;
    }
    free(o);
}

static FakeObj *autorelease_obj(FakeObj *o) {
    if (!o || o->immortal) return o;
    if (t_arpool_depth > 0 && t_arpool_top < ARPOOL_MAX_OBJS) {
        t_arpool_objs[t_arpool_top++] = o;
    }
    return o;
}

void *objc_autoreleasePoolPush(void) {
    if (t_arpool_depth >= ARPOOL_MAX_DEPTH) return NULL;
    size_t idx = t_arpool_depth++;
    t_arpool_marks[idx] = t_arpool_top;
    atomic_fetch_add(&g_active_pools, 1);
    return (void *)(uintptr_t)(idx + 1u);
}

void objc_autoreleasePoolPop(void *pool) {
    if (!pool || t_arpool_depth == 0) return;
    size_t target_depth = (size_t)((uintptr_t)pool) - 1u;
    if (target_depth >= t_arpool_depth) return;
    size_t mark = t_arpool_marks[target_depth];
    while (t_arpool_top > mark) {
        FakeObj *o = t_arpool_objs[--t_arpool_top];
        objc_release(o);
    }
    size_t popped = t_arpool_depth - target_depth;
    t_arpool_depth = target_depth;
    atomic_fetch_sub(&g_active_pools, (int)popped);
}

void *objc_getClass(const char *name) {
    if (name && strcmp(name, "NSString") == 0) {
        return &g_cls_nsstring;
    }
    return NULL;
}

void *sel_registerName(const char *name) {
    if (!name) return NULL;
    for (size_t i = 0; i < NUM_SELS; ++i) {
        if (strcmp(SEL_NAMES[i], name) == 0) {
            return (void *)(uintptr_t)SEL_NAMES[i];
        }
    }
    return NULL;
}

void *dispatch_data_create(const void *buf, size_t size, void *queue, void *destructor) {
    (void)queue; (void)destructor;
    if (!buf || size == 0) return NULL;
    void *d = malloc(size);
    if (!d) return NULL;
    memcpy(d, buf, size);
    atomic_fetch_add(&g_active_dispatch_data, 1);
    return d;
}

void dispatch_release(void *data) {
    if (data) {
        free(data);
        atomic_fetch_sub(&g_active_dispatch_data, 1);
    }
}

void *MTLCopyAllDevices(void) {
    init_singletons();
    int n = metal_count();
    FakeObj *arr = alloc_obj(OBJ_ARRAY, 0);
    if (!arr) return NULL;
    arr->u.arr.count = (uint64_t)n;
    if (n > 0) {
        arr->u.arr.items = (FakeObj **)calloc((size_t)n, sizeof(FakeObj *));
        if (!arr->u.arr.items) {
            free(arr);
            return NULL;
        }
        for (int i = 0; i < n; ++i) {
            arr->u.arr.items[i] = &g_singleton_devices[i];
        }
    }
    atomic_fetch_add(&g_active_arrays, 1);
    return arr;
}

static FakeObj *make_string(const char *text, int autorelease) {
    FakeObj *s = alloc_obj(OBJ_STRING, 0);
    if (!s) return NULL;
    size_t len = text ? strlen(text) : 0;
    s->u.str.cstr = (char *)malloc(len + 1);
    if (!s->u.str.cstr) {
        free(s);
        return NULL;
    }
    if (len > 0) memcpy(s->u.str.cstr, text, len);
    s->u.str.cstr[len] = '\0';
    return autorelease ? autorelease_obj(s) : s;
}

static FakeObj *make_error(int64_t code, const char *desc, int autorelease) {
    FakeObj *e = alloc_obj(OBJ_ERROR, 0);
    if (!e) return NULL;
    e->u.err.code = code;
    e->u.err.desc = make_string(desc, 0);
    return autorelease ? autorelease_obj(e) : e;
}

static int sel_is(const void *sel, const char *name) {
    return sel && strcmp((const char *)sel, name) == 0;
}

/*
 * Non-variadic dispatcher matching all 19 MetalBackend _bind_objc_msg CFUNCTYPEs.
 * Avoiding `...` eliminates x86_64 SysV %al SSE-count mismatches when invoked
 * through non-variadic ctypes function pointers.
 */
uintptr_t objc_msgSend(
    void *obj_ptr,
    void *sel_ptr,
    uintptr_t a1,
    uintptr_t a2,
    uintptr_t a3,
    uintptr_t a4,
    uintptr_t a5,
    uintptr_t a6,
    uintptr_t a7,
    uintptr_t a8,
    uintptr_t a9,
    uintptr_t a10
) {
    (void)a6; (void)a7; (void)a8; (void)a9; (void)a10;
    FakeObj *obj = (FakeObj *)obj_ptr;
    if (!obj || !sel_ptr) return 0;

    if (sel_is(sel_ptr, "retain")) {
        return (uintptr_t)objc_retain(obj);
    }
    if (sel_is(sel_ptr, "release")) {
        objc_release(obj);
        return 0;
    }

    switch (obj->kind) {
        case OBJ_CLASS_NSSTRING:
            if (sel_is(sel_ptr, "stringWithUTF8String:")) {
                return (uintptr_t)make_string((const char *)a1, 1);
            }
            return 0;

        case OBJ_STRING:
            if (sel_is(sel_ptr, "UTF8String")) {
                return (uintptr_t)(obj->u.str.cstr ? obj->u.str.cstr : "");
            }
            return 0;

        case OBJ_ARRAY:
            if (sel_is(sel_ptr, "count")) {
                return (uintptr_t)obj->u.arr.count;
            }
            if (sel_is(sel_ptr, "objectAtIndex:")) {
                uint64_t idx = (uint64_t)a1;
                if (idx < obj->u.arr.count) return (uintptr_t)obj->u.arr.items[idx];
                return 0;
            }
            return 0;

        case OBJ_ARCH:
            if (sel_is(sel_ptr, "name")) {
                return (uintptr_t)obj->u.dev.name_str;
            }
            return 0;

        case OBJ_ERROR:
            if (sel_is(sel_ptr, "code")) {
                return (uintptr_t)obj->u.err.code;
            }
            if (sel_is(sel_ptr, "localizedDescription")) {
                return (uintptr_t)obj->u.err.desc;
            }
            return 0;

        case OBJ_DEVICE:
            if (sel_is(sel_ptr, "respondsToSelector:")) {
                return a1 != 0 ? 1u : 0u;
            }
            if (sel_is(sel_ptr, "name")) {
                return (uintptr_t)obj->u.dev.name_str;
            }
            if (sel_is(sel_ptr, "registryID")) {
                return (uintptr_t)(0x1000003E8ULL + (uint64_t)obj->dev_idx);
            }
            if (sel_is(sel_ptr, "architecture")) {
                return (uintptr_t)obj->u.dev.arch_obj;
            }
            if (sel_is(sel_ptr, "recommendedMaxWorkingSetSize") ||
                sel_is(sel_ptr, "maxBufferLength")) {
                return (uintptr_t)shim_vram_bytes(256);
            }
            if (sel_is(sel_ptr, "currentAllocatedSize")) {
                return 0u;
            }
            if (sel_is(sel_ptr, "hasUnifiedMemory") || sel_is(sel_ptr, "supportsBfloat16")) {
                return 1u;
            }
            if (sel_is(sel_ptr, "isLowPower")) {
                return 0u;
            }
            if (sel_is(sel_ptr, "supportsFamily:")) {
                int64_t fam = (int64_t)a1;
                return (fam == MTL_GPU_FAMILY_APPLE7 || fam == MTL_GPU_FAMILY_APPLE9) ? 1u : 0u;
            }
            if (sel_is(sel_ptr, "newCommandQueue")) {
                FakeObj *q = alloc_obj(OBJ_QUEUE, obj->dev_idx);
                if (!q) return 0;
                atomic_fetch_add(&g_active_queues, 1);
                return (uintptr_t)q;
            }
            if (sel_is(sel_ptr, "newBufferWithLength:options:")) {
                uint64_t len = (uint64_t)a1;
                uint64_t opts = (uint64_t)a2;
                FakeObj *b = alloc_obj(OBJ_BUFFER, obj->dev_idx);
                if (!b) return 0;
                b->u.buf.bytes = calloc(1, len > 0 ? (size_t)len : 1);
                if (!b->u.buf.bytes) {
                    free(b);
                    return 0;
                }
                b->u.buf.length = (size_t)len;
                b->u.buf.is_private = (opts == MTL_RESOURCE_STORAGE_MODE_PRIVATE) ? 1 : 0;
                atomic_fetch_add(&g_active_buffers, 1);
                return (uintptr_t)b;
            }
            if (sel_is(sel_ptr, "newLibraryWithData:error:")) {
                void *data = (void *)a1;
                void **err_out = (void **)a2;
                if (!data) {
                    if (err_out) *err_out = make_error(1, "Empty metallib data", 1);
                    return 0;
                }
                FakeObj *lib = alloc_obj(OBJ_LIBRARY, obj->dev_idx);
                if (!lib) return 0;
                atomic_fetch_add(&g_active_libraries, 1);
                return (uintptr_t)lib;
            }
            if (sel_is(sel_ptr, "newLibraryWithSource:options:error:")) {
                FakeObj *src = (FakeObj *)a1;
                void **err_out = (void **)a3;
                if (!src || src->kind != OBJ_STRING || !src->u.str.cstr || !src->u.str.cstr[0]) {
                    if (err_out) *err_out = make_error(1, "Empty Metal source", 1);
                    return 0;
                }
                FakeObj *lib = alloc_obj(OBJ_LIBRARY, obj->dev_idx);
                if (!lib) return 0;
                atomic_fetch_add(&g_active_libraries, 1);
                return (uintptr_t)lib;
            }
            if (sel_is(sel_ptr, "newComputePipelineStateWithFunction:error:")) {
                void *fn = (void *)a1;
                void **err_out = (void **)a2;
                if (!fn) {
                    if (err_out) *err_out = make_error(1, "Nil Metal function", 1);
                    return 0;
                }
                FakeObj *pso = alloc_obj(OBJ_PIPELINE, obj->dev_idx);
                if (!pso) return 0;
                atomic_fetch_add(&g_active_pipelines, 1);
                return (uintptr_t)pso;
            }
            return 0;

        case OBJ_BUFFER:
            if (sel_is(sel_ptr, "contents")) {
                return (uintptr_t)obj->u.buf.bytes;
            }
            return 0;

        case OBJ_LIBRARY:
            if (sel_is(sel_ptr, "newFunctionWithName:")) {
                FakeObj *nm = (FakeObj *)a1;
                if (!nm || nm->kind != OBJ_STRING || !nm->u.str.cstr || !nm->u.str.cstr[0]) {
                    return 0;
                }
                FakeObj *fn = alloc_obj(OBJ_FUNCTION, obj->dev_idx);
                if (!fn) return 0;
                atomic_fetch_add(&g_active_functions, 1);
                return (uintptr_t)fn;
            }
            return 0;

        case OBJ_QUEUE:
            if (sel_is(sel_ptr, "commandBuffer")) {
                FakeObj *cb = alloc_obj(OBJ_CMD_BUF, obj->dev_idx);
                if (!cb) return 0;
                cb->u.cmd.status = MTL_COMMAND_BUFFER_STATUS_COMPLETED;
                atomic_fetch_add(&g_active_cmd_bufs, 1);
                return (uintptr_t)autorelease_obj(cb);
            }
            return 0;

        case OBJ_CMD_BUF:
            if (sel_is(sel_ptr, "blitCommandEncoder")) {
                FakeObj *enc = alloc_obj(OBJ_BLIT_ENC, obj->dev_idx);
                if (!enc) return 0;
                atomic_fetch_add(&g_active_encoders, 1);
                atomic_fetch_add(&g_unended_encoders, 1);
                return (uintptr_t)autorelease_obj(enc);
            }
            if (sel_is(sel_ptr, "computeCommandEncoder")) {
                FakeObj *enc = alloc_obj(OBJ_COMPUTE_ENC, obj->dev_idx);
                if (!enc) return 0;
                atomic_fetch_add(&g_active_encoders, 1);
                atomic_fetch_add(&g_unended_encoders, 1);
                return (uintptr_t)autorelease_obj(enc);
            }
            if (sel_is(sel_ptr, "commit")) {
                if (shim_fault_is("gpu_timeout") &&
                    shim_fault_target_matches("metal", obj->dev_idx)) {
                    obj->u.cmd.status = MTL_COMMAND_BUFFER_STATUS_ERROR;
                    objc_release(obj->u.cmd.err_obj);
                    obj->u.cmd.err_obj = make_error(
                        MTL_COMMAND_BUFFER_ERROR_TIMEOUT,
                        "Fake Metal command buffer GPU timeout",
                        0
                    );
                }
                return 0;
            }
            if (sel_is(sel_ptr, "waitUntilCompleted")) {
                return 0;
            }
            if (sel_is(sel_ptr, "status")) {
                return (uintptr_t)obj->u.cmd.status;
            }
            if (sel_is(sel_ptr, "error")) {
                return (uintptr_t)obj->u.cmd.err_obj;
            }
            return 0;

        case OBJ_BLIT_ENC:
            if (sel_is(sel_ptr, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:")) {
                FakeObj *src = (FakeObj *)a1;
                uint64_t src_off = (uint64_t)a2;
                FakeObj *dst = (FakeObj *)a3;
                uint64_t dst_off = (uint64_t)a4;
                uint64_t sz = (uint64_t)a5;
                if (src && dst && src->kind == OBJ_BUFFER && dst->kind == OBJ_BUFFER && sz > 0 &&
                    (size_t)src_off + (size_t)sz <= src->u.buf.length &&
                    (size_t)dst_off + (size_t)sz <= dst->u.buf.length) {
                    unsigned char *dst_p = (unsigned char *)dst->u.buf.bytes + dst_off;
                    const unsigned char *src_p = (const unsigned char *)src->u.buf.bytes + src_off;
                    memcpy(dst_p, src_p, (size_t)sz);
                    if (src->u.buf.is_private && !dst->u.buf.is_private &&
                        shim_fault_is("bitflip") &&
                        shim_fault_target_matches("metal", obj->dev_idx)) {
                        dst_p[0] ^= 0x01;
                    }
                }
                return 0;
            }
            if (sel_is(sel_ptr, "endEncoding")) {
                if (!obj->u.enc.ended) {
                    obj->u.enc.ended = 1;
                    atomic_fetch_sub(&g_unended_encoders, 1);
                }
                return 0;
            }
            return 0;

        case OBJ_COMPUTE_ENC:
            if (sel_is(sel_ptr, "setComputePipelineState:") ||
                sel_is(sel_ptr, "setBuffer:offset:atIndex:") ||
                sel_is(sel_ptr, "setBytes:length:atIndex:") ||
                sel_is(sel_ptr, "setThreadgroupMemoryLength:atIndex:")) {
                return 0;
            }
            if (sel_is(sel_ptr, "dispatchThreadgroups:threadsPerThreadgroup:")) {
                FakeMTLSize grid = { 0, 0, 0 };
                FakeMTLSize block = { 0, 0, 0 };
#if defined(__x86_64__)
                grid.width = (uint64_t)a5;
                grid.height = (uint64_t)a6;
                grid.depth = (uint64_t)a7;
                block.width = (uint64_t)a8;
                block.height = (uint64_t)a9;
                block.depth = (uint64_t)a10;
#elif defined(__powerpc64__)
                grid.width = (uint64_t)a1;
                grid.height = (uint64_t)a2;
                grid.depth = (uint64_t)a3;
                block.width = (uint64_t)a4;
                block.height = (uint64_t)a5;
                block.depth = (uint64_t)a6;
#else
                /* AAPCS64 (arm64), RISC-V LP64, and s390x pass >16B structs by indirect pointer. */
                if (a1) memcpy(&grid, (const void *)a1, sizeof(grid));
                if (a2) memcpy(&block, (const void *)a2, sizeof(block));
#endif
                if (grid.width > 0 && block.width > 0) {
                    atomic_fetch_add(&g_dispatch_calls, 1);
                }
                return 0;
            }
            if (sel_is(sel_ptr, "endEncoding")) {
                if (!obj->u.enc.ended) {
                    obj->u.enc.ended = 1;
                    atomic_fetch_sub(&g_unended_encoders, 1);
                }
                return 0;
            }
            return 0;

        default:
            return 0;
    }
}

uint64_t fake_metal_dispatch_count(void) {
    return atomic_load(&g_dispatch_calls);
}
