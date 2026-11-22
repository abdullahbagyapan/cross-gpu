// SPDX-License-Identifier: MIT
#import <Foundation/Foundation.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "shim_common.h"

/* Must match MetalBackend._MTLSize (24-byte struct passed by value via objc_msgSend). */
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

static _Atomic int g_active_queues = 0;
static _Atomic int g_active_buffers = 0;
static _Atomic int g_active_libraries = 0;
static _Atomic int g_active_functions = 0;
static _Atomic int g_active_pipelines = 0;
static _Atomic int g_active_cmd_bufs = 0;
static _Atomic int g_active_encoders = 0;
static _Atomic int g_unended_encoders = 0;
static _Atomic uint64_t g_dispatch_calls = 0;

static int metal_count(void) {
    return shim_env_int("FAKE_METAL_COUNT", 2, 0, SHIM_MAX_GPUS);
}

__attribute__((destructor))
static void check_metal_leaks(void) {
    if (!shim_leak_check_enabled()) return;
    int queues = atomic_load(&g_active_queues);
    int bufs = atomic_load(&g_active_buffers);
    int libs = atomic_load(&g_active_libraries);
    int fns = atomic_load(&g_active_functions);
    int psos = atomic_load(&g_active_pipelines);
    int cmds = atomic_load(&g_active_cmd_bufs);
    int encs = atomic_load(&g_active_encoders);
    int unended = atomic_load(&g_unended_encoders);
    if (queues != 0 || bufs != 0 || libs != 0 || fns != 0 ||
        psos != 0 || cmds != 0 || encs != 0 || unended != 0) {
        fprintf(stderr,
                "[fake_metal] LEAK: queues=%d buffers=%d libs=%d fns=%d psos=%d cmd_bufs=%d encoders=%d unended=%d\n",
                queues, bufs, libs, fns, psos, cmds, encs, unended);
        _exit(99);
    }
}

@interface FakeMTLArchitecture : NSObject {
    NSString *_name;
}
- (instancetype)initWithName:(NSString *)name;
- (NSString *)name;
@end

@implementation FakeMTLArchitecture
- (instancetype)initWithName:(NSString *)name {
    self = [super init];
    if (self) {
        _name = [name copy];
    }
    return self;
}
- (NSString *)name { return _name; }
- (void)dealloc {
    [_name release];
    [super dealloc];
}
@end

@interface FakeMTLBuffer : NSObject {
    void *_bytes;
    size_t _length;
    BOOL _isPrivate;
}
- (instancetype)initWithLength:(uint64_t)length options:(uint64_t)options;
- (void *)contents;
- (void *)rawBytes;
- (size_t)bufferLength;
- (BOOL)isPrivate;
@end

@implementation FakeMTLBuffer
- (instancetype)initWithLength:(uint64_t)length options:(uint64_t)options {
    self = [super init];
    if (self) {
        size_t alloc_sz = length > 0 ? (size_t)length : 1;
        _bytes = calloc(1, alloc_sz);
        if (!_bytes) {
            [self release];
            return nil;
        }
        _length = (size_t)length;
        _isPrivate = (options == MTL_RESOURCE_STORAGE_MODE_PRIVATE) ? YES : NO;
        atomic_fetch_add(&g_active_buffers, 1);
    }
    return self;
}
- (void *)contents { return _bytes; }
- (void *)rawBytes { return _bytes; }
- (size_t)bufferLength { return _length; }
- (BOOL)isPrivate { return _isPrivate; }
- (void)dealloc {
    free(_bytes);
    atomic_fetch_sub(&g_active_buffers, 1);
    [super dealloc];
}
@end

@interface FakeMTLFunction : NSObject {
    NSString *_name;
}
- (instancetype)initWithName:(NSString *)name;
@end

@implementation FakeMTLFunction
- (instancetype)initWithName:(NSString *)name {
    self = [super init];
    if (self) {
        _name = [name copy];
        atomic_fetch_add(&g_active_functions, 1);
    }
    return self;
}
- (void)dealloc {
    [_name release];
    atomic_fetch_sub(&g_active_functions, 1);
    [super dealloc];
}
@end

@interface FakeMTLLibrary : NSObject
- (id)newFunctionWithName:(NSString *)name;
@end

@implementation FakeMTLLibrary
- (instancetype)init {
    self = [super init];
    if (self) {
        atomic_fetch_add(&g_active_libraries, 1);
    }
    return self;
}
- (id)newFunctionWithName:(NSString *)name {
    if (!name || [name length] == 0) return nil;
    return [[FakeMTLFunction alloc] initWithName:name];
}
- (void)dealloc {
    atomic_fetch_sub(&g_active_libraries, 1);
    [super dealloc];
}
@end

@interface FakeMTLComputePipelineState : NSObject
@end

@implementation FakeMTLComputePipelineState
- (instancetype)init {
    self = [super init];
    if (self) {
        atomic_fetch_add(&g_active_pipelines, 1);
    }
    return self;
}
- (void)dealloc {
    atomic_fetch_sub(&g_active_pipelines, 1);
    [super dealloc];
}
@end

@interface FakeMTLBlitCommandEncoder : NSObject {
    int _devIdx;
    BOOL _ended;
}
- (instancetype)initWithDeviceIndex:(int)devIdx;
- (void)copyFromBuffer:(FakeMTLBuffer *)src
          sourceOffset:(uint64_t)srcOff
              toBuffer:(FakeMTLBuffer *)dst
     destinationOffset:(uint64_t)dstOff
                  size:(uint64_t)size;
- (void)endEncoding;
@end

@implementation FakeMTLBlitCommandEncoder
- (instancetype)initWithDeviceIndex:(int)devIdx {
    self = [super init];
    if (self) {
        _devIdx = devIdx;
        _ended = NO;
        atomic_fetch_add(&g_active_encoders, 1);
        atomic_fetch_add(&g_unended_encoders, 1);
    }
    return self;
}
- (void)copyFromBuffer:(FakeMTLBuffer *)src
          sourceOffset:(uint64_t)srcOff
              toBuffer:(FakeMTLBuffer *)dst
     destinationOffset:(uint64_t)dstOff
                  size:(uint64_t)size {
    if (!src || !dst || size == 0) return;
    if ((size_t)srcOff + (size_t)size > [src bufferLength]) return;
    if ((size_t)dstOff + (size_t)size > [dst bufferLength]) return;
    unsigned char *dst_ptr = (unsigned char *)[dst rawBytes] + dstOff;
    const unsigned char *src_ptr = (const unsigned char *)[src rawBytes] + srcOff;
    memcpy(dst_ptr, src_ptr, (size_t)size);
    /* Private -> Shared blit is D2H; inject 1-bit flip when requested. */
    if ([src isPrivate] && ![dst isPrivate] &&
        shim_fault_is("bitflip") && shim_fault_target_matches("metal", _devIdx)) {
        dst_ptr[0] ^= 0x01;
    }
}
- (void)endEncoding {
    if (!_ended) {
        _ended = YES;
        atomic_fetch_sub(&g_unended_encoders, 1);
    }
}
- (void)dealloc {
    atomic_fetch_sub(&g_active_encoders, 1);
    [super dealloc];
}
@end

@interface FakeMTLComputeCommandEncoder : NSObject {
    int _devIdx;
    BOOL _ended;
}
- (instancetype)initWithDeviceIndex:(int)devIdx;
- (void)setComputePipelineState:(id)pso;
- (void)setBuffer:(id)buf offset:(uint64_t)offset atIndex:(uint64_t)idx;
- (void)setBytes:(const void *)bytes length:(uint64_t)length atIndex:(uint64_t)idx;
- (void)setThreadgroupMemoryLength:(uint64_t)length atIndex:(uint64_t)idx;
- (void)dispatchThreadgroups:(FakeMTLSize)grid threadsPerThreadgroup:(FakeMTLSize)block;
- (void)endEncoding;
@end

@implementation FakeMTLComputeCommandEncoder
- (instancetype)initWithDeviceIndex:(int)devIdx {
    self = [super init];
    if (self) {
        _devIdx = devIdx;
        _ended = NO;
        atomic_fetch_add(&g_active_encoders, 1);
        atomic_fetch_add(&g_unended_encoders, 1);
    }
    return self;
}
- (void)setComputePipelineState:(id)pso { (void)pso; }
- (void)setBuffer:(id)buf offset:(uint64_t)offset atIndex:(uint64_t)idx {
    (void)buf; (void)offset; (void)idx;
}
- (void)setBytes:(const void *)bytes length:(uint64_t)length atIndex:(uint64_t)idx {
    (void)bytes; (void)length; (void)idx;
}
- (void)setThreadgroupMemoryLength:(uint64_t)length atIndex:(uint64_t)idx {
    (void)length; (void)idx;
}
- (void)dispatchThreadgroups:(FakeMTLSize)grid threadsPerThreadgroup:(FakeMTLSize)block {
    if (grid.width > 0 && block.width > 0) {
        atomic_fetch_add(&g_dispatch_calls, 1);
    }
}
- (void)endEncoding {
    if (!_ended) {
        _ended = YES;
        atomic_fetch_sub(&g_unended_encoders, 1);
    }
}
- (void)dealloc {
    atomic_fetch_sub(&g_active_encoders, 1);
    [super dealloc];
}
@end

@interface FakeMTLCommandBuffer : NSObject {
    int _devIdx;
    uint64_t _status;
    NSError *_error;
}
- (instancetype)initWithDeviceIndex:(int)devIdx;
- (id)blitCommandEncoder;
- (id)computeCommandEncoder;
- (void)commit;
- (void)waitUntilCompleted;
- (uint64_t)status;
- (NSError *)error;
@end

@implementation FakeMTLCommandBuffer
- (instancetype)initWithDeviceIndex:(int)devIdx {
    self = [super init];
    if (self) {
        _devIdx = devIdx;
        _status = MTL_COMMAND_BUFFER_STATUS_COMPLETED;
        _error = nil;
        atomic_fetch_add(&g_active_cmd_bufs, 1);
    }
    return self;
}
- (id)blitCommandEncoder {
    return [[[FakeMTLBlitCommandEncoder alloc] initWithDeviceIndex:_devIdx] autorelease];
}
- (id)computeCommandEncoder {
    return [[[FakeMTLComputeCommandEncoder alloc] initWithDeviceIndex:_devIdx] autorelease];
}
- (void)commit {
    if (shim_fault_is("gpu_timeout") && shim_fault_target_matches("metal", _devIdx)) {
        _status = MTL_COMMAND_BUFFER_STATUS_ERROR;
        NSDictionary *info = @{ NSLocalizedDescriptionKey: @"Fake Metal command buffer GPU timeout" };
        _error = [[NSError alloc] initWithDomain:@"MTLCommandBufferErrorDomain"
                                            code:MTL_COMMAND_BUFFER_ERROR_TIMEOUT
                                        userInfo:info];
    }
}
- (void)waitUntilCompleted {}
- (uint64_t)status { return _status; }
- (NSError *)error { return _error; }
- (void)dealloc {
    [_error release];
    atomic_fetch_sub(&g_active_cmd_bufs, 1);
    [super dealloc];
}
@end

@interface FakeMTLCommandQueue : NSObject {
    int _devIdx;
}
- (instancetype)initWithDeviceIndex:(int)devIdx;
- (id)commandBuffer;
@end

@implementation FakeMTLCommandQueue
- (instancetype)initWithDeviceIndex:(int)devIdx {
    self = [super init];
    if (self) {
        _devIdx = devIdx;
        atomic_fetch_add(&g_active_queues, 1);
    }
    return self;
}
- (id)commandBuffer {
    return [[[FakeMTLCommandBuffer alloc] initWithDeviceIndex:_devIdx] autorelease];
}
- (void)dealloc {
    atomic_fetch_sub(&g_active_queues, 1);
    [super dealloc];
}
@end

@interface FakeMTLDevice : NSObject {
    int _devIdx;
    NSString *_name;
    FakeMTLArchitecture *_arch;
}
- (instancetype)initWithIndex:(int)idx;
- (NSString *)name;
- (uint64_t)registryID;
- (FakeMTLArchitecture *)architecture;
- (uint64_t)recommendedMaxWorkingSetSize;
- (uint64_t)currentAllocatedSize;
- (uint64_t)maxBufferLength;
- (BOOL)hasUnifiedMemory;
- (BOOL)isLowPower;
- (BOOL)supportsFamily:(int64_t)family;
- (BOOL)supportsBfloat16;
- (id)newCommandQueue;
- (id)newBufferWithLength:(uint64_t)length options:(uint64_t)options;
- (id)newLibraryWithData:(void *)data error:(NSError **)err;
- (id)newLibraryWithSource:(NSString *)source options:(id)options error:(NSError **)err;
- (id)newComputePipelineStateWithFunction:(id)fn error:(NSError **)err;
@end

@implementation FakeMTLDevice
- (instancetype)initWithIndex:(int)idx {
    self = [super init];
    if (self) {
        _devIdx = idx;
        _name = [[NSString alloc] initWithFormat:@"Fake Apple Metal GPU %d", idx];
        _arch = [[FakeMTLArchitecture alloc] initWithName:@"applegpu_g15g"];
    }
    return self;
}
- (NSString *)name { return _name; }
- (uint64_t)registryID { return 0x1000003E8ULL + (uint64_t)_devIdx; }
- (FakeMTLArchitecture *)architecture { return _arch; }
- (uint64_t)recommendedMaxWorkingSetSize { return (uint64_t)shim_vram_bytes(256); }
- (uint64_t)currentAllocatedSize { return 0ULL; }
- (uint64_t)maxBufferLength { return (uint64_t)shim_vram_bytes(256); }
- (BOOL)hasUnifiedMemory { return YES; }
- (BOOL)isLowPower { return NO; }
- (BOOL)supportsFamily:(int64_t)family {
    return (family == MTL_GPU_FAMILY_APPLE7 || family == MTL_GPU_FAMILY_APPLE9) ? YES : NO;
}
- (BOOL)supportsBfloat16 { return YES; }
- (id)newCommandQueue {
    return [[FakeMTLCommandQueue alloc] initWithDeviceIndex:_devIdx];
}
- (id)newBufferWithLength:(uint64_t)length options:(uint64_t)options {
    return [[FakeMTLBuffer alloc] initWithLength:length options:options];
}
- (id)newLibraryWithData:(void *)data error:(NSError **)err {
    if (!data) {
        if (err) {
            *err = [NSError errorWithDomain:@"MTLLibraryErrorDomain"
                                       code:1
                                   userInfo:@{ NSLocalizedDescriptionKey: @"Empty metallib data" }];
        }
        return nil;
    }
    return [[FakeMTLLibrary alloc] init];
}
- (id)newLibraryWithSource:(NSString *)source options:(id)options error:(NSError **)err {
    (void)options;
    if (!source || [source length] == 0) {
        if (err) {
            *err = [NSError errorWithDomain:@"MTLLibraryErrorDomain"
                                       code:1
                                   userInfo:@{ NSLocalizedDescriptionKey: @"Empty Metal source" }];
        }
        return nil;
    }
    return [[FakeMTLLibrary alloc] init];
}
- (id)newComputePipelineStateWithFunction:(id)fn error:(NSError **)err {
    if (!fn) {
        if (err) {
            *err = [NSError errorWithDomain:@"MTLLibraryErrorDomain"
                                       code:1
                                   userInfo:@{ NSLocalizedDescriptionKey: @"Nil Metal function" }];
        }
        return nil;
    }
    return [[FakeMTLComputePipelineState alloc] init];
}
- (void)dealloc {
    [_name release];
    [_arch release];
    [super dealloc];
}
@end

static FakeMTLDevice *g_singleton_devices[SHIM_MAX_GPUS];

void *MTLCopyAllDevices(void) {
    int n = metal_count();
    NSMutableArray *arr = [[NSMutableArray alloc] initWithCapacity:(NSUInteger)n];
    for (int i = 0; i < n; ++i) {
        if (!g_singleton_devices[i]) {
            g_singleton_devices[i] = [[FakeMTLDevice alloc] initWithIndex:i];
        }
        [arr addObject:g_singleton_devices[i]];
    }
    return (void *)arr;
}

uint64_t fake_metal_dispatch_count(void) {
    return atomic_load(&g_dispatch_calls);
}
