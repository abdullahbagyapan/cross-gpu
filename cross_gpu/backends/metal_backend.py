"""Apple Metal (libobjc + Metal.framework) backend plugin via ctypes."""

from __future__ import annotations

import ctypes
import platform
import re
import subprocess
import sys
from typing import Any, Sequence

from cross_gpu.backends.base import (
    NATIVE_BACKEND_PRIORITY,
    BackendContext,
    BackendHardwareError,
    BackendInitError,
    BackendPlugin,
    DeviceBuffer,
    HostBuffer,
    KernelModule,
    bind_c_symbol,
    bind_optional_c_symbol,
    coerce_kernel_arg,
    resolve_host_pointer,
)
from cross_gpu.exit_codes import KernelCompilationError
from cross_gpu.models import BackendDeviceInfo, DeviceCapabilities

LIBOBJC_PATH = "/usr/lib/libobjc.A.dylib"
LIBSYSTEM_PATH = "/usr/lib/libSystem.B.dylib"
METAL_FRAMEWORK_PATH = "/System/Library/Frameworks/Metal.framework/Metal"

APPLE_PCI_VENDOR_ID = 0x106B
AMD_PCI_VENDOR_ID = 0x1002
INTEL_PCI_VENDOR_ID = 0x8086

MTL_RESOURCE_STORAGE_MODE_SHARED = 0x0
MTL_RESOURCE_STORAGE_MODE_PRIVATE = 0x20

MTL_COMMAND_BUFFER_STATUS_ERROR = 5
MTL_HARDWARE_ERROR_CODES: frozenset[int] = frozenset(
    {
        2,   # MTLCommandBufferErrorTimeout
        4,   # MTLCommandBufferErrorBlacklisted
        5,   # MTLCommandBufferErrorNotPermitted
        6,   # MTLCommandBufferErrorOutOfMemory
        7,   # MTLCommandBufferErrorPageFault
        8,   # MTLCommandBufferErrorAccessRevoked
        10,  # MTLCommandBufferErrorStackOverflow
    }
)

MTL_GPU_FAMILY_APPLE7 = 1007
MTL_GPU_FAMILY_APPLE9 = 1009


class _MTLSize(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint64),
        ("height", ctypes.c_uint64),
        ("depth", ctypes.c_uint64),
    ]


def _bind_objc_msg(libobjc: ctypes.CDLL, restype: Any, *argtypes: Any) -> Any:
    """Create an ABI-safe Objective-C objc_msgSend wrapper with explicit argtypes/restype."""
    full_args = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    fn = ctypes.CFUNCTYPE(restype, *full_args)(("objc_msgSend", libobjc))
    fn.argtypes = full_args
    fn.restype = restype
    return fn


class _MetalApi:
    """Explicitly typed ctypes bindings for libobjc, libSystem, and Metal.framework."""

    def __init__(self, libobjc: ctypes.CDLL, libmetal: ctypes.CDLL, libsys: ctypes.CDLL) -> None:
        self.libobjc = libobjc
        self.libmetal = libmetal
        self.libsys = libsys

        self.objc_getClass = bind_c_symbol(
            libobjc, "objc_getClass", ctypes.c_void_p, [ctypes.c_char_p]
        )
        self.sel_registerName = bind_c_symbol(
            libobjc, "sel_registerName", ctypes.c_void_p, [ctypes.c_char_p]
        )
        self.objc_autoreleasePoolPush = bind_optional_c_symbol(
            libobjc, "objc_autoreleasePoolPush", ctypes.c_void_p, []
        )
        self.objc_autoreleasePoolPop = bind_optional_c_symbol(
            libobjc, "objc_autoreleasePoolPop", None, [ctypes.c_void_p]
        )
        self.objc_retain = bind_optional_c_symbol(
            libobjc, "objc_retain", ctypes.c_void_p, [ctypes.c_void_p]
        )
        self.objc_release = bind_optional_c_symbol(
            libobjc, "objc_release", None, [ctypes.c_void_p]
        )

        self.MTLCopyAllDevices = bind_c_symbol(
            libmetal, "MTLCopyAllDevices", ctypes.c_void_p, []
        )
        self.dispatch_data_create = bind_c_symbol(
            libsys,
            "dispatch_data_create",
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.dispatch_release = bind_c_symbol(
            libsys, "dispatch_release", None, [ctypes.c_void_p]
        )

        self.msg_ptr = _bind_objc_msg(libobjc, ctypes.c_void_p)
        self.msg_ptr_u64 = _bind_objc_msg(libobjc, ctypes.c_void_p, ctypes.c_uint64)
        self.msg_ptr_u64_u64 = _bind_objc_msg(
            libobjc, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64
        )
        self.msg_ptr_cstr = _bind_objc_msg(libobjc, ctypes.c_void_p, ctypes.c_char_p)
        self.msg_ptr_ptr = _bind_objc_msg(libobjc, ctypes.c_void_p, ctypes.c_void_p)
        self.msg_ptr_ptr_ptr = _bind_objc_msg(
            libobjc, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )
        self.msg_ptr_ptr_ptr_ptr = _bind_objc_msg(
            libobjc,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self.msg_u64 = _bind_objc_msg(libobjc, ctypes.c_uint64)
        self.msg_i64 = _bind_objc_msg(libobjc, ctypes.c_int64)
        self.msg_cstr = _bind_objc_msg(libobjc, ctypes.c_char_p)
        self.msg_bool = _bind_objc_msg(libobjc, ctypes.c_bool)
        self.msg_bool_ptr = _bind_objc_msg(libobjc, ctypes.c_bool, ctypes.c_void_p)
        self.msg_bool_i64 = _bind_objc_msg(libobjc, ctypes.c_bool, ctypes.c_int64)
        self.msg_void = _bind_objc_msg(libobjc, None)
        self.msg_void_ptr = _bind_objc_msg(libobjc, None, ctypes.c_void_p)
        self.msg_void_ptr_u64_u64 = _bind_objc_msg(
            libobjc, None, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64
        )
        self.msg_void_size_size = _bind_objc_msg(libobjc, None, _MTLSize, _MTLSize)
        self.msg_void_blit = _bind_objc_msg(
            libobjc,
            None,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
        )
        self.msg_void_u64_u64 = _bind_objc_msg(
            libobjc, None, ctypes.c_uint64, ctypes.c_uint64
        )

        self.cls_NSString = self.objc_getClass(b"NSString")
        self.sel_count = self.sel_registerName(b"count")
        self.sel_objectAtIndex = self.sel_registerName(b"objectAtIndex:")
        self.sel_UTF8String = self.sel_registerName(b"UTF8String")
        self.sel_stringWithUTF8String = self.sel_registerName(b"stringWithUTF8String:")
        self.sel_respondsToSelector = self.sel_registerName(b"respondsToSelector:")
        self.sel_name = self.sel_registerName(b"name")
        self.sel_registryID = self.sel_registerName(b"registryID")
        self.sel_architecture = self.sel_registerName(b"architecture")
        self.sel_recommendedMaxWorkingSetSize = self.sel_registerName(
            b"recommendedMaxWorkingSetSize"
        )
        self.sel_currentAllocatedSize = self.sel_registerName(b"currentAllocatedSize")
        self.sel_maxBufferLength = self.sel_registerName(b"maxBufferLength")
        self.sel_hasUnifiedMemory = self.sel_registerName(b"hasUnifiedMemory")
        self.sel_isLowPower = self.sel_registerName(b"isLowPower")
        self.sel_supportsFamily = self.sel_registerName(b"supportsFamily:")
        self.sel_supportsBfloat16 = self.sel_registerName(b"supportsBfloat16")
        self.sel_newCommandQueue = self.sel_registerName(b"newCommandQueue")
        self.sel_newBufferWithLength = self.sel_registerName(
            b"newBufferWithLength:options:"
        )
        self.sel_contents = self.sel_registerName(b"contents")
        self.sel_newLibraryWithData = self.sel_registerName(b"newLibraryWithData:error:")
        self.sel_newLibraryWithSource = self.sel_registerName(
            b"newLibraryWithSource:options:error:"
        )
        self.sel_newFunctionWithName = self.sel_registerName(b"newFunctionWithName:")
        self.sel_newComputePipelineStateWithFunction = self.sel_registerName(
            b"newComputePipelineStateWithFunction:error:"
        )
        self.sel_commandBuffer = self.sel_registerName(b"commandBuffer")
        self.sel_computeCommandEncoder = self.sel_registerName(b"computeCommandEncoder")
        self.sel_blitCommandEncoder = self.sel_registerName(b"blitCommandEncoder")
        self.sel_copyFromBuffer = self.sel_registerName(
            b"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"
        )
        self.sel_setComputePipelineState = self.sel_registerName(
            b"setComputePipelineState:"
        )
        self.sel_setBuffer = self.sel_registerName(b"setBuffer:offset:atIndex:")
        self.sel_setBytes = self.sel_registerName(b"setBytes:length:atIndex:")
        self.sel_setThreadgroupMemoryLength = self.sel_registerName(
            b"setThreadgroupMemoryLength:atIndex:"
        )
        self.sel_dispatchThreadgroups = self.sel_registerName(
            b"dispatchThreadgroups:threadsPerThreadgroup:"
        )
        self.sel_endEncoding = self.sel_registerName(b"endEncoding")
        self.sel_commit = self.sel_registerName(b"commit")
        self.sel_waitUntilCompleted = self.sel_registerName(b"waitUntilCompleted")
        self.sel_status = self.sel_registerName(b"status")
        self.sel_error = self.sel_registerName(b"error")
        self.sel_code = self.sel_registerName(b"code")
        self.sel_localizedDescription = self.sel_registerName(b"localizedDescription")
        self.sel_release = self.sel_registerName(b"release")


def _probe_ioreg_gpu_metadata() -> dict[int, dict[str, Any]]:
    """Best-effort query of IOKit IOAccelerator metadata indexed by registryID."""
    result: dict[int, dict[str, Any]] = {}
    try:
        proc = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return result
    if proc.returncode != 0 or not proc.stdout:
        return result

    blocks = re.split(r"(?=^\+-o\s+)", proc.stdout, flags=re.MULTILINE)
    for block in blocks:
        id_match = re.search(r"\bid\s+(0x[0-9a-fA-F]+|\d+)", block)
        if not id_match:
            continue
        raw_id = id_match.group(1)
        reg_id = int(raw_id, 16) if raw_id.startswith("0x") else int(raw_id)
        meta: dict[str, Any] = {}

        cores_match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', block)
        if cores_match:
            meta["gpu_core_count"] = int(cores_match.group(1))

        src_ver_match = re.search(r'"IOSourceVersion"\s*=\s*"([^"]+)"', block)
        if src_ver_match:
            meta["driver_version"] = src_ver_match.group(1)

        vendor_match = re.search(r'"vendor-id"\s*=\s*<([0-9a-fA-F]+)>', block)
        if vendor_match:
            hex_bytes = bytes.fromhex(vendor_match.group(1))
            if len(hex_bytes) >= 2:
                meta["vendor_id"] = int.from_bytes(hex_bytes[:4], byteorder="little")

        result[reg_id] = meta
    return result


def _infer_vendor(name: str, ioreg_vendor_id: int | None) -> tuple[str, int]:
    if ioreg_vendor_id == APPLE_PCI_VENDOR_ID or "apple" in name.lower():
        return "Apple", APPLE_PCI_VENDOR_ID
    if ioreg_vendor_id == AMD_PCI_VENDOR_ID or "amd" in name.lower() or "radeon" in name.lower():
        return "AMD", AMD_PCI_VENDOR_ID
    if ioreg_vendor_id == INTEL_PCI_VENDOR_ID or "intel" in name.lower():
        return "Intel", INTEL_PCI_VENDOR_ID
    if ioreg_vendor_id:
        return f"0x{ioreg_vendor_id:04x}", ioreg_vendor_id
    return "Apple", APPLE_PCI_VENDOR_ID


class MetalBackend(BackendPlugin):
    """Native Apple Metal backend plugin via Objective-C runtime ctypes bridge."""

    name = "metal"
    priority = NATIVE_BACKEND_PRIORITY

    def __init__(self) -> None:
        super().__init__()
        self._api: _MetalApi | None = None

    def is_available(self) -> bool:
        if sys.platform != "darwin":
            self._load_error = "Metal backend is only available on macOS"
            return False
        if self._api is not None:
            return True
        try:
            libobjc = ctypes.CDLL(LIBOBJC_PATH)
            libmetal = ctypes.CDLL(METAL_FRAMEWORK_PATH)
            libsys = ctypes.CDLL(LIBSYSTEM_PATH)
            self._api = _MetalApi(libobjc, libmetal, libsys)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def _require_api(self) -> _MetalApi:
        if not self.is_available() or self._api is None:
            raise BackendInitError(self._load_error or "Metal runtime unavailable")
        return self._api

    def _nsstring(self, api: _MetalApi, text: str) -> ctypes.c_void_p:
        ptr = api.msg_ptr_cstr(
            api.cls_NSString, api.sel_stringWithUTF8String, text.encode("utf-8")
        )
        if not ptr:
            raise BackendInitError("Failed to create NSString")
        return ptr

    def _from_nsstring(self, api: _MetalApi, ns_obj: ctypes.c_void_p | None) -> str:
        if not ns_obj:
            return ""
        raw = api.msg_cstr(ns_obj, api.sel_UTF8String)
        return raw.decode("utf-8", errors="replace") if raw else ""

    def _nserror_message(self, api: _MetalApi, err_ptr: ctypes.c_void_p) -> tuple[int, str]:
        if not err_ptr:
            return 0, "Unknown Metal error"
        code = int(api.msg_i64(err_ptr, api.sel_code))
        desc_ns = api.msg_ptr(err_ptr, api.sel_localizedDescription)
        return code, self._from_nsstring(api, desc_ns)

    def _release_objc(self, api: _MetalApi, obj: Any) -> None:
        if not obj:
            return
        if api.objc_release is not None:
            api.objc_release(obj)
        else:
            api.msg_void(obj, api.sel_release)

    def _retain_objc(self, api: _MetalApi, obj: Any) -> Any:
        if not obj:
            return obj
        if api.objc_retain is not None:
            return api.objc_retain(obj)
        return obj

    def enumerate_devices(self) -> list[BackendDeviceInfo]:
        api = self._require_api()
        pool = api.objc_autoreleasePoolPush() if api.objc_autoreleasePoolPush else None
        try:
            dev_arr = api.MTLCopyAllDevices()
            if not dev_arr:
                return []
            try:
                count = int(api.msg_u64(dev_arr, api.sel_count))
                ioreg_meta = _probe_ioreg_gpu_metadata()
                os_ver = platform.mac_ver()[0]
                devices: list[BackendDeviceInfo] = []

                for idx in range(count):
                    dev = api.msg_ptr_u64(dev_arr, api.sel_objectAtIndex, idx)
                    if not dev:
                        continue
                    self._retain_objc(api, dev)

                    name = self._from_nsstring(api, api.msg_ptr(dev, api.sel_name)).strip()
                    reg_id = int(api.msg_u64(dev, api.sel_registryID))
                    max_ws = int(api.msg_u64(dev, api.sel_recommendedMaxWorkingSetSize))
                    cur_alloc = 0
                    if api.msg_bool_ptr(dev, api.sel_respondsToSelector, api.sel_currentAllocatedSize):
                        cur_alloc = int(api.msg_u64(dev, api.sel_currentAllocatedSize))

                    arch_name = ""
                    if api.msg_bool_ptr(dev, api.sel_respondsToSelector, api.sel_architecture):
                        arch_obj = api.msg_ptr(dev, api.sel_architecture)
                        if arch_obj:
                            arch_name = self._from_nsstring(
                                api, api.msg_ptr(arch_obj, api.sel_name)
                            )

                    supports_apple7 = bool(
                        api.msg_bool_ptr(dev, api.sel_respondsToSelector, api.sel_supportsFamily)
                        and api.msg_bool_i64(dev, api.sel_supportsFamily, MTL_GPU_FAMILY_APPLE7)
                    )
                    supports_apple9 = bool(
                        api.msg_bool_ptr(dev, api.sel_respondsToSelector, api.sel_supportsFamily)
                        and api.msg_bool_i64(dev, api.sel_supportsFamily, MTL_GPU_FAMILY_APPLE9)
                    )
                    supports_bf16 = supports_apple9
                    if api.msg_bool_ptr(dev, api.sel_respondsToSelector, api.sel_supportsBfloat16):
                        supports_bf16 = bool(api.msg_bool(dev, api.sel_supportsBfloat16))

                    has_unified = bool(
                        api.msg_bool_ptr(dev, api.sel_respondsToSelector, api.sel_hasUnifiedMemory)
                        and api.msg_bool(dev, api.sel_hasUnifiedMemory)
                    )
                    is_low_power = bool(
                        api.msg_bool_ptr(dev, api.sel_respondsToSelector, api.sel_isLowPower)
                        and api.msg_bool(dev, api.sel_isLowPower)
                    )
                    max_buf = int(api.msg_u64(dev, api.sel_maxBufferLength))

                    io_info = ioreg_meta.get(reg_id, {})
                    vendor, vendor_id = _infer_vendor(name, io_info.get("vendor_id"))
                    compute_units = int(io_info.get("gpu_core_count", 0))
                    driver_version = str(io_info.get("driver_version") or os_ver)

                    caps = DeviceCapabilities(
                        fp64=False,
                        fp32=True,
                        fp16=True,
                        bf16=supports_bf16,
                        fp8=False,
                        fp4=False,
                        int64=True,
                        int32=True,
                        int16=True,
                        int8=True,
                        int4=False,
                        tensor_fp32=supports_apple7,
                        tensor_fp16=supports_apple7,
                        tensor_bf16=supports_bf16,
                        tensor_int8=False,
                    )

                    devices.append(
                        BackendDeviceInfo(
                            backend=self.name,
                            priority=self.priority,
                            backend_device_index=idx,
                            native_handle=dev,
                            name=name,
                            vendor=vendor,
                            vendor_id=vendor_id,
                            architecture=arch_name or "metal",
                            compute_capability="metal-3" if supports_apple7 else "metal",
                            total_memory_bytes=max_ws,
                            free_memory_bytes=max(0, max_ws - cur_alloc) if max_ws else None,
                            compute_units=compute_units,
                            driver_version=driver_version,
                            registry_id=reg_id,
                            capabilities=caps,
                            extra={
                                "has_unified_memory": has_unified,
                                "is_low_power": is_low_power,
                                "max_buffer_length": max_buf,
                            },
                        )
                    )
                return devices
            finally:
                self._release_objc(api, dev_arr)
        finally:
            if pool is not None and api.objc_autoreleasePoolPop is not None:
                api.objc_autoreleasePoolPop(pool)

    def create_context(self, device: BackendDeviceInfo) -> BackendContext:
        api = self._require_api()
        dev = device.native_handle
        if not dev:
            raise BackendInitError("Invalid Metal device handle")
        queue = api.msg_ptr(dev, api.sel_newCommandQueue)
        if not queue:
            raise BackendInitError("MTLDevice newCommandQueue returned nil")
        return BackendContext(
            backend=self.name,
            device_info=device,
            handle=dev,
            queue=queue,
            extra={"pending_cmd_bufs": []},
        )

    def destroy_context(self, ctx: BackendContext) -> None:
        if self._api is None:
            return
        pending = ctx.extra.get("pending_cmd_bufs", [])
        for cmd_buf in pending:
            self._release_objc(self._api, cmd_buf)
        pending.clear()
        if ctx.queue:
            self._release_objc(self._api, ctx.queue)
            ctx.queue = None
        ctx.handle = None

    def _wait_cmd_buffer(self, api: _MetalApi, cmd_buf: Any, action: str) -> None:
        api.msg_void(cmd_buf, api.sel_waitUntilCompleted)
        status = int(api.msg_u64(cmd_buf, api.sel_status))
        if status == MTL_COMMAND_BUFFER_STATUS_ERROR:
            err_ptr = api.msg_ptr(cmd_buf, api.sel_error)
            code, msg = self._nserror_message(api, err_ptr)
            detail = f"{action} failed: {msg} (code={code})"
            if code in MTL_HARDWARE_ERROR_CODES:
                raise BackendHardwareError(detail)
            raise BackendInitError(detail)

    def synchronize(self, ctx: BackendContext) -> None:
        api = self._require_api()
        pending: list[Any] = ctx.extra.setdefault("pending_cmd_bufs", [])
        while pending:
            cmd_buf = pending.pop(0)
            try:
                self._wait_cmd_buffer(api, cmd_buf, "MTLCommandBuffer")
            finally:
                self._release_objc(api, cmd_buf)

    def alloc_device(self, ctx: BackendContext, size_bytes: int) -> DeviceBuffer:
        api = self._require_api()
        buf = api.msg_ptr_u64_u64(
            ctx.handle,
            api.sel_newBufferWithLength,
            max(1, size_bytes),
            MTL_RESOURCE_STORAGE_MODE_PRIVATE,
        )
        if not buf:
            raise BackendInitError(f"Failed to allocate Metal private buffer ({size_bytes} bytes)")
        return DeviceBuffer(backend=self.name, handle=buf, size_bytes=size_bytes)

    def free_device(self, ctx: BackendContext, buf: DeviceBuffer) -> None:
        if not buf.handle or self._api is None:
            return
        self._release_objc(self._api, buf.handle)
        buf.handle = None

    def alloc_pinned_host(self, ctx: BackendContext, size_bytes: int) -> HostBuffer:
        api = self._require_api()
        buf = api.msg_ptr_u64_u64(
            ctx.handle,
            api.sel_newBufferWithLength,
            max(1, size_bytes),
            MTL_RESOURCE_STORAGE_MODE_SHARED,
        )
        if not buf:
            raise BackendInitError(f"Failed to allocate Metal shared host buffer ({size_bytes} bytes)")
        contents = int(api.msg_ptr(buf, api.sel_contents) or 0)
        return HostBuffer(
            backend=self.name,
            handle=buf,
            size_bytes=size_bytes,
            pointer=contents,
        )

    def free_pinned_host(self, ctx: BackendContext, buf: HostBuffer) -> None:
        if not buf.handle or self._api is None:
            return
        self._release_objc(self._api, buf.handle)
        buf.handle = None
        buf.pointer = 0

    def _blit_copy(
        self,
        api: _MetalApi,
        ctx: BackendContext,
        src_mtl_buf: Any,
        dst_mtl_buf: Any,
        size_bytes: int,
        action: str,
    ) -> None:
        if size_bytes <= 0:
            return
        pool = api.objc_autoreleasePoolPush() if api.objc_autoreleasePoolPush else None
        try:
            cmd_buf = api.msg_ptr(ctx.queue, api.sel_commandBuffer)
            if not cmd_buf:
                raise BackendInitError("Failed to allocate MTLCommandBuffer for blit")
            self._retain_objc(api, cmd_buf)
            try:
                enc = api.msg_ptr(cmd_buf, api.sel_blitCommandEncoder)
                if not enc:
                    raise BackendInitError("Failed to create MTLBlitCommandEncoder")
                api.msg_void_blit(
                    enc,
                    api.sel_copyFromBuffer,
                    src_mtl_buf,
                    0,
                    dst_mtl_buf,
                    0,
                    size_bytes,
                )
                api.msg_void(enc, api.sel_endEncoding)
                api.msg_void(cmd_buf, api.sel_commit)
                self._wait_cmd_buffer(api, cmd_buf, action)
            finally:
                self._release_objc(api, cmd_buf)
        finally:
            if pool is not None and api.objc_autoreleasePoolPop is not None:
                api.objc_autoreleasePoolPop(pool)

    def memcpy_h2d(
        self,
        ctx: BackendContext,
        dst: DeviceBuffer,
        src: HostBuffer | ctypes.c_void_p | bytes | bytearray | memoryview | int,
        size_bytes: int,
    ) -> None:
        api = self._require_api()
        if size_bytes <= 0:
            return
        if isinstance(src, HostBuffer) and src.handle:
            self._blit_copy(api, ctx, src.handle, dst.handle, size_bytes, "Metal H2D blit")
            return
        src_ptr, keepalive = resolve_host_pointer(src)
        staging = self.alloc_pinned_host(ctx, size_bytes)
        try:
            ctypes.memmove(staging.pointer, src_ptr, size_bytes)
            self._blit_copy(api, ctx, staging.handle, dst.handle, size_bytes, "Metal H2D blit")
        finally:
            self.free_pinned_host(ctx, staging)
            del keepalive

    def memcpy_d2h(
        self,
        ctx: BackendContext,
        dst: HostBuffer | ctypes.c_void_p | bytearray | memoryview | int,
        src: DeviceBuffer,
        size_bytes: int,
    ) -> None:
        api = self._require_api()
        if size_bytes <= 0:
            return
        if isinstance(dst, HostBuffer) and dst.handle:
            self._blit_copy(api, ctx, src.handle, dst.handle, size_bytes, "Metal D2H blit")
            return
        dst_ptr, keepalive = resolve_host_pointer(dst)
        staging = self.alloc_pinned_host(ctx, size_bytes)
        try:
            self._blit_copy(api, ctx, src.handle, staging.handle, size_bytes, "Metal D2H blit")
            ctypes.memmove(dst_ptr, staging.pointer, size_bytes)
        finally:
            self.free_pinned_host(ctx, staging)
            del keepalive

    def memcpy_d2d(
        self,
        ctx: BackendContext,
        dst: DeviceBuffer,
        src: DeviceBuffer,
        size_bytes: int,
    ) -> None:
        api = self._require_api()
        self._blit_copy(api, ctx, src.handle, dst.handle, size_bytes, "Metal D2D blit")

    def load_module(self, ctx: BackendContext, compiled_binary: Any) -> KernelModule:
        api = self._require_api()
        pool = api.objc_autoreleasePoolPush() if api.objc_autoreleasePoolPush else None
        try:
            raw = bytes(compiled_binary.binary_bytes)
            fmt = getattr(compiled_binary, "format", "metallib")
            err_ptr = ctypes.c_void_p(0)

            if fmt == "metallib":
                c_buf = ctypes.create_string_buffer(raw, len(raw))
                dispatch_data = api.dispatch_data_create(c_buf, len(raw), None, None)
                try:
                    lib_ptr = api.msg_ptr_ptr_ptr(
                        ctx.handle,
                        api.sel_newLibraryWithData,
                        dispatch_data,
                        ctypes.byref(err_ptr),
                    )
                finally:
                    api.dispatch_release(dispatch_data)
            else:
                src_ns = self._nsstring(api, raw.decode("utf-8", errors="replace"))
                lib_ptr = api.msg_ptr_ptr_ptr_ptr(
                    ctx.handle,
                    api.sel_newLibraryWithSource,
                    src_ns,
                    None,
                    ctypes.byref(err_ptr),
                )

            if not lib_ptr:
                _, msg = self._nserror_message(api, err_ptr.value)
                raise KernelCompilationError(
                    f"Metal library load failed: {msg}",
                    compiler_log=msg,
                )
            return KernelModule(backend=self.name, handle=lib_ptr)
        finally:
            if pool is not None and api.objc_autoreleasePoolPop is not None:
                api.objc_autoreleasePoolPop(pool)

    def unload_module(self, ctx: BackendContext, module: KernelModule) -> None:
        if self._api is None:
            return
        for pso in module.functions.values():
            self._release_objc(self._api, pso)
        module.functions.clear()
        if module.handle:
            self._release_objc(self._api, module.handle)
            module.handle = None

    def launch_kernel(
        self,
        ctx: BackendContext,
        module: KernelModule,
        entry_point: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[Any],
        shared_mem: int = 0,
    ) -> None:
        api = self._require_api()
        pool = api.objc_autoreleasePoolPush() if api.objc_autoreleasePoolPush else None
        try:
            pso = module.functions.get(entry_point)
            if pso is None:
                fn_ns = self._nsstring(api, entry_point)
                fn_ptr = api.msg_ptr_ptr(module.handle, api.sel_newFunctionWithName, fn_ns)
                if not fn_ptr:
                    raise BackendInitError(f"Metal kernel function {entry_point!r} not found")
                try:
                    err_ptr = ctypes.c_void_p(0)
                    pso = api.msg_ptr_ptr_ptr(
                        ctx.handle,
                        api.sel_newComputePipelineStateWithFunction,
                        fn_ptr,
                        ctypes.byref(err_ptr),
                    )
                    if not pso:
                        _, msg = self._nserror_message(api, err_ptr.value)
                        raise BackendInitError(
                            f"newComputePipelineStateWithFunction({entry_point}) failed: {msg}"
                        )
                    module.functions[entry_point] = pso
                finally:
                    self._release_objc(api, fn_ptr)

            cmd_buf = api.msg_ptr(ctx.queue, api.sel_commandBuffer)
            if not cmd_buf:
                raise BackendInitError("Failed to create Metal commandBuffer")
            self._retain_objc(api, cmd_buf)

            enc = api.msg_ptr(cmd_buf, api.sel_computeCommandEncoder)
            if not enc:
                self._release_objc(api, cmd_buf)
                raise BackendInitError("Failed to create MTLComputeCommandEncoder")

            api.msg_void_ptr(enc, api.sel_setComputePipelineState, pso)

            keepalive: list[Any] = []
            for idx, raw_arg in enumerate(args):
                coerced = coerce_kernel_arg(raw_arg)
                if isinstance(coerced, DeviceBuffer):
                    api.msg_void_ptr_u64_u64(enc, api.sel_setBuffer, coerced.handle, 0, idx)
                else:
                    keepalive.append(coerced)
                    api.msg_void_ptr_u64_u64(
                        enc,
                        api.sel_setBytes,
                        ctypes.cast(ctypes.byref(coerced), ctypes.c_void_p),
                        ctypes.sizeof(coerced),
                        idx,
                    )

            if shared_mem > 0:
                api.msg_void_u64_u64(
                    enc, api.sel_setThreadgroupMemoryLength, shared_mem, 0
                )

            gx, gy, gz = grid
            bx, by, bz = block
            api.msg_void_size_size(
                enc,
                api.sel_dispatchThreadgroups,
                _MTLSize(width=gx, height=gy, depth=gz),
                _MTLSize(width=bx, height=by, depth=bz),
            )
            api.msg_void(enc, api.sel_endEncoding)
            api.msg_void(cmd_buf, api.sel_commit)
            ctx.extra.setdefault("pending_cmd_bufs", []).append(cmd_buf)
            del keepalive
        finally:
            if pool is not None and api.objc_autoreleasePoolPop is not None:
                api.objc_autoreleasePoolPop(pool)
