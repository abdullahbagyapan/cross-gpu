# SPDX-License-Identifier: MIT
"""CUDA Driver API (libcuda.so.1) backend plugin via ctypes."""

from __future__ import annotations

import ctypes
from typing import Any, Sequence

from .base import (
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
    format_pci_bdf,
    format_uuid_bytes,
    load_shared_library,
    resolve_host_pointer,
)
from ..models import BackendDeviceInfo, DeviceCapabilities

CUDA_LIB_CANDIDATES: tuple[str, ...] = ("libcuda.so.1", "libcuda.so")
NVIDIA_PCI_VENDOR_ID = 0x10DE

CUDA_SUCCESS = 0
CUDA_HARDWARE_FAULT_CODES: frozenset[int] = frozenset(
    {
        214,  # CUDA_ERROR_ECC_UNCORRECTABLE
        217,  # CUDA_ERROR_NVLINK_UNCORRECTABLE
        700,  # CUDA_ERROR_ILLEGAL_ADDRESS
        702,  # CUDA_ERROR_LAUNCH_TIMEOUT
        709,  # CUDA_ERROR_CONTEXT_IS_DESTROYED
        710,  # CUDA_ERROR_ASSERT
        714,  # CUDA_ERROR_HARDWARE_STACK_ERROR
        715,  # CUDA_ERROR_ILLEGAL_INSTRUCTION
        716,  # CUDA_ERROR_MISALIGNED_ADDRESS
        717,  # CUDA_ERROR_INVALID_ADDRESS_SPACE
        718,  # CUDA_ERROR_INVALID_PC
        719,  # CUDA_ERROR_LAUNCH_FAILED
    }
)

CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


class _CuUuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


class _CudaApi:
    """Explicitly typed ctypes bindings for the CUDA Driver API."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_size_p = ctypes.POINTER(ctypes.c_size_t)
        c_void_pp = ctypes.POINTER(ctypes.c_void_p)
        c_u64_p = ctypes.POINTER(ctypes.c_uint64)

        self.cuInit = bind_c_symbol(lib, "cuInit", ctypes.c_int, [ctypes.c_uint])
        self.cuDriverGetVersion = bind_c_symbol(
            lib, "cuDriverGetVersion", ctypes.c_int, [c_int_p]
        )
        self.cuDeviceGetCount = bind_c_symbol(
            lib, "cuDeviceGetCount", ctypes.c_int, [c_int_p]
        )
        self.cuDeviceGet = bind_c_symbol(
            lib, "cuDeviceGet", ctypes.c_int, [c_int_p, ctypes.c_int]
        )
        self.cuDeviceGetName = bind_c_symbol(
            lib, "cuDeviceGetName", ctypes.c_int, [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        )
        self.cuDeviceTotalMem = bind_optional_c_symbol(
            lib, "cuDeviceTotalMem_v2", ctypes.c_int, [c_size_p, ctypes.c_int]
        ) or bind_c_symbol(
            lib, "cuDeviceTotalMem", ctypes.c_int, [c_size_p, ctypes.c_int]
        )
        self.cuDeviceGetAttribute = bind_c_symbol(
            lib,
            "cuDeviceGetAttribute",
            ctypes.c_int,
            [c_int_p, ctypes.c_int, ctypes.c_int],
        )
        self.cuDeviceGetUuid = bind_optional_c_symbol(
            lib,
            "cuDeviceGetUuid_v2",
            ctypes.c_int,
            [ctypes.POINTER(_CuUuid), ctypes.c_int],
        ) or bind_optional_c_symbol(
            lib,
            "cuDeviceGetUuid",
            ctypes.c_int,
            [ctypes.POINTER(_CuUuid), ctypes.c_int],
        )
        self.cuDeviceGetPCIBusId = bind_optional_c_symbol(
            lib,
            "cuDeviceGetPCIBusId",
            ctypes.c_int,
            [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        )
        self.cuCtxCreate = bind_optional_c_symbol(
            lib, "cuCtxCreate_v2", ctypes.c_int, [c_void_pp, ctypes.c_uint, ctypes.c_int]
        ) or bind_c_symbol(
            lib, "cuCtxCreate", ctypes.c_int, [c_void_pp, ctypes.c_uint, ctypes.c_int]
        )
        self.cuCtxDestroy = bind_optional_c_symbol(
            lib, "cuCtxDestroy_v2", ctypes.c_int, [ctypes.c_void_p]
        ) or bind_c_symbol(lib, "cuCtxDestroy", ctypes.c_int, [ctypes.c_void_p])
        self.cuCtxSetCurrent = bind_c_symbol(
            lib, "cuCtxSetCurrent", ctypes.c_int, [ctypes.c_void_p]
        )
        self.cuCtxSynchronize = bind_c_symbol(
            lib, "cuCtxSynchronize", ctypes.c_int, []
        )
        self.cuMemGetInfo = bind_optional_c_symbol(
            lib, "cuMemGetInfo_v2", ctypes.c_int, [c_size_p, c_size_p]
        ) or bind_optional_c_symbol(
            lib, "cuMemGetInfo", ctypes.c_int, [c_size_p, c_size_p]
        )
        self.cuMemAlloc = bind_optional_c_symbol(
            lib, "cuMemAlloc_v2", ctypes.c_int, [c_u64_p, ctypes.c_size_t]
        ) or bind_c_symbol(
            lib, "cuMemAlloc", ctypes.c_int, [c_u64_p, ctypes.c_size_t]
        )
        self.cuMemFree = bind_optional_c_symbol(
            lib, "cuMemFree_v2", ctypes.c_int, [ctypes.c_uint64]
        ) or bind_c_symbol(lib, "cuMemFree", ctypes.c_int, [ctypes.c_uint64])
        self.cuMemAllocHost = bind_optional_c_symbol(
            lib, "cuMemAllocHost_v2", ctypes.c_int, [c_void_pp, ctypes.c_size_t]
        ) or bind_c_symbol(
            lib, "cuMemAllocHost", ctypes.c_int, [c_void_pp, ctypes.c_size_t]
        )
        self.cuMemFreeHost = bind_c_symbol(
            lib, "cuMemFreeHost", ctypes.c_int, [ctypes.c_void_p]
        )
        self.cuMemcpyHtoD = bind_optional_c_symbol(
            lib,
            "cuMemcpyHtoD_v2",
            ctypes.c_int,
            [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
        ) or bind_c_symbol(
            lib,
            "cuMemcpyHtoD",
            ctypes.c_int,
            [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
        )
        self.cuMemcpyDtoH = bind_optional_c_symbol(
            lib,
            "cuMemcpyDtoH_v2",
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
        ) or bind_c_symbol(
            lib,
            "cuMemcpyDtoH",
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
        )
        self.cuMemcpyDtoD = bind_optional_c_symbol(
            lib,
            "cuMemcpyDtoD_v2",
            ctypes.c_int,
            [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t],
        ) or bind_c_symbol(
            lib,
            "cuMemcpyDtoD",
            ctypes.c_int,
            [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t],
        )
        self.cuModuleLoadData = bind_c_symbol(
            lib, "cuModuleLoadData", ctypes.c_int, [c_void_pp, ctypes.c_void_p]
        )
        self.cuModuleUnload = bind_c_symbol(
            lib, "cuModuleUnload", ctypes.c_int, [ctypes.c_void_p]
        )
        self.cuModuleGetFunction = bind_c_symbol(
            lib,
            "cuModuleGetFunction",
            ctypes.c_int,
            [c_void_pp, ctypes.c_void_p, ctypes.c_char_p],
        )
        self.cuLaunchKernel = bind_c_symbol(
            lib,
            "cuLaunchKernel",
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_void_p,
                c_void_pp,
                c_void_pp,
            ],
        )
        self.cuGetErrorString = bind_optional_c_symbol(
            lib,
            "cuGetErrorString",
            ctypes.c_int,
            [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
        )


def _cuda_capabilities(major: int, minor: int) -> DeviceCapabilities:
    cc = (major, minor)
    return DeviceCapabilities(
        fp64=cc >= (1, 3),
        fp32=True,
        fp16=cc >= (5, 3),
        bf16=cc >= (8, 0),
        fp8=cc >= (8, 9),
        fp4=cc >= (10, 0),
        int64=True,
        int32=True,
        int16=True,
        int8=cc >= (6, 1),
        int4=cc >= (7, 5),
        tensor_fp32=cc >= (8, 0),
        tensor_fp16=cc >= (7, 0),
        tensor_bf16=cc >= (8, 0),
        tensor_int8=cc >= (7, 2),
    )


class CudaBackend(BackendPlugin):
    """Native CUDA Driver API backend plugin."""

    name = "cuda"
    priority = NATIVE_BACKEND_PRIORITY

    def __init__(self) -> None:
        super().__init__()
        self._api: _CudaApi | None = None
        self._initialized = False

    def is_available(self) -> bool:
        if self._api is not None and self._initialized:
            return True
        self._probe_log.clear()
        try:
            lib = load_shared_library(
                CUDA_LIB_CANDIDATES,
                diag_fn=self._record_probe,
                label=self.name,
            )
            api = _CudaApi(lib)
            rc = api.cuInit(0)
            if rc != CUDA_SUCCESS:
                err_text = self._error_message(api, rc)
                self._load_error = f"cuInit failed: {err_text} (code={rc})"
                self._record_probe(f"probe cuda: {self._load_error}")
                return False
            self._api = api
            self._initialized = True
            self._load_error = None
            return True
        except OSError:
            self._load_error = f"{CUDA_LIB_CANDIDATES[0]} not found"
            return False
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def _require_api(self) -> _CudaApi:
        if not self.is_available() or self._api is None:
            raise BackendInitError(self._load_error or "CUDA driver unavailable")
        return self._api

    def _error_message(self, api: _CudaApi, rc: int) -> str:
        if api.cuGetErrorString is not None:
            msg_ptr = ctypes.c_char_p()
            if api.cuGetErrorString(rc, ctypes.byref(msg_ptr)) == CUDA_SUCCESS and msg_ptr.value:
                return msg_ptr.value.decode("utf-8", errors="replace")
        return f"CUresult({rc})"

    def _check(self, api: _CudaApi, rc: int, action: str) -> None:
        if rc == CUDA_SUCCESS:
            return
        detail = f"{action} failed: {self._error_message(api, rc)} (code={rc})"
        if rc in CUDA_HARDWARE_FAULT_CODES:
            raise BackendHardwareError(detail)
        raise BackendInitError(detail)

    def _get_attr(self, api: _CudaApi, dev: int, attr: int) -> int | None:
        val = ctypes.c_int(0)
        rc = api.cuDeviceGetAttribute(ctypes.byref(val), attr, dev)
        return val.value if rc == CUDA_SUCCESS else None

    def enumerate_devices(self) -> list[BackendDeviceInfo]:
        api = self._require_api()
        count = ctypes.c_int(0)
        self._check(api, api.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        if count.value == 0:
            self._load_error = "no CUDA devices detected"
            return []

        drv_ver_int = ctypes.c_int(0)
        driver_version = ""
        if api.cuDriverGetVersion(ctypes.byref(drv_ver_int)) == CUDA_SUCCESS:
            v = drv_ver_int.value
            driver_version = f"{v // 1000}.{(v % 1000) // 10}"

        devices: list[BackendDeviceInfo] = []
        for idx in range(count.value):
            dev = ctypes.c_int(0)
            self._check(api, api.cuDeviceGet(ctypes.byref(dev), idx), f"cuDeviceGet({idx})")
            dev_handle = dev.value

            name_buf = ctypes.create_string_buffer(256)
            self._check(
                api,
                api.cuDeviceGetName(name_buf, len(name_buf), dev_handle),
                "cuDeviceGetName",
            )
            name = name_buf.value.decode("utf-8", errors="replace").strip()

            total_mem = ctypes.c_size_t(0)
            self._check(
                api,
                api.cuDeviceTotalMem(ctypes.byref(total_mem), dev_handle),
                "cuDeviceTotalMem",
            )

            major = self._get_attr(api, dev_handle, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) or 0
            minor = self._get_attr(api, dev_handle, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) or 0
            mp_count = self._get_attr(api, dev_handle, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) or 0
            warp_size = self._get_attr(api, dev_handle, CU_DEVICE_ATTRIBUTE_WARP_SIZE)
            max_threads = self._get_attr(api, dev_handle, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

            pci_bdf: str | None = None
            if api.cuDeviceGetPCIBusId is not None:
                bdf_buf = ctypes.create_string_buffer(32)
                if api.cuDeviceGetPCIBusId(bdf_buf, len(bdf_buf), dev_handle) == CUDA_SUCCESS:
                    pci_bdf = bdf_buf.value.decode("ascii", errors="ignore").strip().lower() or None
            if pci_bdf is None:
                domain = self._get_attr(api, dev_handle, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
                bus = self._get_attr(api, dev_handle, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
                pci_dev = self._get_attr(api, dev_handle, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
                if domain is not None and bus is not None and pci_dev is not None:
                    pci_bdf = format_pci_bdf(domain, bus, pci_dev, 0)

            uuid_str: str | None = None
            if api.cuDeviceGetUuid is not None:
                raw_uuid = _CuUuid()
                if api.cuDeviceGetUuid(ctypes.byref(raw_uuid), dev_handle) == CUDA_SUCCESS:
                    uuid_str = format_uuid_bytes(bytes(raw_uuid.bytes))

            extra: dict[str, Any] = {}
            if warp_size is not None:
                extra["warp_size"] = warp_size
            if max_threads is not None:
                extra["max_threads_per_block"] = max_threads

            devices.append(
                BackendDeviceInfo(
                    backend=self.name,
                    priority=self.priority,
                    backend_device_index=idx,
                    native_handle=dev_handle,
                    name=name,
                    vendor="NVIDIA",
                    vendor_id=NVIDIA_PCI_VENDOR_ID,
                    architecture=f"sm_{major}{minor}" if major else "",
                    compute_capability=f"{major}.{minor}" if major else "",
                    total_memory_bytes=int(total_mem.value),
                    free_memory_bytes=None,
                    compute_units=mp_count,
                    driver_version=driver_version,
                    pci_bdf=pci_bdf,
                    uuid=uuid_str,
                    capabilities=_cuda_capabilities(major, minor),
                    extra=extra,
                )
            )
        return devices

    def _ensure_current(self, api: _CudaApi, ctx: BackendContext) -> None:
        self._check(api, api.cuCtxSetCurrent(ctx.handle), "cuCtxSetCurrent")

    def create_context(self, device: BackendDeviceInfo) -> BackendContext:
        api = self._require_api()
        ctx_ptr = ctypes.c_void_p()
        self._check(
            api,
            api.cuCtxCreate(ctypes.byref(ctx_ptr), 0, int(device.native_handle)),
            "cuCtxCreate",
        )
        if api.cuMemGetInfo is not None:
            free_b = ctypes.c_size_t(0)
            total_b = ctypes.c_size_t(0)
            if api.cuMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b)) == CUDA_SUCCESS:
                device.free_memory_bytes = int(free_b.value)
        return BackendContext(backend=self.name, device_info=device, handle=ctx_ptr.value)

    def destroy_context(self, ctx: BackendContext) -> None:
        if ctx.handle is None or self._api is None:
            return
        self._api.cuCtxDestroy(ctx.handle)
        ctx.handle = None

    def synchronize(self, ctx: BackendContext) -> None:
        api = self._require_api()
        self._ensure_current(api, ctx)
        self._check(api, api.cuCtxSynchronize(), "cuCtxSynchronize")

    def alloc_device(self, ctx: BackendContext, size_bytes: int) -> DeviceBuffer:
        api = self._require_api()
        self._ensure_current(api, ctx)
        dev_ptr = ctypes.c_uint64(0)
        self._check(
            api,
            api.cuMemAlloc(ctypes.byref(dev_ptr), max(1, size_bytes)),
            "cuMemAlloc",
        )
        return DeviceBuffer(
            backend=self.name,
            handle=dev_ptr.value,
            size_bytes=size_bytes,
            address=dev_ptr.value,
        )

    def free_device(self, ctx: BackendContext, buf: DeviceBuffer) -> None:
        if not buf.handle or self._api is None:
            return
        self._ensure_current(self._api, ctx)
        self._api.cuMemFree(int(buf.handle))
        buf.handle = 0
        buf.address = 0

    def alloc_pinned_host(self, ctx: BackendContext, size_bytes: int) -> HostBuffer:
        api = self._require_api()
        self._ensure_current(api, ctx)
        host_ptr = ctypes.c_void_p()
        self._check(
            api,
            api.cuMemAllocHost(ctypes.byref(host_ptr), max(1, size_bytes)),
            "cuMemAllocHost",
        )
        addr = int(host_ptr.value or 0)
        return HostBuffer(backend=self.name, handle=addr, size_bytes=size_bytes, pointer=addr)

    def free_pinned_host(self, ctx: BackendContext, buf: HostBuffer) -> None:
        if not buf.handle or self._api is None:
            return
        self._ensure_current(self._api, ctx)
        self._api.cuMemFreeHost(ctypes.c_void_p(buf.handle))
        buf.handle = 0
        buf.pointer = 0

    def memcpy_h2d(
        self,
        ctx: BackendContext,
        dst: DeviceBuffer,
        src: HostBuffer | ctypes.c_void_p | bytes | bytearray | memoryview | int,
        size_bytes: int,
    ) -> None:
        api = self._require_api()
        self._ensure_current(api, ctx)
        src_ptr, keepalive = resolve_host_pointer(src)
        self._check(
            api,
            api.cuMemcpyHtoD(int(dst.handle), src_ptr, size_bytes),
            "cuMemcpyHtoD",
        )
        del keepalive

    def memcpy_d2h(
        self,
        ctx: BackendContext,
        dst: HostBuffer | ctypes.c_void_p | bytearray | memoryview | int,
        src: DeviceBuffer,
        size_bytes: int,
    ) -> None:
        api = self._require_api()
        self._ensure_current(api, ctx)
        dst_ptr, keepalive = resolve_host_pointer(dst)
        self._check(
            api,
            api.cuMemcpyDtoH(dst_ptr, int(src.handle), size_bytes),
            "cuMemcpyDtoH",
        )
        del keepalive

    def memcpy_d2d(
        self,
        ctx: BackendContext,
        dst: DeviceBuffer,
        src: DeviceBuffer,
        size_bytes: int,
    ) -> None:
        api = self._require_api()
        self._ensure_current(api, ctx)
        self._check(
            api,
            api.cuMemcpyDtoD(int(dst.handle), int(src.handle), size_bytes),
            "cuMemcpyDtoD",
        )

    def load_module(self, ctx: BackendContext, compiled_binary: Any) -> KernelModule:
        api = self._require_api()
        self._ensure_current(api, ctx)
        raw = bytes(compiled_binary.binary_bytes)
        fmt = getattr(compiled_binary, "format", "ptx")
        if fmt == "ptx" and not raw.endswith(b"\x00"):
            raw += b"\x00"
        image_buf = ctypes.create_string_buffer(raw, len(raw))
        mod_ptr = ctypes.c_void_p()
        self._check(
            api,
            api.cuModuleLoadData(
                ctypes.byref(mod_ptr),
                ctypes.cast(image_buf, ctypes.c_void_p),
            ),
            "cuModuleLoadData",
        )
        return KernelModule(backend=self.name, handle=mod_ptr.value)

    def unload_module(self, ctx: BackendContext, module: KernelModule) -> None:
        if module.handle is None or self._api is None:
            return
        self._ensure_current(self._api, ctx)
        self._api.cuModuleUnload(module.handle)
        module.handle = None
        module.functions.clear()

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
        self._ensure_current(api, ctx)

        func_handle = module.functions.get(entry_point)
        if func_handle is None:
            fn_ptr = ctypes.c_void_p()
            self._check(
                api,
                api.cuModuleGetFunction(
                    ctypes.byref(fn_ptr),
                    module.handle,
                    entry_point.encode("utf-8"),
                ),
                f"cuModuleGetFunction({entry_point})",
            )
            func_handle = fn_ptr.value
            module.functions[entry_point] = func_handle

        packed_storage: list[Any] = []
        param_ptrs = (ctypes.c_void_p * max(1, len(args)))()
        for index, raw_arg in enumerate(args):
            coerced = coerce_kernel_arg(raw_arg)
            if isinstance(coerced, DeviceBuffer):
                c_val = ctypes.c_uint64(int(coerced.handle))
                packed_storage.append(c_val)
                param_ptrs[index] = ctypes.cast(ctypes.byref(c_val), ctypes.c_void_p)
            else:
                packed_storage.append(coerced)
                param_ptrs[index] = ctypes.cast(ctypes.byref(coerced), ctypes.c_void_p)

        params_arg = param_ptrs if args else None
        gx, gy, gz = grid
        bx, by, bz = block
        self._check(
            api,
            api.cuLaunchKernel(
                func_handle,
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
                shared_mem,
                None,
                params_arg,
                None,
            ),
            f"cuLaunchKernel({entry_point})",
        )
