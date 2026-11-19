"""OpenCL C API (libOpenCL.so.1 / OpenCL.framework) backend plugin via ctypes."""

from __future__ import annotations

import ctypes
import sys
from typing import Any, Sequence

from cross_gpu.backends.base import (
    OPENCL_BACKEND_PRIORITY,
    BackendContext,
    BackendHardwareError,
    BackendInitError,
    BackendPlugin,
    DeviceBuffer,
    HostBuffer,
    KernelModule,
    bind_c_symbol,
    bind_optional_c_symbol,
    check_linux_render_node_hint,
    coerce_kernel_arg,
    format_pci_bdf,
    format_uuid_bytes,
    load_shared_library,
    resolve_host_pointer,
)
from cross_gpu.exit_codes import KernelCompilationError
from cross_gpu.models import BackendDeviceInfo, DeviceCapabilities

OPENCL_LIB_CANDIDATES: tuple[str, ...] = (
    ("/System/Library/Frameworks/OpenCL.framework/OpenCL",)
    if sys.platform == "darwin"
    else ("libOpenCL.so.1", "libOpenCL.so")
)

CL_SUCCESS = 0
CL_DEVICE_NOT_FOUND = -1
CL_DEVICE_NOT_AVAILABLE = -2
CL_OUT_OF_RESOURCES = -5
CL_BUILD_PROGRAM_FAILURE = -11
CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14
CL_DEVICE_TYPE_GPU = 1 << 2

CL_HARDWARE_FAULT_CODES: frozenset[int] = frozenset(
    {
        CL_DEVICE_NOT_AVAILABLE,
        CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
    }
)

CL_PLATFORM_NAME = 0x0902

CL_DEVICE_VENDOR_ID = 0x1001
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004
CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C
CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032
CL_DEVICE_HALF_FP_CONFIG = 0x1033
CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035
CL_DEVICE_UUID_KHR = 0x106A

CL_DEVICE_PCI_BUS_ID_NV = 0x4008
CL_DEVICE_PCI_SLOT_ID_NV = 0x4009
CL_DEVICE_PCI_DOMAIN_ID_NV = 0x400A
CL_DEVICE_TOPOLOGY_AMD = 0x4037
CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD = 1
CL_DEVICE_PCI_BUS_INFO_KHR = 0x410F

CL_MEM_READ_WRITE = 1 << 0
CL_MEM_ALLOC_HOST_PTR = 1 << 4
CL_MAP_READ = 1 << 0
CL_MAP_WRITE = 1 << 1
CL_TRUE = 1

CL_PROGRAM_BINARY_SIZES = 0x1165
CL_PROGRAM_BINARIES = 0x1166
CL_PROGRAM_BUILD_LOG = 0x1183


class _ClPciBusInfoKhr(ctypes.Structure):
    _fields_ = [
        ("pci_domain", ctypes.c_uint32),
        ("pci_bus", ctypes.c_uint32),
        ("pci_device", ctypes.c_uint32),
        ("pci_function", ctypes.c_uint32),
    ]


class _ClDeviceTopologyAmdPcie(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("unused", ctypes.c_ubyte * 17),
        ("bus", ctypes.c_ubyte),
        ("device", ctypes.c_ubyte),
        ("function", ctypes.c_ubyte),
    ]


class _OpenCLApi:
    """Explicitly typed ctypes bindings for the OpenCL C API."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib
        c_i32_p = ctypes.POINTER(ctypes.c_int32)
        c_u32_p = ctypes.POINTER(ctypes.c_uint32)
        c_size_p = ctypes.POINTER(ctypes.c_size_t)
        c_void_pp = ctypes.POINTER(ctypes.c_void_p)

        self.clGetPlatformIDs = bind_c_symbol(
            lib, "clGetPlatformIDs", ctypes.c_int32, [ctypes.c_uint32, c_void_pp, c_u32_p]
        )
        self.clGetPlatformInfo = bind_c_symbol(
            lib,
            "clGetPlatformInfo",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p, c_size_p],
        )
        self.clGetDeviceIDs = bind_c_symbol(
            lib,
            "clGetDeviceIDs",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, c_void_pp, c_u32_p],
        )
        self.clGetDeviceInfo = bind_c_symbol(
            lib,
            "clGetDeviceInfo",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p, c_size_p],
        )
        self.clCreateContext = bind_c_symbol(
            lib,
            "clCreateContext",
            ctypes.c_void_p,
            [
                ctypes.c_void_p,
                ctypes.c_uint32,
                c_void_pp,
                ctypes.c_void_p,
                ctypes.c_void_p,
                c_i32_p,
            ],
        )
        self.clReleaseContext = bind_c_symbol(
            lib, "clReleaseContext", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.clCreateCommandQueue = bind_c_symbol(
            lib,
            "clCreateCommandQueue",
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, c_i32_p],
        )
        self.clCreateCommandQueueWithProperties = bind_optional_c_symbol(
            lib,
            "clCreateCommandQueueWithProperties",
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_i32_p],
        )
        self.clReleaseCommandQueue = bind_c_symbol(
            lib, "clReleaseCommandQueue", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.clFinish = bind_c_symbol(
            lib, "clFinish", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.clCreateBuffer = bind_c_symbol(
            lib,
            "clCreateBuffer",
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p, c_i32_p],
        )
        self.clReleaseMemObject = bind_c_symbol(
            lib, "clReleaseMemObject", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.clEnqueueMapBuffer = bind_c_symbol(
            lib,
            "clEnqueueMapBuffer",
            ctypes.c_void_p,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint64,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_uint32,
                c_void_pp,
                c_void_pp,
                c_i32_p,
            ],
        )
        self.clEnqueueUnmapMemObject = bind_c_symbol(
            lib,
            "clEnqueueUnmapMemObject",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                c_void_pp,
                c_void_pp,
            ],
        )
        self.clEnqueueWriteBuffer = bind_c_symbol(
            lib,
            "clEnqueueWriteBuffer",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_uint32,
                c_void_pp,
                c_void_pp,
            ],
        )
        self.clEnqueueReadBuffer = bind_c_symbol(
            lib,
            "clEnqueueReadBuffer",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_uint32,
                c_void_pp,
                c_void_pp,
            ],
        )
        self.clEnqueueCopyBuffer = bind_c_symbol(
            lib,
            "clEnqueueCopyBuffer",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_uint32,
                c_void_pp,
                c_void_pp,
            ],
        )
        self.clCreateProgramWithSource = bind_c_symbol(
            lib,
            "clCreateProgramWithSource",
            ctypes.c_void_p,
            [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_char_p),
                c_size_p,
                c_i32_p,
            ],
        )
        self.clCreateProgramWithBinary = bind_c_symbol(
            lib,
            "clCreateProgramWithBinary",
            ctypes.c_void_p,
            [
                ctypes.c_void_p,
                ctypes.c_uint32,
                c_void_pp,
                c_size_p,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
                c_i32_p,
                c_i32_p,
            ],
        )
        self.clBuildProgram = bind_c_symbol(
            lib,
            "clBuildProgram",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_uint32,
                c_void_pp,
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ],
        )
        self.clGetProgramBuildInfo = bind_c_symbol(
            lib,
            "clGetProgramBuildInfo",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_size_t,
                ctypes.c_void_p,
                c_size_p,
            ],
        )
        self.clGetProgramInfo = bind_c_symbol(
            lib,
            "clGetProgramInfo",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p, c_size_p],
        )
        self.clReleaseProgram = bind_c_symbol(
            lib, "clReleaseProgram", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.clCreateKernel = bind_c_symbol(
            lib,
            "clCreateKernel",
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_char_p, c_i32_p],
        )
        self.clReleaseKernel = bind_c_symbol(
            lib, "clReleaseKernel", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.clSetKernelArg = bind_c_symbol(
            lib,
            "clSetKernelArg",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p],
        )
        self.clEnqueueNDRangeKernel = bind_c_symbol(
            lib,
            "clEnqueueNDRangeKernel",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                c_size_p,
                c_size_p,
                c_size_p,
                ctypes.c_uint32,
                c_void_pp,
                c_void_pp,
            ],
        )


def _normalize_vendor_name(vendor_raw: str) -> str:
    lower = vendor_raw.lower()
    if "nvidia" in lower:
        return "NVIDIA"
    if "advanced micro devices" in lower or "amd" in lower or "ati " in lower:
        return "AMD"
    if "intel" in lower:
        return "Intel"
    if "apple" in lower:
        return "Apple"
    return vendor_raw.strip()


class OpenCLBackend(BackendPlugin):
    """Cross-platform OpenCL backend plugin (priority 50)."""

    name = "opencl"
    priority = OPENCL_BACKEND_PRIORITY

    def __init__(self) -> None:
        super().__init__()
        self._api: _OpenCLApi | None = None

    def is_available(self) -> bool:
        if self._api is not None:
            return True
        self._probe_log.clear()
        try:
            lib = load_shared_library(
                OPENCL_LIB_CANDIDATES,
                diag_fn=self._record_probe,
                label=self.name,
            )
            self._api = _OpenCLApi(lib)
            self._load_error = None
            return True
        except OSError:
            self._load_error = f"{OPENCL_LIB_CANDIDATES[0]} not found"
            return False
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def _require_api(self) -> _OpenCLApi:
        if not self.is_available() or self._api is None:
            raise BackendInitError(self._load_error or "OpenCL runtime unavailable")
        return self._api

    def _check(self, rc: int, action: str, *, runtime_phase: bool = False) -> None:
        if rc == CL_SUCCESS:
            return
        detail = f"{action} failed with cl_int={rc}"
        if rc in CL_HARDWARE_FAULT_CODES or (runtime_phase and rc == CL_OUT_OF_RESOURCES):
            raise BackendHardwareError(detail)
        raise BackendInitError(detail)

    def _get_device_str(self, api: _OpenCLApi, dev: Any, param: int) -> str:
        size = ctypes.c_size_t(0)
        if api.clGetDeviceInfo(dev, param, 0, None, ctypes.byref(size)) != CL_SUCCESS or size.value == 0:
            return ""
        buf = ctypes.create_string_buffer(size.value)
        if api.clGetDeviceInfo(dev, param, size.value, buf, None) != CL_SUCCESS:
            return ""
        return buf.value.decode("utf-8", errors="replace").strip()

    def _get_device_u32(self, api: _OpenCLApi, dev: Any, param: int) -> int | None:
        val = ctypes.c_uint32(0)
        if api.clGetDeviceInfo(dev, param, ctypes.sizeof(val), ctypes.byref(val), None) == CL_SUCCESS:
            return int(val.value)
        return None

    def _get_device_u64(self, api: _OpenCLApi, dev: Any, param: int) -> int | None:
        val = ctypes.c_uint64(0)
        if api.clGetDeviceInfo(dev, param, ctypes.sizeof(val), ctypes.byref(val), None) == CL_SUCCESS:
            return int(val.value)
        return None

    def _get_platform_str(self, api: _OpenCLApi, plat: Any, param: int) -> str:
        size = ctypes.c_size_t(0)
        if api.clGetPlatformInfo(plat, param, 0, None, ctypes.byref(size)) != CL_SUCCESS or size.value == 0:
            return ""
        buf = ctypes.create_string_buffer(size.value)
        if api.clGetPlatformInfo(plat, param, size.value, buf, None) != CL_SUCCESS:
            return ""
        return buf.value.decode("utf-8", errors="replace").strip()

    def _query_pci_bdf(self, api: _OpenCLApi, dev: Any, ext_tokens: set[str]) -> str | None:
        if "cl_khr_pci_bus_info" in ext_tokens:
            info = _ClPciBusInfoKhr()
            if (
                api.clGetDeviceInfo(
                    dev, CL_DEVICE_PCI_BUS_INFO_KHR, ctypes.sizeof(info), ctypes.byref(info), None
                )
                == CL_SUCCESS
            ):
                return format_pci_bdf(
                    info.pci_domain, info.pci_bus, info.pci_device, info.pci_function
                )

        if "cl_nv_device_attribute_query" in ext_tokens:
            bus = self._get_device_u32(api, dev, CL_DEVICE_PCI_BUS_ID_NV)
            slot = self._get_device_u32(api, dev, CL_DEVICE_PCI_SLOT_ID_NV)
            domain = self._get_device_u32(api, dev, CL_DEVICE_PCI_DOMAIN_ID_NV) or 0
            if bus is not None and slot is not None:
                return format_pci_bdf(domain, bus, (slot >> 3) & 0x1F, slot & 0x7)

        if "cl_amd_device_attribute_query" in ext_tokens:
            topo = _ClDeviceTopologyAmdPcie()
            if (
                api.clGetDeviceInfo(
                    dev, CL_DEVICE_TOPOLOGY_AMD, ctypes.sizeof(topo), ctypes.byref(topo), None
                )
                == CL_SUCCESS
                and topo.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD
            ):
                return format_pci_bdf(0, topo.bus, topo.device, topo.function)

        return None

    def _query_uuid(self, api: _OpenCLApi, dev: Any, ext_tokens: set[str]) -> str | None:
        if "cl_khr_device_uuid" not in ext_tokens:
            return None
        raw = (ctypes.c_ubyte * 16)()
        if (
            api.clGetDeviceInfo(
                dev, CL_DEVICE_UUID_KHR, ctypes.sizeof(raw), ctypes.byref(raw), None
            )
            == CL_SUCCESS
        ):
            return format_uuid_bytes(bytes(raw))
        return None

    def enumerate_devices(self) -> list[BackendDeviceInfo]:
        api = self._require_api()
        n_plat = ctypes.c_uint32(0)
        rc = api.clGetPlatformIDs(0, None, ctypes.byref(n_plat))
        if rc != CL_SUCCESS or n_plat.value == 0:
            self._load_error = (
                check_linux_render_node_hint() or "no OpenCL platforms detected"
            )
            return []

        platforms = (ctypes.c_void_p * n_plat.value)()
        self._check(api.clGetPlatformIDs(n_plat.value, platforms, None), "clGetPlatformIDs")

        devices: list[BackendDeviceInfo] = []
        ordinal = 0
        for plat in platforms:
            platform_name = self._get_platform_str(api, plat, CL_PLATFORM_NAME)
            n_dev = ctypes.c_uint32(0)
            rc = api.clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(n_dev))
            if rc == CL_DEVICE_NOT_FOUND or rc != CL_SUCCESS or n_dev.value == 0:
                continue

            dev_ids = (ctypes.c_void_p * n_dev.value)()
            self._check(
                api.clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, n_dev.value, dev_ids, None),
                "clGetDeviceIDs",
            )
            for dev in dev_ids:
                name = self._get_device_str(api, dev, CL_DEVICE_NAME)
                vendor_raw = self._get_device_str(api, dev, CL_DEVICE_VENDOR)
                vendor = _normalize_vendor_name(vendor_raw)
                vendor_id = self._get_device_u32(api, dev, CL_DEVICE_VENDOR_ID)
                driver_ver = self._get_device_str(api, dev, CL_DRIVER_VERSION)
                cl_ver = self._get_device_str(api, dev, CL_DEVICE_VERSION)
                ext_str = self._get_device_str(api, dev, CL_DEVICE_EXTENSIONS)
                ext_tokens = set(ext_str.split())

                total_mem = self._get_device_u64(api, dev, CL_DEVICE_GLOBAL_MEM_SIZE) or 0
                max_alloc = self._get_device_u64(api, dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE) or 0
                cu_count = self._get_device_u32(api, dev, CL_DEVICE_MAX_COMPUTE_UNITS) or 0
                max_wg = self._get_device_u64(api, dev, CL_DEVICE_MAX_WORK_GROUP_SIZE) or 0
                max_clock = self._get_device_u32(api, dev, CL_DEVICE_MAX_CLOCK_FREQUENCY)
                fp64_cfg = self._get_device_u64(api, dev, CL_DEVICE_DOUBLE_FP_CONFIG) or 0
                fp16_cfg = self._get_device_u64(api, dev, CL_DEVICE_HALF_FP_CONFIG) or 0
                unified_mem = bool(self._get_device_u32(api, dev, CL_DEVICE_HOST_UNIFIED_MEMORY) or 0)

                caps = DeviceCapabilities(
                    fp64=("cl_khr_fp64" in ext_tokens or fp64_cfg != 0),
                    fp32=True,
                    fp16=("cl_khr_fp16" in ext_tokens or fp16_cfg != 0),
                    bf16=False,
                    fp8=False,
                    fp4=False,
                    int64=True,
                    int32=True,
                    int16=True,
                    int8=True,
                    int4=False,
                    tensor_fp32=False,
                    tensor_fp16=False,
                    tensor_bf16=False,
                    tensor_int8=False,
                )

                devices.append(
                    BackendDeviceInfo(
                        backend=self.name,
                        priority=self.priority,
                        backend_device_index=ordinal,
                        native_handle={"platform": plat, "device": dev},
                        name=name,
                        vendor=vendor,
                        vendor_id=vendor_id,
                        architecture="opencl",
                        compute_capability=cl_ver,
                        total_memory_bytes=total_mem,
                        free_memory_bytes=None,
                        compute_units=cu_count,
                        driver_version=driver_ver,
                        pci_bdf=self._query_pci_bdf(api, dev, ext_tokens),
                        uuid=self._query_uuid(api, dev, ext_tokens),
                        capabilities=caps,
                        extra={
                            "platform": platform_name,
                            "max_clock_mhz": max_clock,
                            "max_buffer_length": max_alloc,
                            "max_threads_per_block": max_wg,
                            "has_unified_memory": unified_mem,
                        },
                    )
                )
                ordinal += 1
        if not devices:
            self._load_error = (
                check_linux_render_node_hint() or "no OpenCL GPU devices detected"
            )
        return devices

    def create_context(self, device: BackendDeviceInfo) -> BackendContext:
        api = self._require_api()
        dev = device.native_handle["device"]
        err = ctypes.c_int32(0)
        dev_arr = (ctypes.c_void_p * 1)(dev)
        ctx_ptr = api.clCreateContext(None, 1, dev_arr, None, None, ctypes.byref(err))
        self._check(err.value, "clCreateContext")

        q_ptr = None
        if api.clCreateCommandQueueWithProperties is not None:
            q_ptr = api.clCreateCommandQueueWithProperties(ctx_ptr, dev, None, ctypes.byref(err))
        if not q_ptr:
            q_ptr = api.clCreateCommandQueue(ctx_ptr, dev, 0, ctypes.byref(err))
        if err.value != CL_SUCCESS or not q_ptr:
            api.clReleaseContext(ctx_ptr)
            self._check(err.value, "clCreateCommandQueue")

        return BackendContext(
            backend=self.name,
            device_info=device,
            handle=ctx_ptr,
            queue=q_ptr,
            extra={"device": dev},
        )

    def destroy_context(self, ctx: BackendContext) -> None:
        if self._api is None:
            return
        if ctx.queue is not None:
            self._api.clReleaseCommandQueue(ctx.queue)
            ctx.queue = None
        if ctx.handle is not None:
            self._api.clReleaseContext(ctx.handle)
            ctx.handle = None

    def synchronize(self, ctx: BackendContext) -> None:
        api = self._require_api()
        self._check(api.clFinish(ctx.queue), "clFinish", runtime_phase=True)

    def alloc_device(self, ctx: BackendContext, size_bytes: int) -> DeviceBuffer:
        api = self._require_api()
        err = ctypes.c_int32(0)
        mem = api.clCreateBuffer(
            ctx.handle, CL_MEM_READ_WRITE, max(1, size_bytes), None, ctypes.byref(err)
        )
        self._check(err.value, "clCreateBuffer")
        return DeviceBuffer(backend=self.name, handle=mem, size_bytes=size_bytes)

    def free_device(self, ctx: BackendContext, buf: DeviceBuffer) -> None:
        if not buf.handle or self._api is None:
            return
        self._api.clReleaseMemObject(buf.handle)
        buf.handle = None

    def alloc_pinned_host(self, ctx: BackendContext, size_bytes: int) -> HostBuffer:
        api = self._require_api()
        err = ctypes.c_int32(0)
        alloc_sz = max(1, size_bytes)
        mem = api.clCreateBuffer(
            ctx.handle,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            alloc_sz,
            None,
            ctypes.byref(err),
        )
        self._check(err.value, "clCreateBuffer(ALLOC_HOST_PTR)")
        mapped = api.clEnqueueMapBuffer(
            ctx.queue,
            mem,
            CL_TRUE,
            CL_MAP_READ | CL_MAP_WRITE,
            0,
            alloc_sz,
            0,
            None,
            None,
            ctypes.byref(err),
        )
        if err.value != CL_SUCCESS or not mapped:
            api.clReleaseMemObject(mem)
            self._check(err.value, "clEnqueueMapBuffer")
        return HostBuffer(
            backend=self.name,
            handle=mem,
            size_bytes=size_bytes,
            pointer=int(mapped),
        )

    def free_pinned_host(self, ctx: BackendContext, buf: HostBuffer) -> None:
        if not buf.handle or self._api is None:
            return
        if buf.pointer and ctx.queue is not None:
            self._api.clEnqueueUnmapMemObject(
                ctx.queue, buf.handle, ctypes.c_void_p(buf.pointer), 0, None, None
            )
            self._api.clFinish(ctx.queue)
        self._api.clReleaseMemObject(buf.handle)
        buf.handle = None
        buf.pointer = 0

    def memcpy_h2d(
        self,
        ctx: BackendContext,
        dst: DeviceBuffer,
        src: HostBuffer | ctypes.c_void_p | bytes | bytearray | memoryview | int,
        size_bytes: int,
    ) -> None:
        if size_bytes <= 0:
            return
        api = self._require_api()
        src_ptr, keepalive = resolve_host_pointer(src)
        self._check(
            api.clEnqueueWriteBuffer(
                ctx.queue, dst.handle, CL_TRUE, 0, size_bytes, src_ptr, 0, None, None
            ),
            "clEnqueueWriteBuffer",
            runtime_phase=True,
        )
        del keepalive

    def memcpy_d2h(
        self,
        ctx: BackendContext,
        dst: HostBuffer | ctypes.c_void_p | bytearray | memoryview | int,
        src: DeviceBuffer,
        size_bytes: int,
    ) -> None:
        if size_bytes <= 0:
            return
        api = self._require_api()
        dst_ptr, keepalive = resolve_host_pointer(dst)
        self._check(
            api.clEnqueueReadBuffer(
                ctx.queue, src.handle, CL_TRUE, 0, size_bytes, dst_ptr, 0, None, None
            ),
            "clEnqueueReadBuffer",
            runtime_phase=True,
        )
        del keepalive

    def memcpy_d2d(
        self,
        ctx: BackendContext,
        dst: DeviceBuffer,
        src: DeviceBuffer,
        size_bytes: int,
    ) -> None:
        if size_bytes <= 0:
            return
        api = self._require_api()
        self._check(
            api.clEnqueueCopyBuffer(
                ctx.queue, src.handle, dst.handle, 0, 0, size_bytes, 0, None, None
            ),
            "clEnqueueCopyBuffer",
            runtime_phase=True,
        )
        self._check(api.clFinish(ctx.queue), "clFinish", runtime_phase=True)

    def load_module(self, ctx: BackendContext, compiled_binary: Any) -> KernelModule:
        api = self._require_api()
        dev = ctx.extra["device"]
        dev_arr = (ctypes.c_void_p * 1)(dev)
        raw = bytes(compiled_binary.binary_bytes)
        fmt = getattr(compiled_binary, "format", "opencl_source")
        err = ctypes.c_int32(0)

        if fmt == "opencl_binary":
            lengths = (ctypes.c_size_t * 1)(len(raw))
            bin_buf = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
            bin_ptrs = (ctypes.POINTER(ctypes.c_ubyte) * 1)(bin_buf)
            bin_status = (ctypes.c_int32 * 1)(0)
            program = api.clCreateProgramWithBinary(
                ctx.handle, 1, dev_arr, lengths, bin_ptrs, bin_status, ctypes.byref(err)
            )
            self._check(err.value, "clCreateProgramWithBinary")
        else:
            src_arr = (ctypes.c_char_p * 1)(raw)
            lengths = (ctypes.c_size_t * 1)(len(raw))
            program = api.clCreateProgramWithSource(
                ctx.handle, 1, src_arr, lengths, ctypes.byref(err)
            )
            self._check(err.value, "clCreateProgramWithSource")

        flags_str = " ".join(getattr(compiled_binary, "compile_flags", ()))
        rc = api.clBuildProgram(
            program,
            1,
            dev_arr,
            flags_str.encode("utf-8") if flags_str else None,
            None,
            None,
        )
        if rc != CL_SUCCESS:
            log_size = ctypes.c_size_t(0)
            build_log = ""
            if (
                api.clGetProgramBuildInfo(
                    program, dev, CL_PROGRAM_BUILD_LOG, 0, None, ctypes.byref(log_size)
                )
                == CL_SUCCESS
                and log_size.value > 0
            ):
                log_buf = ctypes.create_string_buffer(log_size.value)
                api.clGetProgramBuildInfo(
                    program, dev, CL_PROGRAM_BUILD_LOG, log_size.value, log_buf, None
                )
                build_log = log_buf.value.decode("utf-8", errors="replace")
            api.clReleaseProgram(program)
            raise KernelCompilationError(
                f"clBuildProgram failed with cl_int={rc}",
                compiler_log=build_log,
            )

        return KernelModule(backend=self.name, handle=program)

    def extract_program_binary(self, module: KernelModule) -> bytes:
        """Extract compiled CL_PROGRAM_BINARIES bytes from a built OpenCL program."""
        api = self._require_api()
        sizes = (ctypes.c_size_t * 1)(0)
        self._check(
            api.clGetProgramInfo(
                module.handle, CL_PROGRAM_BINARY_SIZES, ctypes.sizeof(sizes), sizes, None
            ),
            "clGetProgramInfo(CL_PROGRAM_BINARY_SIZES)",
        )
        bin_size = int(sizes[0])
        if bin_size <= 0:
            raise BackendInitError("OpenCL driver returned empty program binary")
        out_buf = (ctypes.c_ubyte * bin_size)()
        ptrs = (ctypes.POINTER(ctypes.c_ubyte) * 1)(out_buf)
        self._check(
            api.clGetProgramInfo(
                module.handle, CL_PROGRAM_BINARIES, ctypes.sizeof(ptrs), ptrs, None
            ),
            "clGetProgramInfo(CL_PROGRAM_BINARIES)",
        )
        return bytes(out_buf)

    def unload_module(self, ctx: BackendContext, module: KernelModule) -> None:
        del ctx
        if self._api is None:
            return
        for k_handle in module.functions.values():
            self._api.clReleaseKernel(k_handle)
        module.functions.clear()
        if module.handle is not None:
            self._api.clReleaseProgram(module.handle)
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
        kernel = module.functions.get(entry_point)
        if kernel is None:
            err = ctypes.c_int32(0)
            kernel = api.clCreateKernel(
                module.handle, entry_point.encode("utf-8"), ctypes.byref(err)
            )
            self._check(err.value, f"clCreateKernel({entry_point})")
            module.functions[entry_point] = kernel

        for idx, raw_arg in enumerate(args):
            coerced = coerce_kernel_arg(raw_arg)
            if isinstance(coerced, DeviceBuffer):
                mem_val = ctypes.c_void_p(coerced.handle)
                self._check(
                    api.clSetKernelArg(
                        kernel,
                        idx,
                        ctypes.sizeof(ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(mem_val), ctypes.c_void_p),
                    ),
                    f"clSetKernelArg({idx})",
                )
            else:
                self._check(
                    api.clSetKernelArg(
                        kernel,
                        idx,
                        ctypes.sizeof(coerced),
                        ctypes.cast(ctypes.byref(coerced), ctypes.c_void_p),
                    ),
                    f"clSetKernelArg({idx})",
                )

        if shared_mem > 0:
            self._check(
                api.clSetKernelArg(kernel, len(args), shared_mem, None),
                "clSetKernelArg(shared_mem)",
            )

        gx, gy, gz = grid
        bx, by, bz = block
        global_work_size = (ctypes.c_size_t * 3)(gx * bx, gy * by, gz * bz)
        local_work_size = (ctypes.c_size_t * 3)(bx, by, bz)
        self._check(
            api.clEnqueueNDRangeKernel(
                ctx.queue,
                kernel,
                3,
                None,
                global_work_size,
                local_work_size,
                0,
                None,
                None,
            ),
            f"clEnqueueNDRangeKernel({entry_point})",
            runtime_phase=True,
        )
