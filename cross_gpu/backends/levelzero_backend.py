# SPDX-License-Identifier: MIT
"""oneAPI Level Zero (libze_loader.so.1) backend plugin via ctypes."""

from __future__ import annotations

import ctypes
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
    check_linux_render_node_hint,
    coerce_kernel_arg,
    format_pci_bdf,
    format_uuid_bytes,
    load_shared_library,
    resolve_host_pointer,
)
from cross_gpu.exit_codes import KernelCompilationError
from cross_gpu.models import BackendDeviceInfo, DeviceCapabilities

LEVELZERO_LIB_CANDIDATES: tuple[str, ...] = ("libze_loader.so.1", "libze_loader.so")

ZE_RESULT_SUCCESS = 0
ZE_RESULT_ERROR_DEVICE_LOST = 0x70000001
ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET = 0x70000005
ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE = 0x70000006

ZE_HARDWARE_FAULT_CODES: frozenset[int] = frozenset(
    {
        ZE_RESULT_ERROR_DEVICE_LOST,
        ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET,
        ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE,
    }
)

ZE_INIT_FLAG_GPU_ONLY = 0x1
ZE_DEVICE_TYPE_GPU = 1

ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES = 0x1
ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3
ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES = 0x5
ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES = 0x7
ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xD
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0xE
ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 0xF
ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15
ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 0x16
ZE_STRUCTURE_TYPE_MODULE_DESC = 0x1B
ZE_STRUCTURE_TYPE_KERNEL_DESC = 0x1D
ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES = 0x10008

ZE_MODULE_FORMAT_IL_SPIRV = 0
ZE_MODULE_FORMAT_NATIVE = 1
ZE_COMMAND_QUEUE_MODE_DEFAULT = 0
ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0
ZE_UINT64_MAX = 0xFFFFFFFFFFFFFFFF

ZE_DEVICE_MODULE_FLAG_FP16 = 1 << 0
ZE_DEVICE_MODULE_FLAG_FP64 = 1 << 1
ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS = 1 << 2
ZE_DEVICE_MODULE_FLAG_DP4A = 1 << 3

VENDOR_NAMES: dict[int, str] = {
    0x8086: "Intel",
    0x10DE: "NVIDIA",
    0x1002: "AMD",
}


class _ZeDriverProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("uuid", ctypes.c_ubyte * 16),
        ("driverVersion", ctypes.c_uint32),
    ]


class _ZeDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),
        ("vendorId", ctypes.c_uint32),
        ("deviceId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subdeviceId", ctypes.c_uint32),
        ("coreClockRate", ctypes.c_uint32),
        ("maxMemAllocSize", ctypes.c_uint64),
        ("maxHardwareContexts", ctypes.c_uint32),
        ("maxCommandQueuePriority", ctypes.c_uint32),
        ("numThreadsPerEU", ctypes.c_uint32),
        ("physicalEUSimdWidth", ctypes.c_uint32),
        ("numEUsPerSubslice", ctypes.c_uint32),
        ("numSubslicesPerSlice", ctypes.c_uint32),
        ("numSlices", ctypes.c_uint32),
        ("timerResolution", ctypes.c_uint64),
        ("timestampValidBits", ctypes.c_uint32),
        ("kernelTimestampValidBits", ctypes.c_uint32),
        ("uuid", ctypes.c_ubyte * 16),
        ("name", ctypes.c_char * 256),
    ]


class _ZeDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("maxClockRate", ctypes.c_uint32),
        ("maxBusWidth", ctypes.c_uint32),
        ("totalSize", ctypes.c_uint64),
        ("name", ctypes.c_char * 256),
    ]


class _ZeDeviceModuleProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("spirvVersionSupported", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("fp16flags", ctypes.c_uint32),
        ("fp32flags", ctypes.c_uint32),
        ("fp64flags", ctypes.c_uint32),
        ("maxArgumentsSize", ctypes.c_uint32),
        ("printfBufferSize", ctypes.c_uint32),
        ("nativeKernelSupported", ctypes.c_ubyte * 16),
    ]


class _ZePciAddressExt(ctypes.Structure):
    _fields_ = [
        ("domain", ctypes.c_uint32),
        ("bus", ctypes.c_uint32),
        ("device", ctypes.c_uint32),
        ("function", ctypes.c_uint32),
    ]


class _ZePciSpeedExt(ctypes.Structure):
    _fields_ = [
        ("genVersion", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("maxBandwidth", ctypes.c_int64),
    ]


class _ZePciExtProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("address", _ZePciAddressExt),
        ("maxSpeed", _ZePciSpeedExt),
    ]


class _ZeContextDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class _ZeCommandQueueDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("ordinal", ctypes.c_uint32),
        ("index", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("priority", ctypes.c_uint32),
    ]


class _ZeCommandListDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("commandQueueGroupOrdinal", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class _ZeDeviceMemAllocDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("ordinal", ctypes.c_uint32),
    ]


class _ZeHostMemAllocDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class _ZeModuleDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("format", ctypes.c_uint32),
        ("inputSize", ctypes.c_size_t),
        ("pInputModule", ctypes.POINTER(ctypes.c_ubyte)),
        ("pBuildFlags", ctypes.c_char_p),
        ("pConstants", ctypes.c_void_p),
    ]


class _ZeKernelDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pKernelName", ctypes.c_char_p),
    ]


class _ZeGroupCount(ctypes.Structure):
    _fields_ = [
        ("groupCountX", ctypes.c_uint32),
        ("groupCountY", ctypes.c_uint32),
        ("groupCountZ", ctypes.c_uint32),
    ]


class _LevelZeroApi:
    """Explicitly typed ctypes bindings for the oneAPI Level Zero C API."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib
        c_u32_p = ctypes.POINTER(ctypes.c_uint32)
        c_size_p = ctypes.POINTER(ctypes.c_size_t)
        c_void_pp = ctypes.POINTER(ctypes.c_void_p)

        self.zeInit = bind_c_symbol(lib, "zeInit", ctypes.c_int32, [ctypes.c_uint32])
        self.zeDriverGet = bind_c_symbol(
            lib, "zeDriverGet", ctypes.c_int32, [c_u32_p, c_void_pp]
        )
        self.zeDriverGetProperties = bind_c_symbol(
            lib,
            "zeDriverGetProperties",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.POINTER(_ZeDriverProperties)],
        )
        self.zeDeviceGet = bind_c_symbol(
            lib, "zeDeviceGet", ctypes.c_int32, [ctypes.c_void_p, c_u32_p, c_void_pp]
        )
        self.zeDeviceGetProperties = bind_c_symbol(
            lib,
            "zeDeviceGetProperties",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.POINTER(_ZeDeviceProperties)],
        )
        self.zeDeviceGetMemoryProperties = bind_c_symbol(
            lib,
            "zeDeviceGetMemoryProperties",
            ctypes.c_int32,
            [ctypes.c_void_p, c_u32_p, ctypes.POINTER(_ZeDeviceMemoryProperties)],
        )
        self.zeDeviceGetModuleProperties = bind_c_symbol(
            lib,
            "zeDeviceGetModuleProperties",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.POINTER(_ZeDeviceModuleProperties)],
        )
        self.zeDevicePciGetPropertiesExt = bind_optional_c_symbol(
            lib,
            "zeDevicePciGetPropertiesExt",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.POINTER(_ZePciExtProperties)],
        )
        self.zeContextCreate = bind_c_symbol(
            lib,
            "zeContextCreate",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.POINTER(_ZeContextDesc), c_void_pp],
        )
        self.zeContextDestroy = bind_c_symbol(
            lib, "zeContextDestroy", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.zeCommandQueueCreate = bind_c_symbol(
            lib,
            "zeCommandQueueCreate",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(_ZeCommandQueueDesc),
                c_void_pp,
            ],
        )
        self.zeCommandQueueDestroy = bind_c_symbol(
            lib, "zeCommandQueueDestroy", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.zeCommandQueueExecuteCommandLists = bind_c_symbol(
            lib,
            "zeCommandQueueExecuteCommandLists",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint32, c_void_pp, ctypes.c_void_p],
        )
        self.zeCommandQueueSynchronize = bind_c_symbol(
            lib,
            "zeCommandQueueSynchronize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint64],
        )
        self.zeCommandListCreate = bind_c_symbol(
            lib,
            "zeCommandListCreate",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(_ZeCommandListDesc),
                c_void_pp,
            ],
        )
        self.zeCommandListDestroy = bind_c_symbol(
            lib, "zeCommandListDestroy", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.zeCommandListReset = bind_c_symbol(
            lib, "zeCommandListReset", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.zeCommandListClose = bind_c_symbol(
            lib, "zeCommandListClose", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.zeCommandListAppendMemoryCopy = bind_c_symbol(
            lib,
            "zeCommandListAppendMemoryCopy",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_uint32,
                c_void_pp,
            ],
        )
        self.zeCommandListAppendLaunchKernel = bind_c_symbol(
            lib,
            "zeCommandListAppendLaunchKernel",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(_ZeGroupCount),
                ctypes.c_void_p,
                ctypes.c_uint32,
                c_void_pp,
            ],
        )
        self.zeMemAllocDevice = bind_c_symbol(
            lib,
            "zeMemAllocDevice",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.POINTER(_ZeDeviceMemAllocDesc),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_void_p,
                c_void_pp,
            ],
        )
        self.zeMemAllocHost = bind_c_symbol(
            lib,
            "zeMemAllocHost",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.POINTER(_ZeHostMemAllocDesc),
                ctypes.c_size_t,
                ctypes.c_size_t,
                c_void_pp,
            ],
        )
        self.zeMemFree = bind_c_symbol(
            lib, "zeMemFree", ctypes.c_int32, [ctypes.c_void_p, ctypes.c_void_p]
        )
        self.zeModuleCreate = bind_c_symbol(
            lib,
            "zeModuleCreate",
            ctypes.c_int32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(_ZeModuleDesc),
                c_void_pp,
                c_void_pp,
            ],
        )
        self.zeModuleDestroy = bind_c_symbol(
            lib, "zeModuleDestroy", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.zeModuleBuildLogGetString = bind_optional_c_symbol(
            lib,
            "zeModuleBuildLogGetString",
            ctypes.c_int32,
            [ctypes.c_void_p, c_size_p, ctypes.c_char_p],
        )
        self.zeModuleBuildLogDestroy = bind_optional_c_symbol(
            lib, "zeModuleBuildLogDestroy", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.zeKernelCreate = bind_c_symbol(
            lib,
            "zeKernelCreate",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.POINTER(_ZeKernelDesc), c_void_pp],
        )
        self.zeKernelDestroy = bind_c_symbol(
            lib, "zeKernelDestroy", ctypes.c_int32, [ctypes.c_void_p]
        )
        self.zeKernelSetGroupSize = bind_c_symbol(
            lib,
            "zeKernelSetGroupSize",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32],
        )
        self.zeKernelSetArgumentValue = bind_c_symbol(
            lib,
            "zeKernelSetArgumentValue",
            ctypes.c_int32,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p],
        )


def _levelzero_capabilities(
    props: _ZeDeviceProperties,
    mod_props: _ZeDeviceModuleProperties,
) -> DeviceCapabilities:
    flags = int(mod_props.flags)
    name_lower = props.name.decode("utf-8", errors="replace").lower()
    is_xe_or_arc = (
        props.vendorId == 0x8086
        and any(tag in name_lower for tag in ("arc", "max", "flex", "xe", "battlemage", "alchemist"))
    )
    return DeviceCapabilities(
        fp64=bool(mod_props.fp64flags != 0 or (flags & ZE_DEVICE_MODULE_FLAG_FP64)),
        fp32=True,
        fp16=bool(mod_props.fp16flags != 0 or (flags & ZE_DEVICE_MODULE_FLAG_FP16)),
        bf16=is_xe_or_arc,
        fp8=False,
        fp4=False,
        int64=True,
        int32=True,
        int16=True,
        int8=bool((flags & ZE_DEVICE_MODULE_FLAG_DP4A) or is_xe_or_arc),
        int4=False,
        tensor_fp32=False,
        tensor_fp16=is_xe_or_arc,
        tensor_bf16=is_xe_or_arc,
        tensor_int8=bool((flags & ZE_DEVICE_MODULE_FLAG_DP4A) or is_xe_or_arc),
    )


class LevelZeroBackend(BackendPlugin):
    """Native oneAPI Level Zero backend plugin."""

    name = "levelzero"
    priority = NATIVE_BACKEND_PRIORITY

    def __init__(self) -> None:
        super().__init__()
        self._api: _LevelZeroApi | None = None
        self._initialized = False

    def is_available(self) -> bool:
        if self._api is not None and self._initialized:
            return True
        self._probe_log.clear()
        try:
            lib = load_shared_library(
                LEVELZERO_LIB_CANDIDATES,
                diag_fn=self._record_probe,
                label=self.name,
            )
            api = _LevelZeroApi(lib)
            rc = api.zeInit(ZE_INIT_FLAG_GPU_ONLY)
            if rc != ZE_RESULT_SUCCESS:
                rc = api.zeInit(0)
            if rc != ZE_RESULT_SUCCESS:
                perm_hint = check_linux_render_node_hint()
                self._load_error = perm_hint or f"zeInit failed with code 0x{rc & 0xFFFFFFFF:08x}"
                self._record_probe(f"probe levelzero: {self._load_error}")
                return False
            self._api = api
            self._initialized = True
            self._load_error = None
            return True
        except OSError:
            self._load_error = f"{LEVELZERO_LIB_CANDIDATES[0]} not found"
            return False
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def _require_api(self) -> _LevelZeroApi:
        if not self.is_available() or self._api is None:
            raise BackendInitError(self._load_error or "Level Zero runtime unavailable")
        return self._api

    def _check(self, rc: int, action: str) -> None:
        if rc == ZE_RESULT_SUCCESS:
            return
        detail = f"{action} failed with ze_result_t=0x{rc & 0xFFFFFFFF:08x}"
        if rc in ZE_HARDWARE_FAULT_CODES:
            raise BackendHardwareError(detail)
        raise BackendInitError(detail)

    def _query_pci_bdf(self, api: _LevelZeroApi, dev_handle: ctypes.c_void_p) -> str | None:
        if api.zeDevicePciGetPropertiesExt is None:
            return None
        for stype in (ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES, 0x10006, 0x1000B):
            pci_props = _ZePciExtProperties()
            pci_props.stype = stype
            rc = api.zeDevicePciGetPropertiesExt(dev_handle, ctypes.byref(pci_props))
            if rc == ZE_RESULT_SUCCESS:
                addr = pci_props.address
                if addr.domain or addr.bus or addr.device or addr.function:
                    return format_pci_bdf(addr.domain, addr.bus, addr.device, addr.function)
                return None
        return None

    def enumerate_devices(self) -> list[BackendDeviceInfo]:
        api = self._require_api()
        drv_count = ctypes.c_uint32(0)
        self._check(api.zeDriverGet(ctypes.byref(drv_count), None), "zeDriverGet(count)")
        if drv_count.value == 0:
            self._load_error = (
                check_linux_render_node_hint() or "no Level Zero GPU drivers detected"
            )
            return []

        drivers = (ctypes.c_void_p * drv_count.value)()
        self._check(api.zeDriverGet(ctypes.byref(drv_count), drivers), "zeDriverGet")

        devices: list[BackendDeviceInfo] = []
        ordinal = 0
        for drv in drivers:
            drv_props = _ZeDriverProperties()
            drv_props.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES
            driver_version = ""
            if api.zeDriverGetProperties(drv, ctypes.byref(drv_props)) == ZE_RESULT_SUCCESS:
                v = int(drv_props.driverVersion)
                driver_version = f"{(v >> 16) & 0xFFFF}.{v & 0xFFFF}"

            dev_count = ctypes.c_uint32(0)
            if api.zeDeviceGet(drv, ctypes.byref(dev_count), None) != ZE_RESULT_SUCCESS or dev_count.value == 0:
                continue
            dev_handles = (ctypes.c_void_p * dev_count.value)()
            self._check(api.zeDeviceGet(drv, ctypes.byref(dev_count), dev_handles), "zeDeviceGet")

            for dev_handle in dev_handles:
                props = _ZeDeviceProperties()
                props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
                self._check(
                    api.zeDeviceGetProperties(dev_handle, ctypes.byref(props)),
                    "zeDeviceGetProperties",
                )
                if props.type != ZE_DEVICE_TYPE_GPU:
                    continue

                mem_count = ctypes.c_uint32(0)
                total_mem = 0
                if (
                    api.zeDeviceGetMemoryProperties(dev_handle, ctypes.byref(mem_count), None)
                    == ZE_RESULT_SUCCESS
                    and mem_count.value > 0
                ):
                    mem_props = (_ZeDeviceMemoryProperties * mem_count.value)()
                    for entry in mem_props:
                        entry.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES
                    if (
                        api.zeDeviceGetMemoryProperties(
                            dev_handle, ctypes.byref(mem_count), mem_props
                        )
                        == ZE_RESULT_SUCCESS
                    ):
                        total_mem = sum(int(m.totalSize) for m in mem_props)

                mod_props = _ZeDeviceModuleProperties()
                mod_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES
                api.zeDeviceGetModuleProperties(dev_handle, ctypes.byref(mod_props))

                compute_units = max(
                    1,
                    int(props.numSlices)
                    * int(props.numSubslicesPerSlice)
                    * int(props.numEUsPerSubslice),
                )
                vendor_id = int(props.vendorId)
                vendor = VENDOR_NAMES.get(vendor_id, f"0x{vendor_id:04x}")
                name = props.name.decode("utf-8", errors="replace").strip()

                devices.append(
                    BackendDeviceInfo(
                        backend=self.name,
                        priority=self.priority,
                        backend_device_index=ordinal,
                        native_handle={"driver": drv, "device": dev_handle},
                        name=name,
                        vendor=vendor,
                        vendor_id=vendor_id,
                        architecture=f"0x{int(props.deviceId):04x}",
                        compute_capability=f"spirv-{mod_props.spirvVersionSupported >> 16}.{mod_props.spirvVersionSupported & 0xFFFF}",
                        total_memory_bytes=total_mem,
                        free_memory_bytes=None,
                        compute_units=compute_units,
                        driver_version=driver_version,
                        pci_bdf=self._query_pci_bdf(api, dev_handle),
                        uuid=format_uuid_bytes(bytes(props.uuid)),
                        capabilities=_levelzero_capabilities(props, mod_props),
                        extra={
                            "device_id": f"0x{int(props.deviceId):04x}",
                            "core_clock_mhz": int(props.coreClockRate),
                            "eu_simd_width": int(props.physicalEUSimdWidth),
                            "threads_per_eu": int(props.numThreadsPerEU),
                            "max_buffer_length": int(props.maxMemAllocSize),
                        },
                    )
                )
                ordinal += 1
        return devices

    def create_context(self, device: BackendDeviceInfo) -> BackendContext:
        api = self._require_api()
        handles = device.native_handle
        drv = handles["driver"]
        dev = handles["device"]

        ctx_desc = _ZeContextDesc(stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC, pNext=None, flags=0)
        ctx_ptr = ctypes.c_void_p()
        self._check(
            api.zeContextCreate(drv, ctypes.byref(ctx_desc), ctypes.byref(ctx_ptr)),
            "zeContextCreate",
        )

        q_desc = _ZeCommandQueueDesc(
            stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
            pNext=None,
            ordinal=0,
            index=0,
            flags=0,
            mode=ZE_COMMAND_QUEUE_MODE_DEFAULT,
            priority=ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
        )
        q_ptr = ctypes.c_void_p()
        rc = api.zeCommandQueueCreate(
            ctx_ptr.value, dev, ctypes.byref(q_desc), ctypes.byref(q_ptr)
        )
        if rc != ZE_RESULT_SUCCESS:
            api.zeContextDestroy(ctx_ptr.value)
            self._check(rc, "zeCommandQueueCreate")

        cl_desc = _ZeCommandListDesc(
            stype=ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
            pNext=None,
            commandQueueGroupOrdinal=0,
            flags=0,
        )
        cl_ptr = ctypes.c_void_p()
        rc = api.zeCommandListCreate(
            ctx_ptr.value, dev, ctypes.byref(cl_desc), ctypes.byref(cl_ptr)
        )
        if rc != ZE_RESULT_SUCCESS:
            api.zeCommandQueueDestroy(q_ptr.value)
            api.zeContextDestroy(ctx_ptr.value)
            self._check(rc, "zeCommandListCreate")

        return BackendContext(
            backend=self.name,
            device_info=device,
            handle=ctx_ptr.value,
            queue=q_ptr.value,
            command_list=cl_ptr.value,
            extra={"device": dev, "driver": drv},
        )

    def destroy_context(self, ctx: BackendContext) -> None:
        if self._api is None:
            return
        if ctx.command_list is not None:
            self._api.zeCommandListDestroy(ctx.command_list)
            ctx.command_list = None
        if ctx.queue is not None:
            self._api.zeCommandQueueDestroy(ctx.queue)
            ctx.queue = None
        if ctx.handle is not None:
            self._api.zeContextDestroy(ctx.handle)
            ctx.handle = None

    def synchronize(self, ctx: BackendContext) -> None:
        api = self._require_api()
        self._check(
            api.zeCommandQueueSynchronize(ctx.queue, ZE_UINT64_MAX),
            "zeCommandQueueSynchronize",
        )

    def alloc_device(self, ctx: BackendContext, size_bytes: int) -> DeviceBuffer:
        api = self._require_api()
        desc = _ZeDeviceMemAllocDesc(
            stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            pNext=None,
            flags=0,
            ordinal=0,
        )
        ptr = ctypes.c_void_p()
        self._check(
            api.zeMemAllocDevice(
                ctx.handle,
                ctypes.byref(desc),
                max(1, size_bytes),
                64,
                ctx.extra["device"],
                ctypes.byref(ptr),
            ),
            "zeMemAllocDevice",
        )
        addr = int(ptr.value or 0)
        return DeviceBuffer(backend=self.name, handle=addr, size_bytes=size_bytes, address=addr)

    def free_device(self, ctx: BackendContext, buf: DeviceBuffer) -> None:
        if not buf.handle or self._api is None or ctx.handle is None:
            return
        self._api.zeMemFree(ctx.handle, ctypes.c_void_p(int(buf.handle)))
        buf.handle = 0
        buf.address = 0

    def alloc_pinned_host(self, ctx: BackendContext, size_bytes: int) -> HostBuffer:
        api = self._require_api()
        desc = _ZeHostMemAllocDesc(
            stype=ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
            pNext=None,
            flags=0,
        )
        ptr = ctypes.c_void_p()
        self._check(
            api.zeMemAllocHost(
                ctx.handle,
                ctypes.byref(desc),
                max(1, size_bytes),
                64,
                ctypes.byref(ptr),
            ),
            "zeMemAllocHost",
        )
        addr = int(ptr.value or 0)
        return HostBuffer(backend=self.name, handle=addr, size_bytes=size_bytes, pointer=addr)

    def free_pinned_host(self, ctx: BackendContext, buf: HostBuffer) -> None:
        if not buf.handle or self._api is None or ctx.handle is None:
            return
        self._api.zeMemFree(ctx.handle, ctypes.c_void_p(int(buf.handle)))
        buf.handle = 0
        buf.pointer = 0

    def _submit_and_sync(self, api: _LevelZeroApi, ctx: BackendContext, *, sync: bool) -> None:
        self._check(api.zeCommandListClose(ctx.command_list), "zeCommandListClose")
        cl_arr = (ctypes.c_void_p * 1)(ctx.command_list)
        self._check(
            api.zeCommandQueueExecuteCommandLists(ctx.queue, 1, cl_arr, None),
            "zeCommandQueueExecuteCommandLists",
        )
        if sync:
            self._check(
                api.zeCommandQueueSynchronize(ctx.queue, ZE_UINT64_MAX),
                "zeCommandQueueSynchronize",
            )

    def _reset_command_list(self, api: _LevelZeroApi, ctx: BackendContext) -> None:
        self._check(
            api.zeCommandQueueSynchronize(ctx.queue, ZE_UINT64_MAX),
            "zeCommandQueueSynchronize",
        )
        self._check(api.zeCommandListReset(ctx.command_list), "zeCommandListReset")

    def memcpy_h2d(
        self,
        ctx: BackendContext,
        dst: DeviceBuffer,
        src: HostBuffer | ctypes.c_void_p | bytes | bytearray | memoryview | int,
        size_bytes: int,
    ) -> None:
        api = self._require_api()
        src_ptr, keepalive = resolve_host_pointer(src)
        self._reset_command_list(api, ctx)
        self._check(
            api.zeCommandListAppendMemoryCopy(
                ctx.command_list,
                ctypes.c_void_p(int(dst.handle)),
                src_ptr,
                size_bytes,
                None,
                0,
                None,
            ),
            "zeCommandListAppendMemoryCopy(H2D)",
        )
        self._submit_and_sync(api, ctx, sync=True)
        del keepalive

    def memcpy_d2h(
        self,
        ctx: BackendContext,
        dst: HostBuffer | ctypes.c_void_p | bytearray | memoryview | int,
        src: DeviceBuffer,
        size_bytes: int,
    ) -> None:
        api = self._require_api()
        dst_ptr, keepalive = resolve_host_pointer(dst)
        self._reset_command_list(api, ctx)
        self._check(
            api.zeCommandListAppendMemoryCopy(
                ctx.command_list,
                dst_ptr,
                ctypes.c_void_p(int(src.handle)),
                size_bytes,
                None,
                0,
                None,
            ),
            "zeCommandListAppendMemoryCopy(D2H)",
        )
        self._submit_and_sync(api, ctx, sync=True)
        del keepalive

    def memcpy_d2d(
        self,
        ctx: BackendContext,
        dst: DeviceBuffer,
        src: DeviceBuffer,
        size_bytes: int,
    ) -> None:
        api = self._require_api()
        self._reset_command_list(api, ctx)
        self._check(
            api.zeCommandListAppendMemoryCopy(
                ctx.command_list,
                ctypes.c_void_p(int(dst.handle)),
                ctypes.c_void_p(int(src.handle)),
                size_bytes,
                None,
                0,
                None,
            ),
            "zeCommandListAppendMemoryCopy(D2D)",
        )
        self._submit_and_sync(api, ctx, sync=True)

    def load_module(self, ctx: BackendContext, compiled_binary: Any) -> KernelModule:
        api = self._require_api()
        raw = bytes(compiled_binary.binary_bytes)
        buf = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        flags_bytes = " ".join(getattr(compiled_binary, "compile_flags", ())).encode("utf-8")
        desc = _ZeModuleDesc(
            stype=ZE_STRUCTURE_TYPE_MODULE_DESC,
            pNext=None,
            format=ZE_MODULE_FORMAT_IL_SPIRV,
            inputSize=len(raw),
            pInputModule=buf,
            pBuildFlags=flags_bytes if flags_bytes else None,
            pConstants=None,
        )
        mod_ptr = ctypes.c_void_p()
        log_ptr = ctypes.c_void_p()
        rc = api.zeModuleCreate(
            ctx.handle,
            ctx.extra["device"],
            ctypes.byref(desc),
            ctypes.byref(mod_ptr),
            ctypes.byref(log_ptr),
        )
        build_log = ""
        if log_ptr.value:
            if api.zeModuleBuildLogGetString is not None:
                sz = ctypes.c_size_t(0)
                if api.zeModuleBuildLogGetString(log_ptr.value, ctypes.byref(sz), None) == ZE_RESULT_SUCCESS and sz.value > 0:
                    log_buf = ctypes.create_string_buffer(sz.value)
                    api.zeModuleBuildLogGetString(log_ptr.value, ctypes.byref(sz), log_buf)
                    build_log = log_buf.value.decode("utf-8", errors="replace")
            if api.zeModuleBuildLogDestroy is not None:
                api.zeModuleBuildLogDestroy(log_ptr.value)
        if rc != ZE_RESULT_SUCCESS:
            if rc in ZE_HARDWARE_FAULT_CODES:
                self._check(rc, "zeModuleCreate")
            raise KernelCompilationError(
                f"zeModuleCreate failed with ze_result_t=0x{rc & 0xFFFFFFFF:08x}",
                compiler_log=build_log,
            )
        return KernelModule(backend=self.name, handle=mod_ptr.value)

    def unload_module(self, ctx: BackendContext, module: KernelModule) -> None:
        if self._api is None:
            return
        for k_handle in module.functions.values():
            self._api.zeKernelDestroy(k_handle)
        module.functions.clear()
        if module.handle is not None:
            self._api.zeModuleDestroy(module.handle)
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
        del shared_mem  # Shared memory size in SPIR-V/Level Zero is declared in kernel local memory args.
        api = self._require_api()

        kernel_handle = module.functions.get(entry_point)
        if kernel_handle is None:
            k_desc = _ZeKernelDesc(
                stype=ZE_STRUCTURE_TYPE_KERNEL_DESC,
                pNext=None,
                flags=0,
                pKernelName=entry_point.encode("utf-8"),
            )
            k_ptr = ctypes.c_void_p()
            self._check(
                api.zeKernelCreate(module.handle, ctypes.byref(k_desc), ctypes.byref(k_ptr)),
                f"zeKernelCreate({entry_point})",
            )
            kernel_handle = k_ptr.value
            module.functions[entry_point] = kernel_handle

        bx, by, bz = block
        self._check(
            api.zeKernelSetGroupSize(kernel_handle, bx, by, bz),
            "zeKernelSetGroupSize",
        )

        for idx, raw_arg in enumerate(args):
            coerced = coerce_kernel_arg(raw_arg)
            if isinstance(coerced, DeviceBuffer):
                ptr_val = ctypes.c_void_p(int(coerced.handle))
                self._check(
                    api.zeKernelSetArgumentValue(
                        kernel_handle,
                        idx,
                        ctypes.sizeof(ctypes.c_void_p),
                        ctypes.cast(ctypes.byref(ptr_val), ctypes.c_void_p),
                    ),
                    f"zeKernelSetArgumentValue({idx})",
                )
            else:
                self._check(
                    api.zeKernelSetArgumentValue(
                        kernel_handle,
                        idx,
                        ctypes.sizeof(coerced),
                        ctypes.cast(ctypes.byref(coerced), ctypes.c_void_p),
                    ),
                    f"zeKernelSetArgumentValue({idx})",
                )

        gx, gy, gz = grid
        group_count = _ZeGroupCount(groupCountX=gx, groupCountY=gy, groupCountZ=gz)
        self._reset_command_list(api, ctx)
        self._check(
            api.zeCommandListAppendLaunchKernel(
                ctx.command_list,
                kernel_handle,
                ctypes.byref(group_count),
                None,
                0,
                None,
            ),
            f"zeCommandListAppendLaunchKernel({entry_point})",
        )
        self._submit_and_sync(api, ctx, sync=False)
