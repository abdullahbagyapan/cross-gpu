"""AMD HIP runtime API (libamdhip64.so) backend plugin via ctypes."""

from __future__ import annotations

import ctypes
import re
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
    format_pci_bdf,
    format_uuid_bytes,
    load_shared_library,
    resolve_host_pointer,
)
from cross_gpu.models import BackendDeviceInfo, DeviceCapabilities

HIP_LIB_CANDIDATES: tuple[str, ...] = (
    "libamdhip64.so",
    "libamdhip64.so.6",
    "libamdhip64.so.5",
)
AMD_PCI_VENDOR_ID = 0x1002

HIP_SUCCESS = 0
HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2
HIP_MEMCPY_DEVICE_TO_DEVICE = 3

HIP_HARDWARE_FAULT_CODES: frozenset[int] = frozenset(
    {
        214,  # hipErrorECCNotCorrectable
        700,  # hipErrorIllegalAddress
        702,  # hipErrorLaunchTimeOut
        709,  # hipErrorContextIsDestroyed
        710,  # hipErrorAssert
        714,  # hipErrorHardwareStackError
        715,  # hipErrorIllegalInstruction
        716,  # hipErrorMisalignedAddress
        719,  # hipErrorLaunchFailure
    }
)


class _HipUuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


class _HipDevicePropR0600(ctypes.Structure):
    """ROCm 6.0+ hipDeviceProp_tR0600 ABI layout."""

    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("uuid", _HipUuid),
        ("luid", ctypes.c_char * 8),
        ("luidDeviceNodeMask", ctypes.c_uint),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("memoryClockRate", ctypes.c_int),
        ("memoryBusWidth", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("textureAlignment", ctypes.c_size_t),
        ("texturePitchAlignment", ctypes.c_size_t),
        ("deviceOverlap", ctypes.c_int),
        ("multiProcessorCount", ctypes.c_int),
        ("kernelExecTimeoutEnabled", ctypes.c_int),
        ("integrated", ctypes.c_int),
        ("canMapHostMemory", ctypes.c_int),
        ("computeMode", ctypes.c_int),
        ("maxTexture1D", ctypes.c_int),
        ("maxTexture1DMipmap", ctypes.c_int),
        ("maxTexture1DLinear", ctypes.c_int),
        ("maxTexture2D", ctypes.c_int * 2),
        ("maxTexture2DMipmap", ctypes.c_int * 2),
        ("maxTexture2DLinear", ctypes.c_int * 3),
        ("maxTexture2DGather", ctypes.c_int * 2),
        ("maxTexture3D", ctypes.c_int * 3),
        ("maxTexture3DAlt", ctypes.c_int * 3),
        ("maxTextureCubemap", ctypes.c_int),
        ("maxTexture1DLayered", ctypes.c_int * 2),
        ("maxTexture2DLayered", ctypes.c_int * 3),
        ("maxTextureCubemapLayered", ctypes.c_int * 2),
        ("surfaceAlignment", ctypes.c_size_t),
        ("concurrentKernels", ctypes.c_int),
        ("ECCEnabled", ctypes.c_int),
        ("pciBusID", ctypes.c_int),
        ("pciDeviceID", ctypes.c_int),
        ("pciDomainID", ctypes.c_int),
        ("tccDriver", ctypes.c_int),
        ("asyncEngineCount", ctypes.c_int),
        ("unifiedAddressing", ctypes.c_int),
        ("memoryClockRateDup", ctypes.c_int),
        ("memoryBusWidthDup", ctypes.c_int),
        ("l2CacheSize", ctypes.c_int),
        ("maxThreadsPerMultiProcessor", ctypes.c_int),
        ("arch", ctypes.c_uint32),
        ("maxSharedMemoryPerMultiProcessor", ctypes.c_size_t),
        ("isMultiGpuBoard", ctypes.c_int),
        ("gcnArch", ctypes.c_int),
        ("gcnArchName", ctypes.c_char * 256),
        ("_tail_pad", ctypes.c_ubyte * 512),
    ]


class _HipDevicePropLegacy(ctypes.Structure):
    """ROCm 5.x hipDeviceProp_t ABI layout (no uuid/luid header fields)."""

    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("textureAlignment", ctypes.c_size_t),
        ("texturePitchAlignment", ctypes.c_size_t),
        ("deviceOverlap", ctypes.c_int),
        ("multiProcessorCount", ctypes.c_int),
        ("kernelExecTimeoutEnabled", ctypes.c_int),
        ("integrated", ctypes.c_int),
        ("canMapHostMemory", ctypes.c_int),
        ("computeMode", ctypes.c_int),
        ("maxTexture1D", ctypes.c_int),
        ("maxTexture2D", ctypes.c_int * 2),
        ("maxTexture3D", ctypes.c_int * 3),
        ("maxTexture1DLayered", ctypes.c_int * 2),
        ("maxTexture2DLayered", ctypes.c_int * 3),
        ("surfaceAlignment", ctypes.c_size_t),
        ("concurrentKernels", ctypes.c_int),
        ("ECCEnabled", ctypes.c_int),
        ("pciBusID", ctypes.c_int),
        ("pciDeviceID", ctypes.c_int),
        ("pciDomainID", ctypes.c_int),
        ("tccDriver", ctypes.c_int),
        ("computePreemptionSupported", ctypes.c_int),
        ("asyncEngineCount", ctypes.c_int),
        ("unifiedAddressing", ctypes.c_int),
        ("memoryClockRate", ctypes.c_int),
        ("memoryBusWidth", ctypes.c_int),
        ("l2CacheSize", ctypes.c_int),
        ("maxThreadsPerMultiProcessor", ctypes.c_int),
        ("arch", ctypes.c_uint32),
        ("maxSharedMemoryPerMultiProcessor", ctypes.c_size_t),
        ("isMultiGpuBoard", ctypes.c_int),
        ("gcnArch", ctypes.c_int),
        ("gcnArchName", ctypes.c_char * 256),
        ("_tail_pad", ctypes.c_ubyte * 512),
    ]


class _HipApi:
    """Explicitly typed ctypes bindings for the HIP runtime API."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_size_p = ctypes.POINTER(ctypes.c_size_t)
        c_void_pp = ctypes.POINTER(ctypes.c_void_p)

        self.hipGetDeviceCount = bind_c_symbol(
            lib, "hipGetDeviceCount", ctypes.c_int, [c_int_p]
        )
        self.hipSetDevice = bind_c_symbol(
            lib, "hipSetDevice", ctypes.c_int, [ctypes.c_int]
        )
        self.hipDriverGetVersion = bind_optional_c_symbol(
            lib, "hipDriverGetVersion", ctypes.c_int, [c_int_p]
        )
        self.hipRuntimeGetVersion = bind_optional_c_symbol(
            lib, "hipRuntimeGetVersion", ctypes.c_int, [c_int_p]
        )
        self.hipGetDevicePropertiesR0600 = bind_optional_c_symbol(
            lib,
            "hipGetDevicePropertiesR0600",
            ctypes.c_int,
            [ctypes.POINTER(_HipDevicePropR0600), ctypes.c_int],
        )
        self.hipGetDeviceProperties = bind_c_symbol(
            lib, "hipGetDeviceProperties", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]
        )
        self.hipDeviceGetPCIBusId = bind_optional_c_symbol(
            lib,
            "hipDeviceGetPCIBusId",
            ctypes.c_int,
            [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        )
        self.hipDeviceGetUuid = bind_optional_c_symbol(
            lib,
            "hipDeviceGetUuid",
            ctypes.c_int,
            [ctypes.POINTER(_HipUuid), ctypes.c_int],
        )
        self.hipMemGetInfo = bind_optional_c_symbol(
            lib, "hipMemGetInfo", ctypes.c_int, [c_size_p, c_size_p]
        )
        self.hipDeviceSynchronize = bind_c_symbol(
            lib, "hipDeviceSynchronize", ctypes.c_int, []
        )
        self.hipMalloc = bind_c_symbol(
            lib, "hipMalloc", ctypes.c_int, [c_void_pp, ctypes.c_size_t]
        )
        self.hipFree = bind_c_symbol(lib, "hipFree", ctypes.c_int, [ctypes.c_void_p])
        self.hipHostMalloc = bind_optional_c_symbol(
            lib, "hipHostMalloc", ctypes.c_int, [c_void_pp, ctypes.c_size_t, ctypes.c_uint]
        )
        self.hipHostFree = bind_optional_c_symbol(
            lib, "hipHostFree", ctypes.c_int, [ctypes.c_void_p]
        )
        self.hipMemcpy = bind_c_symbol(
            lib,
            "hipMemcpy",
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
        )
        self.hipModuleLoadData = bind_c_symbol(
            lib, "hipModuleLoadData", ctypes.c_int, [c_void_pp, ctypes.c_void_p]
        )
        self.hipModuleUnload = bind_c_symbol(
            lib, "hipModuleUnload", ctypes.c_int, [ctypes.c_void_p]
        )
        self.hipModuleGetFunction = bind_c_symbol(
            lib,
            "hipModuleGetFunction",
            ctypes.c_int,
            [c_void_pp, ctypes.c_void_p, ctypes.c_char_p],
        )
        self.hipModuleLaunchKernel = bind_c_symbol(
            lib,
            "hipModuleLaunchKernel",
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
        self.hipGetErrorString = bind_optional_c_symbol(
            lib, "hipGetErrorString", ctypes.c_char_p, [ctypes.c_int]
        )


def _hip_capabilities(gcn_arch: str) -> DeviceCapabilities:
    arch = gcn_arch.split(":", 1)[0].lower()
    m = re.match(r"gfx([0-9a-f]+)", arch)
    gfx_num = int(m.group(1), 16) if m else 0
    is_cdna = 0x908 <= gfx_num <= 0x95F
    is_rdna3_or_newer = gfx_num >= 0x1100
    is_cdna3_or_rdna4 = (0x940 <= gfx_num <= 0x95F) or gfx_num >= 0x1200

    return DeviceCapabilities(
        fp64=True,
        fp32=True,
        fp16=gfx_num >= 0x900 or gfx_num == 0,
        bf16=is_cdna or is_rdna3_or_newer,
        fp8=is_cdna3_or_rdna4,
        fp4=gfx_num >= 0x950,
        int64=True,
        int32=True,
        int16=True,
        int8=gfx_num >= 0x906 or is_rdna3_or_newer,
        int4=False,
        tensor_fp32=(0x90A <= gfx_num <= 0x95F),
        tensor_fp16=is_cdna or is_rdna3_or_newer,
        tensor_bf16=is_cdna or is_rdna3_or_newer,
        tensor_int8=is_cdna or is_rdna3_or_newer,
    )


class HipBackend(BackendPlugin):
    """Native AMD HIP runtime backend plugin."""

    name = "hip"
    priority = NATIVE_BACKEND_PRIORITY

    def __init__(self) -> None:
        super().__init__()
        self._api: _HipApi | None = None

    def is_available(self) -> bool:
        if self._api is not None:
            return True
        try:
            lib = load_shared_library(HIP_LIB_CANDIDATES)
            self._api = _HipApi(lib)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def _require_api(self) -> _HipApi:
        if not self.is_available() or self._api is None:
            raise BackendInitError(self._load_error or "HIP runtime unavailable")
        return self._api

    def _error_message(self, api: _HipApi, rc: int) -> str:
        if api.hipGetErrorString is not None:
            raw = api.hipGetErrorString(rc)
            if raw:
                return raw.decode("utf-8", errors="replace")
        return f"hipError_t({rc})"

    def _check(self, api: _HipApi, rc: int, action: str) -> None:
        if rc == HIP_SUCCESS:
            return
        detail = f"{action} failed: {self._error_message(api, rc)} (code={rc})"
        if rc in HIP_HARDWARE_FAULT_CODES:
            raise BackendHardwareError(detail)
        raise BackendInitError(detail)

    def enumerate_devices(self) -> list[BackendDeviceInfo]:
        api = self._require_api()
        count = ctypes.c_int(0)
        rc = api.hipGetDeviceCount(ctypes.byref(count))
        if rc != HIP_SUCCESS:
            return []

        drv_ver = ""
        ver_int = ctypes.c_int(0)
        if api.hipRuntimeGetVersion is not None and api.hipRuntimeGetVersion(ctypes.byref(ver_int)) == HIP_SUCCESS:
            v = ver_int.value
            drv_ver = f"{v // 10000000}.{(v // 100000) % 100}.{v % 100000}"

        devices: list[BackendDeviceInfo] = []
        for idx in range(count.value):
            if api.hipGetDevicePropertiesR0600 is not None:
                prop = _HipDevicePropR0600()
                self._check(
                    api,
                    api.hipGetDevicePropertiesR0600(ctypes.byref(prop), idx),
                    f"hipGetDevicePropertiesR0600({idx})",
                )
            else:
                prop = _HipDevicePropLegacy()
                self._check(
                    api,
                    api.hipGetDeviceProperties(ctypes.byref(prop), idx),
                    f"hipGetDeviceProperties({idx})",
                )

            name = prop.name.decode("utf-8", errors="replace").strip()
            gcn_arch = prop.gcnArchName.decode("utf-8", errors="replace").split("\x00", 1)[0].strip()

            pci_bdf: str | None = None
            if api.hipDeviceGetPCIBusId is not None:
                bdf_buf = ctypes.create_string_buffer(32)
                if api.hipDeviceGetPCIBusId(bdf_buf, len(bdf_buf), idx) == HIP_SUCCESS:
                    pci_bdf = bdf_buf.value.decode("ascii", errors="ignore").strip().lower() or None
            if pci_bdf is None:
                pci_bdf = format_pci_bdf(prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, 0)

            uuid_str: str | None = None
            if api.hipDeviceGetUuid is not None:
                raw_uuid = _HipUuid()
                if api.hipDeviceGetUuid(ctypes.byref(raw_uuid), idx) == HIP_SUCCESS:
                    uuid_str = format_uuid_bytes(bytes(raw_uuid.bytes))

            devices.append(
                BackendDeviceInfo(
                    backend=self.name,
                    priority=self.priority,
                    backend_device_index=idx,
                    native_handle=idx,
                    name=name,
                    vendor="AMD",
                    vendor_id=AMD_PCI_VENDOR_ID,
                    architecture=gcn_arch.split(":", 1)[0] if gcn_arch else f"gfx{prop.gcnArch}",
                    compute_capability=f"{prop.major}.{prop.minor}",
                    total_memory_bytes=int(prop.totalGlobalMem),
                    free_memory_bytes=None,
                    compute_units=int(prop.multiProcessorCount),
                    driver_version=drv_ver,
                    pci_bdf=pci_bdf,
                    uuid=uuid_str,
                    capabilities=_hip_capabilities(gcn_arch),
                    extra={
                        "warp_size": int(prop.warpSize),
                        "max_threads_per_block": int(prop.maxThreadsPerBlock),
                        "gcn_arch_name": gcn_arch,
                    },
                )
            )
        return devices

    def _activate_device(self, api: _HipApi, ctx: BackendContext) -> None:
        self._check(api, api.hipSetDevice(int(ctx.handle)), f"hipSetDevice({ctx.handle})")

    def create_context(self, device: BackendDeviceInfo) -> BackendContext:
        api = self._require_api()
        dev_idx = int(device.native_handle)
        self._check(api, api.hipSetDevice(dev_idx), f"hipSetDevice({dev_idx})")
        if api.hipMemGetInfo is not None:
            free_b = ctypes.c_size_t(0)
            total_b = ctypes.c_size_t(0)
            if api.hipMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b)) == HIP_SUCCESS:
                device.free_memory_bytes = int(free_b.value)
        return BackendContext(backend=self.name, device_info=device, handle=dev_idx)

    def destroy_context(self, ctx: BackendContext) -> None:
        ctx.handle = None

    def synchronize(self, ctx: BackendContext) -> None:
        api = self._require_api()
        self._activate_device(api, ctx)
        self._check(api, api.hipDeviceSynchronize(), "hipDeviceSynchronize")

    def alloc_device(self, ctx: BackendContext, size_bytes: int) -> DeviceBuffer:
        api = self._require_api()
        self._activate_device(api, ctx)
        ptr = ctypes.c_void_p()
        self._check(
            api,
            api.hipMalloc(ctypes.byref(ptr), max(1, size_bytes)),
            "hipMalloc",
        )
        addr = int(ptr.value or 0)
        return DeviceBuffer(backend=self.name, handle=addr, size_bytes=size_bytes, address=addr)

    def free_device(self, ctx: BackendContext, buf: DeviceBuffer) -> None:
        if not buf.handle or self._api is None:
            return
        self._activate_device(self._api, ctx)
        self._api.hipFree(ctypes.c_void_p(int(buf.handle)))
        buf.handle = 0
        buf.address = 0

    def alloc_pinned_host(self, ctx: BackendContext, size_bytes: int) -> HostBuffer:
        api = self._require_api()
        self._activate_device(api, ctx)
        if api.hipHostMalloc is None:
            raise BackendInitError("hipHostMalloc symbol unavailable")
        ptr = ctypes.c_void_p()
        self._check(
            api,
            api.hipHostMalloc(ctypes.byref(ptr), max(1, size_bytes), 0),
            "hipHostMalloc",
        )
        addr = int(ptr.value or 0)
        return HostBuffer(backend=self.name, handle=addr, size_bytes=size_bytes, pointer=addr)

    def free_pinned_host(self, ctx: BackendContext, buf: HostBuffer) -> None:
        if not buf.handle or self._api is None or self._api.hipHostFree is None:
            return
        self._activate_device(self._api, ctx)
        self._api.hipHostFree(ctypes.c_void_p(int(buf.handle)))
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
        self._activate_device(api, ctx)
        src_ptr, keepalive = resolve_host_pointer(src)
        self._check(
            api,
            api.hipMemcpy(
                ctypes.c_void_p(int(dst.handle)),
                src_ptr,
                size_bytes,
                HIP_MEMCPY_HOST_TO_DEVICE,
            ),
            "hipMemcpy(H2D)",
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
        self._activate_device(api, ctx)
        dst_ptr, keepalive = resolve_host_pointer(dst)
        self._check(
            api,
            api.hipMemcpy(
                dst_ptr,
                ctypes.c_void_p(int(src.handle)),
                size_bytes,
                HIP_MEMCPY_DEVICE_TO_HOST,
            ),
            "hipMemcpy(D2H)",
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
        self._activate_device(api, ctx)
        self._check(
            api,
            api.hipMemcpy(
                ctypes.c_void_p(int(dst.handle)),
                ctypes.c_void_p(int(src.handle)),
                size_bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            ),
            "hipMemcpy(D2D)",
        )

    def load_module(self, ctx: BackendContext, compiled_binary: Any) -> KernelModule:
        api = self._require_api()
        self._activate_device(api, ctx)
        raw = bytes(compiled_binary.binary_bytes)
        image_buf = ctypes.create_string_buffer(raw, len(raw))
        mod_ptr = ctypes.c_void_p()
        self._check(
            api,
            api.hipModuleLoadData(
                ctypes.byref(mod_ptr),
                ctypes.cast(image_buf, ctypes.c_void_p),
            ),
            "hipModuleLoadData",
        )
        return KernelModule(backend=self.name, handle=mod_ptr.value)

    def unload_module(self, ctx: BackendContext, module: KernelModule) -> None:
        if module.handle is None or self._api is None:
            return
        self._activate_device(self._api, ctx)
        self._api.hipModuleUnload(module.handle)
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
        self._activate_device(api, ctx)

        func_handle = module.functions.get(entry_point)
        if func_handle is None:
            fn_ptr = ctypes.c_void_p()
            self._check(
                api,
                api.hipModuleGetFunction(
                    ctypes.byref(fn_ptr),
                    module.handle,
                    entry_point.encode("utf-8"),
                ),
                f"hipModuleGetFunction({entry_point})",
            )
            func_handle = fn_ptr.value
            module.functions[entry_point] = func_handle

        packed_storage: list[Any] = []
        param_ptrs = (ctypes.c_void_p * max(1, len(args)))()
        for index, raw_arg in enumerate(args):
            coerced = coerce_kernel_arg(raw_arg)
            if isinstance(coerced, DeviceBuffer):
                c_val = ctypes.c_void_p(int(coerced.handle))
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
            api.hipModuleLaunchKernel(
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
            f"hipModuleLaunchKernel({entry_point})",
        )
