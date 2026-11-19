# SPDX-License-Identifier: MIT
"""AMD HIP runtime API (libamdhip64.so) backend plugin via ctypes."""

from __future__ import annotations

import ctypes
import re
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
    check_linux_render_node_hint,
    coerce_kernel_arg,
    format_pci_bdf,
    format_uuid_bytes,
    load_shared_library,
    resolve_host_pointer,
)
from ..models import BackendDeviceInfo, DeviceCapabilities

HIP_LIB_CANDIDATES: tuple[str, ...] = (
    "libamdhip64.so.6",
    "libamdhip64.so.5",
    "libamdhip64.so",
)
AMD_PCI_VENDOR_ID = 0x1002

HIP_SUCCESS = 0
HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2
HIP_MEMCPY_DEVICE_TO_DEVICE = 3

# ABI-stable hipDeviceAttribute_t values verified identical across ROCm 5.x and 6.x.
HIP_DEVICE_ATTRIBUTE_INTEGRATED = 16
HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 23
HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 56
HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 61
HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 63
HIP_DEVICE_ATTRIBUTE_PCI_BUS_ID = 67
HIP_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 68
HIP_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 69
HIP_DEVICE_ATTRIBUTE_WARP_SIZE = 87

# Generous opaque buffer size for hipGetDeviceProperties* when scanning gcnArchName only.
_HIP_PROP_SCAN_BUF_SIZE = 8192
_GCN_ARCH_TOKEN_RE = re.compile(rb"^gfx[0-9a-fA-F]{3,5}(?::[a-zA-Z0-9_+-]+)*$")

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


def _extract_gcn_arch_name(raw_bytes: bytes) -> str:
    """Extract null-terminated gfx* architecture string from an opaque hipDeviceProp_t buffer."""
    for token in raw_bytes.split(b"\x00"):
        if token and _GCN_ARCH_TOKEN_RE.match(token):
            return token.decode("ascii", errors="ignore")
    return ""


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
        self.hipDeviceGetName = bind_c_symbol(
            lib, "hipDeviceGetName", ctypes.c_int, [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        )
        self.hipDeviceTotalMem = bind_c_symbol(
            lib, "hipDeviceTotalMem", ctypes.c_int, [c_size_p, ctypes.c_int]
        )
        self.hipDeviceGetAttribute = bind_c_symbol(
            lib, "hipDeviceGetAttribute", ctypes.c_int, [c_int_p, ctypes.c_int, ctypes.c_int]
        )
        self.hipGetDevicePropertiesR0600 = bind_optional_c_symbol(
            lib,
            "hipGetDevicePropertiesR0600",
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_int],
        )
        self.hipGetDeviceProperties = bind_optional_c_symbol(
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
        self._probe_log.clear()
        try:
            lib = load_shared_library(
                HIP_LIB_CANDIDATES,
                diag_fn=self._record_probe,
                label=self.name,
            )
            self._api = _HipApi(lib)
            self._load_error = None
            return True
        except OSError:
            self._load_error = f"{HIP_LIB_CANDIDATES[0]} not found"
            return False
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

    def _get_attr(self, api: _HipApi, dev_idx: int, attr: int) -> int | None:
        val = ctypes.c_int(0)
        rc = api.hipDeviceGetAttribute(ctypes.byref(val), attr, dev_idx)
        return int(val.value) if rc == HIP_SUCCESS else None

    def _probe_gcn_arch_name(self, api: _HipApi, dev_idx: int) -> str:
        """Safely scan an oversized zeroed property buffer for the gfx* arch token."""
        prop_fn = api.hipGetDevicePropertiesR0600 or api.hipGetDeviceProperties
        if prop_fn is None:
            return ""
        raw_buf = ctypes.create_string_buffer(_HIP_PROP_SCAN_BUF_SIZE)
        if prop_fn(ctypes.cast(raw_buf, ctypes.c_void_p), dev_idx) != HIP_SUCCESS:
            return ""
        return _extract_gcn_arch_name(raw_buf.raw)

    def enumerate_devices(self) -> list[BackendDeviceInfo]:
        api = self._require_api()
        count = ctypes.c_int(0)
        rc = api.hipGetDeviceCount(ctypes.byref(count))
        if rc != HIP_SUCCESS or count.value == 0:
            perm_hint = check_linux_render_node_hint(include_kfd=True)
            if perm_hint:
                self._load_error = perm_hint
            elif rc != HIP_SUCCESS:
                self._load_error = f"hipGetDeviceCount failed: {self._error_message(api, rc)} (code={rc})"
            else:
                self._load_error = "no HIP devices detected"
            return []

        drv_ver = ""
        ver_int = ctypes.c_int(0)
        if api.hipRuntimeGetVersion is not None and api.hipRuntimeGetVersion(ctypes.byref(ver_int)) == HIP_SUCCESS:
            v = ver_int.value
            drv_ver = f"{v // 10000000}.{(v // 100000) % 100}.{v % 100000}"

        devices: list[BackendDeviceInfo] = []
        for idx in range(count.value):
            name_buf = ctypes.create_string_buffer(256)
            self._check(
                api,
                api.hipDeviceGetName(name_buf, len(name_buf), idx),
                f"hipDeviceGetName({idx})",
            )
            name = name_buf.value.decode("utf-8", errors="replace").strip()

            total_mem = ctypes.c_size_t(0)
            self._check(
                api,
                api.hipDeviceTotalMem(ctypes.byref(total_mem), idx),
                f"hipDeviceTotalMem({idx})",
            )

            major = self._get_attr(api, idx, HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) or 0
            minor = self._get_attr(api, idx, HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) or 0
            cu_count = self._get_attr(api, idx, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) or 0
            warp_size = self._get_attr(api, idx, HIP_DEVICE_ATTRIBUTE_WARP_SIZE) or 64
            max_threads = self._get_attr(api, idx, HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) or 256
            integrated = bool(self._get_attr(api, idx, HIP_DEVICE_ATTRIBUTE_INTEGRATED) or 0)

            gcn_arch = self._probe_gcn_arch_name(api, idx)
            short_arch = gcn_arch.split(":", 1)[0] if gcn_arch else (
                f"gfx{major}{minor:x}" if major else ""
            )

            pci_bdf: str | None = None
            if api.hipDeviceGetPCIBusId is not None:
                bdf_buf = ctypes.create_string_buffer(32)
                if api.hipDeviceGetPCIBusId(bdf_buf, len(bdf_buf), idx) == HIP_SUCCESS:
                    pci_bdf = bdf_buf.value.decode("ascii", errors="ignore").strip().lower() or None
            if pci_bdf is None:
                domain = self._get_attr(api, idx, HIP_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
                bus = self._get_attr(api, idx, HIP_DEVICE_ATTRIBUTE_PCI_BUS_ID)
                pci_dev = self._get_attr(api, idx, HIP_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
                if domain is not None and bus is not None and pci_dev is not None:
                    pci_bdf = format_pci_bdf(domain, bus, pci_dev, 0)

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
                    architecture=short_arch,
                    compute_capability=f"{major}.{minor}" if major else "",
                    total_memory_bytes=int(total_mem.value),
                    free_memory_bytes=None,
                    compute_units=cu_count,
                    driver_version=drv_ver,
                    pci_bdf=pci_bdf,
                    uuid=uuid_str,
                    capabilities=_hip_capabilities(gcn_arch or short_arch),
                    extra={
                        "warp_size": warp_size,
                        "max_threads_per_block": max_threads,
                        "gcn_arch_name": gcn_arch,
                        "has_unified_memory": integrated,
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
        try:
            if ctx.handle is not None:
                self._api.hipSetDevice(int(ctx.handle))
            self._api.hipFree(ctypes.c_void_p(int(buf.handle)))
        finally:
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
        try:
            if ctx.handle is not None:
                self._api.hipSetDevice(int(ctx.handle))
            self._api.hipHostFree(ctypes.c_void_p(int(buf.handle)))
        finally:
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
        try:
            if ctx.handle is not None:
                self._api.hipSetDevice(int(ctx.handle))
            self._api.hipModuleUnload(module.handle)
        finally:
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
