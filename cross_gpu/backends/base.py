"""Abstract backend plugin interface, ctypes ABI helpers, and buffer handles."""

from __future__ import annotations

from abc import ABC, abstractmethod
import ctypes
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from cross_gpu.exit_codes import BackendHardwareError, BackendInitError
from cross_gpu.models import BackendDeviceInfo

NATIVE_BACKEND_PRIORITY = 100
OPENCL_BACKEND_PRIORITY = 50

DiagCallback = Callable[[str], None]


@dataclass
class BackendContext:
    """Active context/queue state bound to a physical backend device."""

    backend: str
    device_info: BackendDeviceInfo
    handle: Any
    queue: Any = None
    command_list: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceBuffer:
    """Device-side memory allocation handle."""

    backend: str
    handle: Any
    size_bytes: int
    address: int = 0


@dataclass
class HostBuffer:
    """Pinned or host-accessible memory allocation handle."""

    backend: str
    handle: Any
    size_bytes: int
    pointer: int = 0


@dataclass
class KernelModule:
    """Loaded native shader/compute module and cached function handles."""

    backend: str
    handle: Any
    functions: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def load_shared_library(
    candidates: Sequence[str],
    *,
    diag_fn: DiagCallback | None = None,
    label: str = "",
) -> ctypes.CDLL:
    """Load the first available shared library from candidate names/paths."""
    last_err: OSError | None = None
    prefix = f"probe {label}: " if label else "probe: "
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
            if diag_fn is not None:
                diag_fn(f"{prefix}loaded {candidate}")
            return lib
        except OSError as exc:
            last_err = exc
            if diag_fn is not None:
                diag_fn(f"{prefix}{candidate} failed ({exc})")
    primary = candidates[0] if candidates else "<none>"
    raise OSError(f"{primary} not found (tried {', '.join(candidates)}: {last_err})")


def check_linux_render_node_hint(
    *,
    platform_name: str | None = None,
    dri_dir: Path | None = None,
    kfd_path: Path | None = None,
    include_kfd: bool = False,
) -> str | None:
    """Detect Linux /dev/dri/renderD* (and optional /dev/kfd) permission denials."""
    active_platform = sys.platform if platform_name is None else platform_name
    if not active_platform.startswith("linux"):
        return None

    if include_kfd:
        kfd = Path("/dev/kfd") if kfd_path is None else kfd_path
        if kfd.exists() and not os.access(kfd, os.R_OK | os.W_OK):
            return f"{kfd} permission denied — check 'render'/'video' group membership"

    root = Path("/dev/dri") if dri_dir is None else dri_dir
    if not root.is_dir():
        return None
    try:
        render_nodes = sorted(root.glob("renderD*"))
    except OSError:
        return None
    if not render_nodes:
        return None
    accessible = [node for node in render_nodes if os.access(node, os.R_OK | os.W_OK)]
    if not accessible:
        return f"{render_nodes[0]} permission denied — check 'render'/'video' group membership"
    return None


def bind_c_symbol(
    lib: ctypes.CDLL,
    name: str,
    restype: Any,
    argtypes: list[Any],
) -> Any:
    """Bind a native C symbol with explicit ABI argtypes and restype."""
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def bind_optional_c_symbol(
    lib: ctypes.CDLL,
    name: str,
    restype: Any,
    argtypes: list[Any],
) -> Any | None:
    """Bind an optional native C symbol if exported by the loaded library."""
    fn = getattr(lib, name, None)
    if fn is None:
        return None
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def format_pci_bdf(domain: int, bus: int, device: int, function: int) -> str:
    """Format PCIe components into canonical lowercase domain:bus:device.function."""
    return f"{domain & 0xFFFF:04x}:{bus & 0xFF:02x}:{device & 0xFF:02x}.{function & 0x7:x}"


def format_uuid_bytes(raw_bytes: bytes | Sequence[int]) -> str | None:
    """Convert a 16-byte UUID array to canonical lowercase hyphenated UUID."""
    data = bytes(raw_bytes)
    if len(data) != 16 or all(byte == 0 for byte in data):
        return None
    hex_str = data.hex()
    return (
        f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-"
        f"{hex_str[16:20]}-{hex_str[20:32]}"
    )


def resolve_host_pointer(
    buf: HostBuffer | ctypes.c_void_p | ctypes.Array[Any] | bytes | bytearray | memoryview | int,
) -> tuple[ctypes.c_void_p, Any]:
    """Return a c_void_p and backing object to keep alive during C calls."""
    if isinstance(buf, HostBuffer):
        return ctypes.c_void_p(buf.pointer), buf
    if isinstance(buf, int):
        return ctypes.c_void_p(buf), None
    if isinstance(buf, ctypes.c_void_p):
        return buf, buf
    if isinstance(buf, bytes):
        c_buf = ctypes.create_string_buffer(buf, len(buf))
        return ctypes.cast(c_buf, ctypes.c_void_p), c_buf
    if isinstance(buf, bytearray):
        arr = (ctypes.c_ubyte * len(buf)).from_buffer(buf)
        return ctypes.cast(arr, ctypes.c_void_p), arr
    if isinstance(buf, memoryview):
        arr = (ctypes.c_ubyte * buf.nbytes).from_buffer(buf)
        return ctypes.cast(arr, ctypes.c_void_p), arr
    if isinstance(buf, ctypes.Array):
        return ctypes.cast(buf, ctypes.c_void_p), buf
    return ctypes.cast(ctypes.byref(buf), ctypes.c_void_p), buf


def coerce_kernel_arg(arg: Any) -> Any:
    """Normalize a Python/buffer kernel argument into a ctypes value or DeviceBuffer."""
    if isinstance(arg, DeviceBuffer):
        return arg
    if isinstance(arg, ctypes._SimpleCData) or isinstance(arg, (ctypes.Structure, ctypes.Array)):
        return arg
    if isinstance(arg, bool):
        return ctypes.c_uint32(1 if arg else 0)
    if isinstance(arg, int):
        if -0x80000000 <= arg <= 0x7FFFFFFF:
            return ctypes.c_int32(arg)
        if 0 <= arg <= 0xFFFFFFFF:
            return ctypes.c_uint32(arg)
        return ctypes.c_uint64(arg) if arg >= 0 else ctypes.c_int64(arg)
    if isinstance(arg, float):
        return ctypes.c_float(arg)
    raise TypeError(f"Unsupported kernel argument type: {type(arg).__name__}")


class BackendPlugin(ABC):
    """Abstract base class implemented by all 5 GPU backend plugins."""

    name: str = ""
    priority: int = NATIVE_BACKEND_PRIORITY

    def __init__(self) -> None:
        self._load_error: str | None = None
        self._probe_log: list[str] = []

    @property
    def load_error(self) -> str | None:
        """Diagnostic message explaining why this backend is unavailable, if any."""
        return self._load_error

    @property
    def probe_log(self) -> tuple[str, ...]:
        """Recorded SONAME/framework probe attempts for --verbose diagnostics."""
        return tuple(self._probe_log)

    def _record_probe(self, message: str) -> None:
        self._probe_log.append(message)

    @abstractmethod
    def is_available(self) -> bool:
        """Safely probe and load native runtime libraries via ctypes."""

    @abstractmethod
    def enumerate_devices(self) -> list[BackendDeviceInfo]:
        """Enumerate hardware devices exposed by this backend."""

    @abstractmethod
    def create_context(self, device: BackendDeviceInfo) -> BackendContext:
        """Create an execution context/command queue on the target device."""

    @abstractmethod
    def destroy_context(self, ctx: BackendContext) -> None:
        """Release all native context/queue handles."""

    @abstractmethod
    def synchronize(self, ctx: BackendContext) -> None:
        """Block until all queued GPU operations on the context complete."""

    @abstractmethod
    def alloc_device(self, ctx: BackendContext, size_bytes: int) -> DeviceBuffer:
        """Allocate device-resident GPU memory."""

    @abstractmethod
    def free_device(self, ctx: BackendContext, buf: DeviceBuffer) -> None:
        """Free a device-resident memory buffer."""

    @abstractmethod
    def alloc_pinned_host(self, ctx: BackendContext, size_bytes: int) -> HostBuffer:
        """Allocate page-locked / shared host memory."""

    @abstractmethod
    def free_pinned_host(self, ctx: BackendContext, buf: HostBuffer) -> None:
        """Free a host memory buffer allocated by alloc_pinned_host."""

    @abstractmethod
    def memcpy_h2d(
        self,
        ctx: BackendContext,
        dst: DeviceBuffer,
        src: HostBuffer | ctypes.c_void_p | bytes | bytearray | memoryview | int,
        size_bytes: int,
    ) -> None:
        """Copy bytes from host memory to a device buffer."""

    @abstractmethod
    def memcpy_d2h(
        self,
        ctx: BackendContext,
        dst: HostBuffer | ctypes.c_void_p | bytearray | memoryview | int,
        src: DeviceBuffer,
        size_bytes: int,
    ) -> None:
        """Copy bytes from a device buffer to host memory."""

    @abstractmethod
    def memcpy_d2d(
        self,
        ctx: BackendContext,
        dst: DeviceBuffer,
        src: DeviceBuffer,
        size_bytes: int,
    ) -> None:
        """Copy bytes between two device buffers."""

    @abstractmethod
    def load_module(self, ctx: BackendContext, compiled_binary: Any) -> KernelModule:
        """Load a compiled kernel binary/source into the backend runtime."""

    @abstractmethod
    def unload_module(self, ctx: BackendContext, module: KernelModule) -> None:
        """Unload a loaded kernel module."""

    @abstractmethod
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
        """Launch a compute kernel on the context."""


__all__ = [
    "NATIVE_BACKEND_PRIORITY",
    "OPENCL_BACKEND_PRIORITY",
    "BackendContext",
    "BackendHardwareError",
    "BackendInitError",
    "BackendPlugin",
    "DeviceBuffer",
    "DiagCallback",
    "HostBuffer",
    "KernelModule",
    "bind_c_symbol",
    "bind_optional_c_symbol",
    "check_linux_render_node_hint",
    "coerce_kernel_arg",
    "format_pci_bdf",
    "format_uuid_bytes",
    "load_shared_library",
    "resolve_host_pointer",
]
