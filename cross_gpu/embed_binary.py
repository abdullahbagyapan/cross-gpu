"""Runtime kernel loader (via importlib.resources), verifier CLI, and compiler adapters."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import importlib.resources as importlib_resources
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from cross_gpu import __version__
from cross_gpu.backends.base import (
    DiagCallback,
    bind_c_symbol,
    bind_optional_c_symbol,
    load_shared_library,
)
from cross_gpu.exit_codes import (
    ExitCode,
    KernelCompilationError,
    KernelNotFoundError,
    UnsupportedKernelError,
)
from cross_gpu.models import DeviceCapabilities
from cross_gpu.workloads import WORKLOAD_SPECS_BY_NAME, device_supports_workload

NVRTC_LIB_CANDIDATES: tuple[str, ...] = ("libnvrtc.so.12", "libnvrtc.so", "libnvrtc.so.11.2")
HIPRTC_LIB_CANDIDATES: tuple[str, ...] = (
    "libhiprtc.so.6",
    "libhiprtc.so",
    "libhiprtc.so.5",
    "libamdhip64.so.6",
    "libamdhip64.so",
)

BACKEND_SOURCE_CONFIG: dict[str, tuple[str, str]] = {
    "cuda": (".cu", "cuda_source"),
    "hip": (".hip", "hip_source"),
    "levelzero": (".cl", "opencl_source"),
    "metal": (".metal", "msl_source"),
    "opencl": (".cl", "opencl_source"),
}

KERNEL_WORKLOADS: tuple[str, ...] = (
    "bandwidth_d2d",
    "compute_bf16",
    "compute_fp16",
    "compute_fp32",
    "compute_fp4",
    "compute_fp64",
    "compute_fp8",
    "compute_int16",
    "compute_int32",
    "compute_int4",
    "compute_int64",
    "compute_int8",
    "compute_tensor_bf16",
    "compute_tensor_fp16",
    "compute_tensor_fp32",
    "compute_tensor_int8",
    "memory_pattern",
    "memory_sweep",
)

_MULTI_ENTRY_POINTS: dict[str, tuple[str, ...]] = {
    "memory_pattern": ("kernel_memory_pattern_write", "kernel_memory_pattern_verify"),
    "memory_sweep": ("kernel_memory_sweep_write", "kernel_memory_sweep_verify"),
}

SPIRV_MAGIC_LE = b"\x03\x02\x23\x07"
SPIRV_MAGIC_BE = b"\x07\x23\x02\x03"
METALLIB_MAGIC = b"MTLB"

_CAP_MACRO_RE = re.compile(r"^-DCROSS_GPU_HAS_([A-Z0-9_]+)=([01])$")
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_FLAGS_TOKEN_LEN = 96

MemoryCacheKey = tuple[str, str, str, str, tuple[str, ...], bytes]


@dataclass(frozen=True)
class KernelSpec:
    """Metadata describing one raw kernel source file bundled as package data."""

    backend: str
    workload: str
    filename: str
    format: str
    entry_point: str
    entry_points: tuple[str, ...]


@dataclass(frozen=True)
class KernelSource:
    """Loaded raw kernel source bytes/text resolved via importlib.resources."""

    backend: str
    workload: str
    filename: str
    format: str
    entry_point: str
    entry_points: tuple[str, ...]
    source_bytes: bytes
    source_text: str
    size_bytes: int


@dataclass
class CompiledKernelBinary:
    """Compiled or prepared kernel payload ready for BackendPlugin.load_module."""

    backend: str
    workload: str
    entry_point: str
    format: str
    binary_bytes: bytes
    compile_flags: tuple[str, ...] = ()
    compiler_log: str = ""
    cache_key: str = ""


def normalize_workload_name(workload: str) -> str:
    """Normalize a workload slug (hyphen or snake case) into canonical kernel stem."""
    return workload.strip().replace("-", "_")


def default_entry_points(workload: str) -> tuple[str, ...]:
    """Return all canonical kernel entry-point symbol names for a workload."""
    normalized = normalize_workload_name(workload)
    if normalized in _MULTI_ENTRY_POINTS:
        return _MULTI_ENTRY_POINTS[normalized]
    dashed = normalized.replace("_", "-")
    if dashed in WORKLOAD_SPECS_BY_NAME:
        return (WORKLOAD_SPECS_BY_NAME[dashed].default_entry_point,)
    return (f"kernel_{normalized}",)


def default_entry_point(workload: str) -> str:
    """Return primary C/kernel entry point name for a workload slug."""
    return default_entry_points(workload)[0]


def kernel_key(backend: str, workload: str) -> str:
    """Build deterministic display key for a (backend, workload) pair."""
    return f"{backend}:{normalize_workload_name(workload)}"


def _build_kernel_specs() -> dict[tuple[str, str], KernelSpec]:
    catalog: dict[tuple[str, str], KernelSpec] = {}
    for backend in sorted(BACKEND_SOURCE_CONFIG):
        ext, fmt = BACKEND_SOURCE_CONFIG[backend]
        for workload in KERNEL_WORKLOADS:
            eps = default_entry_points(workload)
            catalog[(backend, workload)] = KernelSpec(
                backend=backend,
                workload=workload,
                filename=f"{workload}{ext}",
                format=fmt,
                entry_point=eps[0],
                entry_points=eps,
            )
    return catalog


KERNEL_SPECS: dict[tuple[str, str], KernelSpec] = _build_kernel_specs()

_KERNEL_SOURCE_CACHE: dict[tuple[str, str], KernelSource] = {}


def clear_kernel_source_cache() -> None:
    """Clear the in-memory memoization cache of loaded KernelSource entries."""
    _KERNEL_SOURCE_CACHE.clear()


def _default_kernels_resource_root() -> Any:
    return importlib_resources.files("cross_gpu").joinpath("kernels")


def load_kernel_source(
    backend: str,
    workload: str,
    *,
    specs: Mapping[tuple[str, str], KernelSpec] | None = None,
    resource_root: Any | None = None,
    use_cache: bool = True,
) -> KernelSource:
    """Load raw UTF-8 kernel source directly from packaged cross_gpu/kernels/ data."""
    normalized = normalize_workload_name(workload)
    key = (backend, normalized)
    active_specs = KERNEL_SPECS if specs is None else specs
    spec = active_specs.get(key)
    if spec is None:
        raise KernelNotFoundError(
            f"No kernel registered for backend={backend!r}, workload={workload!r}"
        )

    cacheable = use_cache and specs is None and resource_root is None
    if cacheable and key in _KERNEL_SOURCE_CACHE:
        return _KERNEL_SOURCE_CACHE[key]

    root = _default_kernels_resource_root() if resource_root is None else resource_root
    resource = root.joinpath(spec.backend, spec.filename)
    try:
        source_bytes = resource.read_bytes()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, OSError) as exc:
        raise KernelNotFoundError(
            f"Missing kernel resource {spec.backend}/{spec.filename}: {exc}"
        ) from exc

    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KernelNotFoundError(
            f"Kernel resource {spec.backend}/{spec.filename} is not valid UTF-8: {exc}"
        ) from exc

    loaded = KernelSource(
        backend=spec.backend,
        workload=spec.workload,
        filename=spec.filename,
        format=spec.format,
        entry_point=spec.entry_point,
        entry_points=spec.entry_points,
        source_bytes=source_bytes,
        source_text=source_text,
        size_bytes=len(source_bytes),
    )
    if cacheable:
        _KERNEL_SOURCE_CACHE[key] = loaded
    return loaded


def verify_kernel_resources(
    *,
    specs: Mapping[tuple[str, str], KernelSpec] | None = None,
    resource_root: Any | None = None,
) -> list[str]:
    """Verify every registered kernel loads cleanly, declares entry points, and has no orphans."""
    active_specs = KERNEL_SPECS if specs is None else specs
    root = _default_kernels_resource_root() if resource_root is None else resource_root
    errors: list[str] = []

    registered_files: dict[str, set[str]] = {b: set() for b in BACKEND_SOURCE_CONFIG}
    for (backend, workload), spec in sorted(active_specs.items()):
        registered_files.setdefault(backend, set()).add(spec.filename)
        try:
            loaded = load_kernel_source(
                backend,
                workload,
                specs=active_specs,
                resource_root=root,
                use_cache=False,
            )
        except KernelNotFoundError as exc:
            errors.append(f"{backend}:{workload}: {exc}")
            continue

        if loaded.size_bytes == 0 or not loaded.source_text.strip():
            errors.append(f"{backend}:{workload}: kernel source file is empty")
            continue

        for ep in spec.entry_points:
            if ep not in loaded.source_text:
                errors.append(
                    f"{backend}:{workload}: missing declared entry point {ep!r} in {spec.filename}"
                )

    # Detect orphan kernel files under cross_gpu/kernels/<backend>/ not in KERNEL_SPECS.
    try:
        top_entries = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        errors.append(f"Unable to enumerate kernel resource directory: {exc}")
        return errors

    for entry in top_entries:
        name = entry.name
        if name.startswith(".") or name.lower() == "readme.md":
            continue
        if not entry.is_dir():
            errors.append(f"Orphan file in kernels root: {name}")
            continue
        if name not in registered_files:
            errors.append(f"Unknown backend kernel directory: {name}")
            continue
        expected_files = registered_files[name]
        for child in sorted(entry.iterdir(), key=lambda p: p.name):
            if child.name.startswith(".") or child.name.lower() == "readme.md":
                continue
            if child.name not in expected_files:
                errors.append(f"Orphan kernel file not in KERNEL_SPECS: {name}/{child.name}")

    return errors


def canonicalize_compile_flags(compile_flags: Sequence[str]) -> tuple[str, ...]:
    """Return canonical, whitespace-stripped compile flags tuple."""
    return tuple(str(flag).strip() for flag in compile_flags if str(flag).strip())


def compute_memory_cache_key(
    backend: str,
    target_arch: str,
    driver_version: str,
    workload: str,
    compile_flags: Sequence[str],
    source_bytes: bytes,
) -> MemoryCacheKey:
    """Build plain tuple key for in-memory compiled-kernel memoization (no hashing)."""
    return (
        backend,
        target_arch,
        driver_version,
        normalize_workload_name(workload),
        canonicalize_compile_flags(compile_flags),
        source_bytes,
    )


def _sanitize_token(value: str, fallback: str = "none") -> str:
    cleaned = _UNSAFE_FILENAME_RE.sub("_", value.strip()).strip("_.")
    return cleaned or fallback


def _encode_flags_for_filename(flags: tuple[str, ...]) -> str:
    if not flags:
        return "noflags"

    cap_bits: list[tuple[str, int]] = []
    other_tokens: list[str] = []
    for flag in flags:
        match = _CAP_MACRO_RE.match(flag)
        if match:
            cap_bits.append((match.group(1), int(match.group(2))))
        else:
            other_tokens.append(_sanitize_token(flag, fallback="flag"))

    parts: list[str] = []
    if cap_bits:
        cap_bits.sort(key=lambda item: item[0])
        bitmask = 0
        for idx, (_, enabled) in enumerate(cap_bits):
            if enabled:
                bitmask |= 1 << idx
        parts.append(f"caps{len(cap_bits)}x{bitmask:x}")
    if other_tokens:
        parts.append("_".join(other_tokens))

    joined = "_".join(parts)
    if len(joined) > _MAX_FLAGS_TOKEN_LEN:
        joined = f"{joined[:_MAX_FLAGS_TOKEN_LEN]}_n{len(flags)}"
    return joined


def compute_disk_cache_name(
    backend: str,
    target_arch: str,
    driver_version: str,
    workload: str,
    size_bytes: int,
    compile_flags: Sequence[str],
    *,
    package_version: str = __version__,
) -> str:
    """Build a deterministic, human-readable, filesystem-safe disk cache stem."""
    canonical_flags = canonicalize_compile_flags(compile_flags)
    return "__".join(
        (
            f"v{_sanitize_token(package_version, '0')}",
            _sanitize_token(backend, "backend"),
            _sanitize_token(normalize_workload_name(workload), "workload"),
            _sanitize_token(target_arch, "default"),
            _sanitize_token(driver_version, "unknown"),
            f"{int(size_bytes)}B",
            _encode_flags_for_filename(canonical_flags),
        )
    )


def _safe_home_dir(home: Path | None = None) -> Path | None:
    if home is not None:
        return home
    try:
        return Path.home()
    except (RuntimeError, OSError):
        return None


_TRUTHY_ENV_VALUES: frozenset[str] = frozenset({"1", "true", "yes"})


def is_disk_cache_disabled(env: Mapping[str, str] | None = None) -> bool:
    """Return True when CROSS_GPU_NO_CACHE is set to a truthy value (1, true, yes)."""
    active_env = os.environ if env is None else env
    return active_env.get("CROSS_GPU_NO_CACHE", "").strip().lower() in _TRUTHY_ENV_VALUES


def default_cache_dir(
    *,
    platform_name: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path | None:
    """Resolve the OS-standard runtime binary cache directory, or None if unavailable/disabled.

    - Disk cache disabled: CROSS_GPU_NO_CACHE=1|true|yes -> None (in-memory only)
    - Explicit override: $CROSS_GPU_CACHE_DIR
    - Explicit XDG override (Linux & macOS): $XDG_CACHE_HOME/cross-gpu (if absolute)
    - macOS default: ~/Library/Caches/cross-gpu
    - Linux/POSIX default: ~/.cache/cross-gpu
    """
    active_env = os.environ if env is None else env
    if is_disk_cache_disabled(active_env):
        return None

    explicit = active_env.get("CROSS_GPU_CACHE_DIR", "").strip()
    if explicit:
        return Path(explicit)

    xdg_cache = active_env.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        xdg_path = Path(xdg_cache)
        if xdg_path.is_absolute():
            return xdg_path / "cross-gpu"

    resolved_home = _safe_home_dir(home)
    if resolved_home is None:
        return None

    active_platform = sys.platform if platform_name is None else platform_name
    if active_platform == "darwin":
        return resolved_home / "Library" / "Caches" / "cross-gpu"
    return resolved_home / ".cache" / "cross-gpu"


class RuntimeBinaryCache:
    """In-memory tuple cache + best-effort metadata-validated disk cache."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._memory: dict[MemoryCacheKey, CompiledKernelBinary] = {}
        self._explicit_cache_dir: Path | None = cache_dir

    @property
    def cache_dir(self) -> Path | None:
        """Resolved disk cache directory, or None when disabled/unavailable."""
        if is_disk_cache_disabled():
            return None
        if self._explicit_cache_dir is not None:
            return self._explicit_cache_dir
        return default_cache_dir()

    def clear_memory(self) -> None:
        """Clear in-memory compiled-kernel cache."""
        self._memory.clear()

    def get_with_source(
        self,
        memory_key: MemoryCacheKey,
        disk_name: str | None = None,
        *,
        expected_meta: Mapping[str, Any] | None = None,
    ) -> tuple[CompiledKernelBinary | None, str | None]:
        """Look up compiled binary and return (entry, 'memory'|'disk'|None)."""
        if memory_key in self._memory:
            return self._memory[memory_key], "memory"
        active_dir = self.cache_dir
        if disk_name is None or active_dir is None:
            return None, None
        meta_path = active_dir / f"{disk_name}.json"
        bin_path = active_dir / f"{disk_name}.bin"
        try:
            if not (meta_path.is_file() and bin_path.is_file()):
                return None, None
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw_bytes = bin_path.read_bytes()
            if len(raw_bytes) != int(meta.get("binary_size_bytes", -1)):
                return None, None
            if expected_meta is not None:
                for field_name, expected_val in expected_meta.items():
                    if meta.get(field_name) != expected_val:
                        return None, None
            entry = CompiledKernelBinary(
                backend=str(meta["backend"]),
                workload=str(meta["workload"]),
                entry_point=str(meta["entry_point"]),
                format=str(meta["format"]),
                binary_bytes=raw_bytes,
                compile_flags=tuple(meta.get("compile_flags", ())),
                compiler_log=str(meta.get("compiler_log", "")),
                cache_key=disk_name,
            )
            self._memory[memory_key] = entry
            return entry, "disk"
        except (OSError, KeyError, ValueError, TypeError):
            return None, None

    def get(
        self,
        memory_key: MemoryCacheKey,
        disk_name: str | None = None,
        *,
        expected_meta: Mapping[str, Any] | None = None,
    ) -> CompiledKernelBinary | None:
        entry, _source = self.get_with_source(
            memory_key,
            disk_name,
            expected_meta=expected_meta,
        )
        return entry

    def put(
        self,
        memory_key: MemoryCacheKey,
        compiled: CompiledKernelBinary,
        disk_name: str | None = None,
        *,
        meta_extra: Mapping[str, Any] | None = None,
    ) -> None:
        self._memory[memory_key] = compiled
        active_dir = self.cache_dir
        if disk_name is None or active_dir is None:
            return
        try:
            active_dir.mkdir(parents=True, exist_ok=True)
            meta: dict[str, Any] = {
                "backend": compiled.backend,
                "workload": compiled.workload,
                "entry_point": compiled.entry_point,
                "format": compiled.format,
                "compile_flags": list(compiled.compile_flags),
                "compiler_log": compiled.compiler_log,
                "binary_size_bytes": len(compiled.binary_bytes),
            }
            if meta_extra:
                meta.update(meta_extra)
            (active_dir / f"{disk_name}.bin").write_bytes(compiled.binary_bytes)
            (active_dir / f"{disk_name}.json").write_text(
                json.dumps(meta, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            # Degrade gracefully to in-memory-only caching on read-only filesystems.
            return


_DEFAULT_CACHE = RuntimeBinaryCache()


def _capability_macros(
    capabilities: DeviceCapabilities | dict[str, bool] | None,
) -> list[str]:
    if capabilities is None:
        return []
    cap_dict = (
        capabilities.to_dict()
        if isinstance(capabilities, DeviceCapabilities)
        else dict(capabilities)
    )
    return [
        f"-DCROSS_GPU_HAS_{name.upper()}={1 if enabled else 0}"
        for name, enabled in sorted(cap_dict.items())
    ]


class _NvrtcApi:
    """Explicitly typed ctypes bindings for NVIDIA NVRTC."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        c_void_pp = ctypes.POINTER(ctypes.c_void_p)
        c_size_p = ctypes.POINTER(ctypes.c_size_t)
        c_char_pp = ctypes.POINTER(ctypes.c_char_p)

        self.nvrtcCreateProgram = bind_c_symbol(
            lib,
            "nvrtcCreateProgram",
            ctypes.c_int,
            [c_void_pp, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, c_char_pp, c_char_pp],
        )
        self.nvrtcDestroyProgram = bind_c_symbol(
            lib, "nvrtcDestroyProgram", ctypes.c_int, [c_void_pp]
        )
        self.nvrtcCompileProgram = bind_c_symbol(
            lib, "nvrtcCompileProgram", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, c_char_pp]
        )
        self.nvrtcGetProgramLogSize = bind_c_symbol(
            lib, "nvrtcGetProgramLogSize", ctypes.c_int, [ctypes.c_void_p, c_size_p]
        )
        self.nvrtcGetProgramLog = bind_c_symbol(
            lib, "nvrtcGetProgramLog", ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]
        )
        self.nvrtcGetPTXSize = bind_c_symbol(
            lib, "nvrtcGetPTXSize", ctypes.c_int, [ctypes.c_void_p, c_size_p]
        )
        self.nvrtcGetPTX = bind_c_symbol(
            lib, "nvrtcGetPTX", ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]
        )
        self.nvrtcGetCUBINSize = bind_optional_c_symbol(
            lib, "nvrtcGetCUBINSize", ctypes.c_int, [ctypes.c_void_p, c_size_p]
        )
        self.nvrtcGetCUBIN = bind_optional_c_symbol(
            lib, "nvrtcGetCUBIN", ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]
        )
        self.nvrtcGetErrorString = bind_optional_c_symbol(
            lib, "nvrtcGetErrorString", ctypes.c_char_p, [ctypes.c_int]
        )


class _HiprtcApi:
    """Explicitly typed ctypes bindings for AMD hipRTC."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        c_void_pp = ctypes.POINTER(ctypes.c_void_p)
        c_size_p = ctypes.POINTER(ctypes.c_size_t)
        c_char_pp = ctypes.POINTER(ctypes.c_char_p)

        self.hiprtcCreateProgram = bind_c_symbol(
            lib,
            "hiprtcCreateProgram",
            ctypes.c_int,
            [c_void_pp, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, c_char_pp, c_char_pp],
        )
        self.hiprtcDestroyProgram = bind_c_symbol(
            lib, "hiprtcDestroyProgram", ctypes.c_int, [c_void_pp]
        )
        self.hiprtcCompileProgram = bind_c_symbol(
            lib, "hiprtcCompileProgram", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, c_char_pp]
        )
        self.hiprtcGetProgramLogSize = bind_c_symbol(
            lib, "hiprtcGetProgramLogSize", ctypes.c_int, [ctypes.c_void_p, c_size_p]
        )
        self.hiprtcGetProgramLog = bind_c_symbol(
            lib, "hiprtcGetProgramLog", ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]
        )
        self.hiprtcGetCodeSize = bind_c_symbol(
            lib, "hiprtcGetCodeSize", ctypes.c_int, [ctypes.c_void_p, c_size_p]
        )
        self.hiprtcGetCode = bind_c_symbol(
            lib, "hiprtcGetCode", ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]
        )
        self.hiprtcGetErrorString = bind_optional_c_symbol(
            lib, "hiprtcGetErrorString", ctypes.c_char_p, [ctypes.c_int]
        )


def _compile_cuda_nvrtc(
    source_bytes: bytes,
    workload: str,
    target_arch: str,
    entry_point: str,
    flags: tuple[str, ...],
) -> CompiledKernelBinary:
    try:
        lib = load_shared_library(NVRTC_LIB_CANDIDATES)
    except OSError as exc:
        raise KernelCompilationError(
            f"NVRTC library unavailable for CUDA runtime compilation: {exc}",
            compiler_log=str(exc),
        ) from exc

    api = _NvrtcApi(lib)
    prog = ctypes.c_void_p()
    rc = api.nvrtcCreateProgram(
        ctypes.byref(prog),
        source_bytes,
        f"{workload}.cu".encode("utf-8"),
        0,
        None,
        None,
    )
    if rc != 0:
        raise KernelCompilationError(f"nvrtcCreateProgram failed with code {rc}")

    effective_flags = list(flags)
    if target_arch and not any(f.startswith("--gpu-architecture") for f in effective_flags):
        arch_arg = target_arch if target_arch.startswith(("sm_", "compute_")) else f"compute_{target_arch}"
        effective_flags.append(f"--gpu-architecture={arch_arg}")

    try:
        c_flags = (ctypes.c_char_p * max(1, len(effective_flags)))(
            *[f.encode("utf-8") for f in effective_flags]
        )
        compile_rc = api.nvrtcCompileProgram(
            prog.value, len(effective_flags), c_flags if effective_flags else None
        )
        log_size = ctypes.c_size_t(0)
        log_text = ""
        if api.nvrtcGetProgramLogSize(prog.value, ctypes.byref(log_size)) == 0 and log_size.value > 1:
            log_buf = ctypes.create_string_buffer(log_size.value)
            if api.nvrtcGetProgramLog(prog.value, log_buf) == 0:
                log_text = log_buf.value.decode("utf-8", errors="replace")

        if compile_rc != 0:
            err_name = ""
            if api.nvrtcGetErrorString is not None:
                raw_err = api.nvrtcGetErrorString(compile_rc)
                if raw_err:
                    err_name = raw_err.decode("utf-8", errors="replace")
            raise KernelCompilationError(
                f"nvrtcCompileProgram failed ({err_name or compile_rc})",
                compiler_log=log_text,
            )

        if (
            target_arch.startswith("sm_")
            and api.nvrtcGetCUBINSize is not None
            and api.nvrtcGetCUBIN is not None
        ):
            cubin_size = ctypes.c_size_t(0)
            if api.nvrtcGetCUBINSize(prog.value, ctypes.byref(cubin_size)) == 0 and cubin_size.value > 0:
                cubin_buf = ctypes.create_string_buffer(cubin_size.value)
                if api.nvrtcGetCUBIN(prog.value, cubin_buf) == 0:
                    return CompiledKernelBinary(
                        backend="cuda",
                        workload=workload,
                        entry_point=entry_point,
                        format="cubin",
                        binary_bytes=bytes(cubin_buf.raw[: cubin_size.value]),
                        compile_flags=tuple(effective_flags),
                        compiler_log=log_text,
                    )

        ptx_size = ctypes.c_size_t(0)
        if api.nvrtcGetPTXSize(prog.value, ctypes.byref(ptx_size)) != 0 or ptx_size.value == 0:
            raise KernelCompilationError("nvrtcGetPTXSize failed", compiler_log=log_text)
        ptx_buf = ctypes.create_string_buffer(ptx_size.value)
        if api.nvrtcGetPTX(prog.value, ptx_buf) != 0:
            raise KernelCompilationError("nvrtcGetPTX failed", compiler_log=log_text)
        return CompiledKernelBinary(
            backend="cuda",
            workload=workload,
            entry_point=entry_point,
            format="ptx",
            binary_bytes=bytes(ptx_buf.raw[: ptx_size.value]),
            compile_flags=tuple(effective_flags),
            compiler_log=log_text,
        )
    finally:
        api.nvrtcDestroyProgram(ctypes.byref(prog))


def _compile_hip_hiprtc(
    source_bytes: bytes,
    workload: str,
    target_arch: str,
    entry_point: str,
    flags: tuple[str, ...],
) -> CompiledKernelBinary:
    try:
        lib = load_shared_library(HIPRTC_LIB_CANDIDATES)
    except OSError as exc:
        raise KernelCompilationError(
            f"hipRTC library unavailable for HIP runtime compilation: {exc}",
            compiler_log=str(exc),
        ) from exc

    api = _HiprtcApi(lib)
    prog = ctypes.c_void_p()
    rc = api.hiprtcCreateProgram(
        ctypes.byref(prog),
        source_bytes,
        f"{workload}.hip".encode("utf-8"),
        0,
        None,
        None,
    )
    if rc != 0:
        raise KernelCompilationError(f"hiprtcCreateProgram failed with code {rc}")

    effective_flags = list(flags)
    if target_arch and not any(f.startswith("--offload-arch") for f in effective_flags):
        effective_flags.append(f"--offload-arch={target_arch}")

    try:
        c_flags = (ctypes.c_char_p * max(1, len(effective_flags)))(
            *[f.encode("utf-8") for f in effective_flags]
        )
        compile_rc = api.hiprtcCompileProgram(
            prog.value, len(effective_flags), c_flags if effective_flags else None
        )
        log_size = ctypes.c_size_t(0)
        log_text = ""
        if api.hiprtcGetProgramLogSize(prog.value, ctypes.byref(log_size)) == 0 and log_size.value > 1:
            log_buf = ctypes.create_string_buffer(log_size.value)
            if api.hiprtcGetProgramLog(prog.value, log_buf) == 0:
                log_text = log_buf.value.decode("utf-8", errors="replace")

        if compile_rc != 0:
            raise KernelCompilationError(
                f"hiprtcCompileProgram failed with code {compile_rc}",
                compiler_log=log_text,
            )

        code_size = ctypes.c_size_t(0)
        if api.hiprtcGetCodeSize(prog.value, ctypes.byref(code_size)) != 0 or code_size.value == 0:
            raise KernelCompilationError("hiprtcGetCodeSize failed", compiler_log=log_text)
        code_buf = ctypes.create_string_buffer(code_size.value)
        if api.hiprtcGetCode(prog.value, code_buf) != 0:
            raise KernelCompilationError("hiprtcGetCode failed", compiler_log=log_text)

        return CompiledKernelBinary(
            backend="hip",
            workload=workload,
            entry_point=entry_point,
            format="hsaco",
            binary_bytes=bytes(code_buf.raw[: code_size.value]),
            compile_flags=tuple(effective_flags),
            compiler_log=log_text,
        )
    finally:
        api.hiprtcDestroyProgram(ctypes.byref(prog))


def _compile_levelzero_spirv(
    source_bytes: bytes,
    workload: str,
    target_arch: str,
    entry_point: str,
    flags: tuple[str, ...],
) -> CompiledKernelBinary:
    if source_bytes[:4] in (SPIRV_MAGIC_LE, SPIRV_MAGIC_BE):
        return CompiledKernelBinary(
            backend="levelzero",
            workload=workload,
            entry_point=entry_point,
            format="spirv",
            binary_bytes=source_bytes,
            compile_flags=flags,
        )

    ocloc_bin = shutil.which("ocloc")
    clang_bin = shutil.which("clang")
    llvm_spirv_bin = shutil.which("llvm-spirv")

    with tempfile.TemporaryDirectory(prefix="cross_gpu_l0_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        src_path = tmp_path / f"{workload}.cl"
        src_path.write_bytes(source_bytes)

        if clang_bin and llvm_spirv_bin:
            bc_path = tmp_path / f"{workload}.bc"
            spv_path = tmp_path / f"{workload}.spv"
            clang_cmd = [
                clang_bin,
                "-cc1",
                "-triple",
                "spir64-unknown-unknown",
                "-emit-llvm-bc",
                "-cl-std=CL3.0",
                *[f for f in flags if f.startswith(("-D", "-I"))],
                "-o",
                str(bc_path),
                str(src_path),
            ]
            res1 = subprocess.run(clang_cmd, capture_output=True, text=True, check=False)
            if res1.returncode != 0 or not bc_path.is_file():
                raise KernelCompilationError(
                    f"clang SPIR-V bitcode compilation failed (rc={res1.returncode})",
                    compiler_log=f"{res1.stdout}\n{res1.stderr}".strip(),
                )
            res2 = subprocess.run(
                [llvm_spirv_bin, str(bc_path), "-o", str(spv_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res2.returncode != 0 or not spv_path.is_file():
                raise KernelCompilationError(
                    f"llvm-spirv translation failed (rc={res2.returncode})",
                    compiler_log=f"{res2.stdout}\n{res2.stderr}".strip(),
                )
            return CompiledKernelBinary(
                backend="levelzero",
                workload=workload,
                entry_point=entry_point,
                format="spirv",
                binary_bytes=spv_path.read_bytes(),
                compile_flags=flags,
                compiler_log=f"{res1.stderr}\n{res2.stderr}".strip(),
            )

        if ocloc_bin:
            out_prefix = tmp_path / workload
            cmd = [
                ocloc_bin,
                "compile",
                "-file",
                str(src_path),
                "-spirv_only",
                "-out_dir",
                str(tmp_path),
                "-output",
                out_prefix.name,
            ]
            if target_arch:
                cmd.extend(["-device", target_arch])
            if flags:
                cmd.extend(["-options", " ".join(flags)])
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            spv_candidates = sorted(tmp_path.glob("*.spv"))
            if res.returncode != 0 or not spv_candidates:
                raise KernelCompilationError(
                    f"ocloc SPIR-V compilation failed (rc={res.returncode})",
                    compiler_log=f"{res.stdout}\n{res.stderr}".strip(),
                )
            return CompiledKernelBinary(
                backend="levelzero",
                workload=workload,
                entry_point=entry_point,
                format="spirv",
                binary_bytes=spv_candidates[0].read_bytes(),
                compile_flags=flags,
                compiler_log=f"{res.stdout}\n{res.stderr}".strip(),
            )

    raise KernelCompilationError(
        "No Level Zero SPIR-V compiler toolchain found (need clang + llvm-spirv or ocloc)",
        compiler_log="Missing clang/llvm-spirv and ocloc in PATH",
    )


def _prepare_source_macros(source_bytes: bytes, flags: tuple[str, ...]) -> bytes:
    """Prepend preprocessor #define lines for backends without flag-only macro passing."""
    defines: list[str] = []
    for flag in flags:
        if not flag.startswith("-D"):
            continue
        body = flag[2:]
        if "=" in body:
            name, val = body.split("=", 1)
            defines.append(f"#define {name} {val}")
        elif body:
            defines.append(f"#define {body} 1")
    if not defines:
        return source_bytes
    preamble = ("\n".join(defines) + "\n").encode("utf-8")
    return preamble + source_bytes


def _compile_metal(
    source_bytes: bytes,
    workload: str,
    entry_point: str,
    flags: tuple[str, ...],
) -> CompiledKernelBinary:
    if source_bytes[:4] == METALLIB_MAGIC:
        return CompiledKernelBinary(
            backend="metal",
            workload=workload,
            entry_point=entry_point,
            format="metallib",
            binary_bytes=source_bytes,
            compile_flags=flags,
        )

    prepared_source = _prepare_source_macros(source_bytes, flags)
    xcrun_bin = shutil.which("xcrun") if sys.platform == "darwin" else None
    if not xcrun_bin:
        return CompiledKernelBinary(
            backend="metal",
            workload=workload,
            entry_point=entry_point,
            format="msl_source",
            binary_bytes=prepared_source,
            compile_flags=flags,
        )

    with tempfile.TemporaryDirectory(prefix="cross_gpu_mtl_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        src_path = tmp_path / f"{workload}.metal"
        air_path = tmp_path / f"{workload}.air"
        lib_path = tmp_path / f"{workload}.metallib"
        src_path.write_bytes(prepared_source)

        res1 = subprocess.run(
            [xcrun_bin, "-sdk", "macosx", "metal", "-c", str(src_path), "-o", str(air_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        combined_err = f"{res1.stdout}\n{res1.stderr}".strip()
        if res1.returncode != 0:
            if "unable to find utility" in combined_err.lower():
                return CompiledKernelBinary(
                    backend="metal",
                    workload=workload,
                    entry_point=entry_point,
                    format="msl_source",
                    binary_bytes=prepared_source,
                    compile_flags=flags,
                    compiler_log=combined_err,
                )
            raise KernelCompilationError(
                f"Metal xcrun compilation failed (rc={res1.returncode})",
                compiler_log=combined_err,
            )

        res2 = subprocess.run(
            [xcrun_bin, "-sdk", "macosx", "metallib", str(air_path), "-o", str(lib_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if res2.returncode != 0 or not lib_path.is_file():
            raise KernelCompilationError(
                f"Metal metallib linking failed (rc={res2.returncode})",
                compiler_log=f"{res2.stdout}\n{res2.stderr}".strip(),
            )
        return CompiledKernelBinary(
            backend="metal",
            workload=workload,
            entry_point=entry_point,
            format="metallib",
            binary_bytes=lib_path.read_bytes(),
            compile_flags=flags,
            compiler_log=f"{res1.stderr}\n{res2.stderr}".strip(),
        )


def _compile_opencl(
    source_bytes: bytes,
    workload: str,
    entry_point: str,
    flags: tuple[str, ...],
) -> CompiledKernelBinary:
    return CompiledKernelBinary(
        backend="opencl",
        workload=workload,
        entry_point=entry_point,
        format="opencl_source",
        binary_bytes=source_bytes,
        compile_flags=flags,
    )


def extract_opencl_program_binary(backend: Any, module: Any) -> bytes:
    """Delegate extraction of CL_PROGRAM_BINARIES bytes from a built OpenCL module."""
    return bytes(backend.extract_program_binary(module))


def reload_opencl_program_binary(
    backend: Any,
    ctx: Any,
    workload: str,
    binary_bytes: bytes,
    *,
    entry_point: str | None = None,
) -> Any:
    """Reload cached CL_PROGRAM_BINARIES bytes via clCreateProgramWithBinary."""
    compiled = CompiledKernelBinary(
        backend="opencl",
        workload=workload,
        entry_point=entry_point or default_entry_point(workload),
        format="opencl_binary",
        binary_bytes=binary_bytes,
    )
    return backend.load_module(ctx, compiled)


def compile_kernel_runtime(
    backend: str,
    workload: str,
    target_arch: str = "",
    capabilities: DeviceCapabilities | dict[str, bool] | None = None,
    source_override: bytes | str | None = None,
    extra_flags: Sequence[str] | None = None,
    *,
    driver_version: str = "",
    entry_point: str | None = None,
    cache: RuntimeBinaryCache | None = None,
    specs: Mapping[tuple[str, str], KernelSpec] | None = None,
    resource_root: Any | None = None,
    diag_fn: DiagCallback | None = None,
) -> CompiledKernelBinary:
    """Resolve raw kernel source via importlib.resources, compile at runtime, and cache."""
    if capabilities is not None and workload in WORKLOAD_SPECS_BY_NAME:
        if not device_supports_workload(capabilities, workload):
            spec = WORKLOAD_SPECS_BY_NAME[workload]
            raise UnsupportedKernelError(
                f"Device lacks capability {spec.required_capability!r} required by {workload!r}"
            )

    normalized_workload = normalize_workload_name(workload)
    resolved_entry = entry_point or default_entry_point(workload)

    if source_override is not None:
        raw_bytes = (
            source_override.encode("utf-8")
            if isinstance(source_override, str)
            else bytes(source_override)
        )
        if diag_fn is not None:
            diag_fn(
                f"kernel lookup [{backend}:{normalized_workload}]: "
                f"source_override ({len(raw_bytes)} B)"
            )
    else:
        loaded = load_kernel_source(
            backend,
            workload,
            specs=specs,
            resource_root=resource_root,
        )
        raw_bytes = loaded.source_bytes
        resolved_entry = entry_point or loaded.entry_point
        if diag_fn is not None:
            diag_fn(
                f"kernel lookup [{backend}:{normalized_workload}]: "
                f"importlib.resources cross_gpu/kernels/{loaded.backend}/{loaded.filename} "
                f"({loaded.size_bytes} B)"
            )

    macro_flags = _capability_macros(capabilities)
    combined_flags = canonicalize_compile_flags([*macro_flags, *(extra_flags or ())])
    active_cache = _DEFAULT_CACHE if cache is None else cache

    if diag_fn is not None:
        flags_desc = " ".join(combined_flags) if combined_flags else "<none>"
        diag_fn(
            f"compile flags [{backend}:{normalized_workload}] "
            f"arch={target_arch or 'default'}: {flags_desc}"
        )

    memory_key = compute_memory_cache_key(
        backend=backend,
        target_arch=target_arch,
        driver_version=driver_version,
        workload=normalized_workload,
        compile_flags=combined_flags,
        source_bytes=raw_bytes,
    )

    # Skip disk cache when caller supplies an ad-hoc source_override.
    disk_name: str | None = None
    disk_meta: dict[str, Any] | None = None
    if source_override is None:
        disk_name = compute_disk_cache_name(
            backend=backend,
            target_arch=target_arch,
            driver_version=driver_version,
            workload=normalized_workload,
            size_bytes=len(raw_bytes),
            compile_flags=combined_flags,
        )
        disk_meta = {
            "package_version": __version__,
            "backend": backend,
            "workload": normalized_workload,
            "target_arch": target_arch,
            "driver_version": driver_version,
            "source_size_bytes": len(raw_bytes),
            "input_compile_flags": list(combined_flags),
        }

    cached, hit_source = active_cache.get_with_source(
        memory_key, disk_name, expected_meta=disk_meta
    )
    if cached is not None:
        if diag_fn is not None:
            if hit_source == "disk" and active_cache.cache_dir is not None and disk_name:
                diag_fn(
                    f"cache hit (disk) [{backend}:{normalized_workload}]: "
                    f"{active_cache.cache_dir / f'{disk_name}.bin'}"
                )
            else:
                diag_fn(f"cache hit (memory) [{backend}:{normalized_workload}]")
        return cached

    if diag_fn is not None:
        active_dir = active_cache.cache_dir
        if is_disk_cache_disabled():
            cache_loc = "disabled via CROSS_GPU_NO_CACHE (in-memory only)"
        elif disk_name and active_dir is not None:
            cache_loc = str(active_dir)
        else:
            cache_loc = "in-memory only"
        diag_fn(f"cache miss [{backend}:{normalized_workload}] (cache={cache_loc})")

    try:
        if backend == "cuda":
            result = _compile_cuda_nvrtc(
                raw_bytes, normalized_workload, target_arch, resolved_entry, combined_flags
            )
        elif backend == "hip":
            result = _compile_hip_hiprtc(
                raw_bytes, normalized_workload, target_arch, resolved_entry, combined_flags
            )
        elif backend == "levelzero":
            result = _compile_levelzero_spirv(
                raw_bytes, normalized_workload, target_arch, resolved_entry, combined_flags
            )
        elif backend == "metal":
            result = _compile_metal(raw_bytes, normalized_workload, resolved_entry, combined_flags)
        elif backend == "opencl":
            result = _compile_opencl(raw_bytes, normalized_workload, resolved_entry, combined_flags)
        else:
            raise KernelCompilationError(
                f"Unsupported backend {backend!r} for runtime compilation"
            )
    except KernelCompilationError as exc:
        if diag_fn is not None:
            diag_fn(f"compile failed [{backend}:{normalized_workload}]: {exc}")
            if exc.compiler_log:
                for log_line in exc.compiler_log.splitlines():
                    diag_fn(f"compiler [{backend}:{normalized_workload}]: {log_line}")
        raise

    if diag_fn is not None:
        diag_fn(
            f"compile ok [{backend}:{normalized_workload}]: "
            f"format={result.format} size={len(result.binary_bytes)} B"
        )
        if result.compiler_log:
            for log_line in result.compiler_log.splitlines():
                diag_fn(f"compiler [{backend}:{normalized_workload}]: {log_line}")

    result.cache_key = disk_name or f"memory:{backend}:{normalized_workload}"
    active_cache.put(memory_key, result, disk_name, meta_extra=disk_meta)
    return result


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m cross_gpu.embed_binary",
        description="Verify and list packaged raw GPU kernel sources.",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser(
        "verify",
        help="Verify all registered kernel resources load cleanly and match entry points",
    )
    sub.add_parser(
        "list",
        help="List all registered kernel sources with filename, entry points, and byte size",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.action == "verify":
        errors = verify_kernel_resources()
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return int(ExitCode.INIT_ERROR)
        print(f"Verified {len(KERNEL_SPECS)} packaged kernel source(s): OK")
        return int(ExitCode.SUCCESS)

    if args.action == "list":
        for (backend, workload), _spec in sorted(KERNEL_SPECS.items()):
            loaded = load_kernel_source(backend, workload)
            eps = ",".join(loaded.entry_points)
            print(
                f"{backend}:{workload}\tfile={loaded.filename}"
                f"\tentry_points={eps}\tsize={loaded.size_bytes}"
            )
        return int(ExitCode.SUCCESS)

    return int(ExitCode.USAGE_ERROR)


if __name__ == "__main__":
    sys.exit(_main())
