"""Embedded kernel payload registry, packing CLI, runtime cache, and compiler adapters."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zlib
import ctypes

from cross_gpu.backends.base import bind_c_symbol, bind_optional_c_symbol, load_shared_library
from cross_gpu.exit_codes import (
    ExitCode,
    KernelCompilationError,
    KernelNotFoundError,
    UnsupportedKernelError,
)
from cross_gpu.models import DeviceCapabilities
from cross_gpu.workloads import WORKLOAD_SPECS_BY_NAME, device_supports_workload

KERNELS_DIR = Path(__file__).resolve().parent / "kernels"

NVRTC_LIB_CANDIDATES: tuple[str, ...] = ("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11.2")
HIPRTC_LIB_CANDIDATES: tuple[str, ...] = (
    "libhiprtc.so",
    "libhiprtc.so.6",
    "libhiprtc.so.5",
    "libamdhip64.so",
)

BACKEND_EXTENSION_FORMATS: dict[str, dict[str, str]] = {
    "cuda": {".cu": "cuda_source", ".ptx": "ptx", ".cubin": "cubin"},
    "hip": {".hip": "hip_source", ".cu": "hip_source", ".hsaco": "hsaco"},
    "levelzero": {".cl": "opencl_source", ".spv": "spirv"},
    "metal": {".metal": "msl_source", ".metallib": "metallib"},
    "opencl": {".cl": "opencl_source", ".bin": "opencl_binary"},
}

PRECOMPILED_FORMATS: frozenset[str] = frozenset(
    {"ptx", "cubin", "hsaco", "spirv", "metallib", "opencl_binary"}
)

SPIRV_MAGIC_LE = b"\x03\x02\x23\x07"
SPIRV_MAGIC_BE = b"\x07\x23\x02\x03"
METALLIB_MAGIC = b"MTLB"

# BEGIN EMBEDDED_KERNELS
EMBEDDED_KERNELS: dict[str, dict[str, Any]] = {}
# END EMBEDDED_KERNELS


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


def default_entry_point(workload: str) -> str:
    """Return canonical C/kernel entry point name for a workload slug."""
    if workload in WORKLOAD_SPECS_BY_NAME:
        return WORKLOAD_SPECS_BY_NAME[workload].default_entry_point
    return f"kernel_{workload.replace('-', '_')}"


def kernel_key(backend: str, workload: str) -> str:
    """Build deterministic registry lookup key."""
    return f"{backend}:{workload}"


def compute_cache_key(
    backend: str,
    target_arch: str,
    driver_version: str,
    workload: str,
    source_hash: str,
    compile_flags: Sequence[str],
) -> str:
    """Compute SHA-256 cache key per Section 6 specification."""
    payload = "|".join(
        [
            backend,
            target_arch,
            driver_version,
            workload,
            source_hash,
            " ".join(compile_flags),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_home_dir(home: Path | None = None) -> Path | None:
    if home is not None:
        return home
    try:
        return Path.home()
    except (RuntimeError, OSError):
        return None


def default_cache_dir(
    *,
    platform_name: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path | None:
    """Resolve the OS-standard runtime binary cache directory, or None if unavailable.

    - Explicit override: $CROSS_GPU_CACHE_DIR
    - macOS: ~/Library/Caches/cross-gpu
    - Linux/POSIX: $XDG_CACHE_HOME/cross-gpu (if absolute), fallback ~/.cache/cross-gpu
    """
    active_env = os.environ if env is None else env
    explicit = active_env.get("CROSS_GPU_CACHE_DIR", "").strip()
    if explicit:
        return Path(explicit)

    active_platform = sys.platform if platform_name is None else platform_name
    if active_platform == "darwin":
        resolved_home = _safe_home_dir(home)
        if resolved_home is None:
            return None
        return resolved_home / "Library" / "Caches" / "cross-gpu"

    xdg_cache = active_env.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        xdg_path = Path(xdg_cache)
        if xdg_path.is_absolute():
            return xdg_path / "cross-gpu"

    resolved_home = _safe_home_dir(home)
    if resolved_home is None:
        return None
    return resolved_home / ".cache" / "cross-gpu"


class RuntimeBinaryCache:
    """In-memory + best-effort disk cache for compiled kernel binaries."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._memory: dict[str, CompiledKernelBinary] = {}
        self.cache_dir: Path | None = (
            cache_dir if cache_dir is not None else default_cache_dir()
        )

    def get(self, cache_key: str) -> CompiledKernelBinary | None:
        if cache_key in self._memory:
            return self._memory[cache_key]
        if self.cache_dir is None:
            return None
        meta_path = self.cache_dir / f"{cache_key}.json"
        bin_path = self.cache_dir / f"{cache_key}.bin"
        try:
            if not (meta_path.is_file() and bin_path.is_file()):
                return None
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw_bytes = bin_path.read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() != meta.get("binary_sha256"):
                return None
            entry = CompiledKernelBinary(
                backend=meta["backend"],
                workload=meta["workload"],
                entry_point=meta["entry_point"],
                format=meta["format"],
                binary_bytes=raw_bytes,
                compile_flags=tuple(meta.get("compile_flags", ())),
                compiler_log=meta.get("compiler_log", ""),
                cache_key=cache_key,
            )
            self._memory[cache_key] = entry
            return entry
        except (OSError, KeyError, ValueError, TypeError):
            return None

    def put(self, cache_key: str, compiled: CompiledKernelBinary) -> None:
        self._memory[cache_key] = compiled
        if self.cache_dir is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "backend": compiled.backend,
                "workload": compiled.workload,
                "entry_point": compiled.entry_point,
                "format": compiled.format,
                "compile_flags": list(compiled.compile_flags),
                "compiler_log": compiled.compiler_log,
                "binary_sha256": hashlib.sha256(compiled.binary_bytes).hexdigest(),
            }
            (self.cache_dir / f"{cache_key}.bin").write_bytes(compiled.binary_bytes)
            (self.cache_dir / f"{cache_key}.json").write_text(
                json.dumps(meta, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            # Degrade gracefully on read-only filesystems.
            return


_DEFAULT_CACHE = RuntimeBinaryCache()


def encode_payload(raw_bytes: bytes) -> tuple[str, str]:
    """Compress raw bytes with zlib and encode as base64 + SHA-256 hex."""
    digest = hashlib.sha256(raw_bytes).hexdigest()
    compressed = zlib.compress(raw_bytes, level=9)
    b64 = base64.b64encode(compressed).decode("ascii")
    return b64, digest


def decode_payload(entry: dict[str, Any]) -> bytes:
    """Decode and verify integrity of an embedded kernel payload entry."""
    required = ("backend", "workload", "format", "sha256", "size_bytes", "zlib_b64")
    for field_name in required:
        if field_name not in entry:
            raise ValueError(f"Embedded kernel entry missing field {field_name!r}")
    compressed = base64.b64decode(entry["zlib_b64"].encode("ascii"), validate=True)
    raw = zlib.decompress(compressed)
    if len(raw) != int(entry["size_bytes"]):
        raise ValueError(
            f"Size mismatch for {entry['backend']}:{entry['workload']}: "
            f"expected {entry['size_bytes']}, got {len(raw)}"
        )
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != entry["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {entry['backend']}:{entry['workload']}: "
            f"expected {entry['sha256']}, got {actual_sha}"
        )
    return raw


def get_embedded_kernel(
    backend: str,
    workload: str,
    registry: dict[str, dict[str, Any]] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Retrieve and verify embedded kernel bytes for (backend, workload)."""
    store = EMBEDDED_KERNELS if registry is None else registry
    candidates = (
        kernel_key(backend, workload),
        kernel_key(backend, workload.replace("-", "_")),
    )
    for key in candidates:
        if key in store:
            entry = store[key]
            return decode_payload(entry), entry
    raise KernelNotFoundError(
        f"No kernel registered for backend={backend!r}, workload={workload!r}"
    )


def scan_kernels_directory(kernels_dir: Path = KERNELS_DIR) -> dict[str, dict[str, Any]]:
    """Scan kernels/<backend>/ deterministically and pack all kernel files."""
    packed: dict[str, dict[str, Any]] = {}
    if not kernels_dir.exists():
        return packed

    for backend in sorted(BACKEND_EXTENSION_FORMATS):
        backend_dir = kernels_dir / backend
        if not backend_dir.is_dir():
            continue
        ext_map = BACKEND_EXTENSION_FORMATS[backend]
        for file_path in sorted(backend_dir.iterdir(), key=lambda p: p.name):
            if not file_path.is_file():
                continue
            if file_path.name.startswith(".") or file_path.name.lower() == "readme.md":
                continue
            fmt = ext_map.get(file_path.suffix.lower())
            if fmt is None:
                continue
            workload = file_path.stem
            raw_bytes = file_path.read_bytes()
            b64, sha256_hex = encode_payload(raw_bytes)
            key = kernel_key(backend, workload)
            packed[key] = {
                "backend": backend,
                "workload": workload,
                "filename": file_path.name,
                "format": fmt,
                "entry_point": default_entry_point(workload),
                "sha256": sha256_hex,
                "size_bytes": len(raw_bytes),
                "zlib_b64": b64,
            }
    return packed


def pack_kernels(
    kernels_dir: Path = KERNELS_DIR,
    target_module_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Pack kernel payloads from kernels_dir and rewrite EMBEDDED_KERNELS in module."""
    packed = scan_kernels_directory(kernels_dir)
    target = Path(__file__).resolve() if target_module_path is None else target_module_path
    content = target.read_text(encoding="utf-8")
    begin_marker = "# BEGIN EMBEDDED_KERNELS"
    end_marker = "# END EMBEDDED_KERNELS"
    if begin_marker not in content or end_marker not in content:
        raise RuntimeError(f"Missing EMBEDDED_KERNELS markers in {target}")

    serialized = json.dumps(packed, indent=4, sort_keys=True)
    replacement = (
        f"{begin_marker}\n"
        f"EMBEDDED_KERNELS: dict[str, dict[str, Any]] = {serialized}\n"
        f"{end_marker}"
    )
    prefix = content.split(begin_marker, 1)[0]
    suffix = content.split(end_marker, 1)[1]
    target.write_text(prefix + replacement + suffix, encoding="utf-8")
    return packed


def verify_embedded_kernels(
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Validate integrity and decompressibility of every packed entry."""
    store = EMBEDDED_KERNELS if registry is None else registry
    errors: list[str] = []
    for key, entry in sorted(store.items()):
        try:
            decode_payload(entry)
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    return errors


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

        # Prefer native CUBIN when compiling for a concrete sm_XX target and supported by NVRTC.
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
    source_format: str,
) -> CompiledKernelBinary:
    if source_format == "opencl_binary":
        return CompiledKernelBinary(
            backend="opencl",
            workload=workload,
            entry_point=entry_point,
            format="opencl_binary",
            binary_bytes=source_bytes,
            compile_flags=flags,
        )
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
    registry: dict[str, dict[str, Any]] | None = None,
) -> CompiledKernelBinary:
    """Resolve, compile, and cache a kernel payload for any of the 5 backends."""
    if capabilities is not None and workload in WORKLOAD_SPECS_BY_NAME:
        if not device_supports_workload(capabilities, workload):
            spec = WORKLOAD_SPECS_BY_NAME[workload]
            raise UnsupportedKernelError(
                f"Device lacks capability {spec.required_capability!r} required by {workload!r}"
            )

    resolved_entry = entry_point or default_entry_point(workload)
    source_format = ""
    if source_override is not None:
        raw_bytes = (
            source_override.encode("utf-8")
            if isinstance(source_override, str)
            else bytes(source_override)
        )
    else:
        raw_bytes, meta = get_embedded_kernel(backend, workload, registry=registry)
        source_format = str(meta.get("format", ""))
        resolved_entry = entry_point or str(meta.get("entry_point") or resolved_entry)

    macro_flags = _capability_macros(capabilities)
    combined_flags = tuple([*macro_flags, *(extra_flags or ())])
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    active_cache = _DEFAULT_CACHE if cache is None else cache
    cache_key = compute_cache_key(
        backend=backend,
        target_arch=target_arch,
        driver_version=driver_version,
        workload=workload,
        source_hash=source_hash,
        compile_flags=combined_flags,
    )

    cached = active_cache.get(cache_key)
    if cached is not None:
        return cached

    if source_format in PRECOMPILED_FORMATS:
        result = CompiledKernelBinary(
            backend=backend,
            workload=workload,
            entry_point=resolved_entry,
            format=source_format,
            binary_bytes=raw_bytes,
            compile_flags=combined_flags,
            cache_key=cache_key,
        )
    elif backend == "cuda":
        result = _compile_cuda_nvrtc(
            raw_bytes, workload, target_arch, resolved_entry, combined_flags
        )
    elif backend == "hip":
        result = _compile_hip_hiprtc(
            raw_bytes, workload, target_arch, resolved_entry, combined_flags
        )
    elif backend == "levelzero":
        result = _compile_levelzero_spirv(
            raw_bytes, workload, target_arch, resolved_entry, combined_flags
        )
    elif backend == "metal":
        result = _compile_metal(raw_bytes, workload, resolved_entry, combined_flags)
    elif backend == "opencl":
        result = _compile_opencl(
            raw_bytes, workload, resolved_entry, combined_flags, source_format
        )
    else:
        raise KernelCompilationError(f"Unsupported backend {backend!r} for runtime compilation")

    result.cache_key = cache_key
    active_cache.put(cache_key, result)
    return result


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m cross_gpu.embed_binary",
        description="Pack, verify, and list embedded kernel payloads.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    pack_p = sub.add_parser("pack", help="Pack kernel files from kernels/<backend>/")
    pack_p.add_argument(
        "--kernels-dir",
        type=Path,
        default=KERNELS_DIR,
        help="Root directory containing per-backend kernel subdirectories",
    )

    sub.add_parser("verify", help="Verify integrity of all embedded kernel payloads")
    sub.add_parser("list", help="List all registered embedded kernels")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.action == "pack":
        packed = pack_kernels(kernels_dir=args.kernels_dir)
        print(f"Packed {len(packed)} kernel payload(s) from {args.kernels_dir}")
        return ExitCode.SUCCESS

    if args.action == "verify":
        errors = verify_embedded_kernels()
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return ExitCode.INIT_ERROR
        print(f"Verified {len(EMBEDDED_KERNELS)} embedded kernel payload(s): OK")
        return ExitCode.SUCCESS

    if args.action == "list":
        if not EMBEDDED_KERNELS:
            print("No embedded kernels registered.")
            return ExitCode.SUCCESS
        for key, entry in sorted(EMBEDDED_KERNELS.items()):
            print(
                f"{key}\tbackend={entry['backend']}\tformat={entry['format']}"
                f"\tsize={entry['size_bytes']}\tsha256={entry['sha256']}"
            )
        return ExitCode.SUCCESS

    return ExitCode.USAGE_ERROR


if __name__ == "__main__":
    sys.exit(_main())
