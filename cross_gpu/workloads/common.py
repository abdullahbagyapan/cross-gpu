# SPDX-License-Identifier: MIT
"""Shared workload helpers: geometry, calibration, safe VRAM sizing, and golden verification."""

from __future__ import annotations

import ctypes
import math
import os
import re
import struct
import time
from typing import Any, Callable, Sequence
import zlib

from cross_gpu.backends.base import DeviceBuffer, HostBuffer
from cross_gpu.exit_codes import BackendInitError
from cross_gpu.models import BackendDeviceInfo, VerificationResult

# Watchdog-safe dispatch window (Linux DRM hangcheck & macOS Metal timeout safety).
WATCHDOG_MIN_DISPATCH_S = 0.050
WATCHDOG_TARGET_DISPATCH_S = 0.100
WATCHDOG_MAX_DISPATCH_S = 0.200
CALIBRATION_PROBE_THRESHOLD_S = 0.005
CALIBRATION_MAX_PROBE_ITERS = 64

DEFAULT_MIN_ITERATIONS = 1
DEFAULT_MAX_ITERATIONS = 65_536
NONLINEAR_DOT_MAX_ITERATIONS = 1_024

# Byte-size constants and safe VRAM guardrails.
KIB = 1024
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

VRAM_ALIGNMENT_BYTES = 4096
MIN_STRESS_VRAM_BYTES = 4 * MIB
MIN_BANDWIDTH_BYTES = 1 * MIB

DISCRETE_HEADROOM_MIN_BYTES = 512 * MIB
DISCRETE_HEADROOM_MAX_BYTES = 1 * GIB
DISCRETE_HEADROOM_RATIO = 0.10

# Unified memory shares RAM with the OS/compositor; keep wider headroom to avoid swap stalls.
UNIFIED_HEADROOM_MIN_BYTES = 1 * GIB
UNIFIED_HEADROOM_RATIO = 0.25

MEMORY_PATTERN_VRAM_RATIO = 0.60
MEMORY_SWEEP_VRAM_RATIO = 0.85
BANDWIDTH_MAX_VRAM_RATIO = 0.25
BANDWIDTH_MAX_HOST_RAM_RATIO = 0.25
BANDWIDTH_DISCRETE_CAP_BYTES = 512 * MIB
BANDWIDTH_UNIFIED_CAP_BYTES = 256 * MIB
VRAM_STEPDOWN_FRACTION = 0.10

# Host verification limits.
FULL_VERIFY_BUFFER_LIMIT_BYTES = 16 * MIB
NONLINEAR_VERIFY_BUDGET_OPS = 250_000
MIN_VERIFY_SAMPLES = 16
SENTINEL_SLICE_BYTES = 4096
PAYLOAD_TILE_BYTES = 64 * KIB
PAYLOAD_STAMP_STRIDE_BYTES = 1 * MIB

# Type-appropriate relative and absolute floating-point verification tolerances.
FP64_REL_TOL = 1e-12
FP64_ABS_TOL = 1e-12
FP32_REL_TOL = 1e-5
FP32_ABS_TOL = 1e-5
FP16_REL_TOL = 2e-3
FP16_ABS_TOL = 2e-3
BF16_REL_TOL = 1.5e-2
BF16_ABS_TOL = 3e-2
FP8_REL_TOL = 1e-5
FP8_ABS_TOL = 1e-5
FP4_REL_TOL = 1e-5
FP4_ABS_TOL = 1e-5
TENSOR_FP_REL_TOL = 1e-4
TENSOR_FP_ABS_TOL = 1e-4

# Canonical memory-pattern adversarial modes (matching all 5 backend kernels).
PATTERN_MODE_ALL_ZEROS = 0
PATTERN_MODE_ALL_ONES = 1
PATTERN_MODE_CHECKERBOARD_55 = 2
PATTERN_MODE_CHECKERBOARD_AA = 3
PATTERN_MODE_WALKING_ONES = 4
PATTERN_MODE_WALKING_ZEROS = 5
PATTERN_MODE_PRNG_HASH = 6

PATTERN_MODE_NAMES: tuple[str, ...] = (
    "all_zeros(0x00000000)",
    "all_ones(0xFFFFFFFF)",
    "checkerboard(0x55555555)",
    "checkerboard(0xAAAAAAAA)",
    "walking_ones",
    "walking_zeros",
    "prng_hash",
)

# OCP MXFP4 E2M1 positive lookup table (sign bit 0x8 negates).
_FP4_E2M1_MAGNITUDES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

_SM_PATTERN = re.compile(r"(?:sm|compute)_(\d+)", re.IGNORECASE)
_GFX_PATTERN = re.compile(r"gfx([0-9a-f]+)", re.IGNORECASE)


def _parse_cuda_sm(architecture: str) -> int:
    match = _SM_PATTERN.search(architecture or "")
    return int(match.group(1)) if match else 0


def _parse_hip_gfx(architecture: str) -> int:
    match = _GFX_PATTERN.search((architecture or "").split(":", 1)[0])
    return int(match.group(1), 16) if match else 0


def ops_per_thread_per_iter(workload: str, backend: str, architecture: str = "") -> int:
    """Return exact OPS_PER_THREAD_PER_ITER matching the compiled kernel header."""
    wl = workload.replace("_", "-")
    if wl in ("compute-int64", "compute-int32", "compute-int16", "compute-fp64", "compute-fp32"):
        return 256
    if wl in ("compute-fp16", "compute-bf16"):
        return 512
    if wl == "compute-int8":
        return 1024
    if wl == "compute-int4":
        return 2048 if backend == "hip" else 4096
    if wl == "compute-fp8":
        if backend == "hip":
            gfx = _parse_hip_gfx(architecture)
            return 4096 if gfx >= 0x1200 else 8192
        return 8192
    if wl == "compute-fp4":
        return 32768 if backend == "hip" else 16384
    if wl == "compute-tensor-fp32":
        if backend in ("hip", "metal"):
            return 1024
        return 2048
    if wl == "compute-tensor-fp16":
        if backend == "metal":
            return 1024
        if backend == "cuda":
            sm = _parse_cuda_sm(architecture)
            return 2048 if (0 < sm < 80) else 4096
        return 4096
    if wl == "compute-tensor-bf16":
        if backend == "metal":
            return 1024
        if backend == "hip" and _parse_hip_gfx(architecture) == 0x908:
            return 2048
        return 4096
    if wl == "compute-tensor-int8":
        return 4096 if backend == "hip" else 8192
    raise KeyError(f"No op-count constant defined for workload {workload!r}")


def compute_launch_geometry(
    device_info: BackendDeviceInfo,
    *,
    is_tensor: bool = False,
    is_memory: bool = False,
    target_threads: int | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    """Derive warp/wavefront-aligned (grid, block, total_threads) from device specs."""
    extra = device_info.extra or {}
    default_warp = 64 if (device_info.backend == "hip" and _parse_hip_gfx(device_info.architecture) < 0x1000) else 32
    warp_size = int(extra.get("warp_size") or extra.get("eu_simd_width") or default_warp)
    warp_size = max(16, min(64, warp_size))

    max_block = int(extra.get("max_threads_per_block") or 256)
    preferred_block = min(256, max(warp_size, max_block))
    # Align block size down to a warp/wavefront multiple for uniform MMA execution.
    block_x = max(warp_size, (preferred_block // warp_size) * warp_size)

    if target_threads is not None:
        grid_x = max(1, (target_threads + block_x - 1) // block_x)
    else:
        cu = max(1, int(device_info.compute_units or 16))
        if is_memory:
            grid_x = min(4096, max(128, cu * 32))
        elif is_tensor:
            grid_x = min(2048, max(64, cu * 16))
        else:
            grid_x = min(2048, max(64, cu * 16))

    total_threads = grid_x * block_x
    return (grid_x, 1, 1), (block_x, 1, 1), total_threads


def calibrate_iterations(
    launch_once: Callable[[int], None],
    synchronize: Callable[[], None],
    *,
    target_s: float = WATCHDOG_TARGET_DISPATCH_S,
    min_iters: int = DEFAULT_MIN_ITERATIONS,
    max_iters: int = DEFAULT_MAX_ITERATIONS,
    probe_iters: int = 1,
    round_multiple: int = 1,
    clock_fn: Callable[[], float] = time.perf_counter,
) -> int:
    """Dynamically calibrate iterations per dispatch inside the watchdog-safe window."""
    clamped_min = max(1, int(min_iters))
    clamped_max = max(clamped_min, int(max_iters))
    mult = max(1, int(round_multiple))
    clamped_target = max(WATCHDOG_MIN_DISPATCH_S, min(WATCHDOG_MAX_DISPATCH_S, float(target_s)))

    current_probe = max(clamped_min, min(clamped_max, int(probe_iters)))

    def _measure(iters: int) -> float:
        t0 = clock_fn()
        launch_once(iters)
        synchronize()
        return max(clock_fn() - t0, 1e-9)

    elapsed = _measure(current_probe)

    # Re-probe with a larger batch if initial probe was too short for stable timing.
    if elapsed < CALIBRATION_PROBE_THRESHOLD_S and current_probe < min(clamped_max, CALIBRATION_MAX_PROBE_ITERS):
        stepped_probe = min(clamped_max, CALIBRATION_MAX_PROBE_ITERS, max(current_probe * 8, current_probe + 1))
        if stepped_probe > current_probe:
            current_probe = stepped_probe
            elapsed = _measure(current_probe)

    scaled = int(round(current_probe * (clamped_target / elapsed)))
    clamped = max(clamped_min, min(clamped_max, scaled))
    if mult > 1:
        rounded = max(mult, ((clamped + mult // 2) // mult) * mult)
        clamped = max(clamped_min, min(clamped_max, rounded))
    return clamped


def is_unified_memory_device(device_info: BackendDeviceInfo) -> bool:
    """Return True when the device shares unified system memory with the host OS."""
    extra = device_info.extra or {}
    if extra.get("has_unified_memory") is True:
        return True
    if extra.get("integrated") is True or extra.get("is_low_power") is True:
        return True
    if device_info.vendor.lower() == "apple":
        return True
    return False


def query_host_memory_bytes() -> int | None:
    """Return total physical host RAM in bytes via POSIX sysconf, or None if unavailable."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            return int(page_size * phys_pages)
    except (AttributeError, OSError, ValueError):
        return None
    return None


def _align_down(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value // alignment) * alignment


def compute_safe_vram_budget(
    device_info: BackendDeviceInfo,
    *,
    mode: str,
    alignment: int = VRAM_ALIGNMENT_BYTES,
    min_bytes: int | None = None,
    host_ram_bytes: int | None = None,
) -> int:
    """Compute a safe, headroom-aware VRAM buffer size in bytes for memory/bandwidth workloads."""
    total_vram = int(device_info.total_memory_bytes or 0)
    free_vram = int(device_info.free_memory_bytes or 0)
    available = free_vram if free_vram > 0 else total_vram
    if available <= 0:
        raise BackendInitError(f"Device {device_info.name!r} reports zero available VRAM")

    unified = is_unified_memory_device(device_info)
    if unified:
        raw_headroom = max(UNIFIED_HEADROOM_MIN_BYTES, int(available * UNIFIED_HEADROOM_RATIO))
        # Preserve usable space on small/mocked test devices while respecting OS safety on real GPUs.
        headroom = min(raw_headroom, available // 3)
    else:
        raw_headroom = min(
            DISCRETE_HEADROOM_MAX_BYTES,
            max(DISCRETE_HEADROOM_MIN_BYTES, int((total_vram or available) * DISCRETE_HEADROOM_RATIO)),
        )
        headroom = min(raw_headroom, available // 4)

    usable = max(0, available - headroom)
    max_buf_len = int((device_info.extra or {}).get("max_buffer_length") or 0)

    if mode == "memory-pattern":
        target = int(usable * MEMORY_PATTERN_VRAM_RATIO)
        floor_bytes = MIN_STRESS_VRAM_BYTES if min_bytes is None else min_bytes
    elif mode == "memory-sweep":
        target = int(usable * MEMORY_SWEEP_VRAM_RATIO)
        floor_bytes = MIN_STRESS_VRAM_BYTES if min_bytes is None else min_bytes
    elif mode == "bandwidth-d2d":
        cap = BANDWIDTH_UNIFIED_CAP_BYTES if unified else BANDWIDTH_DISCRETE_CAP_BYTES
        # Allocate two equal buffers (src + dst), so each stays <= 25% of usable VRAM.
        target = min(cap, int(usable * BANDWIDTH_MAX_VRAM_RATIO))
        floor_bytes = MIN_BANDWIDTH_BYTES if min_bytes is None else min_bytes
    elif mode in ("bandwidth-h2d", "bandwidth-d2h"):
        cap = BANDWIDTH_UNIFIED_CAP_BYTES if unified else BANDWIDTH_DISCRETE_CAP_BYTES
        host_ram = query_host_memory_bytes() if host_ram_bytes is None else host_ram_bytes
        limits = [cap, int(usable * BANDWIDTH_MAX_VRAM_RATIO)]
        if host_ram is not None and host_ram > 0:
            limits.append(int(host_ram * BANDWIDTH_MAX_HOST_RAM_RATIO))
        target = min(limits)
        floor_bytes = MIN_BANDWIDTH_BYTES if min_bytes is None else min_bytes
    else:
        raise ValueError(f"Unknown VRAM sizing mode: {mode!r}")

    if max_buf_len > 0:
        target = min(target, max_buf_len)

    aligned = _align_down(target, alignment)
    effective_floor = _align_down(min(floor_bytes, usable), alignment)
    if aligned < max(alignment, effective_floor):
        aligned = _align_down(min(usable, max_buf_len if max_buf_len > 0 else usable), alignment)
    if aligned <= 0:
        raise BackendInitError(
            f"Insufficient VRAM on {device_info.name!r} for {mode} (available={available} bytes)"
        )
    return aligned


def alloc_device_with_stepdown(
    backend: Any,
    backend_ctx: Any,
    target_bytes: int,
    *,
    min_bytes: int = MIN_STRESS_VRAM_BYTES,
    alignment: int = VRAM_ALIGNMENT_BYTES,
    step_fraction: float = VRAM_STEPDOWN_FRACTION,
) -> tuple[DeviceBuffer, int]:
    """Allocate a device buffer, stepping down in 10% increments on allocation refusal."""
    effective_min = max(alignment, _align_down(min(min_bytes, target_bytes), alignment))
    current = _align_down(target_bytes, alignment)
    last_exc: Exception | None = None

    while current >= effective_min:
        try:
            buf = backend.alloc_device(backend_ctx, current)
            return buf, current
        except BackendInitError as exc:
            last_exc = exc
            if current == effective_min:
                break
            next_size = _align_down(int(current * (1.0 - step_fraction)), alignment)
            if next_size >= current:
                next_size = current - alignment
            current = max(effective_min, next_size)

    raise BackendInitError(
        f"Failed to allocate device buffer (target={target_bytes} B, min={effective_min} B): {last_exc}"
    ) from last_exc


def alloc_host_buffer_with_fallback(
    backend: Any,
    backend_ctx: Any,
    size_bytes: int,
) -> tuple[HostBuffer | bytearray, bool]:
    """Allocate pinned host memory, falling back cleanly to a standard host bytearray."""
    try:
        pinned = backend.alloc_pinned_host(backend_ctx, size_bytes)
        if isinstance(pinned, HostBuffer) and pinned.pointer:
            return pinned, True
        # Mock or backend returned a non-dereferenceable handle; fall back to bytearray.
        try:
            backend.free_pinned_host(backend_ctx, pinned)
        except Exception:
            pass
    except (BackendInitError, OSError):
        pass
    return bytearray(size_bytes), False


def free_host_buffer(
    backend: Any,
    backend_ctx: Any,
    buf: HostBuffer | bytearray | None,
    is_pinned: bool,
) -> None:
    """Idempotently release a pinned HostBuffer (standard bytearrays are GC-managed)."""
    if buf is None or not is_pinned:
        return
    if isinstance(buf, HostBuffer):
        backend.free_pinned_host(backend_ctx, buf)


def build_stamped_payload(size_bytes: int, seed: int) -> bytearray:
    """Build a deterministic non-zero payload with unique per-MiB chunk stamps.

    Tiling a 64 KiB PRNG block then stamping unique 64-bit hashes at every 1 MiB
    boundary avoids OS zero-page copy-on-write compression and detects aliased or
    truncated copies in milliseconds without third-party libraries.
    """
    tile_words = PAYLOAD_TILE_BYTES // 4
    words = [
        (golden_pattern_word(i, PATTERN_MODE_PRNG_HASH, seed) | 0x01010101)
        for i in range(tile_words)
    ]
    tile = struct.pack(f"<{tile_words}I", *words)
    repeats = (size_bytes + len(tile) - 1) // len(tile)
    buf = bytearray((tile * repeats)[:size_bytes])

    if size_bytes >= 16:
        offset = 0
        chunk_idx = 0
        while offset + 8 <= size_bytes:
            stamp = golden_splitmix64((seed ^ chunk_idx ^ offset) & 0xFFFFFFFFFFFFFFFF)
            struct.pack_into("<Q", buf, offset, stamp)
            offset += PAYLOAD_STAMP_STRIDE_BYTES
            chunk_idx += 1
        tail_stamp = golden_splitmix64((seed ^ size_bytes ^ 0x5A5A5A5A5A5A5A5A) & 0xFFFFFFFFFFFFFFFF)
        struct.pack_into("<Q", buf, size_bytes - 8, tail_stamp)
    return buf


def write_host_buffer(buf: HostBuffer | bytearray, data: bytes | bytearray | memoryview) -> None:
    """Copy bytes into either a pinned HostBuffer or a host bytearray."""
    n = len(data)
    if isinstance(buf, bytearray):
        buf[:n] = data
        return
    if isinstance(buf, HostBuffer) and buf.pointer:
        c_src = (ctypes.c_ubyte * n).from_buffer(data) if isinstance(data, (bytearray, memoryview)) else ctypes.create_string_buffer(bytes(data), n)
        ctypes.memmove(ctypes.c_void_p(buf.pointer), c_src, n)
        return
    raise BackendInitError("Host buffer has neither a writable bytearray nor a valid C pointer")


def fill_host_buffer_byte(buf: HostBuffer | bytearray, byte_val: int, size_bytes: int) -> None:
    """Fill a host buffer with a sentinel byte value prior to D2H transfers."""
    b = byte_val & 0xFF
    if isinstance(buf, bytearray):
        buf[:size_bytes] = bytes([b]) * size_bytes
        return
    if isinstance(buf, HostBuffer) and buf.pointer:
        ctypes.memset(ctypes.c_void_p(buf.pointer), b, size_bytes)
        return
    raise BackendInitError("Host buffer has neither a writable bytearray nor a valid C pointer")


def checksum_and_sentinels(
    buf: HostBuffer | bytearray | bytes | memoryview,
    size_bytes: int,
) -> tuple[int, bytes, bytes, bytes]:
    """Compute CRC-32 and (head, mid, tail) sentinel slices zero-copy via stdlib."""
    slice_len = min(SENTINEL_SLICE_BYTES, size_bytes)
    mid_start = max(0, (size_bytes - slice_len) // 2)
    tail_start = max(0, size_bytes - slice_len)

    if isinstance(buf, HostBuffer) and buf.pointer:
        raw_view = (ctypes.c_ubyte * size_bytes).from_address(buf.pointer)
        mv = memoryview(raw_view)
        crc = zlib.crc32(mv) & 0xFFFFFFFF
        return (
            crc,
            bytes(mv[:slice_len]),
            bytes(mv[mid_start : mid_start + slice_len]),
            bytes(mv[tail_start : tail_start + slice_len]),
        )

    mv = memoryview(buf)[:size_bytes]
    crc = zlib.crc32(mv) & 0xFFFFFFFF
    return (
        crc,
        bytes(mv[:slice_len]),
        bytes(mv[mid_start : mid_start + slice_len]),
        bytes(mv[tail_start : tail_start + slice_len]),
    )


# ---------------------------------------------------------------------------
# Pure-Python golden reference math & IEEE-754 / OCP bit-level decoders
# ---------------------------------------------------------------------------


def affine_pow_mod(a: int, b: int, steps: int, mask: int) -> tuple[int, int]:
    """Return (A, B) such that applying x -> (a*x + b) & mask `steps` times is (A*x + B) & mask."""
    res_a = 1
    res_b = 0
    cur_a = a & mask
    cur_b = b & mask
    n = int(steps)
    while n > 0:
        if n & 1:
            res_a = (cur_a * res_a) & mask
            res_b = (cur_a * res_b + cur_b) & mask
        cur_b = (cur_a * cur_b + cur_b) & mask
        cur_a = (cur_a * cur_a) & mask
        n >>= 1
    return res_a, res_b


def golden_int64_params(iterations: int, seed_a: int, seed_b: int) -> tuple[int, int, int]:
    """Precompute O(log N) affine parameters (A, B, seed_a) for compute_int64."""
    mask = 0xFFFFFFFFFFFFFFFF
    mul = (seed_a | 1) & mask
    add = (seed_b | 1) & mask
    aff_a, aff_b = affine_pow_mod(mul, add, iterations * 16, mask)
    return aff_a, aff_b, seed_a & mask


def golden_int64(gid: int, iterations: int, seed_a: int, seed_b: int, *, precomputed: tuple[int, int, int] | None = None) -> int:
    """Bit-exact unsigned 64-bit golden output for compute_int64."""
    mask = 0xFFFFFFFFFFFFFFFF
    aff_a, aff_b, sa = golden_int64_params(iterations, seed_a, seed_b) if precomputed is None else precomputed
    r = [
        (aff_a * (((gid + (k + 1) * 0x9E3779B97F4A7C15) ^ sa) & mask) + aff_b) & mask
        for k in range(8)
    ]
    return ((r[0] ^ r[1]) + (r[2] ^ r[3]) + (r[4] ^ r[5]) + (r[6] ^ r[7])) & mask


def golden_int32_params(iterations: int, seed_a: int, seed_b: int) -> tuple[int, int, int]:
    """Precompute O(log N) affine parameters (A, B, seed_a) for compute_int32."""
    mask = 0xFFFFFFFF
    mul = (seed_a | 1) & mask
    add = (seed_b | 1) & mask
    aff_a, aff_b = affine_pow_mod(mul, add, iterations * 16, mask)
    return aff_a, aff_b, seed_a & mask


def golden_int32(gid: int, iterations: int, seed_a: int, seed_b: int, *, precomputed: tuple[int, int, int] | None = None) -> int:
    """Bit-exact unsigned 32-bit golden output for compute_int32."""
    mask = 0xFFFFFFFF
    aff_a, aff_b, sa = golden_int32_params(iterations, seed_a, seed_b) if precomputed is None else precomputed
    r = [
        (aff_a * (((gid + (k + 1) * 0x9E3779B9) ^ sa) & mask) + aff_b) & mask
        for k in range(8)
    ]
    return ((r[0] ^ r[1]) + (r[2] ^ r[3]) + (r[4] ^ r[5]) + (r[6] ^ r[7])) & mask


def golden_int16_params(iterations: int, seed_a: int, seed_b: int) -> tuple[int, int, int]:
    """Precompute O(log N) affine parameters (A, B, seed_a) for compute_int16."""
    mask = 0xFFFF
    mul = (seed_a & mask) | 1
    add = (seed_b & mask) | 1
    aff_a, aff_b = affine_pow_mod(mul, add, iterations * 16, mask)
    return aff_a, aff_b, seed_a & mask


def golden_int16(gid: int, iterations: int, seed_a: int, seed_b: int, *, precomputed: tuple[int, int, int] | None = None) -> int:
    """Bit-exact unsigned 16-bit (stored in u32) golden output for compute_int16."""
    mask = 0xFFFF
    aff_a, aff_b, sa = golden_int16_params(iterations, seed_a, seed_b) if precomputed is None else precomputed
    r = [
        (aff_a * (((gid + (k + 1) * 0x9E37) ^ sa) & mask) + aff_b) & mask
        for k in range(8)
    ]
    return ((r[0] ^ r[1]) + (r[2] ^ r[3]) + (r[4] ^ r[5]) + (r[6] ^ r[7])) & mask


def golden_int8_uchar4_params(iterations: int, seed_a: int, seed_b: int) -> tuple[tuple[int, int, int], ...]:
    """Precompute O(log N) per-lane affine parameters for Metal/OpenCL/LevelZero uchar4 int8."""
    steps = iterations * 16
    lanes: list[tuple[int, int, int]] = []
    for lane in range(4):
        sa_l = (seed_a >> (8 * lane)) & 0xFF
        sb_l = (seed_b >> (8 * lane)) & 0xFF
        aff_a, aff_b = affine_pow_mod(sa_l | 1, sb_l | 1, steps, 0xFF)
        lanes.append((aff_a, aff_b, sa_l))
    return tuple(lanes)


def golden_int8_uchar4(
    gid: int,
    iterations: int,
    seed_a: int,
    seed_b: int,
    *,
    precomputed: tuple[tuple[int, int, int], ...] | None = None,
) -> int:
    """Bit-exact packed uchar4 golden output for Metal/OpenCL/LevelZero compute_int8."""
    lanes = golden_int8_uchar4_params(iterations, seed_a, seed_b) if precomputed is None else precomputed
    packed = 0
    for lane in range(4):
        aff_a, aff_b, sa_l = lanes[lane]
        r = [
            (aff_a * (((gid + (k + 1) * 37 + lane * 17) ^ sa_l) & 0xFF) + aff_b) & 0xFF
            for k in range(8)
        ]
        acc = ((r[0] ^ r[1]) + (r[2] ^ r[3]) + (r[4] ^ r[5]) + (r[6] ^ r[7])) & 0xFF
        packed |= acc << (8 * lane)
    return packed


def golden_int8_dp4a(gid: int, iterations: int, seed_a: int, seed_b: int) -> int:
    """Bit-exact golden output for CUDA dp4a.u32.u32 and HIP udot4 compute_int8."""
    mask = 0xFFFFFFFF
    mul_pack = (seed_a & 0x7F7F7F7F) | 0x01010101
    m0 = mul_pack & 0xFF
    m1 = (mul_pack >> 8) & 0xFF
    m2 = (mul_pack >> 16) & 0xFF
    m3 = (mul_pack >> 24) & 0xFF
    mix = (seed_a ^ seed_b) & mask
    steps = iterations * 16
    r: list[int] = []
    for k in range(8):
        val = (((gid + (k + 1) * 0x9E3779B9) ^ mix) | 1) & mask
        for _ in range(steps):
            dot = (
                (val & 0xFF) * m0
                + ((val >> 8) & 0xFF) * m1
                + ((val >> 16) & 0xFF) * m2
                + ((val >> 24) & 0xFF) * m3
            )
            val = (val + dot) & mask
        r.append(val)
    return ((r[0] ^ r[1]) + (r[2] ^ r[3]) + (r[4] ^ r[5]) + (r[6] ^ r[7])) & mask


def golden_int4_cuda(gid: int, seed_a: int, seed_b: int) -> int:
    """Bit-exact analytical golden output for CUDA mma.sync.aligned.m8n8k32 s4 compute_int4."""
    base = (gid ^ seed_a ^ seed_b) & 0xFFFF
    return (8 * base + 36) & 0xFFFFFFFF


def golden_int4_hip_sdot8(gid: int, iterations: int, seed_a: int, seed_b: int) -> int:
    """Bit-exact golden output for HIP __builtin_amdgcn_sdot8 compute_int4."""
    mask = 0xFFFFFFFF
    mul_pack = (seed_a & 0x77777777) | 0x11111111
    m_nib = [(mul_pack >> (4 * i)) & 0xF for i in range(8)]
    mix = (seed_a ^ seed_b) & mask
    steps = iterations * 16
    r: list[int] = []
    for k in range(8):
        val = (((gid + (k + 1) * 0x9E3779B9) ^ mix) & 0x77777777) & mask
        for _ in range(steps):
            dot = 0
            for i in range(8):
                nib = (val >> (4 * i)) & 0xF
                signed_nib = nib - 16 if nib >= 8 else nib
                dot += signed_nib * m_nib[i]
            val = (val + dot) & mask
        r.append(val)
    return ((r[0] ^ r[1]) + (r[2] ^ r[3]) + (r[4] ^ r[5]) + (r[6] ^ r[7])) & mask


def round_fp32(value: float) -> float:
    """Round a Python float to IEEE-754 binary32 precision."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def round_fp16(value: float) -> float:
    """Round a Python float to IEEE-754 binary16 precision via struct 'e'."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def round_bf16(value: float) -> float:
    """Round a Python float to bfloat16 (IEEE-754 round-to-nearest-even)."""
    if not math.isfinite(value):
        return value
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded_bits))[0]


def decode_fp8_e4m3(byte_val: int, *, fnuz: bool = False) -> float:
    """Decode an 8-bit FP8 E4M3FN (OCP/CUDA/RDNA4) or E4M3FNUZ (AMD CDNA3) byte."""
    b = byte_val & 0xFF
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 3) & 0x0F
    mant = b & 0x07
    if fnuz:
        if b == 0x80:
            return float("nan")
        if exp == 0:
            return sign * (mant / 8.0) * (2.0 ** -7)
        return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 8))
    if exp == 15 and mant == 7:
        return float("nan")
    if exp == 0:
        return sign * (mant / 8.0) * (2.0 ** -6)
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


def decode_fp4_e2m1(nibble_val: int) -> float:
    """Decode an OCP MXFP4 E2M1 4-bit nibble into a Python float."""
    n = nibble_val & 0x0F
    mag = _FP4_E2M1_MAGNITUDES[n & 0x07]
    return -mag if (n & 0x08) else mag


def golden_fp64_table(iterations: int, seed_a: float, seed_b: float) -> tuple[float, ...]:
    """Compute the 16-entry golden table (indexed by gid & 15) for compute_fp64."""
    a = 0.25 * (seed_a - math.floor(seed_a)) + 0.5
    bp = 0.25 * (seed_b - math.floor(seed_b)) + 0.25
    bn = -0.5 * bp
    # Contractive map (a^2 <= 0.5625) reaches FP64 fixed point within 128 pairs.
    pairs = min(iterations * 8, 128)
    orbit: list[float] = []
    for j in range(16):
        x = 0.125 + 0.0625 * j
        for _ in range(pairs):
            x = x * a + bp
            x = x * a + bn
        orbit.append(x)
    return tuple(
        (orbit[(g + 0) & 15] + orbit[(g + 1) & 15])
        + (orbit[(g + 2) & 15] + orbit[(g + 3) & 15])
        + (orbit[(g + 4) & 15] + orbit[(g + 5) & 15])
        + (orbit[(g + 6) & 15] + orbit[(g + 7) & 15])
        for g in range(16)
    )


def golden_fp32_table(iterations: int, seed_a: float, seed_b: float) -> tuple[float, ...]:
    """Compute the 16-entry golden table (indexed by gid & 15) for compute_fp32."""
    a = round_fp32(0.25 * (seed_a - math.floor(seed_a)) + 0.5)
    bp = round_fp32(0.25 * (seed_b - math.floor(seed_b)) + 0.25)
    bn = round_fp32(-0.5 * bp)
    pairs = min(iterations * 8, 96)
    orbit: list[float] = []
    for j in range(16):
        x = round_fp32(0.125 + 0.0625 * j)
        for _ in range(pairs):
            x = round_fp32(x * a + bp)
            x = round_fp32(x * a + bn)
        orbit.append(x)
    return tuple(
        round_fp32(
            round_fp32(orbit[(g + 0) & 15] + orbit[(g + 1) & 15])
            + round_fp32(orbit[(g + 2) & 15] + orbit[(g + 3) & 15])
            + round_fp32(orbit[(g + 4) & 15] + orbit[(g + 5) & 15])
            + round_fp32(orbit[(g + 6) & 15] + orbit[(g + 7) & 15])
        )
        for g in range(16)
    )


def golden_fp16_table(iterations: int, seed_a: float, seed_b: float) -> tuple[float, ...]:
    """Compute the 16-entry golden table (indexed by gid & 15) for compute_fp16."""
    a = round_fp16(0.25 * (seed_a - math.floor(seed_a)) + 0.5)
    bp = round_fp16(0.25 * (seed_b - math.floor(seed_b)) + 0.25)
    bn = round_fp16(-0.5 * bp)
    pairs = min(iterations * 8, 64)
    lane_sums: list[float] = []
    for j in range(16):
        lo = round_fp16(0.125 + 0.0625 * j)
        hi = round_fp16(0.1875 + 0.0625 * j)
        for _ in range(pairs):
            lo = round_fp16(lo * a + bp)
            lo = round_fp16(lo * a + bn)
            hi = round_fp16(hi * a + bp)
            hi = round_fp16(hi * a + bn)
        lane_sums.append(round_fp32(lo + hi))
    return tuple(
        round_fp32(
            round_fp32(lane_sums[(g + 0) & 15] + lane_sums[(g + 1) & 15])
            + round_fp32(lane_sums[(g + 2) & 15] + lane_sums[(g + 3) & 15])
            + round_fp32(lane_sums[(g + 4) & 15] + lane_sums[(g + 5) & 15])
            + round_fp32(lane_sums[(g + 6) & 15] + lane_sums[(g + 7) & 15])
        )
        for g in range(16)
    )


def golden_bf16_table(iterations: int, seed_a: float, seed_b: float) -> tuple[float, ...]:
    """Compute the 16-entry golden table (indexed by gid & 15) for compute_bf16."""
    a = round_bf16(0.25 * (seed_a - math.floor(seed_a)) + 0.5)
    bp = round_bf16(0.25 * (seed_b - math.floor(seed_b)) + 0.25)
    bn = round_bf16(-0.5 * bp)
    pairs = min(iterations * 8, 64)
    lane_sums: list[float] = []
    for j in range(16):
        lo = round_bf16(0.125 + 0.0625 * j)
        hi = round_bf16(0.1875 + 0.0625 * j)
        for _ in range(pairs):
            lo = round_bf16(lo * a + bp)
            lo = round_bf16(lo * a + bn)
            hi = round_bf16(hi * a + bp)
            hi = round_bf16(hi * a + bn)
        lane_sums.append(round_fp32(lo + hi))
    return tuple(
        round_fp32(
            round_fp32(lane_sums[(g + 0) & 15] + lane_sums[(g + 1) & 15])
            + round_fp32(lane_sums[(g + 2) & 15] + lane_sums[(g + 3) & 15])
            + round_fp32(lane_sums[(g + 4) & 15] + lane_sums[(g + 5) & 15])
            + round_fp32(lane_sums[(g + 6) & 15] + lane_sums[(g + 7) & 15])
        )
        for g in range(16)
    )


def golden_fp_mma_table(
    seed_a: float,
    seed_b: float,
    *,
    metal_layout: bool = False,
) -> tuple[float, ...]:
    """Exact analytical golden table for paired (+B, -B) FP8/FP4/Tensor-FP/BF16 kernels."""
    frac_a = seed_a - math.floor(seed_a)
    frac_b = seed_b - math.floor(seed_b)
    if metal_layout:
        base = 0.5 + 0.25 * frac_a + 0.25 * frac_b
        expected = 8.0 * base + 3.0
        return (expected,) * 16
    return tuple(
        16.0 * (0.5 + 0.25 * frac_a + 0.25 * frac_b + 0.0625 * j) + 15.0
        for j in range(16)
    )


def golden_tensor_int8(gid: int, seed_a: int, seed_b: int) -> int:
    """Bit-exact analytical golden output for compute_tensor_int8 across CUDA/HIP/L0/OpenCL."""
    base = (gid ^ seed_a ^ seed_b) & 0xFFFF
    return (16 * base + 136) & 0xFFFFFFFF


def golden_pattern_word(idx: int, mode: int, seed: int) -> int:
    """Bit-exact 32-bit pattern word matching kernel_memory_pattern_{write,verify}."""
    if mode == PATTERN_MODE_ALL_ZEROS:
        return 0x00000000
    if mode == PATTERN_MODE_ALL_ONES:
        return 0xFFFFFFFF
    if mode == PATTERN_MODE_CHECKERBOARD_55:
        return 0x55555555
    if mode == PATTERN_MODE_CHECKERBOARD_AA:
        return 0xAAAAAAAA
    if mode == PATTERN_MODE_WALKING_ONES:
        return (1 << (idx & 31)) & 0xFFFFFFFF
    if mode == PATTERN_MODE_WALKING_ZEROS:
        return (~(1 << (idx & 31))) & 0xFFFFFFFF
    z = ((idx ^ seed ^ (seed >> 32)) + 0x9E3779B9) & 0xFFFFFFFF
    z = ((z ^ (z >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    z = ((z ^ (z >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    return (z ^ (z >> 16)) & 0xFFFFFFFF


def golden_splitmix64(x: int) -> int:
    """Bit-exact 64-bit SplitMix64 hash matching kernel_memory_sweep_{write,verify}."""
    mask = 0xFFFFFFFFFFFFFFFF
    z = (x + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def select_verify_indices(num_elements: int, *, max_samples: int) -> tuple[int, ...] | None:
    """Return None to verify all elements, or deterministic boundary + strided indices."""
    if num_elements <= max_samples:
        return None
    count = max(MIN_VERIFY_SAMPLES, max_samples)
    stride = max(1, (num_elements - 1) // (count - 1))
    indices = list(range(0, num_elements, stride))
    if indices[-1] != num_elements - 1:
        indices.append(num_elements - 1)
    return tuple(indices)


def verify_int_values(
    actual: Sequence[int],
    expected_fn: Callable[[int], int],
    *,
    indices: Sequence[int] | None = None,
    workload_name: str = "",
) -> VerificationResult:
    """Verify unsigned integer buffer elements bit-for-bit against expected_fn."""
    mismatches = 0
    first_detail = ""
    check_iter = range(len(actual)) if indices is None else indices
    for idx in check_iter:
        got = actual[idx]
        exp = expected_fn(idx)
        if got != exp:
            mismatches += 1
            if not first_detail:
                first_detail = f"index={idx} got=0x{got:x} expected=0x{exp:x}"
    if mismatches > 0:
        prefix = f"{workload_name}: " if workload_name else ""
        return VerificationResult(
            passed=False,
            error_count=mismatches,
            message=f"{prefix}{mismatches} bitwise mismatch(es); first at {first_detail}",
        )
    return VerificationResult(passed=True, error_count=0)


def verify_float_values(
    actual: Sequence[float],
    expected_by_mod16: Sequence[float],
    *,
    rel_tol: float,
    abs_tol: float,
    workload_name: str = "",
) -> VerificationResult:
    """Verify floating-point buffer elements, explicitly rejecting NaN/Inf and drift."""
    mismatches = 0
    first_detail = ""
    for idx, got in enumerate(actual):
        exp = expected_by_mod16[idx & 15]
        if not math.isfinite(got):
            mismatches += 1
            if not first_detail:
                first_detail = f"index={idx} non-finite value={got!r} expected={exp:.6g}"
            continue
        tol = max(abs_tol, rel_tol * abs(exp))
        if abs(got - exp) > tol:
            mismatches += 1
            if not first_detail:
                first_detail = (
                    f"index={idx} got={got:.8g} expected={exp:.8g} "
                    f"diff={abs(got - exp):.3g} tol={tol:.3g}"
                )
    if mismatches > 0:
        prefix = f"{workload_name}: " if workload_name else ""
        return VerificationResult(
            passed=False,
            error_count=mismatches,
            message=f"{prefix}{mismatches} numerical mismatch(es); first at {first_detail}",
        )
    return VerificationResult(passed=True, error_count=0)
