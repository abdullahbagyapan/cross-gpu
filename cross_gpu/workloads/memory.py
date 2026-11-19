# SPDX-License-Identifier: MIT
"""Memory pattern, full-capacity sweep, and bandwidth workload executors (Group 4)."""

from __future__ import annotations

import ctypes
import struct
from typing import Any

from cross_gpu.backends.base import DeviceBuffer
from cross_gpu.exit_codes import BackendInitError
from cross_gpu.models import StepMetrics, VerificationResult
from cross_gpu.workloads.common import (
    MIB,
    MIN_BANDWIDTH_BYTES,
    MIN_STRESS_VRAM_BYTES,
    PATTERN_MODE_NAMES,
    VRAM_ALIGNMENT_BYTES,
    VRAM_STEPDOWN_FRACTION,
    alloc_device_with_stepdown,
    alloc_host_buffer_with_fallback,
    build_stamped_payload,
    calibrate_iterations,
    checksum_and_sentinels,
    compute_launch_geometry,
    compute_safe_vram_budget,
    fill_host_buffer_byte,
    free_host_buffer,
    golden_pattern_word,
    golden_splitmix64,
    write_host_buffer,
)
from cross_gpu.workloads.registry import WorkloadExecutionContext, WorkloadExecutor

MEMORY_PATTERN_WRITE_EP = "kernel_memory_pattern_write"
MEMORY_PATTERN_VERIFY_EP = "kernel_memory_pattern_verify"
MEMORY_SWEEP_WRITE_EP = "kernel_memory_sweep_write"
MEMORY_SWEEP_VERIFY_EP = "kernel_memory_sweep_verify"
BANDWIDTH_D2D_EP = "kernel_bandwidth_d2d"

BANDWIDTH_MAX_TRANSFERS_PER_STEP = 32
SENTINEL_SCRUB_BYTE_DST = 0xA5
SENTINEL_SCRUB_BYTE_VERIFY = 0x5A

_INITIAL_PATTERN_SEED = 0x243F6A8885A308D3
_INITIAL_SWEEP_SEED = 0xB7E151628AED2A6A
_INITIAL_BANDWIDTH_SEED = 0x9E3779B97F4A7C15


def _alloc_device_pair_with_stepdown(
    backend: Any,
    backend_ctx: Any,
    target_bytes: int,
    *,
    min_bytes: int,
    alignment: int = VRAM_ALIGNMENT_BYTES,
) -> tuple[DeviceBuffer, DeviceBuffer, int]:
    """Allocate two equal-sized device buffers (src, dst) with leak-free 10% step-down."""
    effective_min = max(alignment, ((min(min_bytes, target_bytes)) // alignment) * alignment)
    current = (target_bytes // alignment) * alignment
    last_exc: Exception | None = None

    while current >= effective_min:
        src_buf: DeviceBuffer | None = None
        try:
            src_buf = backend.alloc_device(backend_ctx, current)
            dst_buf = backend.alloc_device(backend_ctx, current)
            return src_buf, dst_buf, current
        except BackendInitError as exc:
            last_exc = exc
            if src_buf is not None:
                try:
                    backend.free_device(backend_ctx, src_buf)
                except Exception:
                    pass
            if current == effective_min:
                break
            next_size = (int(current * (1.0 - VRAM_STEPDOWN_FRACTION)) // alignment) * alignment
            if next_size >= current:
                next_size = current - alignment
            current = max(effective_min, next_size)

    raise BackendInitError(
        f"Failed to allocate paired D2D device buffers (target={target_bytes} B, min={effective_min} B): {last_exc}"
    ) from last_exc


def _compare_checksum_and_sentinels(
    workload: str,
    expected: tuple[int, bytes, bytes, bytes],
    actual: tuple[int, bytes, bytes, bytes],
    ok_message: str,
) -> VerificationResult:
    exp_crc, exp_head, exp_mid, exp_tail = expected
    act_crc, act_head, act_mid, act_tail = actual
    if act_crc != exp_crc:
        return VerificationResult(
            passed=False,
            error_count=1,
            message=f"{workload}: payload CRC-32 mismatch (got=0x{act_crc:08x}, expected=0x{exp_crc:08x})",
        )
    if act_head != exp_head or act_mid != exp_mid or act_tail != exp_tail:
        return VerificationResult(
            passed=False,
            error_count=1,
            message=f"{workload}: sentinel slice mismatch despite matching CRC-32",
        )
    return VerificationResult(passed=True, error_count=0, message=ok_message)


class MemoryPatternExecutor(WorkloadExecutor):
    """VRAM adversarial bit-pattern retention and fault-detection stress executor."""

    requires_kernel: bool = True

    def prepare(self, ctx: WorkloadExecutionContext, binary: Any) -> None:
        if binary is None:
            raise BackendInitError(f"Missing compiled kernel binary for {self.spec.name!r}")

        backend = ctx.backend
        bctx = ctx.backend_context
        dev_info = ctx.binding.device_info

        ctx.module = backend.load_module(bctx, binary)
        override_bytes = ctx.state.get("target_bytes")
        target_bytes = (
            int(override_bytes)
            if override_bytes is not None
            else compute_safe_vram_budget(dev_info, mode="memory-pattern", alignment=VRAM_ALIGNMENT_BYTES)
        )
        min_bytes = min(MIN_STRESS_VRAM_BYTES, target_bytes)
        buf, alloc_bytes = alloc_device_with_stepdown(
            backend, bctx, target_bytes, min_bytes=min_bytes, alignment=VRAM_ALIGNMENT_BYTES
        )
        ctx.buffers["vram"] = buf
        ctx.buffers["error_count"] = backend.alloc_device(bctx, 4)

        grid, block, _ = compute_launch_geometry(dev_info, is_memory=True)
        ctx.state.update(
            {
                "grid": grid,
                "block": block,
                "buffer_bytes": alloc_bytes,
                "num_words": alloc_bytes // 4,
                "mode_index": 0,
                "seed": _INITIAL_PATTERN_SEED,
                "last_mode": None,
                "last_seed": None,
                "total_errors": 0,
                "failed_detail": "",
                "err_host_buf": bytearray(4),
                "spot_host_buf": bytearray(4),
            }
        )

    def reset_run_state(self, ctx: WorkloadExecutionContext) -> None:
        ctx.state["mode_index"] = 0
        ctx.state["seed"] = golden_splitmix64(int(ctx.state.get("seed", _INITIAL_PATTERN_SEED)))
        ctx.state["total_errors"] = 0
        ctx.state["failed_detail"] = ""

    def run_step(self, ctx: WorkloadExecutionContext) -> StepMetrics:
        backend = ctx.backend
        bctx = ctx.backend_context
        vram_buf = ctx.buffers["vram"]
        err_buf = ctx.buffers["error_count"]

        mode_idx = int(ctx.state["mode_index"])
        mode = mode_idx % len(PATTERN_MODE_NAMES)
        seed = int(ctx.state["seed"])
        num_words = int(ctx.state["num_words"])
        buffer_bytes = int(ctx.state["buffer_bytes"])
        grid = ctx.state["grid"]
        block = ctx.state["block"]

        backend.memcpy_h2d(bctx, err_buf, b"\x00\x00\x00\x00", 4)
        backend.launch_kernel(
            bctx,
            ctx.module,
            MEMORY_PATTERN_WRITE_EP,
            grid,
            block,
            [vram_buf, ctypes.c_uint64(num_words), ctypes.c_uint32(mode), ctypes.c_uint64(seed)],
        )
        backend.synchronize(bctx)

        backend.launch_kernel(
            bctx,
            ctx.module,
            MEMORY_PATTERN_VERIFY_EP,
            grid,
            block,
            [
                vram_buf,
                ctypes.c_uint64(num_words),
                ctypes.c_uint32(mode),
                ctypes.c_uint64(seed),
                err_buf,
            ],
        )
        backend.synchronize(bctx)

        err_host = ctx.state["err_host_buf"]
        backend.memcpy_d2h(bctx, err_host, err_buf, 4)
        err_count = int(struct.unpack("<I", err_host)[0])

        ctx.state["last_mode"] = mode
        ctx.state["last_seed"] = seed
        next_mode_idx = mode_idx + 1
        ctx.state["mode_index"] = next_mode_idx
        if next_mode_idx % len(PATTERN_MODE_NAMES) == 0:
            ctx.state["seed"] = golden_splitmix64(seed)

        if err_count > 0:
            ctx.state["total_errors"] = int(ctx.state["total_errors"]) + err_count
            if not ctx.state["failed_detail"]:
                ctx.state["failed_detail"] = (
                    f"pattern={PATTERN_MODE_NAMES[mode]} (mode={mode}) "
                    f"seed=0x{seed:016x} errors={err_count}"
                )

        # Full write + full verify read traverses 2 * buffer_bytes of VRAM traffic.
        return StepMetrics(operations=float(2 * buffer_bytes), errors=err_count)

    def verify(self, ctx: WorkloadExecutionContext) -> VerificationResult:
        total_errors = int(ctx.state.get("total_errors", 0))
        if total_errors > 0:
            return VerificationResult(
                passed=False,
                error_count=total_errors,
                message=f"memory-pattern: {ctx.state.get('failed_detail') or 'pattern verification error'}",
            )

        last_mode = ctx.state.get("last_mode")
        last_seed = ctx.state.get("last_seed")
        if last_mode is None or last_seed is None:
            return VerificationResult(
                passed=False,
                error_count=1,
                message="memory-pattern: no pattern cycle executed prior to verify",
            )

        # Host spot-check word 0 so a silent no-op verify kernel cannot mask stale VRAM.
        spot_buf = ctx.state["spot_host_buf"]
        ctx.backend.memcpy_d2h(ctx.backend_context, spot_buf, ctx.buffers["vram"], 4)
        got_word0 = int(struct.unpack("<I", spot_buf)[0])
        exp_word0 = golden_pattern_word(0, int(last_mode), int(last_seed))
        if got_word0 != exp_word0:
            return VerificationResult(
                passed=False,
                error_count=1,
                message=(
                    f"memory-pattern: host spot-check mismatch at word 0 "
                    f"(mode={PATTERN_MODE_NAMES[int(last_mode)]}, "
                    f"got=0x{got_word0:08x}, expected=0x{exp_word0:08x})"
                ),
            )

        alloc_mib = int(ctx.state["buffer_bytes"]) / MIB
        return VerificationResult(
            passed=True,
            error_count=0,
            message=f"allocated={alloc_mib:.1f} MiB accounting=2x_rw",
        )

    def cleanup(self, ctx: WorkloadExecutionContext) -> None:
        backend = ctx.backend
        bctx = ctx.backend_context
        vram_buf = ctx.buffers.pop("vram", None)
        err_buf = ctx.buffers.pop("error_count", None)
        try:
            if vram_buf is not None:
                backend.free_device(bctx, vram_buf)
        finally:
            try:
                if err_buf is not None:
                    backend.free_device(bctx, err_buf)
            finally:
                if ctx.module is not None:
                    mod = ctx.module
                    ctx.module = None
                    backend.unload_module(bctx, mod)


class MemorySweepExecutor(WorkloadExecutor):
    """Full-capacity streaming SplitMix64 address-uniqueness VRAM sweep executor."""

    requires_kernel: bool = True

    def prepare(self, ctx: WorkloadExecutionContext, binary: Any) -> None:
        if binary is None:
            raise BackendInitError(f"Missing compiled kernel binary for {self.spec.name!r}")

        backend = ctx.backend
        bctx = ctx.backend_context
        dev_info = ctx.binding.device_info

        ctx.module = backend.load_module(bctx, binary)
        override_bytes = ctx.state.get("target_bytes")
        target_bytes = (
            int(override_bytes)
            if override_bytes is not None
            else compute_safe_vram_budget(dev_info, mode="memory-sweep", alignment=VRAM_ALIGNMENT_BYTES)
        )
        min_bytes = min(MIN_STRESS_VRAM_BYTES, target_bytes)
        buf, alloc_bytes = alloc_device_with_stepdown(
            backend, bctx, target_bytes, min_bytes=min_bytes, alignment=VRAM_ALIGNMENT_BYTES
        )
        ctx.buffers["vram"] = buf
        ctx.buffers["error_count"] = backend.alloc_device(bctx, 4)

        grid, block, _ = compute_launch_geometry(dev_info, is_memory=True)
        ctx.state.update(
            {
                "grid": grid,
                "block": block,
                "buffer_bytes": alloc_bytes,
                "num_qwords": alloc_bytes // 8,
                "seed": _INITIAL_SWEEP_SEED,
                "last_seed": None,
                "total_errors": 0,
                "failed_detail": "",
                "err_host_buf": bytearray(4),
                "spot_host_buf": bytearray(8),
            }
        )

    def reset_run_state(self, ctx: WorkloadExecutionContext) -> None:
        ctx.state["seed"] = golden_splitmix64(int(ctx.state.get("seed", _INITIAL_SWEEP_SEED)))
        ctx.state["total_errors"] = 0
        ctx.state["failed_detail"] = ""

    def run_step(self, ctx: WorkloadExecutionContext) -> StepMetrics:
        backend = ctx.backend
        bctx = ctx.backend_context
        vram_buf = ctx.buffers["vram"]
        err_buf = ctx.buffers["error_count"]

        seed = int(ctx.state["seed"])
        num_qwords = int(ctx.state["num_qwords"])
        buffer_bytes = int(ctx.state["buffer_bytes"])
        grid = ctx.state["grid"]
        block = ctx.state["block"]

        backend.memcpy_h2d(bctx, err_buf, b"\x00\x00\x00\x00", 4)
        backend.launch_kernel(
            bctx,
            ctx.module,
            MEMORY_SWEEP_WRITE_EP,
            grid,
            block,
            [vram_buf, ctypes.c_uint64(num_qwords), ctypes.c_uint64(seed)],
        )
        backend.synchronize(bctx)

        backend.launch_kernel(
            bctx,
            ctx.module,
            MEMORY_SWEEP_VERIFY_EP,
            grid,
            block,
            [vram_buf, ctypes.c_uint64(num_qwords), ctypes.c_uint64(seed), err_buf],
        )
        backend.synchronize(bctx)

        err_host = ctx.state["err_host_buf"]
        backend.memcpy_d2h(bctx, err_host, err_buf, 4)
        err_count = int(struct.unpack("<I", err_host)[0])

        ctx.state["last_seed"] = seed
        ctx.state["seed"] = golden_splitmix64(seed)

        if err_count > 0:
            ctx.state["total_errors"] = int(ctx.state["total_errors"]) + err_count
            if not ctx.state["failed_detail"]:
                ctx.state["failed_detail"] = f"seed=0x{seed:016x} errors={err_count}"

        return StepMetrics(operations=float(2 * buffer_bytes), errors=err_count)

    def verify(self, ctx: WorkloadExecutionContext) -> VerificationResult:
        total_errors = int(ctx.state.get("total_errors", 0))
        if total_errors > 0:
            return VerificationResult(
                passed=False,
                error_count=total_errors,
                message=f"memory-sweep: {ctx.state.get('failed_detail') or 'sweep verification error'}",
            )

        last_seed = ctx.state.get("last_seed")
        if last_seed is None:
            return VerificationResult(
                passed=False,
                error_count=1,
                message="memory-sweep: no sweep pass executed prior to verify",
            )

        spot_buf = ctx.state["spot_host_buf"]
        ctx.backend.memcpy_d2h(ctx.backend_context, spot_buf, ctx.buffers["vram"], 8)
        got_qword0 = int(struct.unpack("<Q", spot_buf)[0])
        exp_qword0 = golden_splitmix64(int(last_seed))
        if got_qword0 != exp_qword0:
            return VerificationResult(
                passed=False,
                error_count=1,
                message=(
                    f"memory-sweep: host spot-check mismatch at qword 0 "
                    f"(got=0x{got_qword0:016x}, expected=0x{exp_qword0:016x})"
                ),
            )

        alloc_mib = int(ctx.state["buffer_bytes"]) / MIB
        return VerificationResult(
            passed=True,
            error_count=0,
            message=f"allocated={alloc_mib:.1f} MiB accounting=2x_rw",
        )

    def cleanup(self, ctx: WorkloadExecutionContext) -> None:
        backend = ctx.backend
        bctx = ctx.backend_context
        vram_buf = ctx.buffers.pop("vram", None)
        err_buf = ctx.buffers.pop("error_count", None)
        try:
            if vram_buf is not None:
                backend.free_device(bctx, vram_buf)
        finally:
            try:
                if err_buf is not None:
                    backend.free_device(bctx, err_buf)
            finally:
                if ctx.module is not None:
                    mod = ctx.module
                    ctx.module = None
                    backend.unload_module(bctx, mod)


class BandwidthD2DExecutor(WorkloadExecutor):
    """Device-to-device memory bandwidth executor (2 * copy_bytes effective DRAM traffic)."""

    requires_kernel: bool = True

    def prepare(self, ctx: WorkloadExecutionContext, binary: Any) -> None:
        backend = ctx.backend
        bctx = ctx.backend_context
        dev_info = ctx.binding.device_info

        if binary is not None:
            ctx.module = backend.load_module(bctx, binary)
            method = BANDWIDTH_D2D_EP
            entry_point = str(getattr(binary, "entry_point", "") or BANDWIDTH_D2D_EP)
        else:
            method = "memcpy_d2d"
            entry_point = ""

        override_bytes = ctx.state.get("target_bytes")
        target_bytes = (
            int(override_bytes)
            if override_bytes is not None
            else compute_safe_vram_budget(dev_info, mode="bandwidth-d2d", alignment=VRAM_ALIGNMENT_BYTES)
        )
        min_bytes = min(MIN_BANDWIDTH_BYTES, target_bytes)
        src_buf, dst_buf, copy_bytes = _alloc_device_pair_with_stepdown(
            backend, bctx, target_bytes, min_bytes=min_bytes, alignment=VRAM_ALIGNMENT_BYTES
        )
        ctx.buffers["src"] = src_buf
        ctx.buffers["dst"] = dst_buf

        host_buf, is_pinned = alloc_host_buffer_with_fallback(backend, bctx, copy_bytes)
        ctx.buffers["host_work"] = host_buf
        ctx.state["host_work_pinned"] = is_pinned

        payload = build_stamped_payload(copy_bytes, _INITIAL_BANDWIDTH_SEED)
        expected = checksum_and_sentinels(payload, copy_bytes)
        write_host_buffer(host_buf, payload)
        backend.memcpy_h2d(bctx, src_buf, host_buf, copy_bytes)

        # Scrub dst with sentinel bytes so a no-op copy cannot pass verification.
        fill_host_buffer_byte(host_buf, SENTINEL_SCRUB_BYTE_DST, copy_bytes)
        backend.memcpy_h2d(bctx, dst_buf, host_buf, copy_bytes)
        backend.synchronize(bctx)

        grid, block, _ = compute_launch_geometry(dev_info, is_memory=True)
        num_vec4 = copy_bytes // 16
        ctx.state.update(
            {
                "method": method,
                "entry_point": entry_point,
                "grid": grid,
                "block": block,
                "copy_bytes": copy_bytes,
                "num_vec4": num_vec4,
                "expected_digest": expected,
            }
        )

        fixed_copies = ctx.state.get("fixed_iterations")
        if fixed_copies is not None:
            copies_per_step = max(1, int(fixed_copies))
        else:
            copies_per_step = calibrate_iterations(
                lambda n: self._dispatch_copies(ctx, n),
                lambda: backend.synchronize(bctx),
                min_iters=1,
                max_iters=BANDWIDTH_MAX_TRANSFERS_PER_STEP,
            )
        ctx.state["copies_per_step"] = copies_per_step

    def _dispatch_copies(self, ctx: WorkloadExecutionContext, count: int) -> None:
        backend = ctx.backend
        bctx = ctx.backend_context
        src_buf = ctx.buffers["src"]
        dst_buf = ctx.buffers["dst"]
        copy_bytes = int(ctx.state["copy_bytes"])
        if ctx.module is not None:
            args = [dst_buf, src_buf, ctypes.c_uint64(int(ctx.state["num_vec4"]))]
            ep = ctx.state["entry_point"]
            grid = ctx.state["grid"]
            block = ctx.state["block"]
            for _ in range(count):
                backend.launch_kernel(bctx, ctx.module, ep, grid, block, args)
        else:
            for _ in range(count):
                backend.memcpy_d2d(bctx, dst_buf, src_buf, copy_bytes)

    def reset_run_state(self, ctx: WorkloadExecutionContext) -> None:
        # Re-scrub dst after warmup so final verify confirms timed copies populated dst.
        host_buf = ctx.buffers.get("host_work")
        dst_buf = ctx.buffers.get("dst")
        copy_bytes = int(ctx.state.get("copy_bytes", 0))
        if host_buf is not None and dst_buf is not None and copy_bytes > 0:
            fill_host_buffer_byte(host_buf, SENTINEL_SCRUB_BYTE_DST, copy_bytes)
            ctx.backend.memcpy_h2d(ctx.backend_context, dst_buf, host_buf, copy_bytes)
            ctx.backend.synchronize(ctx.backend_context)

    def run_step(self, ctx: WorkloadExecutionContext) -> StepMetrics:
        copies = int(ctx.state["copies_per_step"])
        copy_bytes = int(ctx.state["copy_bytes"])
        self._dispatch_copies(ctx, copies)
        # Count effective DRAM traffic: 16-byte uint4 read + 16-byte uint4 write = 2 * copy_bytes.
        return StepMetrics(operations=float(copies * 2 * copy_bytes), errors=0)

    def verify(self, ctx: WorkloadExecutionContext) -> VerificationResult:
        copy_bytes = int(ctx.state["copy_bytes"])
        host_buf = ctx.buffers["host_work"]
        fill_host_buffer_byte(host_buf, SENTINEL_SCRUB_BYTE_VERIFY, copy_bytes)
        ctx.backend.memcpy_d2h(ctx.backend_context, host_buf, ctx.buffers["dst"], copy_bytes)
        actual = checksum_and_sentinels(host_buf, copy_bytes)
        ok_msg = (
            f"method={ctx.state['method']} buffer={copy_bytes / MIB:.1f} MiB "
            f"accounting=2x_rw"
        )
        return _compare_checksum_and_sentinels(
            self.spec.name,
            ctx.state["expected_digest"],
            actual,
            ok_msg,
        )

    def cleanup(self, ctx: WorkloadExecutionContext) -> None:
        backend = ctx.backend
        bctx = ctx.backend_context
        src_buf = ctx.buffers.pop("src", None)
        dst_buf = ctx.buffers.pop("dst", None)
        host_buf = ctx.buffers.pop("host_work", None)
        is_pinned = bool(ctx.state.pop("host_work_pinned", False))
        try:
            if src_buf is not None:
                backend.free_device(bctx, src_buf)
        finally:
            try:
                if dst_buf is not None:
                    backend.free_device(bctx, dst_buf)
            finally:
                try:
                    free_host_buffer(backend, bctx, host_buf, is_pinned)
                finally:
                    if ctx.module is not None:
                        mod = ctx.module
                        ctx.module = None
                        backend.unload_module(bctx, mod)


class BandwidthH2DExecutor(WorkloadExecutor):
    """Host-to-device pinned transfer bandwidth executor (1 * copy_bytes payload)."""

    requires_kernel: bool = False

    def prepare(self, ctx: WorkloadExecutionContext, binary: Any) -> None:
        del binary
        backend = ctx.backend
        bctx = ctx.backend_context
        dev_info = ctx.binding.device_info

        override_bytes = ctx.state.get("target_bytes")
        target_bytes = (
            int(override_bytes)
            if override_bytes is not None
            else compute_safe_vram_budget(dev_info, mode="bandwidth-h2d", alignment=VRAM_ALIGNMENT_BYTES)
        )
        min_bytes = min(MIN_BANDWIDTH_BYTES, target_bytes)
        dev_buf, copy_bytes = alloc_device_with_stepdown(
            backend, bctx, target_bytes, min_bytes=min_bytes, alignment=VRAM_ALIGNMENT_BYTES
        )
        ctx.buffers["dev"] = dev_buf

        host_src, is_pinned = alloc_host_buffer_with_fallback(backend, bctx, copy_bytes)
        ctx.buffers["host_src"] = host_src
        ctx.state["host_src_pinned"] = is_pinned

        payload = build_stamped_payload(copy_bytes, _INITIAL_BANDWIDTH_SEED ^ 0x1111)
        expected = checksum_and_sentinels(payload, copy_bytes)
        write_host_buffer(host_src, payload)

        ctx.state.update(
            {
                "copy_bytes": copy_bytes,
                "expected_digest": expected,
            }
        )

        fixed_xfers = ctx.state.get("fixed_iterations")
        if fixed_xfers is not None:
            transfers_per_step = max(1, int(fixed_xfers))
        else:
            transfers_per_step = calibrate_iterations(
                lambda n: self._dispatch_h2d(ctx, n),
                lambda: backend.synchronize(bctx),
                min_iters=1,
                max_iters=BANDWIDTH_MAX_TRANSFERS_PER_STEP,
            )
        ctx.state["transfers_per_step"] = transfers_per_step

    def _dispatch_h2d(self, ctx: WorkloadExecutionContext, count: int) -> None:
        backend = ctx.backend
        bctx = ctx.backend_context
        dev_buf = ctx.buffers["dev"]
        host_src = ctx.buffers["host_src"]
        copy_bytes = int(ctx.state["copy_bytes"])
        for _ in range(count):
            backend.memcpy_h2d(bctx, dev_buf, host_src, copy_bytes)

    def run_step(self, ctx: WorkloadExecutionContext) -> StepMetrics:
        transfers = int(ctx.state["transfers_per_step"])
        copy_bytes = int(ctx.state["copy_bytes"])
        self._dispatch_h2d(ctx, transfers)
        return StepMetrics(operations=float(transfers * copy_bytes), errors=0)

    def verify(self, ctx: WorkloadExecutionContext) -> VerificationResult:
        copy_bytes = int(ctx.state["copy_bytes"])
        check_buf = ctx.state.get("verify_check_buf")
        if not isinstance(check_buf, bytearray) or len(check_buf) != copy_bytes:
            check_buf = bytearray(copy_bytes)
            ctx.state["verify_check_buf"] = check_buf
        fill_host_buffer_byte(check_buf, SENTINEL_SCRUB_BYTE_VERIFY, copy_bytes)
        ctx.backend.memcpy_d2h(ctx.backend_context, check_buf, ctx.buffers["dev"], copy_bytes)
        actual = checksum_and_sentinels(check_buf, copy_bytes)
        ok_msg = (
            f"buffer={copy_bytes / MIB:.1f} MiB "
            f"pinned={bool(ctx.state.get('host_src_pinned'))} accounting=1x_h2d"
        )
        return _compare_checksum_and_sentinels(
            self.spec.name,
            ctx.state["expected_digest"],
            actual,
            ok_msg,
        )

    def cleanup(self, ctx: WorkloadExecutionContext) -> None:
        backend = ctx.backend
        bctx = ctx.backend_context
        dev_buf = ctx.buffers.pop("dev", None)
        host_src = ctx.buffers.pop("host_src", None)
        is_pinned = bool(ctx.state.pop("host_src_pinned", False))
        ctx.state.pop("verify_check_buf", None)
        try:
            if dev_buf is not None:
                backend.free_device(bctx, dev_buf)
        finally:
            free_host_buffer(backend, bctx, host_src, is_pinned)


class BandwidthD2HExecutor(WorkloadExecutor):
    """Device-to-host pinned transfer bandwidth executor (1 * copy_bytes payload)."""

    requires_kernel: bool = False

    def prepare(self, ctx: WorkloadExecutionContext, binary: Any) -> None:
        del binary
        backend = ctx.backend
        bctx = ctx.backend_context
        dev_info = ctx.binding.device_info

        override_bytes = ctx.state.get("target_bytes")
        target_bytes = (
            int(override_bytes)
            if override_bytes is not None
            else compute_safe_vram_budget(dev_info, mode="bandwidth-d2h", alignment=VRAM_ALIGNMENT_BYTES)
        )
        min_bytes = min(MIN_BANDWIDTH_BYTES, target_bytes)
        dev_buf, copy_bytes = alloc_device_with_stepdown(
            backend, bctx, target_bytes, min_bytes=min_bytes, alignment=VRAM_ALIGNMENT_BYTES
        )
        ctx.buffers["dev"] = dev_buf

        host_dst, is_pinned = alloc_host_buffer_with_fallback(backend, bctx, copy_bytes)
        ctx.buffers["host_dst"] = host_dst
        ctx.state["host_dst_pinned"] = is_pinned

        payload = build_stamped_payload(copy_bytes, _INITIAL_BANDWIDTH_SEED ^ 0x2222)
        expected = checksum_and_sentinels(payload, copy_bytes)
        write_host_buffer(host_dst, payload)
        backend.memcpy_h2d(bctx, dev_buf, host_dst, copy_bytes)
        backend.synchronize(bctx)

        # Scrub host destination prior to timed transfers so a no-op D2H fails verification.
        fill_host_buffer_byte(host_dst, SENTINEL_SCRUB_BYTE_DST, copy_bytes)

        ctx.state.update(
            {
                "copy_bytes": copy_bytes,
                "expected_digest": expected,
            }
        )

        fixed_xfers = ctx.state.get("fixed_iterations")
        if fixed_xfers is not None:
            transfers_per_step = max(1, int(fixed_xfers))
        else:
            transfers_per_step = calibrate_iterations(
                lambda n: self._dispatch_d2h(ctx, n),
                lambda: backend.synchronize(bctx),
                min_iters=1,
                max_iters=BANDWIDTH_MAX_TRANSFERS_PER_STEP,
            )
            fill_host_buffer_byte(host_dst, SENTINEL_SCRUB_BYTE_DST, copy_bytes)
        ctx.state["transfers_per_step"] = transfers_per_step

    def _dispatch_d2h(self, ctx: WorkloadExecutionContext, count: int) -> None:
        backend = ctx.backend
        bctx = ctx.backend_context
        dev_buf = ctx.buffers["dev"]
        host_dst = ctx.buffers["host_dst"]
        copy_bytes = int(ctx.state["copy_bytes"])
        for _ in range(count):
            backend.memcpy_d2h(bctx, host_dst, dev_buf, copy_bytes)

    def reset_run_state(self, ctx: WorkloadExecutionContext) -> None:
        host_dst = ctx.buffers.get("host_dst")
        copy_bytes = int(ctx.state.get("copy_bytes", 0))
        if host_dst is not None and copy_bytes > 0:
            fill_host_buffer_byte(host_dst, SENTINEL_SCRUB_BYTE_DST, copy_bytes)

    def run_step(self, ctx: WorkloadExecutionContext) -> StepMetrics:
        transfers = int(ctx.state["transfers_per_step"])
        copy_bytes = int(ctx.state["copy_bytes"])
        self._dispatch_d2h(ctx, transfers)
        return StepMetrics(operations=float(transfers * copy_bytes), errors=0)

    def verify(self, ctx: WorkloadExecutionContext) -> VerificationResult:
        copy_bytes = int(ctx.state["copy_bytes"])
        host_dst = ctx.buffers["host_dst"]
        actual = checksum_and_sentinels(host_dst, copy_bytes)
        ok_msg = (
            f"buffer={copy_bytes / MIB:.1f} MiB "
            f"pinned={bool(ctx.state.get('host_dst_pinned'))} accounting=1x_d2h"
        )
        return _compare_checksum_and_sentinels(
            self.spec.name,
            ctx.state["expected_digest"],
            actual,
            ok_msg,
        )

    def cleanup(self, ctx: WorkloadExecutionContext) -> None:
        backend = ctx.backend
        bctx = ctx.backend_context
        dev_buf = ctx.buffers.pop("dev", None)
        host_dst = ctx.buffers.pop("host_dst", None)
        is_pinned = bool(ctx.state.pop("host_dst_pinned", False))
        try:
            if dev_buf is not None:
                backend.free_device(bctx, dev_buf)
        finally:
            free_host_buffer(backend, bctx, host_dst, is_pinned)


MEMORY_EXECUTOR_CLASSES: dict[str, type[WorkloadExecutor]] = {
    "memory-pattern": MemoryPatternExecutor,
    "memory-sweep": MemorySweepExecutor,
    "bandwidth-d2d": BandwidthD2DExecutor,
    "bandwidth-h2d": BandwidthH2DExecutor,
    "bandwidth-d2h": BandwidthD2HExecutor,
}
