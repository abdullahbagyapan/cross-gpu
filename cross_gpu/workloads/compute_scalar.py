"""Scalar integer (Group 1) and scalar floating-point / ML (Group 2) workload executors."""

from __future__ import annotations

import ctypes
import struct
from typing import Any, Sequence

from cross_gpu.exit_codes import BackendInitError
from cross_gpu.models import StepMetrics, VerificationResult
from cross_gpu.workloads.common import (
    BF16_ABS_TOL,
    BF16_REL_TOL,
    DEFAULT_MAX_ITERATIONS,
    FP4_ABS_TOL,
    FP4_REL_TOL,
    FP8_ABS_TOL,
    FP8_REL_TOL,
    FP16_ABS_TOL,
    FP16_REL_TOL,
    FP32_ABS_TOL,
    FP32_REL_TOL,
    FP64_ABS_TOL,
    FP64_REL_TOL,
    FULL_VERIFY_BUFFER_LIMIT_BYTES,
    MIN_VERIFY_SAMPLES,
    NONLINEAR_DOT_MAX_ITERATIONS,
    NONLINEAR_VERIFY_BUDGET_OPS,
    calibrate_iterations,
    compute_launch_geometry,
    golden_bf16_table,
    golden_fp16_table,
    golden_fp32_table,
    golden_fp64_table,
    golden_fp_mma_table,
    golden_int4_cuda,
    golden_int4_hip_sdot8,
    golden_int8_dp4a,
    golden_int8_uchar4,
    golden_int8_uchar4_params,
    golden_int16,
    golden_int16_params,
    golden_int32,
    golden_int32_params,
    golden_int64,
    golden_int64_params,
    golden_splitmix64,
    ops_per_thread_per_iter,
    select_verify_indices,
    verify_float_values,
    verify_int_values,
)
from cross_gpu.workloads.registry import WorkloadExecutionContext, WorkloadExecutor

# Dyadic seed pairs exact in FP64, FP32, FP16, and BF16 so analytical values stay drift-free.
_FP_SEED_PAIRS: tuple[tuple[float, float], ...] = (
    (0.75, 0.50),
    (0.625, 0.375),
    (0.50, 0.25),
    (0.875, 0.625),
    (0.375, 0.75),
    (0.25, 0.50),
    (0.625, 0.875),
    (0.50, 0.75),
)

_INT_SEED_BASE_A = 0x123456789ABCDEF1
_INT_SEED_BASE_B = 0xCAFEBABE12345679


class BaseComputeKernelExecutor(WorkloadExecutor):
    """Shared lifecycle for compute and tensor kernels with uniform 5-arg ABI.

    Kernel ABI across all 5 backends:
      (out_buf, uint32 num_elements, uint32 iterations, seed_a, seed_b)
    """

    requires_kernel: bool = True
    elem_size_bytes: int = 4
    struct_fmt: str = "I"
    seed_ctype: Any = ctypes.c_uint32
    is_float_seeds: bool = False
    is_tensor_kernel: bool = False
    max_calibrated_iters: int = DEFAULT_MAX_ITERATIONS
    calibration_round_multiple: int = 1

    def _max_calibration_iters(self, ctx: WorkloadExecutionContext) -> int:
        del ctx
        return self.max_calibrated_iters

    def _seeds_for_step(self, step_index: int, epoch: int) -> tuple[Any, Any]:
        combined = step_index + epoch * 17
        if self.is_float_seeds:
            return _FP_SEED_PAIRS[combined % len(_FP_SEED_PAIRS)]
        a64 = golden_splitmix64((_INT_SEED_BASE_A ^ (combined * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF)
        b64 = golden_splitmix64((_INT_SEED_BASE_B ^ ((combined + 1) * 0xBF58476D1CE4E5B9)) & 0xFFFFFFFFFFFFFFFF)
        if self.seed_ctype is ctypes.c_uint64:
            return a64, b64
        return a64 & 0xFFFFFFFF, b64 & 0xFFFFFFFF

    def _build_kernel_args(
        self,
        ctx: WorkloadExecutionContext,
        iterations: int,
        seed_a: Any,
        seed_b: Any,
    ) -> list[Any]:
        return [
            ctx.buffers["out"],
            ctypes.c_uint32(int(ctx.state["num_elements"])),
            ctypes.c_uint32(int(iterations)),
            self.seed_ctype(seed_a),
            self.seed_ctype(seed_b),
        ]

    def prepare(self, ctx: WorkloadExecutionContext, binary: Any) -> None:
        if binary is None:
            raise BackendInitError(f"Missing compiled kernel binary for {self.spec.name!r}")

        backend = ctx.backend
        bctx = ctx.backend_context
        dev_info = ctx.binding.device_info

        ctx.module = backend.load_module(bctx, binary)
        entry_point = str(getattr(binary, "entry_point", "") or self.spec.default_entry_point)

        target_threads = ctx.state.get("target_threads")
        grid, block, total_threads = compute_launch_geometry(
            dev_info,
            is_tensor=self.is_tensor_kernel,
            target_threads=int(target_threads) if target_threads is not None else None,
        )

        out_bytes = total_threads * self.elem_size_bytes
        ctx.buffers["out"] = backend.alloc_device(bctx, out_bytes)

        ops_per_iter = ops_per_thread_per_iter(
            self.spec.name,
            ctx.binding.backend,
            dev_info.architecture,
        )
        ctx.state.update(
            {
                "entry_point": entry_point,
                "grid": grid,
                "block": block,
                "num_elements": total_threads,
                "ops_per_thread_per_iter": ops_per_iter,
                "step_index": 0,
                "seed_epoch": 0,
                "last_seeds": None,
                "last_iterations": 0,
                "verify_host_buf": bytearray(out_bytes),
            }
        )

        fixed_iters = ctx.state.get("fixed_iterations")
        if fixed_iters is not None:
            iterations = max(1, int(fixed_iters))
        else:
            probe_sa, probe_sb = self._seeds_for_step(0, 0)

            def _launch_probe(probe_iters: int) -> None:
                args = self._build_kernel_args(ctx, probe_iters, probe_sa, probe_sb)
                backend.launch_kernel(bctx, ctx.module, entry_point, grid, block, args)

            iterations = calibrate_iterations(
                _launch_probe,
                lambda: backend.synchronize(bctx),
                min_iters=1,
                max_iters=self._max_calibration_iters(ctx),
                round_multiple=self.calibration_round_multiple,
            )

        ctx.state["iterations"] = iterations

    def reset_run_state(self, ctx: WorkloadExecutionContext) -> None:
        # Advance seed epoch so timed stress dispatches never reuse warmup seed outputs.
        ctx.state["seed_epoch"] = int(ctx.state.get("seed_epoch", 0)) + 1
        ctx.state["step_index"] = 0

    def run_step(self, ctx: WorkloadExecutionContext) -> StepMetrics:
        step_index = int(ctx.state["step_index"])
        epoch = int(ctx.state["seed_epoch"])
        iterations = int(ctx.state["iterations"])
        seed_a, seed_b = self._seeds_for_step(step_index, epoch)

        args = self._build_kernel_args(ctx, iterations, seed_a, seed_b)
        ctx.backend.launch_kernel(
            ctx.backend_context,
            ctx.module,
            ctx.state["entry_point"],
            ctx.state["grid"],
            ctx.state["block"],
            args,
        )

        ctx.state["step_index"] = step_index + 1
        ctx.state["last_seeds"] = (seed_a, seed_b)
        ctx.state["last_iterations"] = iterations

        total_ops = float(
            int(ctx.state["num_elements"])
            * iterations
            * int(ctx.state["ops_per_thread_per_iter"])
        )
        return StepMetrics(operations=total_ops, errors=0)

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[Any],
        iterations: int,
        seed_a: Any,
        seed_b: Any,
    ) -> VerificationResult:
        raise NotImplementedError

    def verify(self, ctx: WorkloadExecutionContext) -> VerificationResult:
        last_seeds = ctx.state.get("last_seeds")
        iterations = int(ctx.state.get("last_iterations", 0))
        if last_seeds is None or iterations <= 0:
            return VerificationResult(
                passed=False,
                error_count=1,
                message=f"{self.spec.name}: no execution step recorded prior to verify",
            )

        num_elements = int(ctx.state["num_elements"])
        out_bytes = num_elements * self.elem_size_bytes
        raw = ctx.state.get("verify_host_buf")
        if not isinstance(raw, bytearray) or len(raw) != out_bytes:
            raw = bytearray(out_bytes)
            ctx.state["verify_host_buf"] = raw

        ctx.backend.memcpy_d2h(ctx.backend_context, raw, ctx.buffers["out"], out_bytes)
        values = struct.unpack(f"<{num_elements}{self.struct_fmt}", raw)
        seed_a, seed_b = last_seeds
        return self._verify_unpacked(ctx, values, iterations, seed_a, seed_b)

    def cleanup(self, ctx: WorkloadExecutionContext) -> None:
        backend = ctx.backend
        bctx = ctx.backend_context
        out_buf = ctx.buffers.pop("out", None)
        try:
            if out_buf is not None:
                backend.free_device(bctx, out_buf)
        finally:
            if ctx.module is not None:
                mod = ctx.module
                ctx.module = None
                backend.unload_module(bctx, mod)
            ctx.state.pop("verify_host_buf", None)


def _full_or_strided_indices(num_elements: int, elem_size_bytes: int) -> tuple[int, ...] | None:
    if num_elements * elem_size_bytes <= FULL_VERIFY_BUFFER_LIMIT_BYTES:
        return None
    max_samples = FULL_VERIFY_BUFFER_LIMIT_BYTES // max(1, elem_size_bytes)
    return select_verify_indices(num_elements, max_samples=max_samples)


# ---------------------------------------------------------------------------
# Group 1: Scalar Integer Compute Executors (GOP/s, GOP)
# ---------------------------------------------------------------------------


class ComputeInt64Executor(BaseComputeKernelExecutor):
    """64-bit unsigned modular MAD compute stress executor."""

    elem_size_bytes = 8
    struct_fmt = "Q"
    seed_ctype = ctypes.c_uint64

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[int],
        iterations: int,
        seed_a: int,
        seed_b: int,
    ) -> VerificationResult:
        params = golden_int64_params(iterations, seed_a, seed_b)
        indices = _full_or_strided_indices(len(values), self.elem_size_bytes)
        return verify_int_values(
            values,
            lambda gid: golden_int64(gid, iterations, seed_a, seed_b, precomputed=params),
            indices=indices,
            workload_name=self.spec.name,
        )


class ComputeInt32Executor(BaseComputeKernelExecutor):
    """32-bit unsigned modular MAD compute stress executor."""

    elem_size_bytes = 4
    struct_fmt = "I"
    seed_ctype = ctypes.c_uint32

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[int],
        iterations: int,
        seed_a: int,
        seed_b: int,
    ) -> VerificationResult:
        params = golden_int32_params(iterations, seed_a, seed_b)
        indices = _full_or_strided_indices(len(values), self.elem_size_bytes)
        return verify_int_values(
            values,
            lambda gid: golden_int32(gid, iterations, seed_a, seed_b, precomputed=params),
            indices=indices,
            workload_name=self.spec.name,
        )


class ComputeInt16Executor(BaseComputeKernelExecutor):
    """16-bit unsigned modular MAD compute stress executor."""

    elem_size_bytes = 4
    struct_fmt = "I"
    seed_ctype = ctypes.c_uint32

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[int],
        iterations: int,
        seed_a: int,
        seed_b: int,
    ) -> VerificationResult:
        params = golden_int16_params(iterations, seed_a, seed_b)
        indices = _full_or_strided_indices(len(values), self.elem_size_bytes)
        return verify_int_values(
            values,
            lambda gid: golden_int16(gid, iterations, seed_a, seed_b, precomputed=params),
            indices=indices,
            workload_name=self.spec.name,
        )


class ComputeInt8Executor(BaseComputeKernelExecutor):
    """8-bit integer compute stress executor (dp4a/udot4 on CUDA/HIP; uchar4 elsewhere)."""

    elem_size_bytes = 4
    struct_fmt = "I"
    seed_ctype = ctypes.c_uint32

    def _uses_dp4a(self, ctx: WorkloadExecutionContext) -> bool:
        return ctx.binding.backend in ("cuda", "hip")

    def _max_calibration_iters(self, ctx: WorkloadExecutionContext) -> int:
        if self._uses_dp4a(ctx):
            return NONLINEAR_DOT_MAX_ITERATIONS
        return self.max_calibrated_iters

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[int],
        iterations: int,
        seed_a: int,
        seed_b: int,
    ) -> VerificationResult:
        if self._uses_dp4a(ctx):
            max_samples = max(
                MIN_VERIFY_SAMPLES,
                NONLINEAR_VERIFY_BUDGET_OPS // max(1, iterations * 16),
            )
            indices = select_verify_indices(len(values), max_samples=max_samples)
            return verify_int_values(
                values,
                lambda gid: golden_int8_dp4a(gid, iterations, seed_a, seed_b),
                indices=indices,
                workload_name=self.spec.name,
            )

        params = golden_int8_uchar4_params(iterations, seed_a, seed_b)
        indices = _full_or_strided_indices(len(values), self.elem_size_bytes)
        return verify_int_values(
            values,
            lambda gid: golden_int8_uchar4(gid, iterations, seed_a, seed_b, precomputed=params),
            indices=indices,
            workload_name=self.spec.name,
        )


class ComputeInt4Executor(BaseComputeKernelExecutor):
    """4-bit packed integer compute stress executor (CUDA s4 MMA; HIP sdot8)."""

    elem_size_bytes = 4
    struct_fmt = "I"
    seed_ctype = ctypes.c_uint32
    is_tensor_kernel = True

    def _max_calibration_iters(self, ctx: WorkloadExecutionContext) -> int:
        if ctx.binding.backend == "hip":
            return NONLINEAR_DOT_MAX_ITERATIONS
        return self.max_calibrated_iters

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[int],
        iterations: int,
        seed_a: int,
        seed_b: int,
    ) -> VerificationResult:
        if ctx.binding.backend == "hip":
            max_samples = max(
                MIN_VERIFY_SAMPLES,
                NONLINEAR_VERIFY_BUDGET_OPS // max(1, iterations * 16),
            )
            indices = select_verify_indices(len(values), max_samples=max_samples)
            return verify_int_values(
                values,
                lambda gid: golden_int4_hip_sdot8(gid, iterations, seed_a, seed_b),
                indices=indices,
                workload_name=self.spec.name,
            )

        indices = _full_or_strided_indices(len(values), self.elem_size_bytes)
        return verify_int_values(
            values,
            lambda gid: golden_int4_cuda(gid, seed_a, seed_b),
            indices=indices,
            workload_name=self.spec.name,
        )


# ---------------------------------------------------------------------------
# Group 2: Scalar Floating-Point & ML Compute Executors (GFLOP/s, GFLOP)
# ---------------------------------------------------------------------------


class ComputeFP64Executor(BaseComputeKernelExecutor):
    """Double-precision IEEE-754 FP64 FMA stress executor."""

    elem_size_bytes = 8
    struct_fmt = "d"
    seed_ctype = ctypes.c_double
    is_float_seeds = True

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[float],
        iterations: int,
        seed_a: float,
        seed_b: float,
    ) -> VerificationResult:
        del ctx
        table = golden_fp64_table(iterations, seed_a, seed_b)
        return verify_float_values(
            values,
            table,
            rel_tol=FP64_REL_TOL,
            abs_tol=FP64_ABS_TOL,
            workload_name=self.spec.name,
        )


class ComputeFP32Executor(BaseComputeKernelExecutor):
    """Single-precision IEEE-754 FP32 FMA stress executor."""

    elem_size_bytes = 4
    struct_fmt = "f"
    seed_ctype = ctypes.c_float
    is_float_seeds = True

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[float],
        iterations: int,
        seed_a: float,
        seed_b: float,
    ) -> VerificationResult:
        del ctx
        table = golden_fp32_table(iterations, seed_a, seed_b)
        return verify_float_values(
            values,
            table,
            rel_tol=FP32_REL_TOL,
            abs_tol=FP32_ABS_TOL,
            workload_name=self.spec.name,
        )


class ComputeFP16Executor(BaseComputeKernelExecutor):
    """Half-precision packed FP16 (half2) FMA stress executor."""

    elem_size_bytes = 4
    struct_fmt = "f"
    seed_ctype = ctypes.c_float
    is_float_seeds = True

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[float],
        iterations: int,
        seed_a: float,
        seed_b: float,
    ) -> VerificationResult:
        del ctx
        table = golden_fp16_table(iterations, seed_a, seed_b)
        return verify_float_values(
            values,
            table,
            rel_tol=FP16_REL_TOL,
            abs_tol=FP16_ABS_TOL,
            workload_name=self.spec.name,
        )


class ComputeBF16Executor(BaseComputeKernelExecutor):
    """Brain-float packed BF16 (bf16x2) FMA stress executor."""

    elem_size_bytes = 4
    struct_fmt = "f"
    seed_ctype = ctypes.c_float
    is_float_seeds = True

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[float],
        iterations: int,
        seed_a: float,
        seed_b: float,
    ) -> VerificationResult:
        del ctx
        table = golden_bf16_table(iterations, seed_a, seed_b)
        return verify_float_values(
            values,
            table,
            rel_tol=BF16_REL_TOL,
            abs_tol=BF16_ABS_TOL,
            workload_name=self.spec.name,
        )


class ComputeFP8Executor(BaseComputeKernelExecutor):
    """Native FP8 hardware MMA stress executor (CUDA SM 8.9+ / AMD CDNA3 & RDNA4)."""

    elem_size_bytes = 4
    struct_fmt = "f"
    seed_ctype = ctypes.c_float
    is_float_seeds = True
    is_tensor_kernel = True

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[float],
        iterations: int,
        seed_a: float,
        seed_b: float,
    ) -> VerificationResult:
        del ctx, iterations
        table = golden_fp_mma_table(seed_a, seed_b, metal_layout=False)
        return verify_float_values(
            values,
            table,
            rel_tol=FP8_REL_TOL,
            abs_tol=FP8_ABS_TOL,
            workload_name=self.spec.name,
        )


class ComputeFP4Executor(BaseComputeKernelExecutor):
    """Native FP4 hardware MMA stress executor (CUDA Blackwell / AMD CDNA4)."""

    elem_size_bytes = 4
    struct_fmt = "f"
    seed_ctype = ctypes.c_float
    is_float_seeds = True
    is_tensor_kernel = True

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[float],
        iterations: int,
        seed_a: float,
        seed_b: float,
    ) -> VerificationResult:
        del ctx, iterations
        table = golden_fp_mma_table(seed_a, seed_b, metal_layout=False)
        return verify_float_values(
            values,
            table,
            rel_tol=FP4_REL_TOL,
            abs_tol=FP4_ABS_TOL,
            workload_name=self.spec.name,
        )


SCALAR_EXECUTOR_CLASSES: dict[str, type[WorkloadExecutor]] = {
    "compute-int64": ComputeInt64Executor,
    "compute-int32": ComputeInt32Executor,
    "compute-int16": ComputeInt16Executor,
    "compute-int8": ComputeInt8Executor,
    "compute-int4": ComputeInt4Executor,
    "compute-fp64": ComputeFP64Executor,
    "compute-fp32": ComputeFP32Executor,
    "compute-fp16": ComputeFP16Executor,
    "compute-fp8": ComputeFP8Executor,
    "compute-fp4": ComputeFP4Executor,
    "compute-bf16": ComputeBF16Executor,
}
