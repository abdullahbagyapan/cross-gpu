"""Hardware matrix / tensor core workload executors (Group 3)."""

from __future__ import annotations

import ctypes
from typing import Sequence

from cross_gpu.models import VerificationResult
from cross_gpu.workloads.common import (
    FULL_VERIFY_BUFFER_LIMIT_BYTES,
    TENSOR_FP_ABS_TOL,
    TENSOR_FP_REL_TOL,
    golden_fp_mma_table,
    golden_tensor_int8,
    select_verify_indices,
    verify_float_values,
    verify_int_values,
)
from cross_gpu.workloads.compute_scalar import BaseComputeKernelExecutor
from cross_gpu.workloads.registry import WorkloadExecutionContext, WorkloadExecutor


class _BaseTensorFloatExecutor(BaseComputeKernelExecutor):
    """Shared base for FP32/TF32, FP16, and BF16 hardware matrix-core stress."""

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
        del iterations
        is_metal = ctx.binding.backend == "metal"
        table = golden_fp_mma_table(seed_a, seed_b, metal_layout=is_metal)
        return verify_float_values(
            values,
            table,
            rel_tol=TENSOR_FP_REL_TOL,
            abs_tol=TENSOR_FP_ABS_TOL,
            workload_name=self.spec.name,
        )


class ComputeTensorFP32Executor(_BaseTensorFloatExecutor):
    """Hardware matrix/tensor FP32/TF32 executor (CUDA TF32, HIP MFMA, Intel XMX, Metal simdgroup)."""


class ComputeTensorFP16Executor(_BaseTensorFloatExecutor):
    """Hardware matrix/tensor FP16 executor (CUDA MMA, HIP MFMA/WMMA, Intel XMX, Metal simdgroup)."""


class ComputeTensorBF16Executor(_BaseTensorFloatExecutor):
    """Hardware matrix/tensor BF16 executor (CUDA MMA, HIP MFMA/WMMA, Intel XMX, Metal simdgroup)."""


class ComputeTensorInt8Executor(BaseComputeKernelExecutor):
    """Hardware matrix/tensor INT8 executor (CUDA s8 MMA, HIP MFMA/WMMA, Intel XMX DPAS)."""

    elem_size_bytes = 4
    struct_fmt = "I"
    seed_ctype = ctypes.c_uint32
    is_float_seeds = False
    is_tensor_kernel = True

    def _verify_unpacked(
        self,
        ctx: WorkloadExecutionContext,
        values: Sequence[int],
        iterations: int,
        seed_a: int,
        seed_b: int,
    ) -> VerificationResult:
        del ctx, iterations
        max_samples = FULL_VERIFY_BUFFER_LIMIT_BYTES // self.elem_size_bytes
        indices = select_verify_indices(len(values), max_samples=max_samples)
        return verify_int_values(
            values,
            lambda gid: golden_tensor_int8(gid, seed_a, seed_b),
            indices=indices,
            workload_name=self.spec.name,
        )


TENSOR_EXECUTOR_CLASSES: dict[str, type[WorkloadExecutor]] = {
    "compute-tensor-fp32": ComputeTensorFP32Executor,
    "compute-tensor-fp16": ComputeTensorFP16Executor,
    "compute-tensor-bf16": ComputeTensorBF16Executor,
    "compute-tensor-int8": ComputeTensorInt8Executor,
}
