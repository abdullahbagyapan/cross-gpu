"""Deterministic exit codes and typed exception hierarchy for cross-gpu."""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable


class ExitCode(IntEnum):
    """Exit codes ordered strictly by ascending severity.

    Numeric value directly encodes aggregation precedence:
    HARDWARE_ERROR (4) > COMPUTE_ERROR (3) > INIT_ERROR (2) > USAGE_ERROR (1) > SUCCESS (0).
    """

    SUCCESS = 0
    USAGE_ERROR = 1
    INIT_ERROR = 2
    COMPUTE_ERROR = 3
    HARDWARE_ERROR = 4


class CrossGpuError(Exception):
    """Base exception carrying a deterministic ExitCode."""

    exit_code: ExitCode = ExitCode.INIT_ERROR

    def __init__(self, message: str = "", *, exit_code: ExitCode | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class UsageError(CrossGpuError):
    """CLI parsing, flag conflict, or invalid user input error."""

    exit_code = ExitCode.USAGE_ERROR


class BackendInitError(CrossGpuError):
    """Backend initialization, device selection, or context creation failure."""

    exit_code = ExitCode.INIT_ERROR


class KernelNotFoundError(CrossGpuError):
    """Requested kernel or workload executor is not registered/available."""

    exit_code = ExitCode.INIT_ERROR


class KernelCompilationError(CrossGpuError):
    """Runtime kernel compilation failure with captured compiler log."""

    exit_code = ExitCode.INIT_ERROR

    def __init__(self, message: str = "", *, compiler_log: str = "") -> None:
        super().__init__(message, exit_code=ExitCode.INIT_ERROR)
        self.compiler_log = compiler_log


class UnsupportedKernelError(CrossGpuError):
    """Device lacks required datatype or ISA capability for the kernel."""

    exit_code = ExitCode.INIT_ERROR


class ComputeError(CrossGpuError):
    """Numerical/bitwise verification mismatch, SDC, or memory pattern failure."""

    exit_code = ExitCode.COMPUTE_ERROR


class BackendHardwareError(CrossGpuError):
    """Fatal device-lost, GPU hang/reset, uncorrectable ECC, or thermal fault."""

    exit_code = ExitCode.HARDWARE_ERROR


def aggregate_exit_codes(codes: Iterable[ExitCode | int]) -> ExitCode:
    """Aggregate exit codes across devices and workloads by highest severity."""
    highest = ExitCode.SUCCESS
    for code in codes:
        candidate = ExitCode(int(code))
        if candidate > highest:
            highest = candidate
    return highest


def error_to_exit_code(exc: BaseException) -> ExitCode:
    """Map an exception instance to its deterministic ExitCode."""
    if isinstance(exc, CrossGpuError):
        return exc.exit_code
    return ExitCode.INIT_ERROR
