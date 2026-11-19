# SPDX-License-Identifier: MIT
"""Unit tests for deterministic ExitCode ordering, aggregation, and error mapping."""

from __future__ import annotations

import unittest

from cross_gpu.exit_codes import (
    BackendHardwareError,
    BackendInitError,
    ComputeError,
    ExitCode,
    KernelCompilationError,
    KernelNotFoundError,
    UnsupportedKernelError,
    UsageError,
    aggregate_exit_codes,
    error_to_exit_code,
)


class TestExitCodes(unittest.TestCase):
    def test_exact_numeric_values(self) -> None:
        self.assertEqual(ExitCode.SUCCESS, 0)
        self.assertEqual(ExitCode.USAGE_ERROR, 1)
        self.assertEqual(ExitCode.INIT_ERROR, 2)
        self.assertEqual(ExitCode.COMPUTE_ERROR, 3)
        self.assertEqual(ExitCode.HARDWARE_ERROR, 4)

    def test_severity_ordering(self) -> None:
        self.assertLess(ExitCode.SUCCESS, ExitCode.USAGE_ERROR)
        self.assertLess(ExitCode.USAGE_ERROR, ExitCode.INIT_ERROR)
        self.assertLess(ExitCode.INIT_ERROR, ExitCode.COMPUTE_ERROR)
        self.assertLess(ExitCode.COMPUTE_ERROR, ExitCode.HARDWARE_ERROR)

    def test_aggregate_exit_codes_precedence(self) -> None:
        self.assertEqual(aggregate_exit_codes([]), ExitCode.SUCCESS)
        self.assertEqual(aggregate_exit_codes([ExitCode.SUCCESS]), ExitCode.SUCCESS)
        self.assertEqual(
            aggregate_exit_codes([ExitCode.SUCCESS, ExitCode.USAGE_ERROR]),
            ExitCode.USAGE_ERROR,
        )
        self.assertEqual(
            aggregate_exit_codes([ExitCode.USAGE_ERROR, ExitCode.INIT_ERROR]),
            ExitCode.INIT_ERROR,
        )
        self.assertEqual(
            aggregate_exit_codes([ExitCode.INIT_ERROR, ExitCode.COMPUTE_ERROR]),
            ExitCode.COMPUTE_ERROR,
        )
        self.assertEqual(
            aggregate_exit_codes([ExitCode.COMPUTE_ERROR, ExitCode.HARDWARE_ERROR, ExitCode.INIT_ERROR]),
            ExitCode.HARDWARE_ERROR,
        )

    def test_exception_to_exit_code_mapping(self) -> None:
        self.assertEqual(error_to_exit_code(UsageError("bad arg")), ExitCode.USAGE_ERROR)
        self.assertEqual(error_to_exit_code(BackendInitError("no gpu")), ExitCode.INIT_ERROR)
        self.assertEqual(error_to_exit_code(KernelNotFoundError("missing")), ExitCode.INIT_ERROR)
        self.assertEqual(
            error_to_exit_code(KernelCompilationError("compile fail", compiler_log="err")),
            ExitCode.INIT_ERROR,
        )
        self.assertEqual(error_to_exit_code(UnsupportedKernelError("no fp8")), ExitCode.INIT_ERROR)
        self.assertEqual(error_to_exit_code(ComputeError("mismatch")), ExitCode.COMPUTE_ERROR)
        self.assertEqual(error_to_exit_code(BackendHardwareError("ecc fault")), ExitCode.HARDWARE_ERROR)
        self.assertEqual(error_to_exit_code(RuntimeError("unexpected")), ExitCode.INIT_ERROR)


if __name__ == "__main__":
    unittest.main()
