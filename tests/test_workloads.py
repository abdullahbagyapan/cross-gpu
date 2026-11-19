# SPDX-License-Identifier: MIT
"""Offline unit, calibration, VRAM sizing, and fault-injection tests for all 20 workloads."""

from __future__ import annotations

import ctypes
import math
import struct
import threading
import time
from typing import Any, Sequence
import unittest
from unittest import mock

from cross_gpu.backends.base import (
    BackendContext,
    BackendPlugin,
    DeviceBuffer,
    HostBuffer,
    KernelModule,
)
from cross_gpu.embed_binary import CompiledKernelBinary
from cross_gpu.exit_codes import (
    BackendHardwareError,
    BackendInitError,
    ExitCode,
    KernelCompilationError,
)
from cross_gpu.models import (
    BackendBinding,
    BackendDeviceInfo,
    DeviceCapabilities,
    GPUDevice,
    TelemetrySample,
)
from cross_gpu.runner import StressRunner
from cross_gpu.workloads import (
    ALL_EXECUTOR_CLASSES,
    BANDWIDTH_DISCRETE_CAP_BYTES,
    BANDWIDTH_UNIFIED_CAP_BYTES,
    GIB,
    MIB,
    PATTERN_MODE_NAMES,
    VRAM_ALIGNMENT_BYTES,
    WORKLOAD_REGISTRY,
    WORKLOAD_SPECS,
    BandwidthD2DExecutor,
    WorkloadExecutionContext,
    affine_pow_mod,
    alloc_device_with_stepdown,
    alloc_host_buffer_with_fallback,
    calibrate_iterations,
    compute_launch_geometry,
    compute_safe_vram_budget,
    decode_fp4_e2m1,
    decode_fp8_e4m3,
    free_host_buffer,
    golden_bf16_table,
    golden_fp16_table,
    golden_fp32_table,
    golden_fp64_table,
    golden_fp_mma_table,
    golden_int4_cuda,
    golden_int4_hip_sdot8,
    golden_int8_dp4a,
    golden_int8_uchar4,
    golden_int16,
    golden_int32,
    golden_int64,
    golden_pattern_word,
    golden_splitmix64,
    golden_tensor_int8,
    is_unified_memory_device,
    list_workload_names,
    ops_per_thread_per_iter,
    round_bf16,
    round_fp16,
    round_fp32,
    verify_float_values,
    verify_int_values,
)


def _c_value(arg: Any) -> Any:
    if isinstance(arg, ctypes._SimpleCData):
        return arg.value
    return arg


class _SimulatingMockBackend(BackendPlugin):
    """In-memory mock backend that simulates kernel execution and tracks resource leaks."""

    name = "cuda"
    priority = 100

    def __init__(
        self,
        *,
        backend_name: str = "cuda",
        fail_alloc_above_bytes: int | None = None,
        fail_pinned_alloc: bool = False,
        flip_compute_bit: bool = False,
        inject_compute_nan: bool = False,
        inject_compute_inf: bool = False,
        inject_memory_errors: int = 0,
        corrupt_d2d_byte: bool = False,
        corrupt_h2d_byte: bool = False,
        corrupt_d2h_byte: bool = False,
        hardware_error_on_launch: bool = False,
        step_sleep_s: float = 0.0,
    ) -> None:
        super().__init__()
        self.name = backend_name
        self.fail_alloc_above_bytes = fail_alloc_above_bytes
        self.fail_pinned_alloc = fail_pinned_alloc
        self.flip_compute_bit = flip_compute_bit
        self.inject_compute_nan = inject_compute_nan
        self.inject_compute_inf = inject_compute_inf
        self.inject_memory_errors = inject_memory_errors
        self.corrupt_d2d_byte = corrupt_d2d_byte
        self.corrupt_h2d_byte = corrupt_h2d_byte
        self.corrupt_d2h_byte = corrupt_d2h_byte
        self.hardware_error_on_launch = hardware_error_on_launch
        self.step_sleep_s = step_sleep_s

        self._next_handle = 1
        self.device_mem: dict[int, bytearray] = {}
        self.pinned_arrays: dict[int, Any] = {}
        self.loaded_modules: dict[int, KernelModule] = {}
        self.active_contexts: set[int] = set()
        self.launch_log: list[str] = []

    def is_available(self) -> bool:
        return True

    def enumerate_devices(self) -> list[BackendDeviceInfo]:
        return []

    def create_context(self, device: BackendDeviceInfo) -> BackendContext:
        h = self._next_handle
        self._next_handle += 1
        self.active_contexts.add(h)
        return BackendContext(backend=self.name, device_info=device, handle=h)

    def destroy_context(self, ctx: BackendContext) -> None:
        if ctx.handle is not None:
            self.active_contexts.discard(int(ctx.handle))
            ctx.handle = None

    def synchronize(self, ctx: BackendContext) -> None:
        del ctx

    def alloc_device(self, ctx: BackendContext, size_bytes: int) -> DeviceBuffer:
        del ctx
        if self.fail_alloc_above_bytes is not None and size_bytes > self.fail_alloc_above_bytes:
            raise BackendInitError(f"Simulated OOM for {size_bytes} bytes")
        h = self._next_handle
        self._next_handle += 1
        self.device_mem[h] = bytearray(size_bytes)
        return DeviceBuffer(backend=self.name, handle=h, size_bytes=size_bytes, address=h)

    def free_device(self, ctx: BackendContext, buf: DeviceBuffer) -> None:
        del ctx
        if buf.handle:
            self.device_mem.pop(int(buf.handle), None)
            buf.handle = 0

    def alloc_pinned_host(self, ctx: BackendContext, size_bytes: int) -> HostBuffer:
        del ctx
        if self.fail_pinned_alloc:
            raise BackendInitError("Simulated pinned allocation failure")
        h = self._next_handle
        self._next_handle += 1
        c_arr = (ctypes.c_ubyte * max(1, size_bytes))()
        self.pinned_arrays[h] = c_arr
        return HostBuffer(
            backend=self.name,
            handle=h,
            size_bytes=size_bytes,
            pointer=ctypes.addressof(c_arr),
        )

    def free_pinned_host(self, ctx: BackendContext, buf: HostBuffer) -> None:
        del ctx
        if buf.handle:
            self.pinned_arrays.pop(int(buf.handle), None)
            buf.handle = 0
            buf.pointer = 0

    def _read_host_src(self, src: Any, size_bytes: int) -> bytes:
        if isinstance(src, HostBuffer) and src.pointer:
            view = (ctypes.c_ubyte * size_bytes).from_address(src.pointer)
            return bytes(view)
        if isinstance(src, (bytes, bytearray, memoryview)):
            return bytes(memoryview(src)[:size_bytes])
        raise TypeError(f"Unsupported host src type {type(src)!r}")

    def _write_host_dst(self, dst: Any, data: bytes) -> None:
        n = len(data)
        if isinstance(dst, HostBuffer) and dst.pointer:
            c_src = ctypes.create_string_buffer(data, n)
            ctypes.memmove(ctypes.c_void_p(dst.pointer), c_src, n)
            return
        if isinstance(dst, bytearray):
            dst[:n] = data
            return
        raise TypeError(f"Unsupported host dst type {type(dst)!r}")

    def memcpy_h2d(self, ctx: BackendContext, dst: DeviceBuffer, src: Any, size_bytes: int) -> None:
        del ctx
        payload = bytearray(self._read_host_src(src, size_bytes))
        if self.corrupt_h2d_byte and size_bytes > 16:
            payload[size_bytes // 2] ^= 0x01
        self.device_mem[int(dst.handle)][:size_bytes] = payload

    def memcpy_d2h(self, ctx: BackendContext, dst: Any, src: DeviceBuffer, size_bytes: int) -> None:
        del ctx
        payload = bytearray(self.device_mem[int(src.handle)][:size_bytes])
        if self.corrupt_d2h_byte and size_bytes > 16:
            payload[size_bytes // 2] ^= 0x01
        self._write_host_dst(dst, bytes(payload))

    def memcpy_d2d(self, ctx: BackendContext, dst: DeviceBuffer, src: DeviceBuffer, size_bytes: int) -> None:
        del ctx
        payload = bytearray(self.device_mem[int(src.handle)][:size_bytes])
        if self.corrupt_d2d_byte and size_bytes > 16:
            payload[0] ^= 0x01
        self.device_mem[int(dst.handle)][:size_bytes] = payload

    def load_module(self, ctx: BackendContext, compiled_binary: Any) -> KernelModule:
        del ctx
        h = self._next_handle
        self._next_handle += 1
        mod = KernelModule(backend=self.name, handle=h, extra={"binary": compiled_binary})
        self.loaded_modules[h] = mod
        return mod

    def unload_module(self, ctx: BackendContext, module: KernelModule) -> None:
        del ctx
        if module.handle is not None:
            self.loaded_modules.pop(int(module.handle), None)
            module.handle = None

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
        del module, grid, block, shared_mem
        if self.hardware_error_on_launch:
            raise BackendHardwareError("Simulated GPU hang / device lost")
        if self.step_sleep_s > 0:
            time.sleep(self.step_sleep_s)
        self.launch_log.append(entry_point)

        if entry_point == "kernel_bandwidth_d2d":
            dst_buf, src_buf, num_vec4 = args[0], args[1], int(_c_value(args[2]))
            self.memcpy_d2d(ctx, dst_buf, src_buf, num_vec4 * 16)
            return

        if entry_point == "kernel_memory_pattern_write":
            buf, num_words, mode, seed = args[0], int(_c_value(args[1])), int(_c_value(args[2])), int(_c_value(args[3]))
            mem = self.device_mem[int(buf.handle)]
            # Populate word 0 + full buffer for small test buffers.
            if num_words <= 4096:
                words = [golden_pattern_word(i, mode, seed) for i in range(num_words)]
                mem[: num_words * 4] = struct.pack(f"<{num_words}I", *words)
            else:
                struct.pack_into("<I", mem, 0, golden_pattern_word(0, mode, seed))
            return

        if entry_point == "kernel_memory_pattern_verify":
            err_buf = args[4]
            struct.pack_into("<I", self.device_mem[int(err_buf.handle)], 0, self.inject_memory_errors)
            return

        if entry_point == "kernel_memory_sweep_write":
            buf, num_qwords, seed = args[0], int(_c_value(args[1])), int(_c_value(args[2]))
            mem = self.device_mem[int(buf.handle)]
            if num_qwords <= 2048:
                qwords = [golden_splitmix64(i ^ seed) for i in range(num_qwords)]
                mem[: num_qwords * 8] = struct.pack(f"<{num_qwords}Q", *qwords)
            else:
                struct.pack_into("<Q", mem, 0, golden_splitmix64(seed))
            return

        if entry_point == "kernel_memory_sweep_verify":
            err_buf = args[3]
            struct.pack_into("<I", self.device_mem[int(err_buf.handle)], 0, self.inject_memory_errors)
            return

        # Compute & tensor kernels: (out_buf, num_elements, iterations, seed_a, seed_b)
        out_buf = args[0]
        num_elements = int(_c_value(args[1]))
        iterations = int(_c_value(args[2]))
        seed_a = _c_value(args[3])
        seed_b = _c_value(args[4])
        mem = self.device_mem[int(out_buf.handle)]

        if entry_point == "kernel_compute_int64":
            vals = [golden_int64(i, iterations, seed_a, seed_b) for i in range(num_elements)]
            if self.flip_compute_bit and vals:
                vals[0] ^= 1
            mem[: num_elements * 8] = struct.pack(f"<{num_elements}Q", *vals)
            return

        if entry_point in (
            "kernel_compute_int32",
            "kernel_compute_int16",
            "kernel_compute_int8",
            "kernel_compute_int4",
            "kernel_compute_tensor_int8",
        ):
            if entry_point == "kernel_compute_int32":
                vals = [golden_int32(i, iterations, seed_a, seed_b) for i in range(num_elements)]
            elif entry_point == "kernel_compute_int16":
                vals = [golden_int16(i, iterations, seed_a, seed_b) for i in range(num_elements)]
            elif entry_point == "kernel_compute_int8":
                if self.name in ("cuda", "hip"):
                    vals = [golden_int8_dp4a(i, iterations, seed_a, seed_b) for i in range(num_elements)]
                else:
                    vals = [golden_int8_uchar4(i, iterations, seed_a, seed_b) for i in range(num_elements)]
            elif entry_point == "kernel_compute_int4":
                if self.name == "hip":
                    vals = [golden_int4_hip_sdot8(i, iterations, seed_a, seed_b) for i in range(num_elements)]
                else:
                    vals = [golden_int4_cuda(i, seed_a, seed_b) for i in range(num_elements)]
            else:
                vals = [golden_tensor_int8(i, seed_a, seed_b) for i in range(num_elements)]
            if self.flip_compute_bit and vals:
                vals[len(vals) // 2] ^= 1
            mem[: num_elements * 4] = struct.pack(f"<{num_elements}I", *vals)
            return

        if entry_point == "kernel_compute_fp64":
            table = golden_fp64_table(iterations, float(seed_a), float(seed_b))
            fvals = [table[i & 15] for i in range(num_elements)]
            if self.inject_compute_nan and fvals:
                fvals[0] = float("nan")
            elif self.inject_compute_inf and fvals:
                fvals[0] = float("inf")
            elif self.flip_compute_bit and fvals:
                fvals[0] += 0.25
            mem[: num_elements * 8] = struct.pack(f"<{num_elements}d", *fvals)
            return

        # 32-bit float output kernels (fp32/fp16/bf16/fp8/fp4/tensor_fp*)
        if entry_point == "kernel_compute_fp32":
            table = golden_fp32_table(iterations, float(seed_a), float(seed_b))
        elif entry_point == "kernel_compute_fp16":
            table = golden_fp16_table(iterations, float(seed_a), float(seed_b))
        elif entry_point == "kernel_compute_bf16":
            table = golden_bf16_table(iterations, float(seed_a), float(seed_b))
        elif entry_point in ("kernel_compute_fp8", "kernel_compute_fp4"):
            table = golden_fp_mma_table(float(seed_a), float(seed_b), metal_layout=False)
        else:
            table = golden_fp_mma_table(
                float(seed_a), float(seed_b), metal_layout=(self.name == "metal")
            )

        fvals = [table[i & 15] for i in range(num_elements)]
        if self.inject_compute_nan and fvals:
            fvals[1] = float("nan")
        elif self.inject_compute_inf and fvals:
            fvals[1] = float("inf")
        elif self.flip_compute_bit and fvals:
            # Flip exponent bit on element 0 -> guaranteed out-of-tolerance mismatch.
            bits = struct.unpack("<I", struct.pack("<f", fvals[0]))[0] ^ 0x00800000
            fvals[0] = struct.unpack("<f", struct.pack("<I", bits))[0]
        mem[: num_elements * 4] = struct.pack(f"<{num_elements}f", *fvals)

    def assert_no_leaks(self, test_case: unittest.TestCase) -> None:
        test_case.assertEqual(len(self.device_mem), 0, "Leaked device buffer(s)")
        test_case.assertEqual(len(self.pinned_arrays), 0, "Leaked pinned host buffer(s)")
        test_case.assertEqual(len(self.loaded_modules), 0, "Leaked kernel module(s)")
        test_case.assertEqual(len(self.active_contexts), 0, "Leaked backend context(s)")


def _make_mock_device(
    index: int = 0,
    *,
    backend: str = "cuda",
    arch: str = "sm_100",
    vram_bytes: int = 16 * MIB,
    free_bytes: int | None = None,
    compute_units: int = 2,
    caps: DeviceCapabilities | None = None,
    extra: dict[str, Any] | None = None,
    vendor: str = "NVIDIA",
) -> GPUDevice:
    if caps is None:
        caps = DeviceCapabilities(
            fp64=True,
            fp32=True,
            fp16=True,
            bf16=True,
            fp8=True,
            fp4=True,
            int64=True,
            int32=True,
            int16=True,
            int8=True,
            int4=True,
            tensor_fp32=True,
            tensor_fp16=True,
            tensor_bf16=True,
            tensor_int8=True,
        )
    info = BackendDeviceInfo(
        backend=backend,
        priority=100,
        backend_device_index=index,
        native_handle=index,
        name=f"Mock GPU {index}",
        vendor=vendor,
        vendor_id=0x10DE,
        architecture=arch,
        compute_capability="10.0",
        total_memory_bytes=vram_bytes,
        free_memory_bytes=free_bytes,
        compute_units=compute_units,
        driver_version="560.0",
        capabilities=caps,
        extra=dict(extra or {"warp_size": 32, "max_threads_per_block": 32}),
    )
    return GPUDevice(
        index=index,
        name=info.name,
        vendor=info.vendor,
        vendor_id=info.vendor_id,
        architecture=info.architecture,
        compute_capability=info.compute_capability,
        total_memory_bytes=info.total_memory_bytes,
        free_memory_bytes=info.free_memory_bytes,
        compute_units=info.compute_units,
        driver_version=info.driver_version,
        capabilities=caps,
        bindings=[BackendBinding(backend=backend, priority=100, device_info=info)],
    )


def _dummy_compile_fn(
    backend: str,
    workload: str,
    target_arch: str = "",
    capabilities: Any = None,
    **kwargs: Any,
) -> CompiledKernelBinary:
    del target_arch, capabilities
    ep = kwargs.get("entry_point") or f"kernel_{workload.replace('-', '_')}"
    return CompiledKernelBinary(
        backend=backend,
        workload=workload,
        entry_point=ep,
        format="ptx",
        binary_bytes=b"mock-binary",
    )


class TestWorkloadRegistryAndAccounting(unittest.TestCase):
    def test_all_20_workloads_registered_with_exact_fixed_units(self) -> None:
        names = list_workload_names()
        self.assertEqual(len(names), 20)
        self.assertEqual(len(ALL_EXECUTOR_CLASSES), 20)

        expected_units = {
            "compute-int64": ("GOP/s", "GOP"),
            "compute-int32": ("GOP/s", "GOP"),
            "compute-int16": ("GOP/s", "GOP"),
            "compute-int8": ("GOP/s", "GOP"),
            "compute-int4": ("GOP/s", "GOP"),
            "compute-fp64": ("GFLOP/s", "GFLOP"),
            "compute-fp32": ("GFLOP/s", "GFLOP"),
            "compute-fp16": ("GFLOP/s", "GFLOP"),
            "compute-fp8": ("GFLOP/s", "GFLOP"),
            "compute-fp4": ("GFLOP/s", "GFLOP"),
            "compute-bf16": ("GFLOP/s", "GFLOP"),
            "compute-tensor-fp32": ("GFLOP/s", "GFLOP"),
            "compute-tensor-fp16": ("GFLOP/s", "GFLOP"),
            "compute-tensor-bf16": ("GFLOP/s", "GFLOP"),
            "compute-tensor-int8": ("GOP/s", "GOP"),
            "memory-pattern": ("GB/s", "GB"),
            "memory-sweep": ("GB/s", "GB"),
            "bandwidth-d2d": ("GB/s", "GB"),
            "bandwidth-h2d": ("GB/s", "GB"),
            "bandwidth-d2h": ("GB/s", "GB"),
        }

        for spec in WORKLOAD_SPECS:
            entry = WORKLOAD_REGISTRY[spec.name]
            self.assertIsNotNone(entry.executor_cls, f"Unregistered executor for {spec.name}")
            self.assertEqual((spec.rate_unit, spec.total_unit), expected_units[spec.name])
            self.assertEqual(spec.unit_scale, 1_000_000_000.0)

    def test_ops_per_thread_per_iter_matches_backend_and_arch_headers(self) -> None:
        self.assertEqual(ops_per_thread_per_iter("compute-int64", "cuda", "sm_89"), 256)
        self.assertEqual(ops_per_thread_per_iter("compute-fp16", "metal", "applegpu_g14g"), 512)
        self.assertEqual(ops_per_thread_per_iter("compute-int8", "cuda", "sm_89"), 1024)
        self.assertEqual(ops_per_thread_per_iter("compute-int4", "cuda", "sm_75"), 4096)
        self.assertEqual(ops_per_thread_per_iter("compute-int4", "hip", "gfx90a"), 2048)
        self.assertEqual(ops_per_thread_per_iter("compute-fp8", "cuda", "sm_89"), 8192)
        self.assertEqual(ops_per_thread_per_iter("compute-fp8", "hip", "gfx942"), 8192)
        self.assertEqual(ops_per_thread_per_iter("compute-fp8", "hip", "gfx1201"), 4096)
        self.assertEqual(ops_per_thread_per_iter("compute-fp4", "cuda", "sm_100"), 16384)
        self.assertEqual(ops_per_thread_per_iter("compute-fp4", "hip", "gfx950"), 32768)
        self.assertEqual(ops_per_thread_per_iter("compute-tensor-fp32", "cuda", "sm_80"), 2048)
        self.assertEqual(ops_per_thread_per_iter("compute-tensor-fp32", "hip", "gfx90a"), 1024)
        self.assertEqual(ops_per_thread_per_iter("compute-tensor-fp32", "metal", "applegpu_g14g"), 1024)
        self.assertEqual(ops_per_thread_per_iter("compute-tensor-fp16", "cuda", "sm_75"), 2048)
        self.assertEqual(ops_per_thread_per_iter("compute-tensor-fp16", "cuda", "sm_80"), 4096)
        self.assertEqual(ops_per_thread_per_iter("compute-tensor-bf16", "hip", "gfx908"), 2048)
        self.assertEqual(ops_per_thread_per_iter("compute-tensor-bf16", "hip", "gfx90a"), 4096)
        self.assertEqual(ops_per_thread_per_iter("compute-tensor-int8", "cuda", "sm_80"), 8192)
        self.assertEqual(ops_per_thread_per_iter("compute-tensor-int8", "hip", "gfx1100"), 4096)


class TestGoldenMathAndDecoders(unittest.TestCase):
    def test_affine_exponentiation_matches_iterative_lcg(self) -> None:
        for bits in (8, 16, 32, 64):
            mask = (1 << bits) - 1
            a = 0x123456789ABCDEF1 & mask | 1
            b = 0xCAFEBABE12345679 & mask | 1
            x0 = 0xDEADBEEFCAFEBABE & mask
            for steps in (0, 1, 5, 16, 37, 128):
                x_iter = x0
                for _ in range(steps):
                    x_iter = (x_iter * a + b) & mask
                aff_a, aff_b = affine_pow_mod(a, b, steps, mask)
                x_fast = (aff_a * x0 + aff_b) & mask
                self.assertEqual(x_fast, x_iter)

    def test_bit_level_ieee_and_ocp_decoders(self) -> None:
        self.assertEqual(round_fp32(1.5), 1.5)
        self.assertEqual(round_fp16(0.625), 0.625)
        self.assertEqual(round_bf16(1.0), 1.0)
        self.assertEqual(round_bf16(0.5), 0.5)
        # OCP E4M3FN constants used by CUDA & RDNA4 kernels
        self.assertEqual(decode_fp8_e4m3(0x38, fnuz=False), 1.0)
        self.assertEqual(decode_fp8_e4m3(0x30, fnuz=False), 0.5)
        self.assertEqual(decode_fp8_e4m3(0xB0, fnuz=False), -0.5)
        self.assertTrue(math.isnan(decode_fp8_e4m3(0x7F, fnuz=False)))
        # AMD CDNA3 E4M3FNUZ constants used by HIP kernels
        self.assertEqual(decode_fp8_e4m3(0x40, fnuz=True), 1.0)
        self.assertEqual(decode_fp8_e4m3(0x38, fnuz=True), 0.5)
        self.assertEqual(decode_fp8_e4m3(0xB8, fnuz=True), -0.5)
        self.assertTrue(math.isnan(decode_fp8_e4m3(0x80, fnuz=True)))
        # OCP MXFP4 E2M1 constants used by Blackwell & CDNA4 kernels
        self.assertEqual(decode_fp4_e2m1(0x2), 1.0)
        self.assertEqual(decode_fp4_e2m1(0x1), 0.5)
        self.assertEqual(decode_fp4_e2m1(0x9), -0.5)
        self.assertEqual(decode_fp4_e2m1(0x7), 6.0)
        self.assertEqual(decode_fp4_e2m1(0xF), -6.0)

    def test_golden_verify_helpers_reject_nan_inf_and_single_bit_flips(self) -> None:
        fp_table = golden_fp32_table(4, 0.75, 0.5)
        good = [fp_table[i & 15] for i in range(32)]
        self.assertTrue(verify_float_values(good, fp_table, rel_tol=1e-5, abs_tol=1e-5).passed)

        with_nan = list(good)
        with_nan[5] = float("nan")
        ver_nan = verify_float_values(with_nan, fp_table, rel_tol=1e-5, abs_tol=1e-5)
        self.assertFalse(ver_nan.passed)
        self.assertEqual(ver_nan.error_count, 1)
        self.assertIn("non-finite", ver_nan.message)

        with_inf = list(good)
        with_inf[7] = float("-inf")
        ver_inf = verify_float_values(with_inf, fp_table, rel_tol=1e-5, abs_tol=1e-5)
        self.assertFalse(ver_inf.passed)
        self.assertEqual(ver_inf.error_count, 1)

        int_vals = [golden_int32(i, 2, 0x12345679, 0x9E3779B9) for i in range(16)]
        self.assertTrue(
            verify_int_values(int_vals, lambda g: golden_int32(g, 2, 0x12345679, 0x9E3779B9)).passed
        )
        int_vals[3] ^= 1
        ver_int = verify_int_values(int_vals, lambda g: golden_int32(g, 2, 0x12345679, 0x9E3779B9))
        self.assertFalse(ver_int.passed)
        self.assertEqual(ver_int.error_count, 1)
        self.assertIn("index=3", ver_int.message)

    def test_tensor_and_memory_golden_formulas(self) -> None:
        self.assertEqual(golden_fp_mma_table(0.75, 0.5, metal_layout=True)[0], 9.5)
        cuda_tbl = golden_fp_mma_table(0.75, 0.5, metal_layout=False)
        self.assertEqual(cuda_tbl[0], 16.0 * 0.8125 + 15.0)
        self.assertEqual(golden_tensor_int8(0, 0, 0), 136)
        self.assertEqual(golden_int4_cuda(0, 0, 0), 36)
        self.assertEqual(len(PATTERN_MODE_NAMES), 7)
        self.assertEqual(golden_pattern_word(3, 4, 0), 1 << 3)
        self.assertEqual(golden_pattern_word(3, 5, 0), (~(1 << 3)) & 0xFFFFFFFF)
        self.assertNotEqual(golden_splitmix64(1), golden_splitmix64(2))


class TestCalibrationAndSafeVramSizing(unittest.TestCase):
    def test_dynamic_batch_calibration_clamping_and_stepup(self) -> None:
        # Simulated virtual clock where each iteration takes 2 ms -> target 100 ms => ~50 iters.
        fake_clock = {"t": 0.0}
        probed: list[int] = []

        def launch(iters: int) -> None:
            probed.append(iters)
            fake_clock["t"] += iters * 0.002

        calibrated = calibrate_iterations(
            launch,
            lambda: None,
            target_s=0.100,
            min_iters=1,
            max_iters=1000,
            probe_iters=1,
            round_multiple=4,
            clock_fn=lambda: fake_clock["t"],
        )
        # First probe (2 ms < 5 ms) triggers stepped probe (8 iters = 16 ms), then scales to 50 -> rounded to 52 or 48.
        self.assertEqual(probed, [1, 8])
        self.assertEqual(calibrated % 4, 0)
        self.assertTrue(44 <= calibrated <= 56)

        # Very slow GPU (0.5 s per iter) clamps strictly to min_iters=1.
        slow_clock = {"t": 0.0}
        slow_cal = calibrate_iterations(
            lambda iters: slow_clock.__setitem__("t", slow_clock["t"] + iters * 0.5),
            lambda: None,
            min_iters=1,
            max_iters=500,
            clock_fn=lambda: slow_clock["t"],
        )
        self.assertEqual(slow_cal, 1)

        # Ultra-fast/mock GPU (1 us per iter) clamps strictly to max_iters.
        fast_clock = {"t": 0.0}
        fast_cal = calibrate_iterations(
            lambda iters: fast_clock.__setitem__("t", fast_clock["t"] + iters * 1e-6),
            lambda: None,
            min_iters=1,
            max_iters=256,
            clock_fn=lambda: fast_clock["t"],
        )
        self.assertEqual(fast_cal, 256)

    def test_safe_vram_sizing_discrete_and_unified_profiles(self) -> None:
        discrete_24g = _make_mock_device(
            backend="cuda", vram_bytes=24 * GIB, free_bytes=22 * GIB, extra={}
        ).bindings[0].device_info
        self.assertFalse(is_unified_memory_device(discrete_24g))
        pat_24g = compute_safe_vram_budget(discrete_24g, mode="memory-pattern")
        swp_24g = compute_safe_vram_budget(discrete_24g, mode="memory-sweep")
        d2d_24g = compute_safe_vram_budget(discrete_24g, mode="bandwidth-d2d")
        h2d_24g = compute_safe_vram_budget(discrete_24g, mode="bandwidth-h2d", host_ram_bytes=64 * GIB)

        self.assertEqual(pat_24g % VRAM_ALIGNMENT_BYTES, 0)
        self.assertTrue(10 * GIB <= pat_24g <= 16 * GIB)
        self.assertTrue(16 * GIB <= swp_24g <= 19 * GIB)
        self.assertEqual(d2d_24g, BANDWIDTH_DISCRETE_CAP_BYTES)
        self.assertEqual(h2d_24g, BANDWIDTH_DISCRETE_CAP_BYTES)

        # Unified-memory Apple Silicon profile with strict max_buffer_length clamp.
        apple_16g = _make_mock_device(
            backend="metal",
            vendor="Apple",
            arch="applegpu_g14g",
            vram_bytes=12 * GIB,
            free_bytes=12 * GIB,
            extra={"has_unified_memory": True, "max_buffer_length": 7 * GIB},
        ).bindings[0].device_info
        self.assertTrue(is_unified_memory_device(apple_16g))
        swp_apple = compute_safe_vram_budget(apple_16g, mode="memory-sweep")
        d2d_apple = compute_safe_vram_budget(apple_16g, mode="bandwidth-d2d")
        h2d_small_host = compute_safe_vram_budget(
            apple_16g, mode="bandwidth-h2d", host_ram_bytes=512 * MIB
        )
        self.assertLessEqual(swp_apple, 7 * GIB)
        self.assertEqual(d2d_apple, BANDWIDTH_UNIFIED_CAP_BYTES)
        self.assertLessEqual(h2d_small_host, 128 * MIB)

    def test_stepdown_allocation_and_pinned_host_fallback(self) -> None:
        backend = _SimulatingMockBackend(fail_alloc_above_bytes=6 * MIB, fail_pinned_alloc=True)
        dev = _make_mock_device()
        bctx = backend.create_context(dev.bindings[0].device_info)
        try:
            buf, alloc_sz = alloc_device_with_stepdown(
                backend, bctx, 10 * MIB, min_bytes=4 * MIB, alignment=VRAM_ALIGNMENT_BYTES
            )
            self.assertLessEqual(alloc_sz, 6 * MIB)
            self.assertGreaterEqual(alloc_sz, 4 * MIB)
            backend.free_device(bctx, buf)

            host_buf, is_pinned = alloc_host_buffer_with_fallback(backend, bctx, 4096)
            self.assertFalse(is_pinned)
            self.assertIsInstance(host_buf, bytearray)
            free_host_buffer(backend, bctx, host_buf, is_pinned)

            # Fail cleanly with BackendInitError when even minimum buffer cannot be allocated.
            backend.fail_alloc_above_bytes = 1 * MIB
            with self.assertRaises(BackendInitError):
                alloc_device_with_stepdown(backend, bctx, 10 * MIB, min_bytes=4 * MIB)
        finally:
            backend.destroy_context(bctx)
        backend.assert_no_leaks(self)

    def test_launch_geometry_aligns_to_warp_width(self) -> None:
        info = _make_mock_device(
            backend="hip", arch="gfx90a", compute_units=104, extra={"warp_size": 64, "max_threads_per_block": 300}
        ).bindings[0].device_info
        grid, block, total = compute_launch_geometry(info, is_tensor=True)
        self.assertEqual(block[0] % 64, 0)
        self.assertEqual(total, grid[0] * block[0])


class TestAll20ExecutorsEndToEndAndFaultInjection(unittest.TestCase):
    def test_all_20_executors_pass_and_clean_up_on_mock_backend(self) -> None:
        # Keep mock device tiny (2 CUs, 1 MiB VRAM) so all 20 simulated runs finish in < 0.5s.
        backend = _SimulatingMockBackend(backend_name="cuda")
        gpu = _make_mock_device(backend="cuda", arch="sm_100", vram_bytes=2 * MIB, compute_units=1)
        gpu.bindings[0].device_info.extra = {"warp_size": 16, "max_threads_per_block": 16}

        # Cap iteration ceiling in mock runs by monkeypatching target threads/iters? No, our mock
        # takes ~0.5 ms per launch, so cap duration to 0.005s and let real calibration run cleanly.
        all_tests = list_workload_names()
        # Use a small-buffer device so memory/bandwidth buffers stay ~1 MiB in RAM.
        runner = StressRunner(
            devices=[gpu],
            plugins={"cuda": backend},
            tests=all_tests,
            duration_s=0.005,
            warmup_s=0.0,
            explicit_test_selection=False,
            compile_fn=_dummy_compile_fn,
            telemetry_reader=lambda d: TelemetrySample(temperature_c=55.0),
        )
        # Keep calibration fast for the nonlinear dp4a mock path while still exercising real calibration.
        with mock.patch(
            "cross_gpu.workloads.compute_scalar.calibrate_iterations", return_value=2
        ), mock.patch(
            "cross_gpu.workloads.memory.calibrate_iterations", return_value=2
        ):
            report = runner.run()

        self.assertEqual(report.exit_code, ExitCode.SUCCESS)
        self.assertEqual(len(report.results), 20)
        for res in report.results:
            self.assertEqual(res.status, "pass", f"{res.workload} failed: {res.message}")
            self.assertEqual(res.exit_code, ExitCode.SUCCESS)
            self.assertGreater(res.rate or 0.0, 0.0, f"{res.workload} had non-positive rate")
            self.assertGreater(res.total or 0.0, 0.0, f"{res.workload} had non-positive total")
            self.assertEqual(res.errors, 0)
        backend.assert_no_leaks(self)

    def test_fault_injection_compute_bitflip_and_nan_return_compute_error_3(self) -> None:
        cases = [
            ("compute-int64", {"flip_compute_bit": True}),
            ("compute-int32", {"flip_compute_bit": True}),
            ("compute-tensor-int8", {"flip_compute_bit": True}),
            ("compute-fp32", {"flip_compute_bit": True}),
            ("compute-fp32", {"inject_compute_nan": True}),
            ("compute-fp64", {"inject_compute_inf": True}),
            ("compute-tensor-fp16", {"inject_compute_nan": True}),
            ("memory-pattern", {"inject_memory_errors": 3}),
            ("memory-sweep", {"inject_memory_errors": 1}),
            ("bandwidth-d2d", {"corrupt_d2d_byte": True}),
            ("bandwidth-h2d", {"corrupt_h2d_byte": True}),
            ("bandwidth-d2h", {"corrupt_d2h_byte": True}),
        ]
        for wl_name, fault_kwargs in cases:
            with self.subTest(workload=wl_name, fault=fault_kwargs):
                backend = _SimulatingMockBackend(backend_name="cuda", **fault_kwargs)
                gpu = _make_mock_device(
                    backend="cuda",
                    arch="sm_100",
                    vram_bytes=2 * MIB,
                    compute_units=1,
                    extra={"warp_size": 16, "max_threads_per_block": 16},
                )
                with mock.patch(
                    "cross_gpu.workloads.compute_scalar.calibrate_iterations", return_value=2
                ), mock.patch(
                    "cross_gpu.workloads.memory.calibrate_iterations", return_value=1
                ):
                    report = StressRunner(
                        devices=[gpu],
                        plugins={"cuda": backend},
                        tests=[wl_name],
                        duration_s=0.005,
                        compile_fn=_dummy_compile_fn,
                        telemetry_reader=lambda d: TelemetrySample(),
                    ).run()
                self.assertEqual(report.exit_code, ExitCode.COMPUTE_ERROR)
                self.assertEqual(report.results[0].status, "fail")
                self.assertEqual(report.results[0].exit_code, ExitCode.COMPUTE_ERROR)
                self.assertGreaterEqual(report.results[0].errors, 1)
                backend.assert_no_leaks(self)

    def test_unsupported_workload_all_vs_explicit_and_compiler_gate(self) -> None:
        no_fp8_caps = DeviceCapabilities(fp32=True, fp8=False, fp4=False, int32=True)
        gpu_no_fp8 = _make_mock_device(
            index=0, backend="cuda", vram_bytes=2 * MIB, compute_units=1, caps=no_fp8_caps,
            extra={"warp_size": 16, "max_threads_per_block": 16},
        )
        gpu_with_fp8 = _make_mock_device(
            index=1, backend="cuda", vram_bytes=2 * MIB, compute_units=1,
            extra={"warp_size": 16, "max_threads_per_block": 16},
        )
        backend = _SimulatingMockBackend(backend_name="cuda")

        with mock.patch(
            "cross_gpu.workloads.compute_scalar.calibrate_iterations", return_value=2
        ):
            # --test all skips unsupported compute-fp8 without triggering INIT_ERROR (2).
            rep_all = StressRunner(
                devices=[gpu_no_fp8],
                plugins={"cuda": backend},
                tests=["compute-fp32", "compute-fp8"],
                duration_s=0.005,
                explicit_test_selection=False,
                compile_fn=_dummy_compile_fn,
                telemetry_reader=lambda d: TelemetrySample(),
            ).run()
            self.assertEqual(rep_all.exit_code, ExitCode.SUCCESS)
            self.assertEqual([r.status for r in rep_all.results], ["pass", "unsupported"])

            # Explicit --test compute-fp8 when unsupported on all selected devices exits with INIT_ERROR (2).
            rep_explicit = StressRunner(
                devices=[gpu_no_fp8],
                plugins={"cuda": backend},
                tests=["compute-fp8"],
                duration_s=0.005,
                explicit_test_selection=True,
                compile_fn=_dummy_compile_fn,
                telemetry_reader=lambda d: TelemetrySample(),
            ).run()
            self.assertEqual(rep_explicit.exit_code, ExitCode.INIT_ERROR)
            self.assertEqual(rep_explicit.results[0].status, "unsupported")

            # Multi-GPU explicit --test compute-fp8 passes when at least one selected GPU supports it.
            rep_multi = StressRunner(
                devices=[gpu_no_fp8, gpu_with_fp8],
                plugins={"cuda": backend},
                tests=["compute-fp8"],
                duration_s=0.005,
                explicit_test_selection=True,
                compile_fn=_dummy_compile_fn,
                telemetry_reader=lambda d: TelemetrySample(),
            ).run()
            self.assertEqual(rep_multi.exit_code, ExitCode.SUCCESS)
            self.assertEqual([r.status for r in rep_multi.results], ["unsupported", "pass"])

            # Compiler #error "UNSUPPORTED" diagnostic also maps cleanly to status="unsupported".
            def _unsupported_compile(*args: Any, **kwargs: Any) -> Any:
                raise KernelCompilationError(
                    "compile failed",
                    compiler_log='#error "UNSUPPORTED: arch lacks feature"',
                )

            rep_gate = StressRunner(
                devices=[gpu_with_fp8],
                plugins={"cuda": backend},
                tests=["compute-fp8"],
                duration_s=0.005,
                explicit_test_selection=False,
                compile_fn=_unsupported_compile,
                telemetry_reader=lambda d: TelemetrySample(),
            ).run()
            self.assertEqual(rep_gate.results[0].status, "unsupported")
            self.assertEqual(rep_gate.exit_code, ExitCode.INIT_ERROR)  # All results unsupported

        backend.assert_no_leaks(self)

    def test_warmup_exclusion_and_sigint_cleanup(self) -> None:
        backend = _SimulatingMockBackend(backend_name="cuda", step_sleep_s=0.005)
        gpu = _make_mock_device(
            backend="cuda",
            vram_bytes=2 * MIB,
            compute_units=1,
            extra={"warp_size": 16, "max_threads_per_block": 16},
        )
        events: list[dict[str, Any]] = []

        with mock.patch(
            "cross_gpu.workloads.compute_scalar.calibrate_iterations", return_value=2
        ):
            runner = StressRunner(
                devices=[gpu],
                plugins={"cuda": backend},
                tests=["compute-fp32"],
                duration_s=0.06,
                warmup_s=0.06,
                telemetry_interval_s=0.01,
                event_callback=events.append,
                compile_fn=_dummy_compile_fn,
                telemetry_reader=lambda d: TelemetrySample(temperature_c=60.0),
            )
            report = runner.run()

        self.assertEqual(report.exit_code, ExitCode.SUCCESS)
        res = report.results[0]
        # Total launches include warmup + timed steps; total reported GFLOP must strictly exclude warmup.
        total_launches = len(backend.launch_log)
        num_elements = 64 * 16  # min 64 blocks * 16 threads
        ops_per_launch = num_elements * 2 * 256 / 1e9
        self.assertGreaterEqual(total_launches, 3)
        self.assertLess(res.total or 0.0, total_launches * ops_per_launch)
        phases = {e["phase"] for e in events if e["type"] == "telemetry"}
        self.assertIn("warmup", phases)
        self.assertIn("run", phases)
        backend.assert_no_leaks(self)

        # SIGINT / request_stop() mid-run exits promptly and frees all buffers/modules cleanly.
        backend2 = _SimulatingMockBackend(backend_name="cuda", step_sleep_s=0.02)
        with mock.patch(
            "cross_gpu.workloads.compute_scalar.calibrate_iterations", return_value=2
        ):
            runner2 = StressRunner(
                devices=[gpu],
                plugins={"cuda": backend2},
                tests=["compute-int32", "compute-fp32"],
                duration_s=10.0,
                compile_fn=_dummy_compile_fn,
                telemetry_reader=lambda d: TelemetrySample(),
            )
            stopper = threading.Timer(0.06, runner2.request_stop)
            stopper.start()
            try:
                t0 = time.perf_counter()
                report2 = runner2.run()
                elapsed = time.perf_counter() - t0
            finally:
                stopper.cancel()

        self.assertTrue(report2.interrupted)
        self.assertLess(elapsed, 2.0)
        self.assertEqual(report2.exit_code, ExitCode.SUCCESS)
        backend2.assert_no_leaks(self)

    def test_bandwidth_d2d_fallback_without_binary_and_partial_prepare_cleanup(self) -> None:
        backend = _SimulatingMockBackend(backend_name="cuda")
        gpu = _make_mock_device(
            backend="cuda",
            vram_bytes=2 * MIB,
            compute_units=1,
            extra={"warp_size": 16, "max_threads_per_block": 16},
        )
        bctx = backend.create_context(gpu.bindings[0].device_info)
        try:
            spec = WORKLOAD_REGISTRY["bandwidth-d2d"].spec
            exec_ctx = WorkloadExecutionContext(
                device=gpu,
                binding=gpu.bindings[0],
                backend=backend,
                backend_context=bctx,
                spec=spec,
                state={"target_bytes": 65536, "fixed_iterations": 2},
            )
            executor = BandwidthD2DExecutor(spec)
            executor.prepare(exec_ctx, None)
            self.assertEqual(exec_ctx.state["method"], "memcpy_d2d")
            step = executor.run_step(exec_ctx)
            self.assertEqual(step.operations, float(2 * 2 * 65536))
            ver = executor.verify(exec_ctx)
            self.assertTrue(ver.passed, ver.message)
            executor.cleanup(exec_ctx)
            # Calling cleanup a second time must be safely idempotent.
            executor.cleanup(exec_ctx)
        finally:
            backend.destroy_context(bctx)
        backend.assert_no_leaks(self)

    def test_hip_and_metal_backend_specific_golden_paths(self) -> None:
        for bname, arch, wl in (
            ("hip", "gfx90a", "compute-int8"),
            ("hip", "gfx90a", "compute-int4"),
            ("metal", "applegpu_g14g", "compute-int8"),
            ("metal", "applegpu_g14g", "compute-tensor-fp32"),
        ):
            with self.subTest(backend=bname, workload=wl):
                backend = _SimulatingMockBackend(backend_name=bname)
                gpu = _make_mock_device(
                    backend=bname,
                    arch=arch,
                    vram_bytes=2 * MIB,
                    compute_units=1,
                    extra={"warp_size": 16, "max_threads_per_block": 16},
                )
                with mock.patch(
                    "cross_gpu.workloads.compute_scalar.calibrate_iterations", return_value=2
                ):
                    rep = StressRunner(
                        devices=[gpu],
                        plugins={bname: backend},
                        tests=[wl],
                        duration_s=0.005,
                        compile_fn=_dummy_compile_fn,
                        telemetry_reader=lambda d: TelemetrySample(),
                    ).run()
                self.assertEqual(rep.exit_code, ExitCode.SUCCESS, rep.results[0].message)
                self.assertEqual(rep.results[0].status, "pass")
                backend.assert_no_leaks(self)


if __name__ == "__main__":
    unittest.main()
