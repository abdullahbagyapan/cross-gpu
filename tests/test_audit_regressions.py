# SPDX-License-Identifier: MIT
"""Pre-v1.0.0 audit regression tests covering ABI safety, SONAME order, layout, and edge cases."""

from __future__ import annotations

import ast
import ctypes
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

from src.backends.base import (
    BackendContext,
    DeviceBuffer,
    HostBuffer,
    KernelModule,
    resolve_host_pointer,
)
from src.backends.cuda_backend import CUDA_LIB_CANDIDATES, CudaBackend
from src.backends.hip_backend import (
    HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    HIP_DEVICE_ATTRIBUTE_INTEGRATED,
    HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    HIP_DEVICE_ATTRIBUTE_PCI_BUS_ID,
    HIP_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
    HIP_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
    HIP_DEVICE_ATTRIBUTE_WARP_SIZE,
    HIP_LIB_CANDIDATES,
    HIP_SUCCESS,
    HipBackend,
    _extract_gcn_arch_name,
)
from src.backends.levelzero_backend import (
    LEVELZERO_LIB_CANDIDATES,
    ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE,
    ZE_RESULT_ERROR_DEVICE_LOST,
    ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET,
    ZE_RESULT_SUCCESS,
    LevelZeroBackend,
)
from src.backends.metal_backend import MetalBackend
from src.backends.opencl_backend import CL_SUCCESS, OPENCL_LIB_CANDIDATES, OpenCLBackend
from src.cli import format_duration
from src.embed_binary import (
    HIPRTC_LIB_CANDIDATES,
    NVRTC_LIB_CANDIDATES,
    CompiledKernelBinary,
)
from src.exit_codes import (
    BackendHardwareError,
    BackendInitError,
    KernelCompilationError,
)
from src.models import BackendDeviceInfo
from src.workloads.common import write_host_buffer


def _is_versioned_so(name: str) -> bool:
    return ".so." in name


def _is_unversioned_so(name: str) -> bool:
    return name.endswith(".so")


class TestAuditRegressions(unittest.TestCase):
    def test_soname_probing_order_versioned_before_unversioned_and_no_distro_paths(self) -> None:
        all_linux_lists = {
            "CUDA_LIB_CANDIDATES": CUDA_LIB_CANDIDATES,
            "HIP_LIB_CANDIDATES": HIP_LIB_CANDIDATES,
            "LEVELZERO_LIB_CANDIDATES": LEVELZERO_LIB_CANDIDATES,
            "NVRTC_LIB_CANDIDATES": NVRTC_LIB_CANDIDATES,
            "HIPRTC_LIB_CANDIDATES": HIPRTC_LIB_CANDIDATES,
        }
        if not sys.platform == "darwin":
            all_linux_lists["OPENCL_LIB_CANDIDATES"] = OPENCL_LIB_CANDIDATES

        forbidden_prefixes = ("/usr/lib", "/usr/local", "/opt/rocm", "/lib64")
        for label, candidates in all_linux_lists.items():
            seen_unversioned = False
            for cand in candidates:
                for prefix in forbidden_prefixes:
                    self.assertFalse(
                        cand.startswith(prefix),
                        f"{label} contains hardcoded distro path: {cand}",
                    )
                if _is_unversioned_so(cand):
                    seen_unversioned = True
                elif _is_versioned_so(cand):
                    self.assertFalse(
                        seen_unversioned,
                        f"{label} probes versioned SONAME {cand!r} after unversioned .so: {candidates}",
                    )

        # Explicitly verify exact ordering of key candidate lists.
        self.assertEqual(HIP_LIB_CANDIDATES, ("libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so"))
        self.assertEqual(NVRTC_LIB_CANDIDATES, ("libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so"))
        self.assertEqual(
            HIPRTC_LIB_CANDIDATES,
            (
                "libhiprtc.so.6",
                "libhiprtc.so.5",
                "libamdhip64.so.6",
                "libamdhip64.so.5",
                "libhiprtc.so",
                "libamdhip64.so",
            ),
        )

    def test_hip_stable_api_enumeration_and_opaque_gcn_arch_scan(self) -> None:
        # Verify opaque gfx* token scanner handles arbitrary struct offsets safely.
        raw_blob = b"AMD Instinct MI250X\x00" + (b"\x00" * 1400) + b"gfx90a:sramecc+:xnack-\x00" + (b"\x00" * 512)
        self.assertEqual(_extract_gcn_arch_name(raw_blob), "gfx90a:sramecc+:xnack-")

        attr_map = {
            HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: 9,
            HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: 0,
            HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: 104,
            HIP_DEVICE_ATTRIBUTE_WARP_SIZE: 64,
            HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: 1024,
            HIP_DEVICE_ATTRIBUTE_INTEGRATED: 0,
            HIP_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: 0,
            HIP_DEVICE_ATTRIBUTE_PCI_BUS_ID: 0x43,
            HIP_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: 0,
        }

        class _FakeHipApi:
            def hipGetDeviceCount(self, out_ptr):
                out_ptr._obj.value = 1
                return HIP_SUCCESS

            def hipRuntimeGetVersion(self, out_ptr):
                out_ptr._obj.value = 60241133
                return HIP_SUCCESS

            def hipDeviceGetName(self, buf, length, idx):
                buf.value = b"AMD Instinct MI250X"
                return HIP_SUCCESS

            def hipDeviceTotalMem(self, out_ptr, idx):
                out_ptr._obj.value = 64 * 1024**3
                return HIP_SUCCESS

            def hipDeviceGetAttribute(self, out_ptr, attr, idx):
                if attr in attr_map:
                    out_ptr._obj.value = attr_map[attr]
                    return HIP_SUCCESS
                return 1

            def hipGetDevicePropertiesR0600(self, ptr, idx):
                ctypes.memmove(ptr, raw_blob, len(raw_blob))
                return HIP_SUCCESS

            hipGetDeviceProperties = None

            def hipDeviceGetPCIBusId(self, buf, length, idx):
                buf.value = b"0000:43:00.0"
                return HIP_SUCCESS

            def hipDeviceGetUuid(self, out_ptr, idx):
                for i in range(16):
                    out_ptr._obj.bytes[i] = i + 1
                return HIP_SUCCESS

            def hipSetDevice(self, dev_idx):
                return 709  # Simulate destroyed/lost device during cleanup

            def hipFree(self, ptr):
                return HIP_SUCCESS

            def hipHostFree(self, ptr):
                return HIP_SUCCESS

            def hipModuleUnload(self, mod):
                return HIP_SUCCESS

            hipGetErrorString = None

        backend = HipBackend()
        backend._api = _FakeHipApi()  # type: ignore[assignment]
        devices = backend.enumerate_devices()
        self.assertEqual(len(devices), 1)
        dev = devices[0]
        self.assertEqual(dev.name, "AMD Instinct MI250X")
        self.assertEqual(dev.architecture, "gfx90a")
        self.assertEqual(dev.compute_units, 104)
        self.assertEqual(dev.total_memory_bytes, 64 * 1024**3)
        self.assertEqual(dev.pci_bdf, "0000:43:00.0")
        self.assertEqual(dev.uuid, "01020304-0506-0708-090a-0b0c0d0e0f10")
        self.assertEqual(dev.extra["warp_size"], 64)
        self.assertEqual(dev.extra["max_threads_per_block"], 1024)
        self.assertFalse(dev.extra["has_unified_memory"])
        self.assertTrue(dev.capabilities.bf16)
        self.assertTrue(dev.capabilities.tensor_fp32)

        # Cleanup must remain exception-safe and zero handles even if hipSetDevice fails.
        ctx = BackendContext(backend="hip", device_info=dev, handle=0)
        dbuf = DeviceBuffer(backend="hip", handle=0x1000, size_bytes=64, address=0x1000)
        hbuf = HostBuffer(backend="hip", handle=0x2000, size_bytes=64, pointer=0x2000)
        mod = KernelModule(backend="hip", handle=0x3000, functions={"k": 0x4000})
        backend.free_device(ctx, dbuf)
        backend.free_pinned_host(ctx, hbuf)
        backend.unload_module(ctx, mod)
        self.assertEqual(dbuf.handle, 0)
        self.assertEqual(hbuf.handle, 0)
        self.assertIsNone(mod.handle)
        self.assertEqual(mod.functions, {})

    def test_levelzero_hardware_fault_codes_and_spirv_flag_filtering(self) -> None:
        self.assertEqual(ZE_RESULT_ERROR_DEVICE_LOST, 0x70000001)
        self.assertEqual(ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET, 0x70000006)
        self.assertEqual(ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE, 0x70000007)

        backend = LevelZeroBackend()
        for fault_code in (0x70000001, 0x70000006, 0x70000007):
            with self.assertRaises(BackendHardwareError):
                backend._check(fault_code, "test_fault")

        # 0x70000005 is ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS -> must map to BackendInitError.
        with self.assertRaises(BackendInitError):
            backend._check(0x70000005, "test_perm")

        captured_flags: list[bytes | None] = []

        class _FakeZeApi:
            zeModuleBuildLogGetString = None
            zeModuleBuildLogDestroy = None

            def zeModuleCreate(self, ctx, dev, desc_ptr, mod_out, log_out):
                captured_flags.append(desc_ptr._obj.pBuildFlags)
                mod_out._obj.value = 0x5555
                log_out._obj.value = 0
                return ZE_RESULT_SUCCESS

        backend._api = _FakeZeApi()  # type: ignore[assignment]
        backend._initialized = True
        dummy_dev = BackendDeviceInfo(backend="levelzero", priority=100, backend_device_index=0, native_handle=None, name="Intel Arc", vendor="Intel")
        ctx = BackendContext(backend="levelzero", device_info=dummy_dev, handle=0x1111, extra={"device": 0x2222})

        with self.assertRaises(KernelCompilationError):
            backend.load_module(
                ctx,
                CompiledKernelBinary(
                    backend="levelzero", workload="compute_fp32", entry_point="k", format="spirv", binary_bytes=b""
                ),
            )

        mod = backend.load_module(
            ctx,
            CompiledKernelBinary(
                backend="levelzero",
                workload="compute_fp32",
                entry_point="k",
                format="spirv",
                binary_bytes=b"\x03\x02\x23\x07\x00\x00\x01\x00",
                compile_flags=("-DCROSS_GPU_HAS_FP32=1", "-ze-opt-large-register-file"),
            ),
        )
        self.assertEqual(mod.handle, 0x5555)
        self.assertEqual(captured_flags, [b"-ze-opt-large-register-file"])

    def test_cuda_cleanup_resilient_on_poisoned_context(self) -> None:
        calls: list[str] = []

        class _FakeCudaApi:
            def cuCtxSetCurrent(self, ctx_handle):
                calls.append("cuCtxSetCurrent")
                return 709  # CUDA_ERROR_CONTEXT_IS_DESTROYED

            def cuMemFree(self, ptr):
                calls.append("cuMemFree")
                return 0

            def cuMemFreeHost(self, ptr):
                calls.append("cuMemFreeHost")
                return 0

            def cuModuleUnload(self, mod):
                calls.append("cuModuleUnload")
                return 0

        backend = CudaBackend()
        backend._api = _FakeCudaApi()  # type: ignore[assignment]
        backend._initialized = True
        dummy_dev = BackendDeviceInfo(backend="cuda", priority=100, backend_device_index=0, native_handle=0, name="GPU", vendor="NVIDIA")
        ctx = BackendContext(backend="cuda", device_info=dummy_dev, handle=0x10)
        dbuf = DeviceBuffer(backend="cuda", handle=0x100, size_bytes=64, address=0x100)
        hbuf = HostBuffer(backend="cuda", handle=0x200, size_bytes=64, pointer=0x200)
        mod = KernelModule(backend="cuda", handle=0x300, functions={"k": 0x400})

        backend.free_device(ctx, dbuf)
        backend.free_pinned_host(ctx, hbuf)
        backend.unload_module(ctx, mod)
        self.assertEqual(dbuf.handle, 0)
        self.assertEqual(hbuf.handle, 0)
        self.assertIsNone(mod.handle)
        self.assertIn("cuMemFree", calls)
        self.assertIn("cuMemFreeHost", calls)
        self.assertIn("cuModuleUnload", calls)

    def test_opencl_size_t_query_and_empty_binary_guard(self) -> None:
        observed_sizes: list[int] = []

        class _FakeClApi:
            def clGetDeviceInfo(self, dev, param, param_size, param_val_ptr, ret_ptr):
                observed_sizes.append(param_size)
                param_val_ptr._obj.value = 1024
                return CL_SUCCESS

        backend = OpenCLBackend()
        backend._api = _FakeClApi()  # type: ignore[assignment]
        val = backend._get_device_size_t(backend._api, 0x1, 0x1004)
        self.assertEqual(val, 1024)
        self.assertEqual(observed_sizes, [ctypes.sizeof(ctypes.c_size_t)])

        dummy_dev = BackendDeviceInfo(backend="opencl", priority=50, backend_device_index=0, native_handle=None, name="CL", vendor="Intel")
        ctx = BackendContext(backend="opencl", device_info=dummy_dev, handle=0x1, extra={"device": 0x2})
        with self.assertRaises(KernelCompilationError):
            backend.load_module(
                ctx,
                CompiledKernelBinary(
                    backend="opencl", workload="compute_fp32", entry_point="k", format="opencl_source", binary_bytes=b""
                ),
            )

    def test_metal_launch_exception_cleanup_and_null_dispatch_data(self) -> None:
        events: list[str] = []

        class _FakeMetalApi:
            sel_commandBuffer = 1
            sel_computeCommandEncoder = 2
            sel_setComputePipelineState = 3
            sel_endEncoding = 4
            sel_commit = 5
            sel_waitUntilCompleted = 6
            sel_status = 7
            sel_release = 8
            sel_retain = 9

            def objc_autoreleasePoolPush(self):
                events.append("pool_push")
                return 0xABCD

            def objc_autoreleasePoolPop(self, pool):
                events.append("pool_pop")

            def objc_retain(self, obj):
                events.append(f"retain:{obj}")
                return obj

            def objc_release(self, obj):
                events.append(f"release:{obj}")

            def dispatch_data_create(self, buf, size, queue, destructor):
                return None  # Simulate allocation failure

            def dispatch_release(self, data):
                events.append("dispatch_release_should_not_run")

            def msg_ptr(self, obj, sel):
                if sel == self.sel_commandBuffer:
                    return 0xCAFE
                if sel == self.sel_computeCommandEncoder:
                    return 0xBEEF
                return 0

            def msg_void_ptr(self, obj, sel, ptr):
                pass

            def msg_void(self, obj, sel):
                if sel == self.sel_endEncoding:
                    events.append(f"endEncoding:{obj}")
                elif sel == self.sel_waitUntilCompleted:
                    events.append(f"wait:{obj}")

            def msg_u64(self, obj, sel):
                return 4  # MTLCommandBufferStatusCompleted

        backend = MetalBackend()
        backend._api = _FakeMetalApi()  # type: ignore[assignment]
        dummy_dev = BackendDeviceInfo(backend="metal", priority=100, backend_device_index=0, native_handle=0x1, name="M2", vendor="Apple")
        ctx = BackendContext(backend="metal", device_info=dummy_dev, handle=0x1, queue=0x2, extra={"pending_cmd_bufs": []})
        mod = KernelModule(backend="metal", handle=0x3, functions={"k": 0x4})

        # Unsupported kernel arg raises TypeError: encoder must end and cmd_buf must be released.
        with mock.patch.object(backend, "is_available", return_value=True):
            with self.assertRaises(TypeError):
                backend.launch_kernel(ctx, mod, "k", (1, 1, 1), (1, 1, 1), args=[object()])
            self.assertIn(f"endEncoding:{0xBEEF}", events)
            self.assertIn(f"release:{0xCAFE}", events)
            self.assertEqual(ctx.extra["pending_cmd_bufs"], [])

            # Null dispatch_data_create must raise BackendInitError without calling dispatch_release(None).
            with self.assertRaises(BackendInitError):
                backend.load_module(
                    ctx,
                    CompiledKernelBinary(
                        backend="metal",
                        workload="compute_fp32",
                        entry_point="k",
                        format="metallib",
                        binary_bytes=b"MTLB\x00\x00\x00\x00",
                    ),
                )
            self.assertNotIn("dispatch_release_should_not_run", events)

            # synchronize() wraps pending waits in an autorelease pool.
            events.clear()
            ctx.extra["pending_cmd_bufs"].append(0xFACE)
            backend.synchronize(ctx)
            self.assertEqual(events, ["pool_push", f"wait:{0xFACE}", f"release:{0xFACE}", "pool_pop"])

    def test_readonly_memoryview_host_pointer_and_write_host_buffer(self) -> None:
        ro_view = memoryview(b"CrossGPU-RO-Buffer")
        self.assertTrue(ro_view.readonly)

        ptr, keepalive = resolve_host_pointer(ro_view)
        self.assertTrue(ptr.value)
        self.assertEqual(ctypes.string_at(ptr.value, len(ro_view)), b"CrossGPU-RO-Buffer")
        del keepalive

        dst_arr = (ctypes.c_ubyte * len(ro_view))()
        hbuf = HostBuffer(
            backend="stub",
            handle=1,
            size_bytes=len(ro_view),
            pointer=ctypes.addressof(dst_arr),
        )
        write_host_buffer(hbuf, ro_view)
        self.assertEqual(bytes(dst_arr), b"CrossGPU-RO-Buffer")

    def test_format_duration_zero_and_submillisecond(self) -> None:
        self.assertEqual(format_duration(0.0), "0s")
        self.assertEqual(format_duration(0.0004), "<1ms")
        self.assertEqual(format_duration(0.001), "1ms")
        self.assertEqual(format_duration(0.25), "250ms")
        self.assertEqual(format_duration(1.0), "1s")

    def test_layout_zero_blobs_zero_cross_gpu_imports_and_spdx_headers(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        self.assertFalse((repo_root / "cross_gpu").exists())
        self.assertFalse((repo_root / "src" / "cross_gpu").exists())
        self.assertTrue((repo_root / "src" / "__init__.py").is_file())

        # Zero absolute cross_gpu imports inside src/
        for py_file in sorted((repo_root / "src").rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            alias.name == "cross_gpu" or alias.name.startswith("cross_gpu."),
                            f"Absolute cross_gpu import in {py_file}:{node.lineno}",
                        )
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    self.assertFalse(
                        node.module == "cross_gpu" or node.module.startswith("cross_gpu."),
                        f"Absolute cross_gpu import in {py_file}:{node.lineno}",
                    )

        # Zero legacy opaque blob or digest artifacts in src/
        forbidden_tokens = ("zlib" + "_b64", "EMBEDDED" + "_KERNELS", "hash" + "lib", "sha" + "256")
        for py_file in sorted((repo_root / "src").rglob("*.py")):
            text = py_file.read_text(encoding="utf-8")
            for tok in forbidden_tokens:
                self.assertNotIn(tok, text, f"Forbidden token {tok!r} in {py_file}")
            first_line = text.splitlines()[0]
            self.assertIn("SPDX-License-Identifier: MIT", first_line, f"Missing SPDX in {py_file}")


if __name__ == "__main__":
    unittest.main()
