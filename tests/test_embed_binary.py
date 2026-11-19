"""Unit tests for embedded kernel packing, verification, caching, and runtime compiler paths."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from cross_gpu.embed_binary import (
    EMBEDDED_KERNELS,
    RuntimeBinaryCache,
    compile_kernel_runtime,
    compute_cache_key,
    decode_payload,
    default_cache_dir,
    default_entry_points,
    encode_payload,
    get_embedded_kernel,
    pack_kernels,
    scan_kernels_directory,
    verify_embedded_kernels,
)
from cross_gpu.exit_codes import (
    KernelCompilationError,
    KernelNotFoundError,
    UnsupportedKernelError,
)
from cross_gpu.models import DeviceCapabilities


class TestEmbedBinary(unittest.TestCase):
    def test_pack_and_verify_roundtrip_in_temp_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            kdir = tmp_path / "kernels"
            (kdir / "cuda").mkdir(parents=True)
            (kdir / "opencl").mkdir(parents=True)
            (kdir / "cuda" / ".gitkeep").write_text("", encoding="utf-8")
            (kdir / "cuda" / "README.md").write_text("docs", encoding="utf-8")
            (kdir / "cuda" / "compute-fp32.cu").write_text(
                'extern "C" __global__ void kernel_compute_fp32() {}',
                encoding="utf-8",
            )
            (kdir / "opencl" / "compute-fp32.cl").write_text(
                "__kernel void kernel_compute_fp32(void) {}",
                encoding="utf-8",
            )

            target_py = tmp_path / "embed_copy.py"
            target_py.write_text(
                "# BEGIN EMBEDDED_KERNELS\nEMBEDDED_KERNELS = {}\n# END EMBEDDED_KERNELS\n",
                encoding="utf-8",
            )

            packed = pack_kernels(kernels_dir=kdir, target_module_path=target_py)
            self.assertEqual(sorted(packed.keys()), ["cuda:compute-fp32", "opencl:compute-fp32"])
            self.assertEqual(verify_embedded_kernels(packed), [])

            raw, meta = get_embedded_kernel("cuda", "compute-fp32", registry=packed)
            self.assertIn(b"kernel_compute_fp32", raw)
            self.assertEqual(meta["format"], "cuda_source")

            # Corrupt SHA-256 and ensure verification catches it.
            corrupt = dict(packed["cuda:compute-fp32"])
            corrupt["sha256"] = "0" * 64
            errors = verify_embedded_kernels({"cuda:compute-fp32": corrupt})
            self.assertEqual(len(errors), 1)
            with self.assertRaises(ValueError):
                decode_payload(corrupt)

    def test_missing_kernel_raises_kernel_not_found(self) -> None:
        with self.assertRaises(KernelNotFoundError):
            get_embedded_kernel("cuda", "compute-fp32", registry={})

        with tempfile.TemporaryDirectory() as tmp:
            cache = RuntimeBinaryCache(Path(tmp))
            with self.assertRaises(KernelNotFoundError):
                compile_kernel_runtime("cuda", "compute-fp32", "sm_89", cache=cache, registry={})

    def test_unsupported_capability_raises_unsupported_kernel_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = RuntimeBinaryCache(Path(tmp))
            with self.assertRaises(UnsupportedKernelError):
                compile_kernel_runtime(
                    "cuda",
                    "compute-fp8",
                    "sm_75",
                    capabilities=DeviceCapabilities(fp8=False),
                    source_override="kernel",
                    cache=cache,
                )

    def test_cache_key_uniqueness_and_readonly_degradation(self) -> None:
        k1 = compute_cache_key("cuda", "sm_89", "12.4", "compute-fp32", "abc", ("-O3",))
        k2 = compute_cache_key("cuda", "sm_80", "12.4", "compute-fp32", "abc", ("-O3",))
        k3 = compute_cache_key("cuda", "sm_89", "12.5", "compute-fp32", "abc", ("-O3",))
        k4 = compute_cache_key("cuda", "sm_89", "12.4", "compute-fp32", "def", ("-O3",))
        self.assertEqual(len({k1, k2, k3, k4}), 4)
        self.assertEqual(len(k1), 64)

        with tempfile.TemporaryDirectory() as tmp:
            cache = RuntimeBinaryCache(Path(tmp))
            with mock.patch("pathlib.Path.mkdir", side_effect=PermissionError("read-only")):
                compiled = compile_kernel_runtime(
                    "opencl",
                    "compute-fp32",
                    "gfx1100",
                    capabilities=DeviceCapabilities(fp32=True),
                    source_override="__kernel void kernel_compute_fp32(void) {}",
                    cache=cache,
                )
            self.assertEqual(compiled.format, "opencl_source")
            # Second lookup hits in-memory cache cleanly.
            cached = compile_kernel_runtime(
                "opencl",
                "compute-fp32",
                "gfx1100",
                capabilities=DeviceCapabilities(fp32=True),
                source_override="__kernel void kernel_compute_fp32(void) {}",
                cache=cache,
            )
            self.assertIs(compiled, cached)

    def test_runtime_compile_error_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = RuntimeBinaryCache(Path(tmp))
            with mock.patch(
                "cross_gpu.embed_binary.load_shared_library",
                side_effect=OSError("missing libnvrtc"),
            ):
                with self.assertRaises(KernelCompilationError) as ctx:
                    compile_kernel_runtime(
                        "cuda",
                        "compute-fp32",
                        "sm_89",
                        source_override="__global__ void k() {}",
                        cache=cache,
                    )
                self.assertIn("missing libnvrtc", ctx.exception.compiler_log)

            with mock.patch(
                "cross_gpu.embed_binary.load_shared_library",
                side_effect=OSError("missing libhiprtc"),
            ):
                with self.assertRaises(KernelCompilationError) as ctx:
                    compile_kernel_runtime(
                        "hip",
                        "compute-fp32",
                        "gfx1100",
                        source_override="__global__ void k() {}",
                        cache=cache,
                    )
                self.assertIn("missing libhiprtc", ctx.exception.compiler_log)

            with mock.patch("shutil.which", return_value=None):
                with self.assertRaises(KernelCompilationError):
                    compile_kernel_runtime(
                        "levelzero",
                        "compute-fp32",
                        "",
                        source_override="__kernel void k() {}",
                        cache=cache,
                    )

            # Metal gracefully passes through MSL source when xcrun metal is missing.
            with mock.patch("shutil.which", return_value=None):
                mtl = compile_kernel_runtime(
                    "metal",
                    "compute-fp32",
                    "applegpu",
                    source_override="kernel void k() {}",
                    cache=cache,
                )
                self.assertEqual(mtl.format, "msl_source")
                self.assertIn(b"kernel void k() {}", mtl.binary_bytes)

    def test_cli_verify_and_list(self) -> None:
        res_verify = subprocess.run(
            [sys.executable, "-m", "cross_gpu.embed_binary", "verify"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_verify.returncode, 0, res_verify.stderr)

        res_list = subprocess.run(
            [sys.executable, "-m", "cross_gpu.embed_binary", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_list.returncode, 0, res_list.stderr)

    def test_encode_decode_helpers_and_empty_dir_scan(self) -> None:
        b64, sha = encode_payload(b"hello")
        raw = decode_payload(
            {
                "backend": "opencl",
                "workload": "test",
                "format": "opencl_source",
                "sha256": sha,
                "size_bytes": 5,
                "zlib_b64": b64,
            }
        )
        self.assertEqual(raw, b"hello")
        with tempfile.TemporaryDirectory() as empty_dir:
            self.assertEqual(scan_kernels_directory(Path(empty_dir)), {})

    def test_os_standard_cache_directories_and_sandbox_fallback(self) -> None:
        fake_home = Path("/fake/home")

        # Linux: respects absolute XDG_CACHE_HOME
        self.assertEqual(
            default_cache_dir(
                platform_name="linux",
                env={"XDG_CACHE_HOME": "/var/cache/custom"},
                home=fake_home,
            ),
            Path("/var/cache/custom/cross-gpu"),
        )

        # Linux: ignores relative XDG_CACHE_HOME per XDG spec and falls back to ~/.cache
        self.assertEqual(
            default_cache_dir(
                platform_name="linux",
                env={"XDG_CACHE_HOME": "relative/cache"},
                home=fake_home,
            ),
            fake_home / ".cache" / "cross-gpu",
        )

        # Linux: unset XDG_CACHE_HOME falls back to ~/.cache/cross-gpu
        self.assertEqual(
            default_cache_dir(platform_name="linux", env={}, home=fake_home),
            fake_home / ".cache" / "cross-gpu",
        )

        # macOS: ~/Library/Caches/cross-gpu
        self.assertEqual(
            default_cache_dir(platform_name="darwin", env={}, home=fake_home),
            fake_home / "Library" / "Caches" / "cross-gpu",
        )

        # Explicit CROSS_GPU_CACHE_DIR override takes precedence on any platform
        self.assertEqual(
            default_cache_dir(
                platform_name="darwin",
                env={"CROSS_GPU_CACHE_DIR": "/tmp/cg-override"},
                home=fake_home,
            ),
            Path("/tmp/cg-override"),
        )

        # Unset/unavailable HOME degrades safely to None (in-memory-only cache)
        with mock.patch("pathlib.Path.home", side_effect=RuntimeError("no home")):
            self.assertIsNone(default_cache_dir(platform_name="linux", env={}))
            self.assertIsNone(default_cache_dir(platform_name="darwin", env={}))
            with mock.patch.dict("os.environ", {}, clear=True):
                mem_only = RuntimeBinaryCache()
                self.assertIsNone(mem_only.cache_dir)
                compiled = compile_kernel_runtime(
                    "opencl",
                    "compute-fp32",
                    "arch",
                    capabilities=DeviceCapabilities(fp32=True),
                    source_override="__kernel void kernel_compute_fp32(void) {}",
                    cache=mem_only,
                )
                self.assertIs(mem_only.get(compiled.cache_key), compiled)


EXPECTED_BACKENDS: tuple[str, ...] = ("cuda", "hip", "levelzero", "metal", "opencl")
EXPECTED_KERNEL_WORKLOADS: tuple[str, ...] = (
    "compute_int64",
    "compute_int32",
    "compute_int16",
    "compute_int8",
    "compute_int4",
    "compute_fp64",
    "compute_fp32",
    "compute_fp16",
    "compute_fp8",
    "compute_fp4",
    "compute_bf16",
    "compute_tensor_fp32",
    "compute_tensor_fp16",
    "compute_tensor_bf16",
    "compute_tensor_int8",
    "memory_pattern",
    "memory_sweep",
    "bandwidth_d2d",
)


def ref_compute_int64(gid: int, iterations: int, seed_a: int, seed_b: int) -> int:
    mask = 0xFFFFFFFFFFFFFFFF
    mul = (seed_a | 1) & mask
    add = (seed_b | 1) & mask
    r = [((gid + (k + 1) * 0x9E3779B97F4A7C15) ^ seed_a) & mask for k in range(8)]
    for _ in range(iterations * 16):
        r = [(x * mul + add) & mask for x in r]
    return ((r[0] ^ r[1]) + (r[2] ^ r[3]) + (r[4] ^ r[5]) + (r[6] ^ r[7])) & mask


def ref_compute_int32(gid: int, iterations: int, seed_a: int, seed_b: int) -> int:
    mask = 0xFFFFFFFF
    mul = (seed_a | 1) & mask
    add = (seed_b | 1) & mask
    r = [((gid + (k + 1) * 0x9E3779B9) ^ seed_a) & mask for k in range(8)]
    for _ in range(iterations * 16):
        r = [(x * mul + add) & mask for x in r]
    return ((r[0] ^ r[1]) + (r[2] ^ r[3]) + (r[4] ^ r[5]) + (r[6] ^ r[7])) & mask


def ref_compute_int16(gid: int, iterations: int, seed_a: int, seed_b: int) -> int:
    mask = 0xFFFF
    mul = (seed_a & mask) | 1
    add = (seed_b & mask) | 1
    r = [(((gid + (k + 1) * 0x9E37) ^ seed_a) & mask) for k in range(8)]
    for _ in range(iterations * 16):
        r = [(x * mul + add) & mask for x in r]
    return ((r[0] ^ r[1]) + (r[2] ^ r[3]) + (r[4] ^ r[5]) + (r[6] ^ r[7])) & mask


def ref_compute_int8_uchar4(gid: int, iterations: int, seed_a: int, seed_b: int) -> int:
    mul = [((seed_a >> (8 * lane)) & 0xFF) | 1 for lane in range(4)]
    add = [((seed_b >> (8 * lane)) & 0xFF) | 1 for lane in range(4)]
    r = [
        [
            (((gid + (k + 1) * 37 + lane * 17) ^ (seed_a >> (8 * lane))) & 0xFF)
            for lane in range(4)
        ]
        for k in range(8)
    ]
    for _ in range(iterations * 16):
        r = [[(x[lane] * mul[lane] + add[lane]) & 0xFF for lane in range(4)] for x in r]
    acc = [
        (
            (r[0][lane] ^ r[1][lane])
            + (r[2][lane] ^ r[3][lane])
            + (r[4][lane] ^ r[5][lane])
            + (r[6][lane] ^ r[7][lane])
        )
        & 0xFF
        for lane in range(4)
    ]
    return acc[0] | (acc[1] << 8) | (acc[2] << 16) | (acc[3] << 24)


def ref_compute_int8_dp4a(gid: int, iterations: int, seed_a: int, seed_b: int) -> int:
    mask = 0xFFFFFFFF
    mul_pack = (seed_a & 0x7F7F7F7F) | 0x01010101
    mul_bytes = [(mul_pack >> (8 * i)) & 0xFF for i in range(4)]
    mix = (seed_a ^ seed_b) & mask
    r = [(((gid + (k + 1) * 0x9E3779B9) ^ mix) | 1) & mask for k in range(8)]
    for _ in range(iterations * 16):
        for k in range(8):
            val = r[k]
            dot = sum(((val >> (8 * i)) & 0xFF) * mul_bytes[i] for i in range(4))
            r[k] = (val + dot) & mask
    return ((r[0] ^ r[1]) + (r[2] ^ r[3]) + (r[4] ^ r[5]) + (r[6] ^ r[7])) & mask


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def ref_compute_fp32(gid: int, iterations: int, seed_a: float, seed_b: float) -> float:
    a = _f32(0.25 * (seed_a - math.floor(seed_a)) + 0.5)
    bp = _f32(0.25 * (seed_b - math.floor(seed_b)) + 0.25)
    bn = _f32(-0.5 * bp)
    r = [_f32(0.125 + 0.0625 * ((gid + k) & 15)) for k in range(8)]
    for _ in range(iterations * 8):
        r = [_f32(x * a + bp) for x in r]
        r = [_f32(x * a + bn) for x in r]
    return _f32(_f32(r[0] + r[1]) + _f32(r[2] + r[3]) + _f32(r[4] + r[5]) + _f32(r[6] + r[7]))


def ref_compute_fp64(gid: int, iterations: int, seed_a: float, seed_b: float) -> float:
    a = 0.25 * (seed_a - math.floor(seed_a)) + 0.5
    bp = 0.25 * (seed_b - math.floor(seed_b)) + 0.25
    bn = -0.5 * bp
    r = [0.125 + 0.0625 * ((gid + k) & 15) for k in range(8)]
    for _ in range(iterations * 8):
        r = [x * a + bp for x in r]
        r = [x * a + bn for x in r]
    return (r[0] + r[1]) + (r[2] + r[3]) + (r[4] + r[5]) + (r[6] + r[7])


def ref_pattern_word(idx: int, mode: int, seed: int) -> int:
    if mode == 0:
        return 0x00000000
    if mode == 1:
        return 0xFFFFFFFF
    if mode == 2:
        return 0x55555555
    if mode == 3:
        return 0xAAAAAAAA
    if mode == 4:
        return (1 << (idx & 31)) & 0xFFFFFFFF
    if mode == 5:
        return (~(1 << (idx & 31))) & 0xFFFFFFFF
    z = ((idx ^ seed ^ (seed >> 32)) + 0x9E3779B9) & 0xFFFFFFFF
    z = ((z ^ (z >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    z = ((z ^ (z >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    return (z ^ (z >> 16)) & 0xFFFFFFFF


def ref_splitmix64(x: int) -> int:
    m = 0xFFFFFFFFFFFFFFFF
    z = (x + 0x9E3779B97F4A7C15) & m
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
    return (z ^ (z >> 31)) & m


class TestPackedKernels(unittest.TestCase):
    def test_all_90_kernels_packed_verified_and_in_sync_with_disk(self) -> None:
        self.assertEqual(verify_embedded_kernels(EMBEDDED_KERNELS), [])
        disk_packed = scan_kernels_directory()
        self.assertEqual(len(EMBEDDED_KERNELS), 90)
        self.assertEqual(EMBEDDED_KERNELS, disk_packed)

        for backend in EXPECTED_BACKENDS:
            for workload in EXPECTED_KERNEL_WORKLOADS:
                key = f"{backend}:{workload}"
                self.assertIn(key, EMBEDDED_KERNELS)
                # Both snake_case and hyphenated canonical names resolve cleanly.
                raw_snake, meta_snake = get_embedded_kernel(backend, workload)
                raw_dash, meta_dash = get_embedded_kernel(backend, workload.replace("_", "-"))
                self.assertEqual(raw_snake, raw_dash)
                self.assertEqual(meta_snake["sha256"], meta_dash["sha256"])

    def test_kernel_entry_points_headers_and_unsupported_gates(self) -> None:
        for backend in EXPECTED_BACKENDS:
            for workload in EXPECTED_KERNEL_WORKLOADS:
                raw, meta = get_embedded_kernel(backend, workload)
                text = raw.decode("utf-8")
                expected_eps = default_entry_points(workload)
                self.assertEqual(tuple(meta["entry_points"]), expected_eps)
                self.assertIn("Entry point(s):", text)
                self.assertTrue(
                    "OPS_PER_THREAD_PER_ITER:" in text or "BYTES_PER_ELEMENT:" in text,
                    f"Missing op/byte formula comment in {backend}:{workload}",
                )

                is_pure_unsupported_stub = "kernel_" not in text.split("*/", 1)[-1]
                if is_pure_unsupported_stub:
                    self.assertIn('#error "UNSUPPORTED:', text, f"{backend}:{workload}")
                else:
                    for ep in expected_eps:
                        self.assertIn(ep, text, f"Missing symbol {ep} in {backend}:{workload}")

    def test_golden_reference_math_determinism_and_bounds(self) -> None:
        self.assertNotEqual(
            ref_compute_int64(0, 2, 0x123456789ABCDEF1, 0xCAFEBABE12345679),
            ref_compute_int64(1, 2, 0x123456789ABCDEF1, 0xCAFEBABE12345679),
        )
        self.assertNotEqual(
            ref_compute_int32(0, 2, 0x12345679, 0x9E3779B9),
            ref_compute_int32(1, 2, 0x12345679, 0x9E3779B9),
        )
        self.assertNotEqual(
            ref_compute_int16(0, 2, 0x12345679, 0x9E3779B9),
            ref_compute_int16(1, 2, 0x12345679, 0x9E3779B9),
        )
        self.assertNotEqual(
            ref_compute_int8_uchar4(0, 2, 0x12345679, 0x9E3779B9),
            ref_compute_int8_uchar4(1, 2, 0x12345679, 0x9E3779B9),
        )
        self.assertNotEqual(
            ref_compute_int8_dp4a(0, 2, 0x12345679, 0x9E3779B9),
            ref_compute_int8_dp4a(1, 2, 0x12345679, 0x9E3779B9),
        )

        # Contractive FP32 and FP64 recurrences remain strictly finite and bounded.
        for iters in (1, 10, 200):
            v32 = ref_compute_fp32(3, iters, 0.75, 0.5)
            v64 = ref_compute_fp64(3, iters, 0.75, 0.5)
            self.assertTrue(math.isfinite(v32) and 0.1 < v32 < 16.0)
            self.assertTrue(math.isfinite(v64) and 0.1 < v64 < 16.0)

        # Memory pattern modes 0..6 and SplitMix64 address hash checks.
        self.assertEqual(ref_pattern_word(7, 0, 123), 0x00000000)
        self.assertEqual(ref_pattern_word(7, 1, 123), 0xFFFFFFFF)
        self.assertEqual(ref_pattern_word(7, 2, 123), 0x55555555)
        self.assertEqual(ref_pattern_word(7, 3, 123), 0xAAAAAAAA)
        self.assertEqual(ref_pattern_word(5, 4, 123), 1 << 5)
        self.assertEqual(ref_pattern_word(5, 5, 123), (~(1 << 5)) & 0xFFFFFFFF)
        self.assertNotEqual(ref_pattern_word(0, 6, 42), ref_pattern_word(1, 6, 42))
        self.assertNotEqual(ref_splitmix64(0 ^ 42), ref_splitmix64(1 ^ 42))

    def test_host_metal_live_compile_and_golden_execution(self) -> None:
        from cross_gpu.backends.metal_backend import MetalBackend

        backend = MetalBackend()
        if not backend.is_available():
            self.skipTest("Metal backend unavailable on this host")
        devices = backend.enumerate_devices()
        if not devices:
            self.skipTest("No Metal GPU detected")

        dev = devices[0]
        ctx = backend.create_context(dev)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cache = RuntimeBinaryCache(Path(tmp))
                N = 64

                # Bit-exact scalar integer verification on live Metal GPU
                int_cases = [
                    ("compute_int64", 8, ref_compute_int64, 0x123456789ABCDEF1, 0xCAFEBABE12345679),
                    ("compute_int32", 4, ref_compute_int32, 0x12345679, 0x9E3779B9),
                    ("compute_int16", 4, ref_compute_int16, 0x12345679, 0x9E3779B9),
                    ("compute_int8", 4, ref_compute_int8_uchar4, 0x12345679, 0x9E3779B9),
                ]
                for wl, elem_sz, ref_fn, sa, sb in int_cases:
                    compiled = compile_kernel_runtime("metal", wl, dev.architecture, cache=cache)
                    mod = backend.load_module(ctx, compiled)
                    out_buf = backend.alloc_device(ctx, N * elem_sz)
                    try:
                        seed_cls = ctypes.c_uint64 if elem_sz == 8 else ctypes.c_uint32
                        backend.launch_kernel(
                            ctx,
                            mod,
                            f"kernel_{wl}",
                            (2, 1, 1),
                            (32, 1, 1),
                            [out_buf, ctypes.c_uint32(N), ctypes.c_uint32(2), seed_cls(sa), seed_cls(sb)],
                        )
                        backend.synchronize(ctx)
                        raw = bytearray(N * elem_sz)
                        backend.memcpy_d2h(ctx, raw, out_buf, N * elem_sz)
                        vals = struct.unpack(f"<{N}{'Q' if elem_sz == 8 else 'I'}", raw)
                        expected = tuple(ref_fn(gid, 2, sa, sb) for gid in range(N))
                        self.assertEqual(vals, expected, f"Metal mismatch for {wl}")
                    finally:
                        backend.free_device(ctx, out_buf)
                        backend.unload_module(ctx, mod)

                # Scalar FP32 + FP16 + Tensor FP32/FP16 verification
                for wl in ("compute_fp32", "compute_fp16", "compute_tensor_fp32", "compute_tensor_fp16"):
                    compiled = compile_kernel_runtime("metal", wl, dev.architecture, cache=cache)
                    mod = backend.load_module(ctx, compiled)
                    out_buf = backend.alloc_device(ctx, N * 4)
                    try:
                        backend.launch_kernel(
                            ctx,
                            mod,
                            f"kernel_{wl}",
                            (2, 1, 1),
                            (32, 1, 1),
                            [out_buf, ctypes.c_uint32(N), ctypes.c_uint32(2), ctypes.c_float(0.75), ctypes.c_float(0.5)],
                        )
                        backend.synchronize(ctx)
                        raw = bytearray(N * 4)
                        backend.memcpy_d2h(ctx, raw, out_buf, N * 4)
                        vals = struct.unpack(f"<{N}f", raw)
                        if wl == "compute_fp32":
                            for gid, val in enumerate(vals):
                                self.assertAlmostEqual(val, ref_compute_fp32(gid, 2, 0.75, 0.5), delta=1e-5)
                        elif wl.startswith("compute_tensor_"):
                            self.assertEqual(vals, (9.5,) * N)
                        else:
                            self.assertTrue(all(math.isfinite(v) and 0.5 < v < 16.0 for v in vals))
                    finally:
                        backend.free_device(ctx, out_buf)
                        backend.unload_module(ctx, mod)

                # Memory pattern + deliberate corruption detection
                mp_compiled = compile_kernel_runtime("metal", "memory_pattern", dev.architecture, cache=cache)
                mp_mod = backend.load_module(ctx, mp_compiled)
                buf = backend.alloc_device(ctx, 128 * 4)
                err_buf = backend.alloc_device(ctx, 4)
                try:
                    backend.memcpy_h2d(ctx, err_buf, struct.pack("<I", 0), 4)
                    backend.launch_kernel(ctx, mp_mod, "kernel_memory_pattern_write", (4, 1, 1), (32, 1, 1), [buf, ctypes.c_uint64(128), ctypes.c_uint32(6), ctypes.c_uint64(42)])
                    backend.launch_kernel(ctx, mp_mod, "kernel_memory_pattern_verify", (4, 1, 1), (32, 1, 1), [buf, ctypes.c_uint64(128), ctypes.c_uint32(6), ctypes.c_uint64(42), err_buf])
                    backend.synchronize(ctx)
                    out_err = bytearray(4)
                    backend.memcpy_d2h(ctx, out_err, err_buf, 4)
                    self.assertEqual(struct.unpack("<I", out_err)[0], 0)
                    # Corrupt word 0 and confirm verify catches exactly 1 error.
                    host_copy = bytearray(128 * 4)
                    backend.memcpy_d2h(ctx, host_copy, buf, 128 * 4)
                    self.assertEqual(struct.unpack("<I", host_copy[:4])[0], ref_pattern_word(0, 6, 42))
                    host_copy[0] ^= 0xFF
                    backend.memcpy_h2d(ctx, buf, bytes(host_copy), 128 * 4)
                    backend.launch_kernel(ctx, mp_mod, "kernel_memory_pattern_verify", (4, 1, 1), (32, 1, 1), [buf, ctypes.c_uint64(128), ctypes.c_uint32(6), ctypes.c_uint64(42), err_buf])
                    backend.synchronize(ctx)
                    backend.memcpy_d2h(ctx, out_err, err_buf, 4)
                    self.assertEqual(struct.unpack("<I", out_err)[0], 1)
                finally:
                    backend.free_device(ctx, buf)
                    backend.free_device(ctx, err_buf)
                    backend.unload_module(ctx, mp_mod)

                # Memory sweep + golden SplitMix64 check
                ms_compiled = compile_kernel_runtime("metal", "memory_sweep", dev.architecture, cache=cache)
                ms_mod = backend.load_module(ctx, ms_compiled)
                sbuf = backend.alloc_device(ctx, 64 * 8)
                serr = backend.alloc_device(ctx, 4)
                try:
                    backend.memcpy_h2d(ctx, serr, struct.pack("<I", 0), 4)
                    backend.launch_kernel(ctx, ms_mod, "kernel_memory_sweep_write", (2, 1, 1), (32, 1, 1), [sbuf, ctypes.c_uint64(64), ctypes.c_uint64(99)])
                    backend.launch_kernel(ctx, ms_mod, "kernel_memory_sweep_verify", (2, 1, 1), (32, 1, 1), [sbuf, ctypes.c_uint64(64), ctypes.c_uint64(99), serr])
                    backend.synchronize(ctx)
                    raw_q = bytearray(64 * 8)
                    raw_e = bytearray(4)
                    backend.memcpy_d2h(ctx, raw_q, sbuf, 64 * 8)
                    backend.memcpy_d2h(ctx, raw_e, serr, 4)
                    self.assertEqual(struct.unpack("<I", raw_e)[0], 0)
                    self.assertEqual(struct.unpack("<Q", raw_q[:8])[0], ref_splitmix64(0 ^ 99))
                finally:
                    backend.free_device(ctx, sbuf)
                    backend.free_device(ctx, serr)
                    backend.unload_module(ctx, ms_mod)

                # Optional Metal 3.1+ BF16 scalar & tensor verification (or clean UNSUPPORTED on older macOS)
                for wl in ("compute_bf16", "compute_tensor_bf16"):
                    compiled = compile_kernel_runtime("metal", wl, dev.architecture, cache=cache)
                    try:
                        mod = backend.load_module(ctx, compiled)
                    except KernelCompilationError as err:
                        self.assertIn("UNSUPPORTED", err.compiler_log)
                        continue
                    out_buf = backend.alloc_device(ctx, N * 4)
                    try:
                        backend.launch_kernel(
                            ctx,
                            mod,
                            f"kernel_{wl}",
                            (2, 1, 1),
                            (32, 1, 1),
                            [out_buf, ctypes.c_uint32(N), ctypes.c_uint32(2), ctypes.c_float(0.75), ctypes.c_float(0.5)],
                        )
                        backend.synchronize(ctx)
                        raw = bytearray(N * 4)
                        backend.memcpy_d2h(ctx, raw, out_buf, N * 4)
                        vals = struct.unpack(f"<{N}f", raw)
                        if wl == "compute_tensor_bf16":
                            self.assertEqual(vals, (9.5,) * N)
                        else:
                            self.assertTrue(all(math.isfinite(v) and 0.5 < v < 16.0 for v in vals))
                    finally:
                        backend.free_device(ctx, out_buf)
                        backend.unload_module(ctx, mod)

                # Device-to-device streaming copy verification
                bw_compiled = compile_kernel_runtime("metal", "bandwidth_d2d", dev.architecture, cache=cache)
                bw_mod = backend.load_module(ctx, bw_compiled)
                num_vec4 = 64
                bw_bytes = num_vec4 * 16
                src_buf = backend.alloc_device(ctx, bw_bytes)
                dst_buf = backend.alloc_device(ctx, bw_bytes)
                try:
                    payload = bytes((i * 37 + 11) & 0xFF for i in range(bw_bytes))
                    backend.memcpy_h2d(ctx, src_buf, payload, bw_bytes)
                    backend.launch_kernel(
                        ctx, bw_mod, "kernel_bandwidth_d2d", (2, 1, 1), (32, 1, 1),
                        [dst_buf, src_buf, ctypes.c_uint64(num_vec4)],
                    )
                    backend.synchronize(ctx)
                    out_bw = bytearray(bw_bytes)
                    backend.memcpy_d2h(ctx, out_bw, dst_buf, bw_bytes)
                    self.assertEqual(bytes(out_bw), payload)
                finally:
                    backend.free_device(ctx, src_buf)
                    backend.free_device(ctx, dst_buf)
                    backend.unload_module(ctx, bw_mod)

                # Explicitly unsupported Metal kernels must fail with UNSUPPORTED
                for bad_wl in ("compute_fp64", "compute_fp8", "compute_fp4", "compute_int4", "compute_tensor_int8"):
                    raw_src, _ = get_embedded_kernel("metal", bad_wl)
                    compiled = compile_kernel_runtime("metal", bad_wl, dev.architecture, source_override=raw_src, cache=cache)
                    with self.assertRaises(KernelCompilationError) as err_ctx:
                        backend.load_module(ctx, compiled)
                    self.assertIn("UNSUPPORTED", err_ctx.exception.compiler_log)
        finally:
            backend.destroy_context(ctx)

    def test_host_opencl_live_compile_and_golden_execution(self) -> None:
        from cross_gpu.backends.opencl_backend import OpenCLBackend

        backend = OpenCLBackend()
        if not backend.is_available():
            self.skipTest("OpenCL backend unavailable on this host")
        devices = backend.enumerate_devices()
        if not devices:
            self.skipTest("No OpenCL GPU detected")

        dev = devices[0]
        ctx = backend.create_context(dev)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cache = RuntimeBinaryCache(Path(tmp))
                N = 64
                for wl, elem_sz, ref_fn, sa, sb in (
                    ("compute_int64", 8, ref_compute_int64, 0x123456789ABCDEF1, 0xCAFEBABE12345679),
                    ("compute_int32", 4, ref_compute_int32, 0x12345679, 0x9E3779B9),
                    ("compute_int16", 4, ref_compute_int16, 0x12345679, 0x9E3779B9),
                    ("compute_int8", 4, ref_compute_int8_uchar4, 0x12345679, 0x9E3779B9),
                ):
                    compiled = compile_kernel_runtime("opencl", wl, dev.architecture, dev.capabilities, cache=cache)
                    mod = backend.load_module(ctx, compiled)
                    out_buf = backend.alloc_device(ctx, N * elem_sz)
                    try:
                        seed_cls = ctypes.c_uint64 if elem_sz == 8 else ctypes.c_uint32
                        backend.launch_kernel(
                            ctx,
                            mod,
                            f"kernel_{wl}",
                            (2, 1, 1),
                            (32, 1, 1),
                            [out_buf, ctypes.c_uint32(N), ctypes.c_uint32(2), seed_cls(sa), seed_cls(sb)],
                        )
                        backend.synchronize(ctx)
                        raw = bytearray(N * elem_sz)
                        backend.memcpy_d2h(ctx, raw, out_buf, N * elem_sz)
                        vals = struct.unpack(f"<{N}{'Q' if elem_sz == 8 else 'I'}", raw)
                        expected = tuple(ref_fn(gid, 2, sa, sb) for gid in range(N))
                        self.assertEqual(vals, expected, f"OpenCL mismatch for {wl}")
                    finally:
                        backend.free_device(ctx, out_buf)
                        backend.unload_module(ctx, mod)

                fp32_compiled = compile_kernel_runtime("opencl", "compute_fp32", dev.architecture, dev.capabilities, cache=cache)
                fp32_mod = backend.load_module(ctx, fp32_compiled)
                fp32_buf = backend.alloc_device(ctx, N * 4)
                try:
                    backend.launch_kernel(
                        ctx,
                        fp32_mod,
                        "kernel_compute_fp32",
                        (2, 1, 1),
                        (32, 1, 1),
                        [fp32_buf, ctypes.c_uint32(N), ctypes.c_uint32(2), ctypes.c_float(0.75), ctypes.c_float(0.5)],
                    )
                    backend.synchronize(ctx)
                    raw_f = bytearray(N * 4)
                    backend.memcpy_d2h(ctx, raw_f, fp32_buf, N * 4)
                    for gid, val in enumerate(struct.unpack(f"<{N}f", raw_f)):
                        self.assertAlmostEqual(val, ref_compute_fp32(gid, 2, 0.75, 0.5), delta=1e-5)
                finally:
                    backend.free_device(ctx, fp32_buf)
                    backend.unload_module(ctx, fp32_mod)

                # Memory pattern + sweep + bandwidth_d2d on live OpenCL GPU
                mp_mod = backend.load_module(
                    ctx, compile_kernel_runtime("opencl", "memory_pattern", dev.architecture, dev.capabilities, cache=cache)
                )
                pbuf = backend.alloc_device(ctx, 128 * 4)
                perr = backend.alloc_device(ctx, 4)
                try:
                    backend.memcpy_h2d(ctx, perr, struct.pack("<I", 0), 4)
                    backend.launch_kernel(ctx, mp_mod, "kernel_memory_pattern_write", (4, 1, 1), (32, 1, 1), [pbuf, ctypes.c_uint64(128), ctypes.c_uint32(6), ctypes.c_uint64(42)])
                    backend.launch_kernel(ctx, mp_mod, "kernel_memory_pattern_verify", (4, 1, 1), (32, 1, 1), [pbuf, ctypes.c_uint64(128), ctypes.c_uint32(6), ctypes.c_uint64(42), perr])
                    backend.synchronize(ctx)
                    out_p = bytearray(128 * 4)
                    out_e = bytearray(4)
                    backend.memcpy_d2h(ctx, out_p, pbuf, 128 * 4)
                    backend.memcpy_d2h(ctx, out_e, perr, 4)
                    self.assertEqual(struct.unpack("<I", out_e)[0], 0)
                    self.assertEqual(struct.unpack("<I", out_p[:4])[0], ref_pattern_word(0, 6, 42))
                finally:
                    backend.free_device(ctx, pbuf)
                    backend.free_device(ctx, perr)
                    backend.unload_module(ctx, mp_mod)

                ms_mod = backend.load_module(
                    ctx, compile_kernel_runtime("opencl", "memory_sweep", dev.architecture, dev.capabilities, cache=cache)
                )
                sbuf = backend.alloc_device(ctx, 64 * 8)
                serr = backend.alloc_device(ctx, 4)
                try:
                    backend.memcpy_h2d(ctx, serr, struct.pack("<I", 0), 4)
                    backend.launch_kernel(ctx, ms_mod, "kernel_memory_sweep_write", (2, 1, 1), (32, 1, 1), [sbuf, ctypes.c_uint64(64), ctypes.c_uint64(99)])
                    backend.launch_kernel(ctx, ms_mod, "kernel_memory_sweep_verify", (2, 1, 1), (32, 1, 1), [sbuf, ctypes.c_uint64(64), ctypes.c_uint64(99), serr])
                    backend.synchronize(ctx)
                    raw_q = bytearray(64 * 8)
                    raw_e = bytearray(4)
                    backend.memcpy_d2h(ctx, raw_q, sbuf, 64 * 8)
                    backend.memcpy_d2h(ctx, raw_e, serr, 4)
                    self.assertEqual(struct.unpack("<I", raw_e)[0], 0)
                    self.assertEqual(struct.unpack("<Q", raw_q[:8])[0], ref_splitmix64(0 ^ 99))
                finally:
                    backend.free_device(ctx, sbuf)
                    backend.free_device(ctx, serr)
                    backend.unload_module(ctx, ms_mod)

                bw_mod = backend.load_module(
                    ctx, compile_kernel_runtime("opencl", "bandwidth_d2d", dev.architecture, dev.capabilities, cache=cache)
                )
                bw_bytes = 64 * 16
                src_buf = backend.alloc_device(ctx, bw_bytes)
                dst_buf = backend.alloc_device(ctx, bw_bytes)
                try:
                    payload = bytes((i * 53 + 7) & 0xFF for i in range(bw_bytes))
                    backend.memcpy_h2d(ctx, src_buf, payload, bw_bytes)
                    backend.launch_kernel(ctx, bw_mod, "kernel_bandwidth_d2d", (2, 1, 1), (32, 1, 1), [dst_buf, src_buf, ctypes.c_uint64(64)])
                    backend.synchronize(ctx)
                    out_bw = bytearray(bw_bytes)
                    backend.memcpy_d2h(ctx, out_bw, dst_buf, bw_bytes)
                    self.assertEqual(bytes(out_bw), payload)
                finally:
                    backend.free_device(ctx, src_buf)
                    backend.free_device(ctx, dst_buf)
                    backend.unload_module(ctx, bw_mod)
        finally:
            backend.destroy_context(ctx)

    def test_host_nvrtc_or_hiprtc_compile_when_present(self) -> None:
        from cross_gpu.backends.cuda_backend import CudaBackend
        from cross_gpu.backends.hip_backend import HipBackend

        compiled_any = False
        with tempfile.TemporaryDirectory() as tmp:
            cache = RuntimeBinaryCache(Path(tmp))
            cuda = CudaBackend()
            if cuda.is_available() and cuda.enumerate_devices():
                dev = cuda.enumerate_devices()[0]
                res = compile_kernel_runtime("cuda", "compute_fp32", dev.architecture, dev.capabilities, cache=cache)
                self.assertIn(res.format, ("ptx", "cubin"))
                self.assertGreater(len(res.binary_bytes), 0)
                compiled_any = True

            hip = HipBackend()
            if hip.is_available() and hip.enumerate_devices():
                dev = hip.enumerate_devices()[0]
                res = compile_kernel_runtime("hip", "compute_fp32", dev.architecture, dev.capabilities, cache=cache)
                self.assertEqual(res.format, "hsaco")
                self.assertGreater(len(res.binary_bytes), 0)
                compiled_any = True

        if not compiled_any:
            self.skipTest("Neither CUDA/NVRTC nor HIP/hipRTC GPU toolchain available on this host")


if __name__ == "__main__":
    unittest.main()
