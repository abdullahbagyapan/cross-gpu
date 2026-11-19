# SPDX-License-Identifier: MIT
"""Unit tests for importlib.resources kernel loading, verification, caching, and runtime compilation."""

from __future__ import annotations

import ctypes
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from cross_gpu.embed_binary import (
    KERNEL_SPECS,
    KernelSpec,
    RuntimeBinaryCache,
    _main as embed_binary_main,
    clear_kernel_source_cache,
    compile_kernel_runtime,
    compute_disk_cache_name,
    compute_memory_cache_key,
    default_cache_dir,
    default_entry_points,
    is_disk_cache_disabled,
    load_kernel_source,
    verify_kernel_resources,
)
from cross_gpu.exit_codes import (
    ExitCode,
    KernelCompilationError,
    KernelNotFoundError,
    UnsupportedKernelError,
)
from cross_gpu.models import DeviceCapabilities


class TestEmbedBinary(unittest.TestCase):
    def test_load_kernel_source_memoization_and_clear_cache(self) -> None:
        clear_kernel_source_cache()
        src1 = load_kernel_source("cuda", "compute-fp32")
        src2 = load_kernel_source("cuda", "compute-fp32")
        src3 = load_kernel_source("cuda", "compute_fp32")
        self.assertIs(src1, src2)
        self.assertIs(src1, src3)

        uncached = load_kernel_source("cuda", "compute-fp32", use_cache=False)
        self.assertIsNot(src1, uncached)
        self.assertEqual(src1, uncached)

        clear_kernel_source_cache()
        reloaded = load_kernel_source("cuda", "compute-fp32")
        self.assertIsNot(src1, reloaded)
        self.assertEqual(src1, reloaded)

    def test_missing_and_corrupt_kernel_raises_kernel_not_found(self) -> None:
        with self.assertRaises(KernelNotFoundError) as ctx_load:
            load_kernel_source("cuda", "compute-fp32", specs={})
        self.assertEqual(ctx_load.exception.exit_code, ExitCode.INIT_ERROR)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = RuntimeBinaryCache(root / "cache")
            with self.assertRaises(KernelNotFoundError) as ctx_compile:
                compile_kernel_runtime("cuda", "compute-fp32", "sm_89", cache=cache, specs={})
            self.assertEqual(ctx_compile.exception.exit_code, ExitCode.INIT_ERROR)

            # Registered spec whose underlying resource file is missing.
            with self.assertRaises(KernelNotFoundError) as ctx_missing_file:
                load_kernel_source("cuda", "compute-fp32", resource_root=root)
            self.assertEqual(ctx_missing_file.exception.exit_code, ExitCode.INIT_ERROR)

            with self.assertRaises(KernelNotFoundError) as ctx_compile_missing:
                compile_kernel_runtime(
                    "cuda", "compute-fp32", "sm_89", cache=cache, resource_root=root
                )
            self.assertEqual(ctx_compile_missing.exception.exit_code, ExitCode.INIT_ERROR)

            # Corrupt non-UTF-8 bytes in resource file.
            bad_file = root / "cuda" / "compute_fp32.cu"
            bad_file.parent.mkdir(parents=True, exist_ok=True)
            bad_file.write_bytes(b"\xff\xfe\xfd")
            with self.assertRaises(KernelNotFoundError) as ctx_bad_utf8:
                load_kernel_source("cuda", "compute-fp32", resource_root=root)
            self.assertEqual(ctx_bad_utf8.exception.exit_code, ExitCode.INIT_ERROR)

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

    def test_memory_cache_key_plain_tuple_differentiates_all_dimensions(self) -> None:
        base = compute_memory_cache_key(
            "cuda", "sm_89", "12.4", "compute-fp32", ("-O3",), b"kernel_a"
        )
        self.assertIsInstance(base, tuple)
        self.assertEqual(
            base,
            ("cuda", "sm_89", "12.4", "compute_fp32", ("-O3",), b"kernel_a"),
        )
        # Canonical normalization of workload slug and flag whitespace.
        canonical_same = compute_memory_cache_key(
            "cuda", "sm_89", "12.4", "compute_fp32", (" -O3 ", ""), b"kernel_a"
        )
        self.assertEqual(base, canonical_same)

        variants = [
            base,
            compute_memory_cache_key("hip", "sm_89", "12.4", "compute-fp32", ("-O3",), b"kernel_a"),
            compute_memory_cache_key("cuda", "sm_80", "12.4", "compute-fp32", ("-O3",), b"kernel_a"),
            compute_memory_cache_key("cuda", "sm_89", "12.5", "compute-fp32", ("-O3",), b"kernel_a"),
            compute_memory_cache_key("cuda", "sm_89", "12.4", "compute-fp64", ("-O3",), b"kernel_a"),
            compute_memory_cache_key("cuda", "sm_89", "12.4", "compute-fp32", ("-O2",), b"kernel_a"),
            compute_memory_cache_key("cuda", "sm_89", "12.4", "compute-fp32", ("-O3",), b"kernel_b"),
        ]
        self.assertEqual(len(set(variants)), len(variants))

    def test_disk_cache_name_differentiates_dimensions_and_fits_name_max(self) -> None:
        base = compute_disk_cache_name(
            "cuda", "sm_89", "12.4", "compute-fp32", 1024, ("-O3",), package_version="0.1.0"
        )
        variants = [
            base,
            compute_disk_cache_name(
                "cuda", "sm_89", "12.4", "compute-fp32", 1024, ("-O3",), package_version="0.2.0"
            ),
            compute_disk_cache_name(
                "hip", "sm_89", "12.4", "compute-fp32", 1024, ("-O3",), package_version="0.1.0"
            ),
            compute_disk_cache_name(
                "cuda", "sm_80", "12.4", "compute-fp32", 1024, ("-O3",), package_version="0.1.0"
            ),
            compute_disk_cache_name(
                "cuda", "sm_89", "12.5", "compute-fp32", 1024, ("-O3",), package_version="0.1.0"
            ),
            compute_disk_cache_name(
                "cuda", "sm_89", "12.4", "compute-fp64", 1024, ("-O3",), package_version="0.1.0"
            ),
            compute_disk_cache_name(
                "cuda", "sm_89", "12.4", "compute-fp32", 2048, ("-O3",), package_version="0.1.0"
            ),
            compute_disk_cache_name(
                "cuda", "sm_89", "12.4", "compute-fp32", 1024, ("-O2",), package_version="0.1.0"
            ),
        ]
        self.assertEqual(len(set(variants)), len(variants))

        # Full capability flag matrix + complex arch/driver strings remain safe and < 255 bytes.
        caps_all = DeviceCapabilities(
            int64=True,
            int32=True,
            int16=True,
            int8=True,
            int4=True,
            fp64=True,
            fp32=True,
            fp16=True,
            fp8=True,
            fp4=True,
            bf16=True,
            tensor_fp32=True,
            tensor_fp16=True,
            tensor_bf16=True,
            tensor_int8=True,
        ).to_dict()
        flags_all = tuple(f"-DCROSS_GPU_HAS_{k.upper()}=1" for k in sorted(caps_all))
        flags_one_off = tuple(
            f"-DCROSS_GPU_HAS_{k.upper()}={0 if k == 'fp8' else 1}" for k in sorted(caps_all)
        )
        name_all = compute_disk_cache_name(
            "opencl",
            "Apple M2 / Pro (14.4)",
            "OpenCL 1.2 (May 19 2026)",
            "compute-tensor-bf16",
            4096,
            flags_all + ("-cl-std=CL3.0", "-DFOO=bar"),
        )
        name_one_off = compute_disk_cache_name(
            "opencl",
            "Apple M2 / Pro (14.4)",
            "OpenCL 1.2 (May 19 2026)",
            "compute-tensor-bf16",
            4096,
            flags_one_off + ("-cl-std=CL3.0", "-DFOO=bar"),
        )
        self.assertNotEqual(name_all, name_one_off)
        self.assertLess(len(name_all) + len(".json"), 255)
        self.assertRegex(name_all, r"^[A-Za-z0-9._-]+$")

    def test_disk_cache_hit_invalidation_and_source_override_behavior(self) -> None:
        caps = DeviceCapabilities(fp32=True)
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "disk_cache"
            cache1 = RuntimeBinaryCache(cache_dir)
            compiled = compile_kernel_runtime(
                "opencl",
                "compute-fp32",
                "gfx1100",
                capabilities=caps,
                driver_version="24.10",
                cache=cache1,
            )
            bin_files = sorted(cache_dir.glob("*.bin"))
            json_files = sorted(cache_dir.glob("*.json"))
            self.assertEqual(len(bin_files), 1)
            self.assertEqual(len(json_files), 1)

            # Fresh RuntimeBinaryCache instance reloads directly from disk without recompiling.
            cache2 = RuntimeBinaryCache(cache_dir)
            with mock.patch(
                "cross_gpu.embed_binary._compile_opencl",
                side_effect=AssertionError("should have hit disk cache"),
            ):
                reloaded = compile_kernel_runtime(
                    "opencl",
                    "compute-fp32",
                    "gfx1100",
                    capabilities=caps,
                    driver_version="24.10",
                    cache=cache2,
                )
            self.assertEqual(reloaded.binary_bytes, compiled.binary_bytes)
            self.assertEqual(reloaded.compile_flags, compiled.compile_flags)
            self.assertEqual(reloaded.entry_point, compiled.entry_point)

            # Truncated .bin artifact is rejected and cleanly rebuilt.
            bin_files[0].write_bytes(compiled.binary_bytes[:-1])
            cache3 = RuntimeBinaryCache(cache_dir)
            rebuilt = compile_kernel_runtime(
                "opencl",
                "compute-fp32",
                "gfx1100",
                capabilities=caps,
                driver_version="24.10",
                cache=cache3,
            )
            self.assertEqual(rebuilt.binary_bytes, compiled.binary_bytes)

            # Stale package_version metadata in sidecar .json invalidates the disk entry.
            meta = json.loads(json_files[0].read_text(encoding="utf-8"))
            meta["package_version"] = "0.0.0-stale"
            json_files[0].write_text(json.dumps(meta), encoding="utf-8")
            cache4 = RuntimeBinaryCache(cache_dir)
            from cross_gpu.embed_binary import _compile_opencl as real_compile_opencl

            with mock.patch(
                "cross_gpu.embed_binary._compile_opencl", wraps=real_compile_opencl
            ) as spy:
                recompiled = compile_kernel_runtime(
                    "opencl",
                    "compute-fp32",
                    "gfx1100",
                    capabilities=caps,
                    driver_version="24.10",
                    cache=cache4,
                )
                self.assertEqual(spy.call_count, 1)
                self.assertEqual(recompiled.binary_bytes, compiled.binary_bytes)

        # source_override skips disk cache completely while differentiating in-memory entries.
        with tempfile.TemporaryDirectory() as tmp:
            override_dir = Path(tmp) / "override_cache"
            cache = RuntimeBinaryCache(override_dir)
            c1 = compile_kernel_runtime(
                "opencl",
                "compute-fp32",
                "gfx1100",
                capabilities=caps,
                source_override="__kernel void kernel_compute_fp32(void) { /* v1 */ }",
                cache=cache,
            )
            c1_again = compile_kernel_runtime(
                "opencl",
                "compute-fp32",
                "gfx1100",
                capabilities=caps,
                source_override="__kernel void kernel_compute_fp32(void) { /* v1 */ }",
                cache=cache,
            )
            c2 = compile_kernel_runtime(
                "opencl",
                "compute-fp32",
                "gfx1100",
                capabilities=caps,
                source_override="__kernel void kernel_compute_fp32(void) { /* v2 */ }",
                cache=cache,
            )
            self.assertIs(c1, c1_again)
            self.assertNotEqual(c1.binary_bytes, c2.binary_bytes)
            self.assertFalse(override_dir.exists())

    def test_readonly_cache_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = RuntimeBinaryCache(Path(tmp))
            with mock.patch("pathlib.Path.mkdir", side_effect=PermissionError("read-only")):
                compiled = compile_kernel_runtime(
                    "opencl",
                    "compute-fp32",
                    "gfx1100",
                    capabilities=DeviceCapabilities(fp32=True),
                    cache=cache,
                )
            self.assertEqual(compiled.format, "opencl_source")
            # Second lookup hits in-memory cache cleanly.
            cached = compile_kernel_runtime(
                "opencl",
                "compute-fp32",
                "gfx1100",
                capabilities=DeviceCapabilities(fp32=True),
                cache=cache,
            )
            self.assertIs(compiled, cached)

        # Also degrades cleanly when directory exists but file writes fail with PermissionError.
        with tempfile.TemporaryDirectory() as tmp:
            cache = RuntimeBinaryCache(Path(tmp))
            with mock.patch("pathlib.Path.write_bytes", side_effect=PermissionError("ro fs")):
                compiled = compile_kernel_runtime(
                    "opencl",
                    "compute-fp32",
                    "gfx1100",
                    capabilities=DeviceCapabilities(fp32=True),
                    cache=cache,
                )
            cached = compile_kernel_runtime(
                "opencl",
                "compute-fp32",
                "gfx1100",
                capabilities=DeviceCapabilities(fp32=True),
                cache=cache,
            )
            self.assertIs(compiled, cached)

    def test_verify_kernel_resources_catches_missing_empty_symbol_and_orphans(self) -> None:
        spec = KernelSpec(
            backend="cuda",
            workload="compute_fp32",
            filename="compute_fp32.cu",
            format="cuda_source",
            entry_point="kernel_compute_fp32",
            entry_points=("kernel_compute_fp32",),
        )
        specs = {("cuda", "compute_fp32"): spec}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cuda_dir = root / "cuda"
            cuda_dir.mkdir()
            (root / "README.md").write_text("# ignored readme\n", encoding="utf-8")
            (cuda_dir / ".DS_Store").write_bytes(b"ignored")

            # Missing kernel file
            errs_missing = verify_kernel_resources(specs=specs, resource_root=root)
            self.assertTrue(any("Missing kernel resource" in e for e in errs_missing), errs_missing)

            # Empty file
            kpath = cuda_dir / "compute_fp32.cu"
            kpath.write_text("   \n", encoding="utf-8")
            errs_empty = verify_kernel_resources(specs=specs, resource_root=root)
            self.assertTrue(any("empty" in e for e in errs_empty), errs_empty)

            # Missing declared entry point symbol
            kpath.write_text("extern \"C\" __global__ void other_name(void) {}\n", encoding="utf-8")
            errs_ep = verify_kernel_resources(specs=specs, resource_root=root)
            self.assertTrue(any("missing declared entry point" in e for e in errs_ep), errs_ep)

            # Valid kernel passes cleanly
            kpath.write_text(
                "extern \"C\" __global__ void kernel_compute_fp32(void) {}\n", encoding="utf-8"
            )
            self.assertEqual(verify_kernel_resources(specs=specs, resource_root=root), [])

            # Orphan file under backend dir, orphan file in kernels root, and unknown backend dir
            (cuda_dir / "orphan_extra.cu").write_text("// orphan\n", encoding="utf-8")
            (root / "stray.txt").write_text("stray\n", encoding="utf-8")
            (root / "vulkan").mkdir()
            errs_orphans = verify_kernel_resources(specs=specs, resource_root=root)
            self.assertTrue(
                any("Orphan kernel file" in e and "cuda/orphan_extra.cu" in e for e in errs_orphans),
                errs_orphans,
            )
            self.assertTrue(
                any("Orphan file in kernels root: stray.txt" in e for e in errs_orphans),
                errs_orphans,
            )
            self.assertTrue(
                any("Unknown backend kernel directory: vulkan" in e for e in errs_orphans),
                errs_orphans,
            )

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
        self.assertIn("Verified 90 packaged kernel source(s): OK", res_verify.stdout)

        res_list = subprocess.run(
            [sys.executable, "-m", "cross_gpu.embed_binary", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res_list.returncode, 0, res_list.stderr)
        lines = [line for line in res_list.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 90)
        for line in lines:
            self.assertIn("file=", line)
            self.assertIn("entry_points=", line)
            self.assertIn("size=", line)

        # Obsolete pack action must no longer be accepted.
        res_pack = subprocess.run(
            [sys.executable, "-m", "cross_gpu.embed_binary", "pack"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(res_pack.returncode, 0)

        # Verification failure exits with ExitCode.INIT_ERROR (2).
        stderr_buf = io.StringIO()
        with mock.patch(
            "cross_gpu.embed_binary.verify_kernel_resources",
            return_value=["cuda:compute_fp32: missing"],
        ):
            with mock.patch("sys.stderr", stderr_buf):
                rc = embed_binary_main(["verify"])
        self.assertEqual(rc, int(ExitCode.INIT_ERROR))
        self.assertIn("ERROR: cuda:compute_fp32: missing", stderr_buf.getvalue())

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

        # macOS: ~/Library/Caches/cross-gpu by default, or $XDG_CACHE_HOME/cross-gpu if explicitly set
        self.assertEqual(
            default_cache_dir(platform_name="darwin", env={}, home=fake_home),
            fake_home / "Library" / "Caches" / "cross-gpu",
        )
        self.assertEqual(
            default_cache_dir(
                platform_name="darwin",
                env={"XDG_CACHE_HOME": "/tmp/mac-xdg-cache"},
                home=fake_home,
            ),
            Path("/tmp/mac-xdg-cache/cross-gpu"),
        )
        self.assertEqual(
            default_cache_dir(
                platform_name="darwin",
                env={"XDG_CACHE_HOME": "relative/cache"},
                home=fake_home,
            ),
            fake_home / "Library" / "Caches" / "cross-gpu",
        )

        # CROSS_GPU_NO_CACHE=1|true|yes disables disk cache on all platforms
        for truthy in ("1", "true", "TRUE", "Yes", " yes "):
            self.assertTrue(is_disk_cache_disabled({"CROSS_GPU_NO_CACHE": truthy}))
            self.assertIsNone(
                default_cache_dir(
                    platform_name="linux",
                    env={
                        "CROSS_GPU_NO_CACHE": truthy,
                        "XDG_CACHE_HOME": "/var/cache/custom",
                        "CROSS_GPU_CACHE_DIR": "/tmp/cg-override",
                    },
                    home=fake_home,
                )
            )
        for falsey in ("", "0", "false", "no"):
            self.assertFalse(is_disk_cache_disabled({"CROSS_GPU_NO_CACHE": falsey}))

        # Explicit CROSS_GPU_CACHE_DIR override takes precedence on any platform
        self.assertEqual(
            default_cache_dir(
                platform_name="darwin",
                env={"CROSS_GPU_CACHE_DIR": "/tmp/cg-override"},
                home=fake_home,
            ),
            Path("/tmp/cg-override"),
        )

        # CROSS_GPU_NO_CACHE=1 completely bypasses disk reads & writes while keeping memory cache
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "nocache_dir"
            cache = RuntimeBinaryCache(cache_dir)
            with mock.patch.dict("os.environ", {"CROSS_GPU_NO_CACHE": "1"}, clear=False):
                self.assertIsNone(cache.cache_dir)
                compiled = compile_kernel_runtime(
                    "opencl",
                    "compute-fp32",
                    "gfx1100",
                    capabilities=DeviceCapabilities(fp32=True),
                    cache=cache,
                )
                compiled_mem = compile_kernel_runtime(
                    "opencl",
                    "compute-fp32",
                    "gfx1100",
                    capabilities=DeviceCapabilities(fp32=True),
                    cache=cache,
                )
                self.assertIs(compiled, compiled_mem)
                self.assertFalse(cache_dir.exists())

            # Populate disk cache when CROSS_GPU_NO_CACHE is unset, then confirm CROSS_GPU_NO_CACHE=true skips reading it
            with mock.patch.dict("os.environ", {"CROSS_GPU_NO_CACHE": "0"}, clear=False):
                fresh_populate = RuntimeBinaryCache(cache_dir)
                compile_kernel_runtime(
                    "opencl",
                    "compute-fp32",
                    "gfx1100",
                    capabilities=DeviceCapabilities(fp32=True),
                    cache=fresh_populate,
                )
                self.assertEqual(len(list(cache_dir.glob("*.bin"))), 1)

            from cross_gpu.embed_binary import _compile_opencl as real_compile_opencl

            with mock.patch.dict("os.environ", {"CROSS_GPU_NO_CACHE": "TRUE"}, clear=False):
                bypass_cache = RuntimeBinaryCache(cache_dir)
                with mock.patch(
                    "cross_gpu.embed_binary._compile_opencl", wraps=real_compile_opencl
                ) as spy:
                    compile_kernel_runtime(
                        "opencl",
                        "compute-fp32",
                        "gfx1100",
                        capabilities=DeviceCapabilities(fp32=True),
                        cache=bypass_cache,
                    )
                    self.assertEqual(spy.call_count, 1)

        # Unset/unavailable HOME degrades safely to None (in-memory-only cache)
        with mock.patch("pathlib.Path.home", side_effect=RuntimeError("no home")):
            self.assertIsNone(default_cache_dir(platform_name="linux", env={}))
            self.assertIsNone(default_cache_dir(platform_name="darwin", env={}))
            with mock.patch.dict("os.environ", {}, clear=True):
                mem_only = RuntimeBinaryCache()
                self.assertIsNone(mem_only.cache_dir)
                src = "__kernel void kernel_compute_fp32(void) {}"
                compiled = compile_kernel_runtime(
                    "opencl",
                    "compute-fp32",
                    "arch",
                    capabilities=DeviceCapabilities(fp32=True),
                    source_override=src,
                    cache=mem_only,
                )
                compiled_again = compile_kernel_runtime(
                    "opencl",
                    "compute-fp32",
                    "arch",
                    capabilities=DeviceCapabilities(fp32=True),
                    source_override=src,
                    cache=mem_only,
                )
                self.assertIs(compiled, compiled_again)


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


class TestPackagedKernels(unittest.TestCase):
    def test_all_90_kernels_load_via_importlib_resources(self) -> None:
        self.assertEqual(verify_kernel_resources(), [])
        self.assertEqual(len(KERNEL_SPECS), 90)

        for backend in EXPECTED_BACKENDS:
            for workload in EXPECTED_KERNEL_WORKLOADS:
                self.assertIn((backend, workload), KERNEL_SPECS)
                # Both snake_case and hyphenated canonical names resolve cleanly.
                src_snake = load_kernel_source(backend, workload)
                src_dash = load_kernel_source(backend, workload.replace("_", "-"))
                self.assertEqual(src_snake.source_bytes, src_dash.source_bytes)
                self.assertGreater(src_snake.size_bytes, 0)

    def test_kernel_entry_points_headers_and_unsupported_gates(self) -> None:
        for backend in EXPECTED_BACKENDS:
            for workload in EXPECTED_KERNEL_WORKLOADS:
                loaded = load_kernel_source(backend, workload)
                text = loaded.source_text
                expected_eps = default_entry_points(workload)
                self.assertEqual(loaded.entry_points, expected_eps)
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

    @unittest.skipUnless(
        os.environ.get("CROSS_GPU_HW_TESTS") == "1",
        "Live host GPU tests skipped unless CROSS_GPU_HW_TESTS=1 is set",
    )
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
                    loaded = load_kernel_source("metal", bad_wl)
                    try:
                        compiled = compile_kernel_runtime(
                            "metal",
                            bad_wl,
                            dev.architecture,
                            source_override=loaded.source_bytes,
                            cache=cache,
                        )
                        backend.load_module(ctx, compiled)
                    except KernelCompilationError as err:
                        self.assertIn("UNSUPPORTED", err.compiler_log)
                    else:
                        self.fail(f"Expected KernelCompilationError for {bad_wl}")
        finally:
            backend.destroy_context(ctx)

    @unittest.skipUnless(
        os.environ.get("CROSS_GPU_HW_TESTS") == "1",
        "Live host GPU tests skipped unless CROSS_GPU_HW_TESTS=1 is set",
    )
    def test_host_opencl_live_compile_and_golden_execution(self) -> None:
        if sys.platform == "darwin":
            self.skipTest("Apple OpenCL is deprecated and does not support modern compute pipelines")

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

    @unittest.skipUnless(
        os.environ.get("CROSS_GPU_HW_TESTS") == "1",
        "Live host GPU tests skipped unless CROSS_GPU_HW_TESTS=1 is set",
    )
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
