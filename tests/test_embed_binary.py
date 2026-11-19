"""Unit tests for embedded kernel packing, verification, caching, and runtime compiler paths."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from cross_gpu.embed_binary import (
    RuntimeBinaryCache,
    compile_kernel_runtime,
    compute_cache_key,
    decode_payload,
    default_cache_dir,
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


if __name__ == "__main__":
    unittest.main()
