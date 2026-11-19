# SPDX-License-Identifier: MIT
"""Unit tests for cross-backend physical GPU deduplication and priority resolution."""

from __future__ import annotations

import unittest

from src.dedup import (
    deduplicate_devices,
    normalize_pci_bdf,
    normalize_uuid,
    resolve_device_backend,
)
from src.exit_codes import BackendInitError
from src.models import BackendDeviceInfo, DeviceCapabilities


def _make_dev(
    backend: str,
    priority: int,
    idx: int,
    *,
    name: str = "NVIDIA GeForce RTX 4090",
    vendor: str = "NVIDIA",
    vendor_id: int | None = 0x10DE,
    mem: int = 24 * 1024**3,
    pci_bdf: str | None = None,
    uuid: str | None = None,
    registry_id: int | None = None,
    cu: int = 128,
) -> BackendDeviceInfo:
    return BackendDeviceInfo(
        backend=backend,
        priority=priority,
        backend_device_index=idx,
        native_handle=idx,
        name=name,
        vendor=vendor,
        vendor_id=vendor_id,
        total_memory_bytes=mem,
        compute_units=cu,
        pci_bdf=pci_bdf,
        uuid=uuid,
        registry_id=registry_id,
        capabilities=DeviceCapabilities(fp64=True, fp16=True),
    )


class TestDedup(unittest.TestCase):
    def test_pci_bdf_normalization(self) -> None:
        self.assertEqual(normalize_pci_bdf("0000:0A:00.0"), "0000:0a:00.0")
        self.assertEqual(normalize_pci_bdf("1:0.0"), "0000:01:00.0")
        self.assertEqual(normalize_pci_bdf("pci@0000:65:00.1"), "0000:65:00.1")
        self.assertIsNone(normalize_pci_bdf(""))
        self.assertIsNone(normalize_pci_bdf("invalid"))

    def test_uuid_normalization(self) -> None:
        self.assertEqual(
            normalize_uuid("GPU-12345678-1234-5678-1234-56789ABCDEF0"),
            "12345678-1234-5678-1234-56789abcdef0",
        )
        self.assertIsNone(normalize_uuid("00000000-0000-0000-0000-000000000000"))

    def test_dedup_by_bdf_prefers_native_100_over_opencl_50(self) -> None:
        opencl_dev = _make_dev(
            "opencl",
            50,
            0,
            pci_bdf="0000:01:00.0",
            uuid=None,
        )
        cuda_dev = _make_dev(
            "cuda",
            100,
            0,
            pci_bdf="01:00.0",
            uuid="GPU-11111111-2222-3333-4444-555555555555",
        )
        gpus = deduplicate_devices([opencl_dev, cuda_dev])
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].index, 0)
        self.assertEqual(gpus[0].active_backend, "cuda")
        self.assertEqual(gpus[0].available_backends, ["cuda", "opencl"])
        self.assertEqual(gpus[0].pci_bdf, "0000:01:00.0")
        self.assertEqual(gpus[0].uuid, "11111111-2222-3333-4444-555555555555")

    def test_dedup_by_uuid_when_bdf_missing(self) -> None:
        hip_dev = _make_dev(
            "hip",
            100,
            0,
            name="AMD Radeon RX 7900 XTX",
            vendor="AMD",
            vendor_id=0x1002,
            uuid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        )
        ocl_dev = _make_dev(
            "opencl",
            50,
            0,
            name="gfx1100",
            vendor="AMD",
            vendor_id=0x1002,
            uuid="AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
        )
        gpus = deduplicate_devices([ocl_dev, hip_dev])
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].active_backend, "hip")
        self.assertEqual(gpus[0].available_backends, ["hip", "opencl"])

    def test_identical_multi_gpu_with_distinct_bdfs_stay_separate(self) -> None:
        cuda0 = _make_dev("cuda", 100, 0, pci_bdf="0000:01:00.0")
        cuda1 = _make_dev("cuda", 100, 1, pci_bdf="0000:02:00.0")
        ocl0 = _make_dev("opencl", 50, 0, pci_bdf="0000:02:00.0")
        ocl1 = _make_dev("opencl", 50, 1, pci_bdf="0000:01:00.0")

        gpus = deduplicate_devices([ocl0, ocl1, cuda0, cuda1])
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0].pci_bdf, "0000:01:00.0")
        self.assertEqual(gpus[1].pci_bdf, "0000:02:00.0")
        self.assertEqual(gpus[0].available_backends, ["cuda", "opencl"])
        self.assertEqual(gpus[1].available_backends, ["cuda", "opencl"])

    def test_apple_silicon_metal_plus_opencl_fallback_dedup(self) -> None:
        metal_dev = _make_dev(
            "metal",
            100,
            0,
            name="Apple M2",
            vendor="Apple",
            vendor_id=0x106B,
            mem=12713115648,
            registry_id=0x1000003E8,
        )
        ocl_dev = _make_dev(
            "opencl",
            50,
            0,
            name="Apple M2",
            vendor="Apple",
            vendor_id=0x1027F00,
            mem=12713115648,
            registry_id=None,
        )
        gpus = deduplicate_devices([ocl_dev, metal_dev])
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].active_backend, "metal")
        self.assertEqual(gpus[0].available_backends, ["metal", "opencl"])
        self.assertEqual(gpus[0].registry_id, 0x1000003E8)

    def test_identical_multi_gpu_ordinal_fallback_pairing(self) -> None:
        l0_0 = _make_dev("levelzero", 100, 0, name="Intel Arc A770", vendor="Intel", vendor_id=0x8086, mem=16 * 1024**3, pci_bdf="0000:03:00.0")
        l0_1 = _make_dev("levelzero", 100, 1, name="Intel Arc A770", vendor="Intel", vendor_id=0x8086, mem=16 * 1024**3, pci_bdf="0000:04:00.0")
        ocl_0 = _make_dev("opencl", 50, 0, name="Intel(R) Arc(TM) A770", vendor="Intel(R) Corporation", vendor_id=0x8086, mem=16 * 1024**3)
        ocl_1 = _make_dev("opencl", 50, 1, name="Intel(R) Arc(TM) A770", vendor="Intel(R) Corporation", vendor_id=0x8086, mem=16 * 1024**3)

        gpus = deduplicate_devices([l0_0, l0_1, ocl_0, ocl_1])
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0].available_backends, ["levelzero", "opencl"])
        self.assertEqual(gpus[1].available_backends, ["levelzero", "opencl"])

    def test_resolve_device_backend_auto_and_explicit(self) -> None:
        gpu = deduplicate_devices([
            _make_dev("opencl", 50, 0, pci_bdf="0000:01:00.0"),
            _make_dev("cuda", 100, 0, pci_bdf="0000:01:00.0"),
        ])[0]

        self.assertEqual(resolve_device_backend(gpu, "auto").backend, "cuda")
        self.assertEqual(resolve_device_backend(gpu, "opencl").backend, "opencl")
        with self.assertRaises(BackendInitError):
            resolve_device_backend(gpu, "hip")


if __name__ == "__main__":
    unittest.main()
