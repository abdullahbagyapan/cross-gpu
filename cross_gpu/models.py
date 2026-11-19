# SPDX-License-Identifier: MIT
"""Core domain dataclasses for devices, telemetry, and workload results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cross_gpu.exit_codes import ExitCode


@dataclass
class DeviceCapabilities:
    """Hardware datatype and matrix/tensor execution capabilities."""

    fp64: bool = False
    fp32: bool = True
    fp16: bool = False
    bf16: bool = False
    fp8: bool = False
    fp4: bool = False
    int64: bool = True
    int32: bool = True
    int16: bool = True
    int8: bool = True
    int4: bool = False
    tensor_fp32: bool = False
    tensor_fp16: bool = False
    tensor_bf16: bool = False
    tensor_int8: bool = False

    def to_dict(self) -> dict[str, bool]:
        """Return capability flags as a plain dictionary."""
        return asdict(self)


@dataclass
class BackendDeviceInfo:
    """Hardware device descriptor returned by a single backend plugin."""

    backend: str
    priority: int
    backend_device_index: int
    native_handle: Any
    name: str
    vendor: str
    vendor_id: int | None = None
    architecture: str = ""
    compute_capability: str = ""
    total_memory_bytes: int = 0
    free_memory_bytes: int | None = None
    compute_units: int = 0
    driver_version: str = ""
    pci_bdf: str | None = None
    uuid: str | None = None
    registry_id: int | None = None
    capabilities: DeviceCapabilities = field(default_factory=DeviceCapabilities)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize backend-specific properties (excluding raw pointer handle)."""
        return {
            "backend": self.backend,
            "priority": self.priority,
            "backend_device_index": self.backend_device_index,
            "name": self.name,
            "vendor": self.vendor,
            "vendor_id": f"0x{self.vendor_id:04x}" if self.vendor_id is not None else None,
            "architecture": self.architecture,
            "compute_capability": self.compute_capability,
            "total_memory_bytes": self.total_memory_bytes,
            "free_memory_bytes": self.free_memory_bytes,
            "compute_units": self.compute_units,
            "driver_version": self.driver_version,
            "pci_bdf": self.pci_bdf,
            "uuid": self.uuid,
            "registry_id": self.registry_id,
            "capabilities": self.capabilities.to_dict(),
            "extra": dict(self.extra),
        }


@dataclass
class BackendBinding:
    """Associates a backend plugin and its device descriptor with a physical GPU."""

    backend: str
    priority: int
    device_info: BackendDeviceInfo


@dataclass
class GPUDevice:
    """Canonical physical GPU device after cross-backend deduplication."""

    index: int
    name: str
    vendor: str
    vendor_id: int | None = None
    architecture: str = ""
    compute_capability: str = ""
    total_memory_bytes: int = 0
    free_memory_bytes: int | None = None
    compute_units: int = 0
    driver_version: str = ""
    pci_bdf: str | None = None
    uuid: str | None = None
    registry_id: int | None = None
    capabilities: DeviceCapabilities = field(default_factory=DeviceCapabilities)
    bindings: list[BackendBinding] = field(default_factory=list)

    @property
    def active_backend(self) -> str:
        """Highest-priority backend bound to this physical device."""
        if not self.bindings:
            return ""
        return self.bindings[0].backend

    @property
    def available_backends(self) -> list[str]:
        """All backend names bound to this device, sorted by priority descending."""
        return [binding.backend for binding in self.bindings]

    def get_binding(self, backend_name: str) -> BackendBinding | None:
        """Return the binding for a specific backend, or None if not bound."""
        for binding in self.bindings:
            if binding.backend == backend_name:
                return binding
        return None

    def to_list_dict(self) -> dict[str, Any]:
        """Return concise JSON representation for `cross-gpu list`."""
        return {
            "index": self.index,
            "name": self.name,
            "vendor": self.vendor,
            "active_backend": self.active_backend,
            "available_backends": self.available_backends,
            "total_memory_bytes": self.total_memory_bytes,
            "compute_units": self.compute_units,
            "pci_bdf": self.pci_bdf,
            "uuid": self.uuid,
            "registry_id": self.registry_id,
        }

    def to_info_dict(self) -> dict[str, Any]:
        """Return detailed JSON representation for `cross-gpu info`."""
        return {
            "index": self.index,
            "name": self.name,
            "vendor": self.vendor,
            "vendor_id": f"0x{self.vendor_id:04x}" if self.vendor_id is not None else None,
            "architecture": self.architecture,
            "compute_capability": self.compute_capability,
            "active_backend": self.active_backend,
            "available_backends": self.available_backends,
            "total_memory_bytes": self.total_memory_bytes,
            "free_memory_bytes": self.free_memory_bytes,
            "compute_units": self.compute_units,
            "driver_version": self.driver_version,
            "pci_bdf": self.pci_bdf,
            "uuid": self.uuid,
            "registry_id": self.registry_id,
            "capabilities": self.capabilities.to_dict(),
            "backends": [binding.device_info.to_dict() for binding in self.bindings],
        }


@dataclass
class TelemetrySample:
    """1 Hz hardware telemetry sample with nullable unavailable sensors."""

    temperature_c: float | None = None
    power_w: float | None = None
    gpu_util_pct: float | None = None
    mem_util_pct: float | None = None
    core_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None
    fan_speed_pct: float | None = None
    fan_speed_rpm: int | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        """Return all 8 telemetry fields in canonical order."""
        return {
            "temperature_c": self.temperature_c,
            "power_w": self.power_w,
            "gpu_util_pct": self.gpu_util_pct,
            "mem_util_pct": self.mem_util_pct,
            "core_clock_mhz": self.core_clock_mhz,
            "mem_clock_mhz": self.mem_clock_mhz,
            "fan_speed_pct": self.fan_speed_pct,
            "fan_speed_rpm": self.fan_speed_rpm,
        }

    def to_usable_dict(self) -> dict[str, float | int]:
        """Return only non-None telemetry fields in canonical order."""
        return {key: val for key, val in self.to_dict().items() if val is not None}


@dataclass
class StepMetrics:
    """Raw accounting returned by a single workload execution step."""

    operations: float
    elapsed_s: float = 0.0
    errors: int = 0


@dataclass
class VerificationResult:
    """Outcome of a workload's numerical/bitwise verification pass."""

    passed: bool
    error_count: int = 0
    message: str = ""


@dataclass
class WorkloadResult:
    """Aggregated result for one workload run on one GPU device."""

    device: int
    device_name: str
    backend: str
    workload: str
    category: str
    status: str
    duration_s: float
    rate: float | None
    rate_unit: str
    total: float | None
    total_unit: str
    errors: int = 0
    message: str = ""
    reason: str = ""
    exit_code: ExitCode = ExitCode.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """Return NDJSON-serializable workload result payload."""
        payload: dict[str, Any] = {
            "device": self.device,
            "device_name": self.device_name,
            "backend": self.backend,
            "workload": self.workload,
            "category": self.category,
            "status": self.status,
            "duration_s": round(self.duration_s, 6),
            "rate": round(self.rate, 4) if self.rate is not None else None,
            "rate_unit": self.rate_unit,
            "total": round(self.total, 4) if self.total is not None else None,
            "total_unit": self.total_unit,
            "errors": self.errors,
            "message": self.message,
            "exit_code": int(self.exit_code),
        }
        resolved_reason = self.reason or (self.message if self.status == "unsupported" else "")
        if resolved_reason:
            payload["reason"] = resolved_reason
        elif self.status == "unsupported":
            payload["reason"] = f"Workload {self.workload!r} is unsupported on {self.backend}"
        return payload
