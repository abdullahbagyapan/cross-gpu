"""Canonical 20-workload metadata registry and WorkloadExecutor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from cross_gpu.models import (
    BackendBinding,
    BackendDeviceInfo,
    DeviceCapabilities,
    GPUDevice,
    StepMetrics,
    VerificationResult,
)

SI_UNIT_SCALE = 1_000_000_000.0


@dataclass(frozen=True)
class WorkloadSpec:
    """Immutable metadata describing one registered stress workload."""

    name: str
    category: str
    rate_unit: str
    total_unit: str
    unit_scale: float = SI_UNIT_SCALE
    required_capability: str | None = None
    default_entry_point: str = ""
    requires_kernel: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not self.default_entry_point:
            object.__setattr__(
                self,
                "default_entry_point",
                f"kernel_{self.name.replace('-', '_')}",
            )


@dataclass
class WorkloadExecutionContext:
    """Runtime execution context handed to a WorkloadExecutor."""

    device: GPUDevice
    binding: BackendBinding
    backend: Any
    backend_context: Any
    spec: WorkloadSpec
    compiled_binary: Any = None
    module: Any = None
    buffers: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)


class WorkloadExecutor(ABC):
    """Abstract base class implemented by all 20 concrete workload executors."""

    requires_kernel: bool = True

    def __init__(self, spec: WorkloadSpec) -> None:
        self.spec = spec

    def check_support(self, device: GPUDevice | BackendDeviceInfo) -> bool:
        """Return True if the device supports the workload's required capability."""
        return device_supports_workload(device, self.spec)

    def reset_run_state(self, ctx: WorkloadExecutionContext) -> None:
        """Reset transient per-phase counters between warmup and timed execution."""
        del ctx

    @abstractmethod
    def prepare(self, ctx: WorkloadExecutionContext, binary: Any) -> None:
        """Allocate device/host buffers, load modules, and calibrate dispatch size."""

    @abstractmethod
    def run_step(self, ctx: WorkloadExecutionContext) -> StepMetrics:
        """Execute one calibrated kernel/transfer batch and return raw ops/bytes."""

    @abstractmethod
    def verify(self, ctx: WorkloadExecutionContext) -> VerificationResult:
        """Validate numerical or bitwise integrity against golden references."""

    @abstractmethod
    def cleanup(self, ctx: WorkloadExecutionContext) -> None:
        """Idempotently release all device buffers, host buffers, and modules."""


@dataclass
class WorkloadEntry:
    """Pairs a WorkloadSpec with its registered WorkloadExecutor class."""

    spec: WorkloadSpec
    executor_cls: type[WorkloadExecutor] | None = None


WORKLOAD_SPECS: tuple[WorkloadSpec, ...] = (
    # Scalar Integer (GOP/s, GOP — 10^9 ops)
    WorkloadSpec(
        name="compute-int64",
        category="compute-int",
        rate_unit="GOP/s",
        total_unit="GOP",
        required_capability="int64",
        description="64-bit integer scalar compute stress",
    ),
    WorkloadSpec(
        name="compute-int32",
        category="compute-int",
        rate_unit="GOP/s",
        total_unit="GOP",
        required_capability="int32",
        description="32-bit integer scalar compute stress",
    ),
    WorkloadSpec(
        name="compute-int16",
        category="compute-int",
        rate_unit="GOP/s",
        total_unit="GOP",
        required_capability="int16",
        description="16-bit integer scalar compute stress",
    ),
    WorkloadSpec(
        name="compute-int8",
        category="compute-int",
        rate_unit="GOP/s",
        total_unit="GOP",
        required_capability="int8",
        description="8-bit integer scalar/packed compute stress",
    ),
    WorkloadSpec(
        name="compute-int4",
        category="compute-int",
        rate_unit="GOP/s",
        total_unit="GOP",
        required_capability="int4",
        description="4-bit integer packed compute stress",
    ),
    # Scalar Floating-Point & ML (GFLOP/s, GFLOP — 10^9 FLOPs)
    WorkloadSpec(
        name="compute-fp64",
        category="compute-fp",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        required_capability="fp64",
        description="Double-precision FP64 compute stress",
    ),
    WorkloadSpec(
        name="compute-fp32",
        category="compute-fp",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        required_capability="fp32",
        description="Single-precision FP32 compute stress",
    ),
    WorkloadSpec(
        name="compute-fp16",
        category="compute-fp",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        required_capability="fp16",
        description="Half-precision FP16 compute stress",
    ),
    WorkloadSpec(
        name="compute-fp8",
        category="compute-fp",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        required_capability="fp8",
        description="8-bit floating-point FP8 compute stress",
    ),
    WorkloadSpec(
        name="compute-fp4",
        category="compute-fp",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        required_capability="fp4",
        description="4-bit floating-point FP4 compute stress",
    ),
    WorkloadSpec(
        name="compute-bf16",
        category="compute-fp",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        required_capability="bf16",
        description="Brain float BF16 compute stress",
    ),
    # Hardware Matrix / Tensor
    WorkloadSpec(
        name="compute-tensor-fp32",
        category="compute-tensor",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        required_capability="tensor_fp32",
        description="Hardware matrix/tensor FP32/TF32 compute stress",
    ),
    WorkloadSpec(
        name="compute-tensor-fp16",
        category="compute-tensor",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        required_capability="tensor_fp16",
        description="Hardware matrix/tensor FP16 compute stress",
    ),
    WorkloadSpec(
        name="compute-tensor-bf16",
        category="compute-tensor",
        rate_unit="GFLOP/s",
        total_unit="GFLOP",
        required_capability="tensor_bf16",
        description="Hardware matrix/tensor BF16 compute stress",
    ),
    WorkloadSpec(
        name="compute-tensor-int8",
        category="compute-tensor",
        rate_unit="GOP/s",
        total_unit="GOP",
        required_capability="tensor_int8",
        description="Hardware matrix/tensor INT8 compute stress",
    ),
    # Memory & Bandwidth (GB/s, GB — SI 10^9 bytes)
    WorkloadSpec(
        name="memory-pattern",
        category="memory",
        rate_unit="GB/s",
        total_unit="GB",
        required_capability=None,
        description="VRAM bit-pattern retention and fault-detection stress",
    ),
    WorkloadSpec(
        name="memory-sweep",
        category="memory",
        rate_unit="GB/s",
        total_unit="GB",
        required_capability=None,
        description="VRAM full-capacity streaming address sweep",
    ),
    WorkloadSpec(
        name="bandwidth-d2d",
        category="memory",
        rate_unit="GB/s",
        total_unit="GB",
        required_capability=None,
        requires_kernel=True,
        description="Device-to-device memory copy bandwidth stress",
    ),
    WorkloadSpec(
        name="bandwidth-h2d",
        category="memory",
        rate_unit="GB/s",
        total_unit="GB",
        required_capability=None,
        requires_kernel=False,
        description="Host-to-device transfer bandwidth stress",
    ),
    WorkloadSpec(
        name="bandwidth-d2h",
        category="memory",
        rate_unit="GB/s",
        total_unit="GB",
        required_capability=None,
        requires_kernel=False,
        description="Device-to-host transfer bandwidth stress",
    ),
)

WORKLOAD_SPECS_BY_NAME: dict[str, WorkloadSpec] = {
    spec.name: spec for spec in WORKLOAD_SPECS
}

WORKLOAD_REGISTRY: dict[str, WorkloadEntry] = {
    spec.name: WorkloadEntry(spec=spec, executor_cls=None) for spec in WORKLOAD_SPECS
}


def capabilities_as_dict(
    capabilities: DeviceCapabilities | dict[str, bool] | GPUDevice | BackendDeviceInfo | None,
) -> dict[str, bool]:
    """Normalize a capability container or device into a capability flag dict."""
    if capabilities is None:
        return {}
    if isinstance(capabilities, (GPUDevice, BackendDeviceInfo)):
        return capabilities.capabilities.to_dict()
    if isinstance(capabilities, DeviceCapabilities):
        return capabilities.to_dict()
    return dict(capabilities)


def device_supports_workload(
    device_or_caps: GPUDevice | BackendDeviceInfo | DeviceCapabilities | dict[str, bool],
    spec_or_name: WorkloadSpec | str,
) -> bool:
    """Return True when the device/capabilities satisfy a workload's requirement."""
    spec = (
        WORKLOAD_SPECS_BY_NAME[spec_or_name]
        if isinstance(spec_or_name, str)
        else spec_or_name
    )
    if spec.required_capability is None:
        return True
    caps = capabilities_as_dict(device_or_caps)
    return bool(caps.get(spec.required_capability, False))


_CUDA_MIN_SM: dict[str, str] = {
    "fp64": "1.3",
    "fp16": "5.3",
    "int8": "6.1",
    "tensor_fp16": "7.0",
    "tensor_int8": "7.2",
    "int4": "7.5",
    "bf16": "8.0",
    "tensor_fp32": "8.0",
    "tensor_bf16": "8.0",
    "fp8": "8.9",
    "fp4": "10.0",
}

_METAL_UNSUPPORTED_CAPS: frozenset[str] = frozenset(
    {"fp64", "fp8", "fp4", "int4", "tensor_int8"}
)

_OPENCL_EXT_BY_CAP: dict[str, str] = {
    "fp64": "cl_khr_fp64",
    "fp16": "cl_khr_fp16",
}


def explain_unsupported_workload(
    device: GPUDevice | BackendDeviceInfo,
    spec_or_name: WorkloadSpec | str,
) -> str:
    """Return a concrete, backend- and hardware-specific reason why a workload is unsupported."""
    spec = (
        WORKLOAD_SPECS_BY_NAME[spec_or_name]
        if isinstance(spec_or_name, str)
        else spec_or_name
    )
    cap = spec.required_capability
    backend = device.backend if isinstance(device, BackendDeviceInfo) else device.active_backend
    arch = device.architecture
    cc = device.compute_capability
    dev_label = arch or device.name

    if cap is None:
        return f"{backend or 'Device'} does not support workload {spec.name}"

    if backend == "cuda":
        min_sm = _CUDA_MIN_SM.get(cap)
        if min_sm:
            detected = f" (detected SM {cc})" if cc else (f" (detected {arch})" if arch else "")
            return f"requires CUDA SM >= {min_sm}{detected}"
        return f"CUDA device {device.name!r} lacks native {cap} support"

    if backend == "hip":
        det = f" (detected {dev_label})" if dev_label else ""
        if cap == "fp8":
            return f"requires AMD CDNA3 (gfx94x+) or RDNA4 (gfx12xx+) for native fp8{det}"
        if cap == "fp4":
            return f"requires AMD CDNA4 (gfx950+) for native fp4{det}"
        if cap == "int4":
            return f"HIP backend lacks native int4 support{det}"
        if cap == "tensor_fp32":
            return f"requires AMD CDNA2+ MFMA architecture for tensor_fp32{det}"
        if cap in ("bf16", "tensor_fp16", "tensor_bf16", "tensor_int8"):
            return f"requires AMD CDNA or RDNA3+ architecture for native {cap}{det}"
        if cap == "fp16":
            return f"requires AMD gfx900+ for native fp16{det}"
        if cap == "int8":
            return f"requires AMD gfx906+ or RDNA3+ for native int8 dot-product{det}"
        return f"HIP device {device.name!r} lacks native {cap} support"

    if backend == "metal":
        if cap in _METAL_UNSUPPORTED_CAPS:
            return f"Metal lacks native {cap}"
        if cap in ("bf16", "tensor_bf16"):
            return f"requires Apple9+ GPU / Metal 3.1+ bfloat16 support (detected {dev_label})"
        if cap in ("tensor_fp32", "tensor_fp16"):
            return f"requires Apple7+ GPU family for simdgroup_matrix {cap} (detected {dev_label})"
        return f"Metal device {device.name!r} lacks native {cap} support"

    if backend == "opencl":
        ext = _OPENCL_EXT_BY_CAP.get(cap)
        if ext:
            return f"OpenCL device missing {ext} extension"
        if cap.startswith("tensor_"):
            return f"OpenCL device missing Intel XMX/DPAS extension for {cap}"
        return f"OpenCL lacks native {cap} support on {device.name}"

    if backend == "levelzero":
        if cap in ("fp64", "fp16"):
            return f"Level Zero device lacks native {cap} module support"
        if cap in ("bf16", "tensor_fp16", "tensor_bf16", "tensor_int8"):
            return f"requires Intel Xe/Arc XMX DPAS hardware for {cap} (detected {device.name})"
        return f"Level Zero lacks native {cap} support on {device.name}"

    return f"Device {device.name!r} lacks required capability {cap!r}"


def register_workload_executor(
    name: str,
    executor_cls: type[WorkloadExecutor] | None,
    *,
    registry: dict[str, WorkloadEntry] | None = None,
) -> None:
    """Register or clear a WorkloadExecutor implementation for a canonical workload."""
    target_registry = WORKLOAD_REGISTRY if registry is None else registry
    if name not in target_registry:
        raise KeyError(f"Unknown workload name: {name!r}")
    target_registry[name].executor_cls = executor_cls


def list_workload_names() -> list[str]:
    """Return all 20 canonical workload names in registry order."""
    return [spec.name for spec in WORKLOAD_SPECS]
