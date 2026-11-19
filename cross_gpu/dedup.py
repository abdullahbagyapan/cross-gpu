"""Cross-backend physical GPU deduplication and backend priority resolution."""

from __future__ import annotations

import re
from typing import Callable, Iterable

from cross_gpu.exit_codes import BackendInitError
from cross_gpu.models import BackendBinding, BackendDeviceInfo, GPUDevice

_BDF_PATTERN = re.compile(
    r"^(?:pci@)?(?:([0-9a-fA-F]{1,8}):)?([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})\.([0-7])$"
)
_HEX32_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")
_GIB = 1024 * 1024 * 1024
_MIB = 1024 * 1024

_CANONICAL_VENDOR_IDS: dict[int, str] = {
    0x10DE: "nvidia",
    0x1002: "amd",
    0x8086: "intel",
    0x106B: "apple",
}


def normalize_pci_bdf(bdf: str | None) -> str | None:
    """Normalize a PCIe bus/device/function string to lowercase domain:bus:dev.fn."""
    if not bdf:
        return None
    cleaned = bdf.strip()
    match = _BDF_PATTERN.match(cleaned)
    if not match:
        return None
    domain = int(match.group(1), 16) if match.group(1) else 0
    bus = int(match.group(2), 16)
    device = int(match.group(3), 16)
    function = int(match.group(4), 16)
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def normalize_uuid(uuid: str | None) -> str | None:
    """Normalize a GPU UUID/LUID string for cross-backend comparison."""
    if not uuid:
        return None
    cleaned = uuid.strip().lower()
    if cleaned.startswith("gpu-"):
        cleaned = cleaned[4:]
    compact = cleaned.replace("-", "").replace("{", "").replace("}", "")
    if _HEX32_PATTERN.match(compact):
        if compact == "0" * 32:
            return None
        return (
            f"{compact[0:8]}-{compact[8:12]}-{compact[12:16]}-"
            f"{compact[16:20]}-{compact[20:32]}"
        )
    return cleaned or None


def canonical_vendor_key(vendor: str, vendor_id: int | None) -> str:
    """Return a canonical vendor key tolerant of OpenCL vendor ID quirks (e.g. Apple)."""
    if vendor_id is not None and vendor_id in _CANONICAL_VENDOR_IDS:
        return _CANONICAL_VENDOR_IDS[vendor_id]
    lower = vendor.strip().lower()
    if "nvidia" in lower:
        return "nvidia"
    if "advanced micro devices" in lower or "amd" in lower or lower.startswith("ati"):
        return "amd"
    if "intel" in lower:
        return "intel"
    if "apple" in lower:
        return "apple"
    if vendor_id is not None and vendor_id > 0:
        return f"0x{vendor_id:04x}"
    return lower


def normalize_device_name(name: str) -> str:
    """Normalize GPU model name across driver backends for fallback matching."""
    cleaned = name.lower()
    for token in ("(tm)", "(r)", "™", "®"):
        cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for prefix in ("nvidia ", "amd ", "intel ", "ati "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    return cleaned


def memory_bucket(total_memory_bytes: int) -> int:
    """Bucket total device memory so minor driver reservation diffs still match."""
    if total_memory_bytes <= 0:
        return 0
    if total_memory_bytes >= _GIB // 2:
        return max(1, round(total_memory_bytes / _GIB))
    if total_memory_bytes >= _MIB:
        return max(1, round(total_memory_bytes / _MIB))
    return total_memory_bytes


def fallback_fingerprint(info: BackendDeviceInfo) -> tuple[str, str, int]:
    """Build (vendor, normalized_name, memory_bucket) fallback identity."""
    return (
        canonical_vendor_key(info.vendor, info.vendor_id),
        normalize_device_name(info.name),
        memory_bucket(info.total_memory_bytes),
    )


class _PhysicalGroup:
    """Mutable accumulator representing one deduplicated physical GPU."""

    def __init__(self, first: BackendDeviceInfo) -> None:
        self.pci_bdf: str | None = normalize_pci_bdf(first.pci_bdf)
        self.uuid: str | None = normalize_uuid(first.uuid)
        self.registry_id: int | None = first.registry_id
        self.fingerprint: tuple[str, str, int] = fallback_fingerprint(first)
        self.infos: list[BackendDeviceInfo] = [first]
        self.match_reasons: list[str] = []

    def has_backend(self, backend_name: str) -> bool:
        return any(item.backend == backend_name for item in self.infos)

    def strong_ids_conflict(
        self,
        cand_bdf: str | None,
        cand_uuid: str | None,
        cand_reg: int | None,
    ) -> bool:
        if self.pci_bdf and cand_bdf and self.pci_bdf != cand_bdf:
            return True
        if self.uuid and cand_uuid and self.uuid != cand_uuid:
            return True
        if (
            self.registry_id is not None
            and cand_reg is not None
            and self.registry_id != cand_reg
        ):
            return True
        return False

    def add(self, info: BackendDeviceInfo, match_reason: str = "") -> None:
        cand_bdf = normalize_pci_bdf(info.pci_bdf)
        cand_uuid = normalize_uuid(info.uuid)
        if not self.pci_bdf and cand_bdf:
            self.pci_bdf = cand_bdf
        if not self.uuid and cand_uuid:
            self.uuid = cand_uuid
        if self.registry_id is None and info.registry_id is not None:
            self.registry_id = info.registry_id
        if match_reason:
            self.match_reasons.append(match_reason)
        self.infos.append(info)


def _find_matching_group(
    groups: list[_PhysicalGroup],
    info: BackendDeviceInfo,
) -> tuple[_PhysicalGroup, str] | None:
    cand_bdf = normalize_pci_bdf(info.pci_bdf)
    cand_uuid = normalize_uuid(info.uuid)
    cand_reg = info.registry_id
    cand_fp = fallback_fingerprint(info)

    # 1. PCIe BDF match (highest priority)
    if cand_bdf is not None:
        for group in groups:
            if group.pci_bdf == cand_bdf and not group.strong_ids_conflict(
                cand_bdf, cand_uuid, cand_reg
            ):
                return group, f"PCIe BDF {cand_bdf}"

    # 2. Hardware UUID / registryID match
    if cand_uuid is not None:
        for group in groups:
            if group.uuid == cand_uuid and not group.strong_ids_conflict(
                cand_bdf, cand_uuid, cand_reg
            ):
                return group, f"UUID {cand_uuid}"
    if cand_reg is not None:
        for group in groups:
            if group.registry_id == cand_reg and not group.strong_ids_conflict(
                cand_bdf, cand_uuid, cand_reg
            ):
                return group, f"registryID 0x{cand_reg:x}"

    # 3. Ordinal fallback fingerprint when no conflicting strong IDs exist
    for group in groups:
        if group.has_backend(info.backend):
            continue
        if group.strong_ids_conflict(cand_bdf, cand_uuid, cand_reg):
            continue
        # Only fall back when at least one side lacks the strong ID present on the other.
        has_shared_strong_id_type = (
            (group.pci_bdf is not None and cand_bdf is not None)
            or (group.uuid is not None and cand_uuid is not None)
            or (group.registry_id is not None and cand_reg is not None)
        )
        if has_shared_strong_id_type:
            continue
        if group.fingerprint == cand_fp:
            return group, f"fallback fingerprint {cand_fp[0]}/{cand_fp[1]}/{cand_fp[2]}"

    return None


def _build_gpu_device(index: int, group: _PhysicalGroup) -> GPUDevice:
    sorted_infos = sorted(
        group.infos,
        key=lambda item: (-item.priority, item.backend, item.backend_device_index),
    )
    primary = sorted_infos[0]

    free_mem = primary.free_memory_bytes
    compute_units = primary.compute_units
    architecture = primary.architecture
    compute_capability = primary.compute_capability
    driver_version = primary.driver_version
    vendor_id = primary.vendor_id
    pci_bdf = normalize_pci_bdf(primary.pci_bdf) or group.pci_bdf
    uuid = normalize_uuid(primary.uuid) or group.uuid
    registry_id = primary.registry_id if primary.registry_id is not None else group.registry_id

    for info in sorted_infos[1:]:
        if free_mem is None and info.free_memory_bytes is not None:
            free_mem = info.free_memory_bytes
        if not compute_units and info.compute_units:
            compute_units = info.compute_units
        if not architecture and info.architecture:
            architecture = info.architecture
        if not compute_capability and info.compute_capability:
            compute_capability = info.compute_capability
        if not driver_version and info.driver_version:
            driver_version = info.driver_version
        if vendor_id is None and info.vendor_id is not None:
            vendor_id = info.vendor_id

    bindings = [
        BackendBinding(
            backend=info.backend,
            priority=info.priority,
            device_info=info,
        )
        for info in sorted_infos
    ]

    return GPUDevice(
        index=index,
        name=primary.name,
        vendor=primary.vendor,
        vendor_id=vendor_id,
        architecture=architecture,
        compute_capability=compute_capability,
        total_memory_bytes=primary.total_memory_bytes,
        free_memory_bytes=free_mem,
        compute_units=compute_units,
        driver_version=driver_version,
        pci_bdf=pci_bdf,
        uuid=uuid,
        registry_id=registry_id,
        capabilities=primary.capabilities,
        bindings=bindings,
    )


def deduplicate_devices(
    backend_devices: Iterable[BackendDeviceInfo],
    *,
    diag_fn: Callable[[str], None] | None = None,
) -> list[GPUDevice]:
    """Deduplicate BackendDeviceInfo entries into canonical indexed GPUDevices."""
    ordered_inputs = sorted(
        backend_devices,
        key=lambda item: (-item.priority, item.backend, item.backend_device_index),
    )
    groups: list[_PhysicalGroup] = []
    for info in ordered_inputs:
        matched = _find_matching_group(groups, info)
        if matched is not None:
            group, reason = matched
            primary = group.infos[0]
            group.add(info, reason)
            if diag_fn is not None:
                diag_fn(
                    f"dedup: merged {info.backend} ({info.priority}) into "
                    f"{primary.backend} ({primary.priority}) for {primary.name!r} "
                    f"via {reason}"
                )
        else:
            groups.append(_PhysicalGroup(info))

    devices = [_build_gpu_device(idx, grp) for idx, grp in enumerate(groups)]
    if diag_fn is not None:
        for dev, grp in zip(devices, groups):
            winner = dev.bindings[0]
            if len(dev.bindings) > 1:
                others = ", ".join(f"{b.backend} ({b.priority})" for b in dev.bindings[1:])
                reasons = "; ".join(grp.match_reasons)
                diag_fn(
                    f"dedup: GPU {dev.index} {dev.name!r} selected "
                    f"{winner.backend} ({winner.priority}) over {others} [{reasons}]"
                )
            else:
                diag_fn(
                    f"dedup: GPU {dev.index} {dev.name!r} bound to "
                    f"{winner.backend} ({winner.priority})"
                )
    return devices


def resolve_device_backend(
    device: GPUDevice,
    backend_preference: str = "auto",
) -> BackendBinding:
    """Resolve the effective backend binding for a GPU according to --backend."""
    if not device.bindings:
        raise BackendInitError(f"Device {device.index} ({device.name}) has no bound backends")
    if backend_preference == "auto":
        return device.bindings[0]
    binding = device.get_binding(backend_preference)
    if binding is None:
        raise BackendInitError(
            f"Requested backend {backend_preference!r} is unavailable for device "
            f"{device.index} ({device.name}); available: {device.available_backends}"
        )
    return binding
