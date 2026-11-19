# SPDX-License-Identifier: MIT
"""1 Hz drift-compensated telemetry sampler and platform reader router."""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable

from cross_gpu.models import BackendDeviceInfo, GPUDevice, TelemetrySample
from cross_gpu.telemetry.readers import (
    merge_samples,
    read_linux_sysfs,
    read_macos,
    read_nvml,
    sanitize_metrics_dict,
    sanitize_sample,
)


def read_telemetry(device: GPUDevice | BackendDeviceInfo | None) -> TelemetrySample:
    """Read a single TelemetrySample for the given GPU using platform readers."""
    if device is None:
        return TelemetrySample()

    try:
        if sys.platform == "darwin":
            return read_macos(device.registry_id)

        if sys.platform.startswith("linux"):
            vendor_lower = device.vendor.lower()
            is_nvidia = device.vendor_id == 0x10DE or "nvidia" in vendor_lower
            if is_nvidia:
                nvml_sample = read_nvml(device.pci_bdf)
                sysfs_sample = read_linux_sysfs(device.pci_bdf)
                return merge_samples(nvml_sample, sysfs_sample)
            return read_linux_sysfs(device.pci_bdf)
    except Exception:
        return TelemetrySample()

    return TelemetrySample()


class TelemetrySampler:
    """Background 1 Hz telemetry sampler using drift-compensated time.monotonic()."""

    def __init__(
        self,
        device: GPUDevice | BackendDeviceInfo,
        callback: Callable[[TelemetrySample, float], None],
        *,
        interval_s: float = 1.0,
        reader_fn: Callable[[GPUDevice | BackendDeviceInfo | None], TelemetrySample] = read_telemetry,
    ) -> None:
        self.device = device
        self.callback = callback
        self.interval_s = max(0.01, float(interval_s))
        self.reader_fn = reader_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_monotonic = 0.0

    def start(self) -> None:
        """Start the 1 Hz background telemetry polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._start_monotonic = time.monotonic()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"cross-gpu-telemetry-{getattr(self.device, 'index', 0)}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Signal the background thread to stop and join cleanly."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def _run_loop(self) -> None:
        next_tick = self._start_monotonic
        while not self._stop_event.is_set():
            now = time.monotonic()
            elapsed = max(0.0, now - self._start_monotonic)
            try:
                sample = self.reader_fn(self.device)
            except Exception:
                sample = TelemetrySample()
            if self._stop_event.is_set():
                break
            try:
                self.callback(sample, elapsed)
            except Exception:
                pass

            next_tick += self.interval_s
            after = time.monotonic()
            if next_tick < after:
                # Skip missed intervals without burst catch-up.
                missed = int((after - next_tick) // self.interval_s) + 1
                next_tick += missed * self.interval_s
            sleep_for = max(0.0, next_tick - after)
            if self._stop_event.wait(timeout=sleep_for):
                break


__all__ = ["TelemetrySampler", "read_telemetry", "sanitize_metrics_dict", "sanitize_sample"]
