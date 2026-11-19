# SPDX-License-Identifier: MIT
"""Native ctypes GPU backend plugins and discovery registry."""

from __future__ import annotations

from .base import (
    NATIVE_BACKEND_PRIORITY,
    OPENCL_BACKEND_PRIORITY,
    BackendContext,
    BackendPlugin,
    DeviceBuffer,
    HostBuffer,
    KernelModule,
)
from .registry import (
    discover_backends,
    enumerate_all_backend_devices,
    get_backend_plugin,
)

__all__ = [
    "BackendContext",
    "BackendPlugin",
    "DeviceBuffer",
    "HostBuffer",
    "KernelModule",
    "NATIVE_BACKEND_PRIORITY",
    "OPENCL_BACKEND_PRIORITY",
    "discover_backends",
    "enumerate_all_backend_devices",
    "get_backend_plugin",
]
