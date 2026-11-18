"""Dynamic discovery, priority sorting, and isolated enumeration of backend plugins."""

from __future__ import annotations

import importlib
from typing import Iterable

from cross_gpu.backends.base import BackendPlugin
from cross_gpu.models import BackendDeviceInfo

BUILTIN_PLUGIN_SPECS: tuple[tuple[str, str], ...] = (
    ("cross_gpu.backends.cuda_backend", "CudaBackend"),
    ("cross_gpu.backends.hip_backend", "HipBackend"),
    ("cross_gpu.backends.levelzero_backend", "LevelZeroBackend"),
    ("cross_gpu.backends.metal_backend", "MetalBackend"),
    ("cross_gpu.backends.opencl_backend", "OpenCLBackend"),
)


def discover_backends(
    *,
    only_available: bool = True,
    plugin_specs: Iterable[tuple[str, str]] = BUILTIN_PLUGIN_SPECS,
) -> list[BackendPlugin]:
    """Instantiate built-in backend plugins safely, sorted descending by priority."""
    plugins: list[BackendPlugin] = []
    for module_name, class_name in plugin_specs:
        try:
            module = importlib.import_module(module_name)
            plugin_cls = getattr(module, class_name)
            plugin: BackendPlugin = plugin_cls()
        except Exception:
            continue

        if only_available:
            try:
                if not plugin.is_available():
                    continue
            except Exception:
                continue

        plugins.append(plugin)

    plugins.sort(key=lambda item: (-item.priority, item.name))
    return plugins


def get_backend_plugin(
    name: str,
    plugins: Iterable[BackendPlugin] | None = None,
) -> BackendPlugin | None:
    """Find an available plugin by name."""
    candidates = discover_backends(only_available=True) if plugins is None else list(plugins)
    for plugin in candidates:
        if plugin.name == name:
            return plugin
    return None


def enumerate_all_backend_devices(
    plugins: Iterable[BackendPlugin] | None = None,
) -> tuple[list[BackendDeviceInfo], dict[str, BackendPlugin]]:
    """Enumerate devices from all available plugins without failing on a broken plugin."""
    active_plugins = discover_backends(only_available=True) if plugins is None else list(plugins)
    devices: list[BackendDeviceInfo] = []
    plugin_map: dict[str, BackendPlugin] = {}

    for plugin in active_plugins:
        try:
            if not plugin.is_available():
                continue
            found = plugin.enumerate_devices()
        except Exception:
            continue
        plugin_map[plugin.name] = plugin
        devices.extend(found)

    return devices, plugin_map
