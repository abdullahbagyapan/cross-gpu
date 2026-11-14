# Backend ABI

The public ABI is declared in `include/crossgpu_backend.h`. The current value of
`CROSSGPU_BACKEND_ABI_VERSION` is 1. Plugins are shared libraries discovered at
runtime and must export:

```c
const crossgpu_backend_plugin_t *crossgpu_backend_get_api(
    uint32_t host_abi_version);
```

Return `NULL` for an unsupported host ABI. For ABI 1, return a descriptor with
`abi_version` set to `CROSSGPU_BACKEND_ABI_VERSION` and `struct_size` at least
`sizeof(crossgpu_backend_plugin_t)`.

## Required descriptor fields

The loader requires a nonzero `kind`, a nonempty `name`, and these callbacks:

- `enumerate_devices`
- `device_open` and `device_close`
- `workload_begin`, `workload_step`, and `workload_end`

`initialize` and `shutdown` are optional. If `initialize` is present and fails,
the loader rejects the plugin. The plugin should release any partial state before
returning failure. If `shutdown` is present, the loader calls it before unloading
the library.

Plugins may use a listed backend kind or another nonzero value, including
`CROSSGPU_BACKEND_CUSTOM`. Loading an unknown kind does not make its devices
selectable: the core's compiled preference registry must approve the plugin's
vendor, kind, and name. See [architecture](ARCHITECTURE.md#backend-selection-and-fallback).

## Device enumeration

Enumeration is a two-pass operation:

```c
size_t count = 0;
plugin->enumerate_devices(NULL, 0, &count, error, error_size);

/* Allocate count crossgpu_device_info_t entries, then: */
plugin->enumerate_devices(devices, count, &count, error, error_size);
```

On success, `device_count` reports the total visible count, and the plugin writes
no more than `capacity` entries. A `NULL` `devices` pointer is valid only with a
zero capacity. Set `crossgpu_device_info_t.struct_size` to the struct size.

## Physical identity

The host merges candidates only when `identity_kind`, `identity_namespace`, and
`identity_value` are all nonempty and identical. `backend_device_id` is for
diagnostics only and is never part of merging.

Use `CROSSGPU_IDENTITY_NAMESPACE_PCI_BDF` for a real PCI
domain:bus:device.function address. If a backend cannot report an identifier
that is trustworthy and comparable with another backend, leave
`identity_kind` as `CROSSGPU_IDENTITY_NONE` and the namespace and value empty.
Do not synthesize an identity from a product name, vendor, device index, or PCI
product/device ID.

The ABI also provides `CROSSGPU_IDENTITY_NAMESPACE_APPLE_IOREGISTRY` for the
macOS IORegistry identity space. Other UUID, OS, and platform identity namespaces
must explicitly identify their authority and encoding; identifiers from different
namespaces are not equivalent merely because they have a similar format.

## Lifecycle and workloads

For a selected backend, the core opens the device, then invokes
`workload_begin`, one or more `workload_step` calls, and `workload_end` for each
selected test before closing the device. Backends must not report a successful
step until their submitted work has completed.

`crossgpu_metrics_t` is a per-step delta:

- `operations` counts floating-point operations.
- `bytes_read` and `bytes_written` count device-memory traffic.
- `errors` counts validation failures.
- `iterations` counts completed workload batches.

`crossgpu_workload_config_t.memory_bytes` is zero for the current CLI, meaning
the backend selects its own default working set. `batch_iterations` is reserved
for workload tuning.

Different GPU workers can call the same plugin concurrently. `device_open` must
return independent per-device state, and plugin-global mutable state must be
synchronized.

## Compatibility

Do not change a published ABI struct or callback signature without increasing
`CROSSGPU_BACKEND_ABI_VERSION`. An incompatible revision requires a new ABI
version and matching host/plugin negotiation.
