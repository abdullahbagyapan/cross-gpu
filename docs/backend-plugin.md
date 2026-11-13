# CrossGPU backend plugin ABI

CrossGPU uses a versioned C ABI so the main executable does not need to link
against vendor SDKs or runtimes. Each backend is an independently built shared
library.

## Required artifact and symbol

Linux:

```text
libcrossgpu_backend_NAME.so
```

macOS:

```text
libcrossgpu_backend_NAME.dylib
```

Windows:

```text
crossgpu_backend_NAME.dll
```

The library exports:

```c
const crossgpu_backend_plugin_t *crossgpu_backend_get_api(uint32_t host_abi_version);
```

Return `NULL` if the requested ABI version is unsupported. Set `abi_version` to
`CROSSGPU_BACKEND_ABI_VERSION` and `struct_size` to
`sizeof(crossgpu_backend_plugin_t)`.

The initial public ABI version is 1. ABI v1 defines typed, namespaced
physical-device identity metadata (`identity_kind`, `identity_namespace`, and
`identity_value`) and a separate backend-local diagnostic ID
(`backend_device_id`). All plugins must be built against the same ABI version
as the host. Incompatible changes to this contract require an ABI version bump.

## Device enumeration contract

The host first queries the required device count:

```c
size_t count = 0;
plugin->enumerate_devices(NULL, 0, &count, error, error_size);
```

The host allocates `count` entries and calls the function again:

```c
plugin->enumerate_devices(devices, count, &count, error, error_size);
```

The plugin sets `*device_count` to the total number of devices visible through
that backend, even when `capacity` is smaller. It writes at most `capacity`
entries. A `devices == NULL` pointer is valid only when `capacity == 0`.

## Lifecycle

1. The host loads the shared library and calls `crossgpu_backend_get_api()`.
2. The host validates the ABI, then calls `initialize()` once for the plugin.
3. The host queries and retrieves the backend's device list.
4. Device candidates are grouped only when their identity kind, namespace,
   and value match exactly. Backend-local IDs never participate in merging.
5. The host ranks backend candidates using the vendor preference registry.
6. The worker attempts `device_open()` in preference order. If it fails, the next
   candidate is tried (for example, CUDA first, then OpenCL for an NVIDIA GPU).
7. Each selected workload follows `workload_begin()`, repeated
   `workload_step()` calls, and `workload_end()`.
8. The host calls `device_close()` and, at shutdown, `shutdown()`.

A failed `initialize()` must clean up any resources it allocated before
returning. A failed `device_open()` must leave `*device_context == NULL` and
release partially created resources. A failed `workload_begin()` must leave
`*workload_context == NULL` and release partial allocations.

## Backend kind and plugin name

Known backends use the public enum values:

- `CROSSGPU_BACKEND_OPENCL`
- `CROSSGPU_BACKEND_CUDA`
- `CROSSGPU_BACKEND_HIP`
- `CROSSGPU_BACKEND_METAL`
- `CROSSGPU_BACKEND_LEVEL_ZERO`
- `CROSSGPU_BACKEND_CUSTOM`

A new backend may use `CROSSGPU_BACKEND_CUSTOM` or another nonzero enum value
that it reserves for its plugin. The host deliberately does not reject
unrecognized nonzero backend kinds. Give the plugin a stable, lowercase `name`
(such as `vulkan`) and users can select it with:

```sh
cross-gpu run --backend vulkan --test compute
```

No CLI parser change is required for a plugin-specific name. Automatic
vendor preference is separate and is configured in the core vendor registry.

## Vendor IDs

`crossgpu_device_info_t.vendor_id` uses `crossgpu_vendor_id_t`. Use named enum
values for known vendors, for example:

- `CROSSGPU_VENDOR_NVIDIA`
- `CROSSGPU_VENDOR_AMD`
- `CROSSGPU_VENDOR_INTEL`
- `CROSSGPU_VENDOR_APPLE`

Unknown PCI vendor IDs may be preserved with an explicit cast:

```c
device.vendor_id = (crossgpu_vendor_id_t)reported_vendor_id;
```

The host maps vendor IDs, plugin ABI kinds, and plugin runtime names to
preference scores using the `backend_preferences` table in `src/registry.c`.
The table is the single
source of backend-selection policy. A row must match the vendor, expected ABI
backend kind, and plugin runtime name. Exact vendor rows override wildcard
rows, and a missing row rejects that plugin for the device. A wildcard OpenCL
row provides generic compatibility for vendors without an explicit
preference. Add preference rows to prioritize a new backend for a vendor; do
not add backend-kind conditions to `src/backend_manager.c`. The plugin's
`name` and `kind` must match its preference row (for example, `level-zero` and
`CROSSGPU_BACKEND_LEVEL_ZERO`).

## Device identity

ABI v1 separates a physical identity from a backend-local diagnostic ID. Set
`device_info.identity_kind`, `identity_namespace`, and `identity_value` only
when the identifier is genuinely comparable with identifiers reported by
other backend APIs. The core treats the namespace and value as opaque strings;
it does not parse prefixes, normalize names, or guess whether two IDs belong to
the same identity domain.

The host merges two backend candidates only when all three identity fields
match exactly. Intel Level Zero and OpenCL may share the PCI BDF namespace when
both report the actual PCI address from their respective runtime APIs:

```c
device.identity_kind = CROSSGPU_IDENTITY_PCI_BDF;
snprintf(device.identity_namespace, sizeof(device.identity_namespace), "%s",
         CROSSGPU_IDENTITY_NAMESPACE_PCI_BDF);
snprintf(device.identity_value, sizeof(device.identity_value),
         "%04x:%02x:%02x.%x", domain, bus, device_number, function);
```

Supported identity categories are:

| Kind | Intended identity |
|---|---|
| `CROSSGPU_IDENTITY_PCI_BDF` | PCI domain:bus:device.function address |
| `CROSSGPU_IDENTITY_DEVICE_UUID` | Device UUID; namespace identifies the UUID authority/format |
| `CROSSGPU_IDENTITY_OS_DEVICE` | OS-provided physical-device identifier |
| `CROSSGPU_IDENTITY_PLATFORM_DEVICE` | Platform-provided identifier with a documented shared namespace |

Use the public PCI namespace constant for PCI BDF values. For UUID, OS, and
platform identities, choose a stable namespace that identifies the exact ID
authority and encoding. Two APIs may use the same namespace only when their
reported values refer to the same physical-device identity space. UUIDs from
different vendors or APIs must not be treated as equivalent merely because
both are UUID-shaped.

If a backend cannot provide a trustworthy cross-backend identity, leave
`identity_kind` as `CROSSGPU_IDENTITY_NONE` and leave the namespace/value empty.
Put any useful API-local identifier in `backend_device_id`; it is displayed for
diagnostics but is never used for merging. Do not synthesize a physical identity
from a product name, vendor name, device index, or PCI product/device ID.

For example, CUDA/HIP and OpenCL can share the PCI BDF namespace when each API
reports the actual PCI address. Metal reports the macOS IORegistry ID in the
`apple.iokit.registry-id` namespace. The macOS OpenCL plugin bridges to that
same ID space using CGL: it associates each OpenCL `cl_device_id` with the CGL
renderer using Apple's documented renderer-ID conversion, then reads the CGL
64-bit registry-ID properties. It assigns the identity only when the mapping
is available and unambiguous. Another macOS backend may merge with Metal only
if it can obtain that same IORegistry identity; matching an Apple GPU model
name is not sufficient.

References for the macOS bridge:

- Apple Technical Note TN2335, *Selecting a GPU for OpenCL on the Mac Pro
  (Late 2013)*, documents deriving `cl_device_id` from the CGL renderer ID by
  clearing the low byte: https://developer.apple.com/library/archive/technotes/tn2335/_index.html
- Apple documents `MTLDevice.registryID` as the GPU's registry identifier:
  https://developer.apple.com/documentation/metal/mtldevice/registryid
- CGL's `kCGLRPRegistryIDLow` and `kCGLRPRegistryIDHigh` properties expose the
  two halves of the IORegistry GPU ID on macOS 10.13 and later.

## Threading

Different physical GPU workers may call the same plugin concurrently. Each
`device_open()` must return an independent context/queue/stream. Plugin-global
state should be limited to immutable dispatch tables and enumeration data, or
protected with synchronization.

## Workload metrics

`crossgpu_metrics_t` is a per-step delta:

- `operations`: floating-point operations completed by compute workload
- `bytes_read`: bytes read from device memory
- `bytes_written`: bytes written to device memory
- `errors`: detected data-validation errors in this step
- `iterations`: completed workload batches (normally one per successful step)

A backend must not report success until its commands have completed.

## ABI compatibility

Do not change a published ABI struct or callback signature without incrementing
`CROSSGPU_BACKEND_ABI_VERSION`. Any future incompatible ABI revision must use a new ABI version because
`crossgpu_device_info_t` carries typed identity metadata and a distinct
backend-local device ID. Plugins must check `host_abi_version` and return `NULL`
when they cannot implement the requested version.
