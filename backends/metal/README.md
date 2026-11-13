# Metal backend plugin

Build on macOS with Xcode Command Line Tools and the Metal SDK:

```sh
make WITH_METAL=1
```

This produces `build/backends/libcrossgpu_backend_metal.dylib`. The plugin uses
Objective-C only at the Metal boundary; the CrossGPU core and public plugin ABI
remain C11. The core executable does not link against Metal or Foundation.

The plugin embeds its Metal Shading Language kernels and compiles them when a
Metal device is opened. It provides the `compute`, `memory`, and `bandwidth`
workloads. Metal devices report `MTLDevice.registryID` using the
`apple.iokit.registry-id` namespace. The macOS OpenCL plugin obtains the same
registry identity through CGL renderer metadata, allowing the host to merge
the Metal and OpenCL candidates without matching by GPU name.
