# Outstanding work

This list records work that is not implemented in the current source tree. It is
not a release schedule.

## Reliability and ABI hardening

- Retry device enumeration when a backend's count changes between the size query
  and the data query.
- Add fault-injection coverage for malformed plugin data, allocation failures,
  changing device counts, and workload-initialization failures.
- Define and test whether a workload-initialization failure should try a fallback
  candidate; failures after a workload begins must remain visible.
- Improve signal handling and inter-thread cancellation synchronization.

## Backend correctness and coverage

- Replace OpenCL's fixed-size extension-string buffer with a length-aware query
  and exact extension-token matching.
- Obtain actual maximum allocation limits from CUDA and HIP instead of reporting
  total device memory as the limit.
- Validate every optional backend against its vendor SDK, driver, and
  representative hardware.
- Validate macOS OpenCL-to-Metal identity mapping on physical hardware.
- Validate CUDA, HIP, and Level Zero plugin builds and workloads on native
  Windows SDK installations and representative hardware.

## Workload controls and reporting

- Expose configurable working-set sizes or percentages with safety headroom.
- Add chunked memory sweeps and additional deterministic or seeded patterns.
- Add host/device transfer bandwidth tests distinct from the current
  device-to-device bandwidth workload.
- Add machine-readable output and documented stable exit-code/result semantics.

## Monitoring and distribution

- Design optional telemetry support for temperature, power, utilization, clocks,
  and fan data without making unavailable sensors mandatory.
- Add reproducible release artifacts and platform-specific installation guidance.
