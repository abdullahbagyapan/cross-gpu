# SPDX-License-Identifier: MIT
# Fish completion for cross-gpu

set -l __cross_gpu_subcommands list info run
set -l __cross_gpu_backends auto cuda hip levelzero metal opencl
set -l __cross_gpu_devices all 0 1 2 3
set -l __cross_gpu_durations 10s 30s 60s 5m 10m 1h
set -l __cross_gpu_warmups 5s 10s 30s
set -l __cross_gpu_tests \
    all \
    compute-int64 \
    compute-int32 \
    compute-int16 \
    compute-int8 \
    compute-int4 \
    compute-fp64 \
    compute-fp32 \
    compute-fp16 \
    compute-fp8 \
    compute-fp4 \
    compute-bf16 \
    compute-tensor-fp32 \
    compute-tensor-fp16 \
    compute-tensor-bf16 \
    compute-tensor-int8 \
    memory-pattern \
    memory-sweep \
    bandwidth-d2d \
    bandwidth-h2d \
    bandwidth-d2h

complete -c cross-gpu -f

# Global flags and subcommands
complete -c cross-gpu -n "not __fish_seen_subcommand_from $__cross_gpu_subcommands" -s h -l help -d 'Show help message and exit'
complete -c cross-gpu -n "not __fish_seen_subcommand_from $__cross_gpu_subcommands" -l version -d 'Show version information and exit'
complete -c cross-gpu -n "not __fish_seen_subcommand_from $__cross_gpu_subcommands" -s v -l verbose -d 'Write detailed diagnostics to stderr'
complete -c cross-gpu -n "not __fish_seen_subcommand_from $__cross_gpu_subcommands" -a list -d 'List detected GPU devices'
complete -c cross-gpu -n "not __fish_seen_subcommand_from $__cross_gpu_subcommands" -a info -d 'Show detailed information for detected GPUs'
complete -c cross-gpu -n "not __fish_seen_subcommand_from $__cross_gpu_subcommands" -a run -d 'Run GPU stress workloads'

# Shared flags across list, info, and run
complete -c cross-gpu -n "__fish_seen_subcommand_from $__cross_gpu_subcommands" -s h -l help -d 'Show help message and exit'
complete -c cross-gpu -n "__fish_seen_subcommand_from $__cross_gpu_subcommands" -l format -xa 'text json' -d 'Output format'
complete -c cross-gpu -n "__fish_seen_subcommand_from $__cross_gpu_subcommands" -l json -d 'Shortcut for --format json'
complete -c cross-gpu -n "__fish_seen_subcommand_from $__cross_gpu_subcommands" -s v -l verbose -d 'Write detailed diagnostics to stderr'

# run-specific flags and values
complete -c cross-gpu -n "__fish_seen_subcommand_from run" -l device -xa "$__cross_gpu_devices" -d 'Target device index or all'
complete -c cross-gpu -n "__fish_seen_subcommand_from run" -l test -xa "$__cross_gpu_tests" -d 'Workload name, comma-separated list, or all'
complete -c cross-gpu -n "__fish_seen_subcommand_from run" -l duration -xa "$__cross_gpu_durations" -d 'Timed execution duration per workload'
complete -c cross-gpu -n "__fish_seen_subcommand_from run" -l warmup -xa "$__cross_gpu_warmups" -d 'Warmup duration before each workload'
complete -c cross-gpu -n "__fish_seen_subcommand_from run" -l backend -xa "$__cross_gpu_backends" -d 'Target backend plugin'
