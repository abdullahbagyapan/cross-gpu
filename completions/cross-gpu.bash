# SPDX-License-Identifier: MIT
# Bash completion for cross-gpu

_cross_gpu_completions() {
    local cur prev words cword
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev=""
    if (( COMP_CWORD > 0 )); then
        prev="${COMP_WORDS[COMP_CWORD-1]}"
    fi

    local subcommands="list info run"
    local global_flags="--help --version -v --verbose"
    local shared_flags="--format --json -v --verbose --help"
    local run_flags="--device --test --duration --warmup --backend --format --json -v --verbose --help"
    local formats="text json"
    local devices="all 0 1 2 3"
    local durations="250ms 500ms 10s 30s 60s 5m 10m 1h"
    local warmups="250ms 500ms 5s 10s 30s"
    local backends="auto cuda hip levelzero metal opencl"
    local tests="all compute-int64 compute-int32 compute-int16 compute-int8 compute-int4 compute-fp64 compute-fp32 compute-fp16 compute-fp8 compute-fp4 compute-bf16 compute-tensor-fp32 compute-tensor-fp16 compute-tensor-bf16 compute-tensor-int8 memory-pattern memory-sweep bandwidth-d2d bandwidth-h2d bandwidth-d2h"

    local cmd=""
    local i
    for (( i=1; i<COMP_CWORD; i++ )); do
        case "${COMP_WORDS[i]}" in
            list|info|run)
                cmd="${COMP_WORDS[i]}"
                break
                ;;
        esac
    done

    if [[ -z "${cmd}" ]]; then
        if [[ "${cur}" == -* ]]; then
            COMPREPLY=( $(compgen -W "${global_flags}" -- "${cur}") )
        else
            COMPREPLY=( $(compgen -W "${subcommands}" -- "${cur}") )
        fi
        return 0
    fi

    case "${prev}" in
        --format)
            COMPREPLY=( $(compgen -W "${formats}" -- "${cur}") )
            return 0
            ;;
        --device)
            COMPREPLY=( $(compgen -W "${devices}" -- "${cur}") )
            return 0
            ;;
        --duration)
            COMPREPLY=( $(compgen -W "${durations}" -- "${cur}") )
            return 0
            ;;
        --warmup)
            if [[ "${cur}" == -* ]]; then
                COMPREPLY=( $(compgen -W "${run_flags}" -- "${cur}") )
            else
                COMPREPLY=( $(compgen -W "${warmups}" -- "${cur}") )
            fi
            return 0
            ;;
        --backend)
            COMPREPLY=( $(compgen -W "${backends}" -- "${cur}") )
            return 0
            ;;
        --test)
            if [[ "${cur}" == *,* ]]; then
                local prefix="${cur%,*},"
                local last="${cur##*,}"
                local item
                COMPREPLY=()
                for item in ${tests}; do
                    [[ "${item}" == "all" ]] && continue
                    if [[ "${item}" == "${last}"* ]]; then
                        COMPREPLY+=( "${prefix}${item}" )
                    fi
                done
            else
                COMPREPLY=( $(compgen -W "${tests}" -- "${cur}") )
            fi
            return 0
            ;;
    esac

    case "${cmd}" in
        list|info)
            COMPREPLY=( $(compgen -W "${shared_flags}" -- "${cur}") )
            ;;
        run)
            COMPREPLY=( $(compgen -W "${run_flags}" -- "${cur}") )
            ;;
    esac
    return 0
}

complete -F _cross_gpu_completions cross-gpu
