#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# CrossGPU Docker-Only Multi-Backend & Multi-Architecture Emulation Test Harness
# ==============================================================================
# TESTING-ONLY script (NOT part of the CrossGPU CLI tool or packaged distributions).
#
# Hard design rules:
#   - Docker is the ONLY dependency (zero host Python/C/make/QEMU/Rosetta assumptions).
#   - Targets all 5 host CPU architectures (little-endian + big-endian s390x):
#       amd64 (linux/amd64), arm64 (linux/arm64), riscv64 (linux/riscv64),
#       ppc64le (linux/ppc64le), and s390x (linux/s390x, big-endian).
#   - Targets all 5 GPU backends in src/backends/:
#       cuda (+ NVRTC), hip (+ HIPRTC), levelzero, opencl (ABI shim + real PoCL LLVM),
#       metal (portable C11 Objective-C objc_msgSend ABI shim), plus 5-GPU dedup.
#   - Exercises multi-distro libc/Python/FHS packaging (Alpine/Debian/Fedora/Arch)
#     and adversarial container constraints (offline, read-only, noexec /tmp, UID 65534).
#   - Mounts the repository strictly read-only (/repo:ro) and writes all per-step
#     logs + summary.md into an isolated log directory.
#   - Compatible with Bash 3.2+ and modern Linux Bash.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="crossgpu-emul-$$-$(date +%Y%m%d%H%M%S)"

# Defaults (no host OS/arch detection anywhere)
DURATION="150ms"
DISTRO_PLATFORM="linux/amd64"
LOG_ROOT=""
SKIP_BINFMT="0"
FAIL_FAST="0"
DRY_RUN="0"
VERBOSE="0"
LIST_ONLY="0"
BINFMT_DONE="0"
DOCKER_BIN=""
DOCKER_NCPU="1"
DOCKER_ARCH=""

SUITES=()
TARGETS=()
SELECTED_TARGETS=()
RESULTS_TARGET=()
RESULTS_STATUS=()
RESULTS_ELAPSED=()
RESULTS_DETAIL=()

ALL_TARGETS=(
  arch-amd64
  arch-arm64
  arch-riscv64
  arch-ppc64le
  arch-s390x
  backend-cuda
  backend-hip
  backend-levelzero
  backend-opencl
  backend-metal
  backend-multigpu
  backend-pocl
  distro-alpine
  distro-debian
  distro-fedora
  distro-arch
  adversarial
)

# ---------------------------------------------------------------------------
# Color & Formatting Helpers
# ---------------------------------------------------------------------------
USE_COLOR="0"
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  USE_COLOR="1"
fi

c_green()  { if [[ "${USE_COLOR}" == "1" ]]; then printf '\033[32;1m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }
c_red()    { if [[ "${USE_COLOR}" == "1" ]]; then printf '\033[31;1m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }
c_yellow() { if [[ "${USE_COLOR}" == "1" ]]; then printf '\033[33;1m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }
c_cyan()   { if [[ "${USE_COLOR}" == "1" ]]; then printf '\033[36;1m%s\033[0m' "$1"; else printf '%s' "$1"; fi; }

decode_rc() {
  local code="$1"
  case "${code}" in
    99)  echo "99 (SHIM_RESOURCE_LEAK)" ;;
    124) echo "124 (TIMEOUT)" ;;
    132) echo "132 (SIGILL)" ;;
    134) echo "134 (SIGABRT)" ;;
    136) echo "136 (SIGFPE)" ;;
    137) echo "137 (SIGKILL/OOM)" ;;
    139) echo "139 (SIGSEGV)" ;;
    143) echo "143 (SIGTERM)" ;;
    *)
      if [[ "${code}" =~ ^[0-9]+$ ]] && (( code > 128 && code < 192 )); then
        echo "${code} (SIG$((code - 128)))"
      else
        echo "${code}"
      fi
      ;;
  esac
}

# ---------------------------------------------------------------------------
# Signal / Orphan Container Cleanup
# ---------------------------------------------------------------------------
cleanup_containers() {
  if [[ -n "${DOCKER_BIN}" ]]; then
    local ids=""
    ids="$("${DOCKER_BIN}" ps -q --filter "label=cross-gpu-emulate=${RUN_ID}" 2>/dev/null || true)"
    if [[ -n "${ids}" ]]; then
      printf '\nStopping active emulation container(s) for run %s...\n' "${RUN_ID}" >&2
      # shellcheck disable=SC2086
      "${DOCKER_BIN}" rm -f ${ids} >/dev/null 2>&1 || true
    fi
    "${DOCKER_BIN}" volume rm -f "${RUN_ID}-wheels" >/dev/null 2>&1 || true
  fi
}

on_interrupt() {
  cleanup_containers
  printf '\nInterrupted by signal. Partial logs (if any): %s\n' "${LOG_ROOT:-none}" >&2
  exit 130
}

trap on_interrupt INT TERM

# ---------------------------------------------------------------------------
# Target Metadata
# ---------------------------------------------------------------------------
target_suite() {
  case "$1" in
    arch-amd64|arch-arm64|arch-riscv64|arch-ppc64le|arch-s390x) echo "arch" ;;
    backend-cuda|backend-hip|backend-levelzero|backend-opencl|backend-metal|backend-multigpu|backend-pocl) echo "backends" ;;
    distro-alpine|distro-debian|distro-fedora|distro-arch) echo "distros" ;;
    adversarial) echo "adversarial" ;;
    *) return 1 ;;
  esac
}

target_desc() {
  case "$1" in
    arch-amd64)        echo "linux/amd64 (x86_64, little-endian) 76 tests, 90 kernels, JSON & offline build" ;;
    arch-arm64)        echo "linux/arm64 (arm64, little-endian) 76 tests, 90 kernels, JSON & offline build" ;;
    arch-riscv64)      echo "linux/riscv64 (riscv64, little-endian) 76 tests, 90 kernels, JSON & build" ;;
    arch-ppc64le)      echo "linux/ppc64le (ppc64le, little-endian) 76 tests, 90 kernels, JSON & build" ;;
    arch-s390x)        echo "linux/s390x (s390x, big-endian) struct/ABI endianness, 76 tests & build" ;;
    backend-cuda)      echo "CUDA + NVRTC versioned SONAME shim (list/info, pass, bitflip=3, ecc_ue=4, leaks)" ;;
    backend-hip)       echo "HIP + HIPRTC versioned SONAME shim (gfx1100 scan, pass, bitflip=3, ecc_ue=4, leaks)" ;;
    backend-levelzero) echo "Level Zero versioned SONAME shim (struct ABI, SPIR-V, pass, bitflip=3, lost=4)" ;;
    backend-opencl)    echo "OpenCL versioned SONAME shim (PCI BDF, MapBuffer, pass, bitflip=3, lost=4, leaks)" ;;
    backend-metal)     echo "Metal C11 ObjC objc_msgSend ABI shim (Linux guard, 24B _MTLSize, pass/3/4, leaks)" ;;
    backend-multigpu)  echo "Heterogeneous 5-GPU dedup (9->5 across CUDA/HIP/L0/OpenCL) + isolated faults 3/4" ;;
    backend-pocl)      echo "Real PoCL OpenCL ICD: no-CPU-fallback gate, 36 .cl builds & golden execution" ;;
    distro-alpine)     echo "alpine:latest (musl) unit tests, completion syntax, offline build & FHS install" ;;
    distro-debian)     echo "debian:trixie-slim (glibc) unit tests, completion syntax, build & FHS install" ;;
    distro-fedora)     echo "fedora:latest (glibc) unit tests, completion syntax, offline build & FHS install" ;;
    distro-arch)       echo "archlinux:latest (pacman --disable-sandbox) tests, offline build & FHS install" ;;
    adversarial)       echo "python:3.12-slim offline, --read-only, noexec /tmp, UID 65534, no /dev/dri" ;;
    *) return 1 ;;
  esac
}

is_valid_target() {
  local cand="$1"
  local t
  for t in "${ALL_TARGETS[@]}"; do
    if [[ "${t}" == "${cand}" ]]; then
      return 0
    fi
  done
  return 1
}

add_target_unique() {
  local cand="$1"
  local existing
  if [[ "${#SELECTED_TARGETS[@]}" -gt 0 ]]; then
    for existing in "${SELECTED_TARGETS[@]}"; do
      if [[ "${existing}" == "${cand}" ]]; then
        return 0
      fi
    done
  fi
  SELECTED_TARGETS+=("${cand}")
}

# ---------------------------------------------------------------------------
# CLI Help & Listing
# ---------------------------------------------------------------------------
usage() {
  cat <<'EOF'
CrossGPU Docker-Only Multi-Backend & Multi-Architecture Emulation Harness (testing only)

Usage:
  ./scripts/emulate.sh [OPTIONS]

Options:
  --list                     List all emulation suites and targets
  --suite SUITE[,SUITE...]   Select suite(s): arch, backends, distros, adversarial, all
                             (default when no --suite/--target is passed: all)
  --target NAME[,NAME...]    Select specific target(s) (repeatable or comma-separated)
  --distro-platform PLAT     Docker --platform for backends, distros, and adversarial
                             [default: linux/amd64]
  --duration TIME            Timed CLI workload duration inside containers [default: 150ms]
  --log-dir DIR              Directory for per-step logs and summary.md
                             [default: ~/.cache/cross-gpu-emulation/<timestamp>]
  --skip-binfmt              Skip automatic tonistiigi/binfmt multi-arch QEMU registration
  --fail-fast                Stop immediately on the first failed target
  --dry-run                  Print selected targets and Docker settings without running
  -v, --verbose              Print container stderr live in addition to log files
  -h, --help                 Show this help message

Suites & Targets:
  arch        : arch-amd64, arch-arm64, arch-riscv64, arch-ppc64le, arch-s390x
  backends    : backend-cuda, backend-hip, backend-levelzero, backend-opencl,
                backend-metal, backend-multigpu, backend-pocl
  distros     : distro-alpine, distro-debian, distro-fedora, distro-arch
  adversarial : adversarial
  all         : every target above

Examples:
  ./scripts/emulate.sh --list
  ./scripts/emulate.sh --suite backends
  ./scripts/emulate.sh --suite arch
  ./scripts/emulate.sh --target backend-cuda,backend-metal,backend-multigpu,backend-pocl
  ./scripts/emulate.sh --target arch-s390x,arch-arm64,arch-riscv64
  ./scripts/emulate.sh --suite distros --distro-platform linux/arm64
  ./scripts/emulate.sh --suite all
EOF
}

print_list() {
  printf '%s\n' "CrossGPU Docker-Only Emulation Targets (testing harness; Docker is the only dependency)"
  printf '%s\n' "========================================================================================"
  printf '%-18s %-12s %s\n' "TARGET" "SUITE" "DESCRIPTION"
  printf '%s\n' "----------------------------------------------------------------------------------------"
  local t
  for t in "${ALL_TARGETS[@]}"; do
    printf '%-18s %-12s %s\n' "${t}" "$(target_suite "${t}")" "$(target_desc "${t}")"
  done
}

# ---------------------------------------------------------------------------
# Docker Pre-Flight & binfmt_misc Setup (Zero Host Assumptions)
# ---------------------------------------------------------------------------
locate_and_verify_docker() {
  local extra_dir
  for extra_dir in "/opt/homebrew/bin" "/usr/local/bin"; do
    if [[ -d "${extra_dir}" && ":${PATH}:" != *":${extra_dir}:"* ]]; then
      PATH="${PATH}:${extra_dir}"
    fi
  done
  export PATH

  DOCKER_BIN="$(command -v docker 2>/dev/null || true)"
  if [[ -z "${DOCKER_BIN}" ]]; then
    printf 'Error: docker CLI not found on PATH.\n' >&2
    printf 'Docker is the sole dependency of this emulation test harness.\n' >&2
    return 2
  fi

  local info_out=""
  local info_err=""
  info_err="$(mktemp)"
  if ! info_out="$("${DOCKER_BIN}" info --format '{{.NCPU}}|{{.Architecture}}|{{.ServerVersion}}' 2>"${info_err}")"; then
    printf 'Error: unable to reach the Docker daemon via %s.\n' "${DOCKER_BIN}" >&2
    cat "${info_err}" >&2
    rm -f "${info_err}"
    return 2
  fi
  rm -f "${info_err}"

  local ncpu_raw="${info_out%%|*}"
  local rest="${info_out#*|}"
  DOCKER_ARCH="${rest%%|*}"
  local server_ver="${rest#*|}"

  if [[ "${ncpu_raw}" =~ ^[0-9]+$ ]] && (( ncpu_raw >= 1 )); then
    DOCKER_NCPU="${ncpu_raw}"
  else
    DOCKER_NCPU="1"
  fi

  printf 'Docker daemon ready: %s (server=%s, engine_arch=%s, engine_ncpu=%s)\n' \
    "${DOCKER_BIN}" "${server_ver}" "${DOCKER_ARCH}" "${DOCKER_NCPU}"
}

ensure_binfmt() {
  if [[ "${SKIP_BINFMT}" == "1" || "${BINFMT_DONE}" == "1" ]]; then
    return 0
  fi
  BINFMT_DONE="1"

  local bf_dir="${LOG_ROOT}/00_binfmt"
  mkdir -p "${bf_dir}"
  local probe_out="${bf_dir}/probe_binfmt.stdout.log"
  local probe_err="${bf_dir}/probe_binfmt.stderr.log"
  local probe_rc=0

  printf '==> Checking multi-arch binfmt_misc handlers (tonistiigi/binfmt)...\n'
  "${DOCKER_BIN}" run --privileged --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    tonistiigi/binfmt >"${probe_out}" 2>"${probe_err}" || probe_rc=$?

  local missing=()
  if [[ "${probe_rc}" -eq 0 ]]; then
    local status_txt
    status_txt="$(cat "${probe_out}")"
    # Preserve native/Rosetta amd64 handling; only install architectures not already handled.
    if [[ "${status_txt}" != *'"linux/amd64"'* && "${status_txt}" != *'"rosetta"'* && "${status_txt}" != *'"qemu-x86_64"'* ]]; then
      missing+=("amd64")
    fi
    if [[ "${status_txt}" != *'"linux/arm64"'* && "${status_txt}" != *'"qemu-aarch64"'* ]]; then
      missing+=("arm64")
    fi
    if [[ "${status_txt}" != *'"linux/riscv64"'* && "${status_txt}" != *'"qemu-riscv64"'* ]]; then
      missing+=("riscv64")
    fi
    if [[ "${status_txt}" != *'"linux/ppc64le"'* && "${status_txt}" != *'"qemu-ppc64le"'* ]]; then
      missing+=("ppc64le")
    fi
    if [[ "${status_txt}" != *'"linux/s390x"'* && "${status_txt}" != *'"qemu-s390x"'* ]]; then
      missing+=("s390x")
    fi
  else
    missing=("amd64" "arm64" "riscv64" "ppc64le" "s390x")
  fi

  if [[ "${#missing[@]}" -eq 0 ]]; then
    printf '  [%s] all required binfmt_misc architectures already registered\n\n' "$(c_green PASS)"
    return 0
  fi

  local install_csv=""
  local arch_item
  for arch_item in "${missing[@]}"; do
    if [[ -z "${install_csv}" ]]; then
      install_csv="${arch_item}"
    else
      install_csv="${install_csv},${arch_item}"
    fi
  done

  local out="${bf_dir}/install_binfmt.stdout.log"
  local err="${bf_dir}/install_binfmt.stderr.log"
  printf '%s\n' "${DOCKER_BIN} run --privileged --rm --label cross-gpu-emulate=${RUN_ID} tonistiigi/binfmt --install ${install_csv}" \
    > "${bf_dir}/install_binfmt.cmd"
  printf '0\n' > "${bf_dir}/install_binfmt.expected_rc"

  printf '  Installing missing binfmt handler(s): %s\n' "${install_csv}"
  local rc=0
  "${DOCKER_BIN}" run --privileged --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    tonistiigi/binfmt --install "${install_csv}" >"${out}" 2>"${err}" || rc=$?
  printf '%s\n' "${rc}" > "${bf_dir}/install_binfmt.rc"

  if [[ "${rc}" -eq 0 ]]; then
    printf '  [%s] binfmt_misc registered for %s\n\n' "$(c_green PASS)" "${install_csv}"
    if [[ "${VERBOSE}" == "1" && -s "${out}" ]]; then
      cat "${out}"
    fi
  else
    printf '  [%s] binfmt_misc registration returned rc=%s (continuing; if your daemon already has binfmt or uses rootless Docker, pass --skip-binfmt)\n' \
      "$(c_yellow WARN)" "${rc}" >&2
    if [[ -s "${err}" ]]; then
      tail -n 20 "${err}" >&2
    fi
    printf '\n'
  fi
}

# ---------------------------------------------------------------------------
# Shared In-Container Bash Logging & Diagnostic Prelude
# ---------------------------------------------------------------------------
emit_container_bash_prelude() {
  cat <<'EOF'
trap 'chmod -R a+rX /logs 2>/dev/null || true' EXIT

decode_rc() {
  local code="$1"
  case "${code}" in
    99)  echo "99 (SHIM_RESOURCE_LEAK)" ;;
    124) echo "124 (TIMEOUT)" ;;
    132) echo "132 (SIGILL)" ;;
    134) echo "134 (SIGABRT)" ;;
    136) echo "136 (SIGFPE)" ;;
    137) echo "137 (SIGKILL/OOM)" ;;
    139) echo "139 (SIGSEGV)" ;;
    143) echo "143 (SIGTERM)" ;;
    *)
      if (( code > 128 && code < 192 )); then
        echo "${code} (SIG$((code - 128)))"
      else
        echo "${code}"
      fi
      ;;
  esac
}

run_logged_expect() {
  local expected="$1"; shift
  local name="$1"; shift
  local out="/logs/${name}.stdout.log"
  local err="/logs/${name}.stderr.log"
  local rc=0
  printf '%s\n' "$*" > "/logs/${name}.cmd"
  printf '%s\n' "${expected}" > "/logs/${name}.expected_rc"
  "$@" >"${out}" 2>"${err}" || rc=$?
  printf '%s\n' "${rc}" > "/logs/${name}.rc"

  if [[ "${rc}" -eq "${expected}" ]]; then
    printf '  [PASS] %s (rc=%s)\n' "${name}" "${rc}"
    return 0
  fi

  local rc_desc
  rc_desc="$(decode_rc "${rc}")"
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    printf '::error title=%s failed (expected rc=%s, got %s)::Command: %s\n' \
      "${name}" "${expected}" "${rc_desc}" "$*"
  fi
  printf '%s\n' "================================================================"
  printf '  [FAIL] %s\n' "${name}"
  printf '  Command : %s\n' "$*"
  printf '  Exit    : %s (expected %s)\n' "${rc_desc}" "${expected}"
  printf '%s\n' "  --- stdout (${name}.stdout.log) ---"
  if [[ -s "${out}" ]]; then tail -n 200 "${out}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "  --- stderr (${name}.stderr.log) ---"
  if [[ -s "${err}" ]]; then tail -n 200 "${err}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "================================================================"
  return 1
}

run_logged() {
  local name="$1"; shift
  run_logged_expect 0 "${name}" "$@"
}

run_cli_json_expect() {
  local expected="$1"; shift
  local name="$1"; shift
  if run_logged_expect "${expected}" "${name}" "$@"; then
    return 0
  fi
  if [[ ! -s "/logs/${name}.stdout.log" && ! -s "/logs/${name}.stderr.log" ]]; then
    local diag_cmd=()
    local arg
    for arg in "$@"; do
      if [[ "${arg}" != "--json" ]]; then
        diag_cmd+=("${arg}")
      fi
    done
    diag_cmd+=("-v")
    printf '%s\n' "  >>> Silent --json failure detected; diagnostic retry with -v for ${name}:"
    run_logged_expect 0 "${name}_diag_verbose" "${diag_cmd[@]}" || true
  fi
  return 1
}

assert_empty_stderr() {
  local name="$1"
  if [[ -s "/logs/${name}.stderr.log" ]]; then
    printf '  [FAIL] %s polluted stderr in --json mode:\n' "${name}" >&2
    cat "/logs/${name}.stderr.log" >&2
    return 1
  fi
}
EOF
}

# ---------------------------------------------------------------------------
# Docker Execution Wrapper
# ---------------------------------------------------------------------------
run_docker_script() {
  local target="$1"
  local log_dir="$2"
  local script_file="$3"
  shift 3

  local container_out="${log_dir}/_container.stdout.log"
  local container_err="${log_dir}/_container.stderr.log"
  printf '%s\n' "${DOCKER_BIN} $*" > "${log_dir}/_container.cmd"
  printf '0\n' > "${log_dir}/_container.expected_rc"

  local rc=0
  "${DOCKER_BIN}" "$@" < "${script_file}" > "${container_out}" 2> "${container_err}" || rc=$?
  cat "${container_out}"
  if [[ "${VERBOSE}" == "1" || "${rc}" -ne 0 ]]; then
    if [[ -s "${container_err}" ]]; then
      printf '%s\n' "--- container stderr (${target}) ---" >&2
      cat "${container_err}" >&2
    fi
  fi

  printf '%s\n' "${rc}" > "${log_dir}/_container.rc"
  return "${rc}"
}

# ---------------------------------------------------------------------------
# Suite 1: Foreign CPU Architectures & Endianness (arch)
# ---------------------------------------------------------------------------
run_target_arch() {
  local target="$1"
  local log_dir="$2"
  local plat=""
  local expected_arch=""
  local expected_endian=""

  case "${target}" in
    arch-amd64)   plat="linux/amd64";   expected_arch="x86_64";  expected_endian="little" ;;
    arch-arm64)   plat="linux/arm64";   expected_arch="aarch64"; expected_endian="little" ;;
    arch-riscv64) plat="linux/riscv64"; expected_arch="riscv64"; expected_endian="little" ;;
    arch-ppc64le) plat="linux/ppc64le"; expected_arch="ppc64le"; expected_endian="little" ;;
    arch-s390x)   plat="linux/s390x";   expected_arch="s390x";   expected_endian="big" ;;
    *) printf 'Internal error: unknown arch target %s\n' "${target}" >&2; return 1 ;;
  esac

  local script_file
  script_file="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 python3-setuptools python3-wheel python3-build'

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info

run_logged "01_env" python3 -c '
import os, platform, sys
mach = platform.machine()
bo = sys.byteorder
exp_arch = os.environ["EXPECTED_ARCH"]
exp_endian = os.environ["EXPECTED_ENDIAN"]
print(f"machine={mach} byteorder={bo} python={sys.version.split()[0]}")
assert mach == exp_arch, f"expected machine {exp_arch}, got {mach}"
assert bo == exp_endian, f"expected byteorder {exp_endian}, got {bo}"
'
run_logged "02_compileall" python3 -m compileall -q src tests
run_logged "03_embed_verify" python3 -m src.embed_binary verify
run_logged "04_unittest" python3 -m unittest discover -s tests -v

run_cli_json_expect 0 "05_list_json" python3 -m src.cli list --json
assert_empty_stderr "05_list_json"
run_logged "05_list_json_check" python3 -m json.tool /logs/05_list_json.stdout.log

run_cli_json_expect 0 "06_info_json" python3 -m src.cli info --json
assert_empty_stderr "06_info_json"
run_logged "06_info_json_check" python3 -m json.tool /logs/06_info_json.stdout.log

run_logged "07_list_verbose_json" python3 -m src.cli list --json -v
run_logged "07_list_verbose_json_check" python3 -m json.tool /logs/07_list_verbose_json.stdout.log

run_logged "08_offline_build" python3 -m build --no-isolation --outdir /tmp/dist .
run_logged "09_verify_wheel" python3 -c '
import glob, zipfile
wheels = glob.glob("/tmp/dist/cross_gpu-*.whl")
assert len(wheels) == 1, f"Expected 1 wheel, found: {wheels}"
with zipfile.ZipFile(wheels[0]) as zf:
    names = zf.namelist()
kernels = [n for n in names if n.startswith("cross_gpu/kernels/") and n.endswith((".cu", ".hip", ".cl", ".metal"))]
assert len(kernels) == 90, f"Expected 90 kernels in wheel, got {len(kernels)}"
assert not any(n.startswith("src/") or "/telemetry/" in n for n in names), names
print("OK wheel:", wheels[0], "kernels:", len(kernels), "entries:", len(names))
'
EOF
  } > "${script_file}"

  local rc=0
  run_docker_script "${target}" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    --platform "${plat}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
    -e "EXPECTED_ARCH=${expected_arch}" \
    -e "EXPECTED_ENDIAN=${expected_endian}" \
    debian:trixie-slim bash -euo pipefail || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

# ---------------------------------------------------------------------------
# Suite 2: GPU Backends Emulation (CUDA, HIP, Level Zero, OpenCL, Metal, Multi-GPU, PoCL)
# ---------------------------------------------------------------------------
run_target_backend_cuda() {
  local log_dir="$1"
  local script_file
  script_file="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 gcc libc6-dev'

cp -a /repo /tmp/work
cd /tmp/work
mkdir -p /tmp/shims

CC_FLAGS=(-std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC -I/repo/scripts/shims)
run_logged "01_build_fake_cuda"  gcc "${CC_FLAGS[@]}" -Wl,-soname,libcuda.so.1   -o /tmp/shims/libcuda.so.1   /repo/scripts/shims/fake_cuda.c
run_logged "02_build_fake_nvrtc" gcc "${CC_FLAGS[@]}" -Wl,-soname,libnvrtc.so.12 -o /tmp/shims/libnvrtc.so.12 /repo/scripts/shims/fake_nvrtc.c

run_cli_json_expect 0 "03_cuda_list_json" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_VRAM_MIB=64 \
  python3 -m src.cli list --json
assert_empty_stderr "03_cuda_list_json"
run_logged "03_cuda_list_assert" python3 -c '
import json, sys
devs = json.load(open(sys.argv[1], encoding="utf-8"))["devices"]
assert len(devs) == 2 and all(d["active_backend"] == "cuda" for d in devs), devs
print("OK CUDA list:", [(d["index"], d["name"], d["pci_bdf"]) for d in devs])
' /logs/03_cuda_list_json.stdout.log

run_cli_json_expect 0 "04_cuda_info_json" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_VRAM_MIB=64 \
  python3 -m src.cli info --json
assert_empty_stderr "04_cuda_info_json"
run_logged "04_cuda_info_assert" python3 -c '
import json, sys
devs = json.load(open(sys.argv[1], encoding="utf-8"))["devices"]
assert devs[0]["name"] == "Fake CUDA GPU 0" and devs[0]["architecture"] == "sm_89", devs[0]
assert devs[0]["pci_bdf"] == "0000:01:00.0" and devs[1]["pci_bdf"] == "0000:02:00.0", devs
print("OK CUDA info:", devs[0]["architecture"], devs[0]["driver_version"])
' /logs/04_cuda_info_json.stdout.log

run_logged "05_nvrtc_compile_and_module_leak_check" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
  python3 -c '
from src.backends.cuda_backend import CudaBackend
from src.embed_binary import compile_kernel_runtime
b = compile_kernel_runtime("cuda", "compute-fp32", "sm_89")
assert b.format == "ptx" and b"kernel_fake" in b.binary_bytes, b
cb = CudaBackend()
ctx = cb.create_context(cb.enumerate_devices()[0])
try:
    mod = cb.load_module(ctx, b)
    cb.unload_module(ctx, mod)
finally:
    cb.destroy_context(ctx)
print("OK NVRTC compile + cuModuleLoadData/Unload (zero leaks)")
'

run_cli_json_expect 0 "06_cuda_pass_leak_check" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
  python3 -m src.cli run --backend cuda --device all --test bandwidth-h2d,bandwidth-d2h --duration "${RUN_DURATION}" --json
assert_empty_stderr "06_cuda_pass_leak_check"
run_logged "06_cuda_pass_assert" python3 -c '
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
assert doc["exit_code"] == 0 and len(doc["results"]) == 4, doc
assert all(r["status"] == "passed" for r in doc["results"]), doc["results"]
print("OK CUDA 2-GPU H2D/D2H pass (4/4, zero leaks)")
' /logs/06_cuda_pass_leak_check.stdout.log

run_cli_json_expect 3 "07_cuda_bitflip_rc3" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_GPU_FAULT=bitflip FAKE_GPU_FAULT_BACKEND=cuda FAKE_GPU_FAULT_DEVICE=0 \
  python3 -m src.cli run --backend cuda --device 0 --test bandwidth-d2h --duration "${RUN_DURATION}" --json
assert_empty_stderr "07_cuda_bitflip_rc3"
run_logged "07_cuda_bitflip_json" python3 -m json.tool /logs/07_cuda_bitflip_rc3.stdout.log

run_cli_json_expect 4 "08_cuda_ecc_ue_rc4" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_GPU_FAULT=ecc_ue FAKE_GPU_FAULT_BACKEND=cuda FAKE_GPU_FAULT_DEVICE=0 \
  python3 -m src.cli run --backend cuda --device 0 --test bandwidth-h2d --duration "${RUN_DURATION}" --json
assert_empty_stderr "08_cuda_ecc_ue_rc4"
run_logged "08_cuda_ecc_ue_json" python3 -m json.tool /logs/08_cuda_ecc_ue_rc4.stdout.log
EOF
  } > "${script_file}"

  local rc=0
  run_docker_script "backend-cuda" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
    -e "RUN_DURATION=${DURATION}" \
    debian:trixie-slim bash -euo pipefail || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

run_target_backend_hip() {
  local log_dir="$1"
  local script_file
  script_file="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 gcc libc6-dev'

cp -a /repo /tmp/work
cd /tmp/work
mkdir -p /tmp/shims

CC_FLAGS=(-std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC -I/repo/scripts/shims)
run_logged "01_build_fake_hip"    gcc "${CC_FLAGS[@]}" -Wl,-soname,libamdhip64.so.6 -o /tmp/shims/libamdhip64.so.6 /repo/scripts/shims/fake_hip.c
run_logged "02_build_fake_hiprtc" gcc "${CC_FLAGS[@]}" -Wl,-soname,libhiprtc.so.6   -o /tmp/shims/libhiprtc.so.6   /repo/scripts/shims/fake_hiprtc.c

run_cli_json_expect 0 "03_hip_info_json" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_VRAM_MIB=64 FAKE_HIP_COUNT=2 \
  python3 -m src.cli info --json
assert_empty_stderr "03_hip_info_json"
run_logged "03_hip_info_assert" python3 -c '
import json, sys
devs = json.load(open(sys.argv[1], encoding="utf-8"))["devices"]
assert len(devs) == 2 and all(d["active_backend"] == "hip" for d in devs), devs
assert devs[0]["name"] == "Fake AMD HIP GPU 0" and devs[0]["architecture"] == "gfx1100", devs[0]
assert devs[0]["pci_bdf"] == "0000:03:00.0" and devs[0]["capabilities"]["bf16"] is True, devs[0]
print("OK HIP info:", devs[0]["name"], devs[0]["architecture"], devs[0]["pci_bdf"])
' /logs/03_hip_info_json.stdout.log

run_logged "04_hiprtc_compile_and_module_leak_check" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
  python3 -c '
from src.backends.hip_backend import HipBackend
from src.embed_binary import compile_kernel_runtime
b = compile_kernel_runtime("hip", "compute-fp32", "gfx1100")
assert b.format == "hsaco" and b.binary_bytes[:4] == b"\x7fELF", b
hb = HipBackend()
ctx = hb.create_context(hb.enumerate_devices()[0])
try:
    mod = hb.load_module(ctx, b)
    hb.unload_module(ctx, mod)
finally:
    hb.destroy_context(ctx)
print("OK HIPRTC compile + hipModuleLoadData/Unload (zero leaks)")
'

run_cli_json_expect 0 "05_hip_pass_leak_check" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 FAKE_HIP_COUNT=2 \
  python3 -m src.cli run --backend hip --device all --test bandwidth-h2d,bandwidth-d2h --duration "${RUN_DURATION}" --json
assert_empty_stderr "05_hip_pass_leak_check"
run_logged "05_hip_pass_assert" python3 -c '
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
assert doc["exit_code"] == 0 and len(doc["results"]) == 4, doc
assert all(r["status"] == "passed" for r in doc["results"]), doc["results"]
print("OK HIP 2-GPU H2D/D2H pass (4/4, zero leaks)")
' /logs/05_hip_pass_leak_check.stdout.log

run_cli_json_expect 3 "06_hip_bitflip_rc3" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_GPU_FAULT=bitflip FAKE_GPU_FAULT_BACKEND=hip FAKE_GPU_FAULT_DEVICE=0 \
  python3 -m src.cli run --backend hip --device 0 --test bandwidth-d2h --duration "${RUN_DURATION}" --json
assert_empty_stderr "06_hip_bitflip_rc3"
run_logged "06_hip_bitflip_json" python3 -m json.tool /logs/06_hip_bitflip_rc3.stdout.log

run_cli_json_expect 4 "07_hip_ecc_ue_rc4" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_GPU_FAULT=ecc_ue FAKE_GPU_FAULT_BACKEND=hip FAKE_GPU_FAULT_DEVICE=0 \
  python3 -m src.cli run --backend hip --device 0 --test bandwidth-h2d --duration "${RUN_DURATION}" --json
assert_empty_stderr "07_hip_ecc_ue_rc4"
run_logged "07_hip_ecc_ue_json" python3 -m json.tool /logs/07_hip_ecc_ue_rc4.stdout.log
EOF
  } > "${script_file}"

  local rc=0
  run_docker_script "backend-hip" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
    -e "RUN_DURATION=${DURATION}" \
    debian:trixie-slim bash -euo pipefail || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

run_target_backend_levelzero() {
  local log_dir="$1"
  local script_file
  script_file="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 gcc libc6-dev'

cp -a /repo /tmp/work
cd /tmp/work
mkdir -p /tmp/shims

CC_FLAGS=(-std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC -I/repo/scripts/shims)
run_logged "01_build_fake_levelzero" gcc "${CC_FLAGS[@]}" -Wl,-soname,libze_loader.so.1 -o /tmp/shims/libze_loader.so.1 /repo/scripts/shims/fake_levelzero.c

run_cli_json_expect 0 "02_l0_info_json" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_VRAM_MIB=64 FAKE_L0_COUNT=2 \
  python3 -m src.cli info --json
assert_empty_stderr "02_l0_info_json"
run_logged "02_l0_info_assert" python3 -c '
import json, sys
devs = json.load(open(sys.argv[1], encoding="utf-8"))["devices"]
assert len(devs) == 2 and all(d["active_backend"] == "levelzero" for d in devs), devs
assert devs[0]["name"] == "Fake Intel Level Zero GPU 0", devs[0]
assert devs[0]["pci_bdf"] == "0000:04:00.0" and devs[0]["vendor"] == "Intel", devs[0]
print("OK Level Zero info:", devs[0]["name"], devs[0]["pci_bdf"], devs[0]["compute_capability"])
' /logs/02_l0_info_json.stdout.log

run_logged "03_l0_spirv_module_lifecycle_leak_check" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
  python3 -c '
import ctypes
from src.backends.levelzero_backend import LevelZeroBackend
from src.embed_binary import CompiledKernelBinary
lb = LevelZeroBackend()
ctx = lb.create_context(lb.enumerate_devices()[0])
try:
    b = CompiledKernelBinary(
        backend="levelzero", workload="compute_fp32", entry_point="kernel_compute_fp32",
        format="spirv", binary_bytes=b"\x03\x02\x23\x07\x00\x00\x01\x00",
        compile_flags=("-DCROSS_GPU_HAS_FP32=1", "-ze-opt-large-register-file"),
    )
    mod = lb.load_module(ctx, b)
    dbuf = lb.alloc_device(ctx, 256)
    try:
        lb.launch_kernel(ctx, mod, "kernel_compute_fp32", (2, 1, 1), (32, 1, 1), [dbuf, ctypes.c_uint32(64)])
        lb.synchronize(ctx)
    finally:
        lb.free_device(ctx, dbuf)
        lb.unload_module(ctx, mod)
finally:
    lb.destroy_context(ctx)
print("OK Level Zero SPIR-V load/launch/unload (zero leaks)")
'

run_cli_json_expect 0 "04_l0_pass_leak_check" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 FAKE_L0_COUNT=2 \
  python3 -m src.cli run --backend levelzero --device all --test bandwidth-h2d,bandwidth-d2h --duration "${RUN_DURATION}" --json
assert_empty_stderr "04_l0_pass_leak_check"
run_logged "04_l0_pass_assert" python3 -c '
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
assert doc["exit_code"] == 0 and len(doc["results"]) == 4, doc
assert all(r["status"] == "passed" for r in doc["results"]), doc["results"]
print("OK Level Zero 2-GPU H2D/D2H pass (4/4, zero leaks)")
' /logs/04_l0_pass_leak_check.stdout.log

run_cli_json_expect 3 "05_l0_bitflip_rc3" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_GPU_FAULT=bitflip FAKE_GPU_FAULT_BACKEND=levelzero FAKE_GPU_FAULT_DEVICE=0 \
  python3 -m src.cli run --backend levelzero --device 0 --test bandwidth-d2h --duration "${RUN_DURATION}" --json
assert_empty_stderr "05_l0_bitflip_rc3"
run_logged "05_l0_bitflip_json" python3 -m json.tool /logs/05_l0_bitflip_rc3.stdout.log

run_cli_json_expect 4 "06_l0_device_lost_rc4" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_GPU_FAULT=device_lost FAKE_GPU_FAULT_BACKEND=levelzero FAKE_GPU_FAULT_DEVICE=0 \
  python3 -m src.cli run --backend levelzero --device 0 --test bandwidth-h2d --duration "${RUN_DURATION}" --json
assert_empty_stderr "06_l0_device_lost_rc4"
run_logged "06_l0_device_lost_json" python3 -m json.tool /logs/06_l0_device_lost_rc4.stdout.log
EOF
  } > "${script_file}"

  local rc=0
  run_docker_script "backend-levelzero" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
    -e "RUN_DURATION=${DURATION}" \
    debian:trixie-slim bash -euo pipefail || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

run_target_backend_opencl() {
  local log_dir="$1"
  local script_file
  script_file="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 gcc libc6-dev'

cp -a /repo /tmp/work
cd /tmp/work
mkdir -p /tmp/shims

CC_FLAGS=(-std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC -I/repo/scripts/shims)
run_logged "01_build_fake_opencl" gcc "${CC_FLAGS[@]}" -Wl,-soname,libOpenCL.so.1 -o /tmp/shims/libOpenCL.so.1 /repo/scripts/shims/fake_opencl.c

run_cli_json_expect 0 "02_opencl_info_json" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_VRAM_MIB=64 FAKE_CL_COUNT=2 \
  python3 -m src.cli info --json
assert_empty_stderr "02_opencl_info_json"
run_logged "02_opencl_info_assert" python3 -c '
import json, sys
devs = json.load(open(sys.argv[1], encoding="utf-8"))["devices"]
assert len(devs) == 2 and all(d["active_backend"] == "opencl" for d in devs), devs
assert devs[0]["name"] == "Fake OpenCL GPU 0" and devs[0]["pci_bdf"] == "0000:01:00.0", devs[0]
assert devs[0]["capabilities"]["fp64"] is True, devs[0]
print("OK OpenCL info:", devs[0]["name"], devs[0]["pci_bdf"], devs[0]["compute_capability"])
' /logs/02_opencl_info_json.stdout.log

run_logged "03_opencl_program_kernel_lifecycle_leak_check" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 FAKE_CL_COUNT=1 \
  python3 -c '
import ctypes
from src.backends.opencl_backend import OpenCLBackend
from src.embed_binary import CompiledKernelBinary
ob = OpenCLBackend()
ctx = ob.create_context(ob.enumerate_devices()[0])
try:
    b = CompiledKernelBinary(
        backend="opencl", workload="compute_fp32", entry_point="kernel_compute_fp32",
        format="opencl_source", binary_bytes=b"__kernel void kernel_compute_fp32(__global float* p, uint n) {}",
    )
    mod = ob.load_module(ctx, b)
    extracted = ob.extract_program_binary(mod)
    assert extracted.startswith(b"CLBIN"), extracted
    dbuf = ob.alloc_device(ctx, 256)
    try:
        ob.launch_kernel(ctx, mod, "kernel_compute_fp32", (2, 1, 1), (32, 1, 1), [dbuf, ctypes.c_uint32(64)])
        ob.synchronize(ctx)
    finally:
        ob.free_device(ctx, dbuf)
        ob.unload_module(ctx, mod)
finally:
    ob.destroy_context(ctx)
print("OK OpenCL program build/binary-extract/launch/release (zero leaks)")
'

run_cli_json_expect 0 "04_opencl_pass_leak_check" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 FAKE_CL_COUNT=2 \
  python3 -m src.cli run --backend opencl --device all --test bandwidth-h2d,bandwidth-d2h --duration "${RUN_DURATION}" --json
assert_empty_stderr "04_opencl_pass_leak_check"
run_logged "04_opencl_pass_assert" python3 -c '
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
assert doc["exit_code"] == 0 and len(doc["results"]) == 4, doc
assert all(r["status"] == "passed" for r in doc["results"]), doc["results"]
print("OK OpenCL 2-GPU H2D/D2H pass (4/4, zero leaks)")
' /logs/04_opencl_pass_leak_check.stdout.log

run_cli_json_expect 3 "05_opencl_bitflip_rc3" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 FAKE_CL_COUNT=2 \
      FAKE_GPU_FAULT=bitflip FAKE_GPU_FAULT_BACKEND=opencl FAKE_GPU_FAULT_DEVICE=0 \
  python3 -m src.cli run --backend opencl --device 0 --test bandwidth-d2h --duration "${RUN_DURATION}" --json
assert_empty_stderr "05_opencl_bitflip_rc3"
run_logged "05_opencl_bitflip_json" python3 -m json.tool /logs/05_opencl_bitflip_rc3.stdout.log

run_cli_json_expect 4 "06_opencl_device_lost_rc4" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 FAKE_CL_COUNT=2 \
      FAKE_GPU_FAULT=device_lost FAKE_GPU_FAULT_BACKEND=opencl FAKE_GPU_FAULT_DEVICE=0 \
  python3 -m src.cli run --backend opencl --device 0 --test bandwidth-h2d --duration "${RUN_DURATION}" --json
assert_empty_stderr "06_opencl_device_lost_rc4"
run_logged "06_opencl_device_lost_json" python3 -m json.tool /logs/06_opencl_device_lost_rc4.stdout.log
EOF
  } > "${script_file}"

  local rc=0
  run_docker_script "backend-opencl" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
    -e "RUN_DURATION=${DURATION}" \
    debian:trixie-slim bash -euo pipefail || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

run_target_backend_metal() {
  local log_dir="$1"
  local script_file
  script_file="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 gcc libc6-dev'

cp -a /repo /tmp/work
cd /tmp/work
mkdir -p /tmp/shims

CC_FLAGS=(-std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC -I/repo/scripts/shims)
run_logged "01_build_fake_metal" gcc "${CC_FLAGS[@]}" -o /tmp/shims/libfake_metal.so /repo/scripts/shims/fake_metal.c

# 1. Verify production Linux platform guard rejects Metal cleanly
run_logged "02_metal_linux_guard" python3 -c '
from src.backends.metal_backend import MetalBackend
mb = MetalBackend()
assert mb.is_available() is False
assert mb.load_error == "macOS only", mb.load_error
print("OK Metal Linux platform guard:", mb.load_error)
'

# 2. Exercise full MetalBackend Objective-C objc_msgSend bridge + 24B _MTLSize dispatch + zero leaks
run_logged "03_metal_list_info_and_dispatch_leak_check" \
  env FAKE_METAL_LIB=/tmp/shims/libfake_metal.so CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
  python3 -c '
import ctypes, io, json, os
from unittest import mock
import src.backends.metal_backend as mb
from src.backends.opencl_backend import OpenCLBackend
from src.cli import main
from src.embed_binary import CompiledKernelBinary

lib_path = os.environ["FAKE_METAL_LIB"]
with (
    mock.patch.object(mb.sys, "platform", "darwin"),
    mock.patch.object(mb, "LIBOBJC_PATH", lib_path),
    mock.patch.object(mb, "METAL_FRAMEWORK_PATH", lib_path),
    mock.patch.object(mb, "LIBSYSTEM_PATH", lib_path),
    mock.patch.object(OpenCLBackend, "is_available", return_value=False),
):
    out, err = io.StringIO(), io.StringIO()
    rc = main(["list", "--json"], stdout=out, stderr=err)
    assert rc == 0 and err.getvalue() == "", (rc, err.getvalue())
    devs = json.loads(out.getvalue())["devices"]
    assert len(devs) == 2 and all(d["active_backend"] == "metal" for d in devs), devs

    out_i, err_i = io.StringIO(), io.StringIO()
    rc_i = main(["info", "--json"], stdout=out_i, stderr=err_i)
    assert rc_i == 0 and err_i.getvalue() == "", (rc_i, err_i.getvalue())
    idevs = json.loads(out_i.getvalue())["devices"]
    assert idevs[0]["name"] == "Fake Apple Metal GPU 0", idevs[0]
    assert idevs[0]["architecture"] == "applegpu_g15g", idevs[0]
    assert idevs[0]["registry_id"] == 0x1000003E8, idevs[0]
    assert idevs[0]["capabilities"]["bf16"] is True, idevs[0]
    assert idevs[0]["capabilities"]["tensor_fp16"] is True, idevs[0]
    assert idevs[0]["capabilities"]["fp64"] is False, idevs[0]

    backend = mb.MetalBackend()
    bdevs = backend.enumerate_devices()
    ctx = backend.create_context(bdevs[0])
    try:
        for fmt, payload in (("metallib", b"MTLB\x00\x00\x00\x00"), ("msl_source", b"kernel void k() {}")):
            mod = backend.load_module(
                ctx,
                CompiledKernelBinary(backend="metal", workload="compute_fp32", entry_point="k", format=fmt, binary_bytes=payload),
            )
            dbuf = backend.alloc_device(ctx, 256)
            try:
                backend.launch_kernel(ctx, mod, "k", (4, 1, 1), (32, 1, 1), [dbuf, ctypes.c_uint32(128)], shared_mem=64)
                backend.synchronize(ctx)
            finally:
                backend.free_device(ctx, dbuf)
                backend.unload_module(ctx, mod)
    finally:
        backend.destroy_context(ctx)

    shim = ctypes.CDLL(lib_path)
    shim.fake_metal_dispatch_count.restype = ctypes.c_uint64
    assert shim.fake_metal_dispatch_count() == 2
    print("OK Metal ObjC ABI bridge: list/info + metallib/msl_source + 24B _MTLSize dispatch (zero leaks)")
'

# 3. 2-GPU Metal H2D/D2H blit pass + zero leaks
run_logged_expect 0 "04_metal_2gpu_pass_leak_check" \
  env FAKE_METAL_LIB=/tmp/shims/libfake_metal.so CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
  python3 -c '
import io, json, os, sys
from unittest import mock
import src.backends.metal_backend as mb
from src.backends.opencl_backend import OpenCLBackend
from src.cli import main

lib_path = os.environ["FAKE_METAL_LIB"]
with (
    mock.patch.object(mb.sys, "platform", "darwin"),
    mock.patch.object(mb, "LIBOBJC_PATH", lib_path),
    mock.patch.object(mb, "METAL_FRAMEWORK_PATH", lib_path),
    mock.patch.object(mb, "LIBSYSTEM_PATH", lib_path),
    mock.patch.object(OpenCLBackend, "is_available", return_value=False),
):
    out, err = io.StringIO(), io.StringIO()
    rc = main(["run", "--device", "all", "--backend", "metal", "--test", "bandwidth-h2d,bandwidth-d2h", "--duration", os.environ["RUN_DURATION"], "--json"], stdout=out, stderr=err)
    assert err.getvalue() == "", err.getvalue()
    doc = json.loads(out.getvalue())
    assert len(doc["results"]) == 4 and all(r["status"] == "passed" for r in doc["results"]), doc
    print("OK Metal 2-GPU H2D/D2H pass (4/4, zero leaks)")
    sys.exit(int(rc))
'

# 4. Targeted Metal GPU 0 D2H bitflip -> COMPUTE_ERROR (3) + zero leaks
run_logged_expect 3 "05_metal_bitflip_rc3" \
  env FAKE_METAL_LIB=/tmp/shims/libfake_metal.so CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_GPU_FAULT=bitflip FAKE_GPU_FAULT_BACKEND=metal FAKE_GPU_FAULT_DEVICE=0 \
  python3 -c '
import io, json, os, sys
from unittest import mock
import src.backends.metal_backend as mb
from src.backends.opencl_backend import OpenCLBackend
from src.cli import main

lib_path = os.environ["FAKE_METAL_LIB"]
with (
    mock.patch.object(mb.sys, "platform", "darwin"),
    mock.patch.object(mb, "LIBOBJC_PATH", lib_path),
    mock.patch.object(mb, "METAL_FRAMEWORK_PATH", lib_path),
    mock.patch.object(mb, "LIBSYSTEM_PATH", lib_path),
    mock.patch.object(OpenCLBackend, "is_available", return_value=False),
):
    out, err = io.StringIO(), io.StringIO()
    rc = main(["run", "--device", "all", "--backend", "metal", "--test", "bandwidth-d2h", "--duration", os.environ["RUN_DURATION"], "--json"], stdout=out, stderr=err)
    assert err.getvalue() == "", err.getvalue()
    doc = json.loads(out.getvalue())
    failed = [r["device"] for r in doc["results"] if r["status"] == "failed"]
    passed = [r["device"] for r in doc["results"] if r["status"] == "passed"]
    assert failed == [0] and passed == [1], doc["results"]
    sys.exit(int(rc))
'

# 5. Targeted Metal GPU 1 MTLCommandBufferErrorTimeout -> HARDWARE_ERROR (4) + zero leaks
run_logged_expect 4 "06_metal_timeout_rc4" \
  env FAKE_METAL_LIB=/tmp/shims/libfake_metal.so CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_GPU_FAULT=gpu_timeout FAKE_GPU_FAULT_BACKEND=metal FAKE_GPU_FAULT_DEVICE=1 \
  python3 -c '
import io, json, os, sys
from unittest import mock
import src.backends.metal_backend as mb
from src.backends.opencl_backend import OpenCLBackend
from src.cli import main

lib_path = os.environ["FAKE_METAL_LIB"]
with (
    mock.patch.object(mb.sys, "platform", "darwin"),
    mock.patch.object(mb, "LIBOBJC_PATH", lib_path),
    mock.patch.object(mb, "METAL_FRAMEWORK_PATH", lib_path),
    mock.patch.object(mb, "LIBSYSTEM_PATH", lib_path),
    mock.patch.object(OpenCLBackend, "is_available", return_value=False),
):
    out, err = io.StringIO(), io.StringIO()
    rc = main(["run", "--device", "all", "--backend", "metal", "--test", "bandwidth-h2d", "--duration", os.environ["RUN_DURATION"], "--json"], stdout=out, stderr=err)
    assert err.getvalue() == "", err.getvalue()
    doc = json.loads(out.getvalue())
    hw = [r["device"] for r in doc["results"] if r["status"] == "hardware_error"]
    assert hw == [1], doc["results"]
    sys.exit(int(rc))
'
EOF
  } > "${script_file}"

  local rc=0
  run_docker_script "backend-metal" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
    -e "RUN_DURATION=${DURATION}" \
    debian:trixie-slim bash -euo pipefail || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

run_target_backend_multigpu() {
  local log_dir="$1"
  local script_file
  script_file="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 gcc libc6-dev'

cp -a /repo /tmp/work
cd /tmp/work
mkdir -p /tmp/shims

CC_FLAGS=(-std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC -I/repo/scripts/shims)
run_logged "01_build_fake_cuda"      gcc "${CC_FLAGS[@]}" -Wl,-soname,libcuda.so.1      -o /tmp/shims/libcuda.so.1      /repo/scripts/shims/fake_cuda.c
run_logged "02_build_fake_nvrtc"     gcc "${CC_FLAGS[@]}" -Wl,-soname,libnvrtc.so.12    -o /tmp/shims/libnvrtc.so.12    /repo/scripts/shims/fake_nvrtc.c
run_logged "03_build_fake_hip"       gcc "${CC_FLAGS[@]}" -Wl,-soname,libamdhip64.so.6  -o /tmp/shims/libamdhip64.so.6  /repo/scripts/shims/fake_hip.c
run_logged "04_build_fake_hiprtc"    gcc "${CC_FLAGS[@]}" -Wl,-soname,libhiprtc.so.6    -o /tmp/shims/libhiprtc.so.6    /repo/scripts/shims/fake_hiprtc.c
run_logged "05_build_fake_levelzero" gcc "${CC_FLAGS[@]}" -Wl,-soname,libze_loader.so.1 -o /tmp/shims/libze_loader.so.1 /repo/scripts/shims/fake_levelzero.c
run_logged "06_build_fake_opencl"    gcc "${CC_FLAGS[@]}" -Wl,-soname,libOpenCL.so.1    -o /tmp/shims/libOpenCL.so.1    /repo/scripts/shims/fake_opencl.c

# 1. Heterogeneous 5-GPU dedup (9 raw handles -> 5 physical GPUs)
run_cli_json_expect 0 "07_5gpu_hetero_list" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_VRAM_MIB=64 \
  python3 -m src.cli list --json
assert_empty_stderr "07_5gpu_hetero_list"
run_logged "07_assert_5gpu_dedup" python3 -c '
import collections, json, sys
devs = json.load(open(sys.argv[1], encoding="utf-8"))["devices"]
assert len(devs) == 5, f"Expected 5 deduplicated GPUs, got {len(devs)}: {devs}"
counts = dict(collections.Counter(d["active_backend"] for d in devs))
expected = {"cuda": 2, "hip": 1, "levelzero": 1, "opencl": 1}
assert counts == expected, f"Dedup mismatch: got {counts}, expected {expected}"
for d in devs:
    if d["active_backend"] != "opencl":
        assert d["available_backends"] == [d["active_backend"], "opencl"], d
print("OK 9->5 dedup:", counts)
' /logs/07_5gpu_hetero_list.stdout.log

# 2. Homogeneous 5-GPU CUDA rig
run_cli_json_expect 0 "08_5gpu_cuda_only_list" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_CUDA_COUNT=5 FAKE_HIP_COUNT=0 FAKE_L0_COUNT=0 FAKE_CL_COUNT=0 \
  python3 -m src.cli list --json
assert_empty_stderr "08_5gpu_cuda_only_list"
run_logged "08_assert_5gpu_cuda_only" python3 -c '
import json, sys
devs = json.load(open(sys.argv[1], encoding="utf-8"))["devices"]
assert len(devs) == 5 and all(d["active_backend"] == "cuda" for d in devs), devs
print("OK 5 homogeneous CUDA GPUs")
' /logs/08_5gpu_cuda_only_list.stdout.log

# 3. Concurrent 5-GPU pass + destructor leak check across all 4 backends
run_cli_json_expect 0 "09_5gpu_all_pass_leak_check" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
  python3 -m src.cli run --device all --test bandwidth-h2d,bandwidth-d2h --duration "${RUN_DURATION}" --json
assert_empty_stderr "09_5gpu_all_pass_leak_check"
run_logged "09_assert_5gpu_all_pass" python3 -c '
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(doc["results"]) == 10 and all(r["status"] == "passed" for r in doc["results"]), doc
print("OK 5-GPU concurrent H2D/D2H pass (10/10, zero leaks)")
' /logs/09_5gpu_all_pass_leak_check.stdout.log

# 4. Isolated fault injection across HIP (3), Level Zero (4), and OpenCL (3)
run_cli_json_expect 3 "10_5gpu_hip_bitflip_rc3" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_GPU_FAULT=bitflip FAKE_GPU_FAULT_BACKEND=hip \
  python3 -m src.cli run --device all --test bandwidth-d2h --duration "${RUN_DURATION}" --json
assert_empty_stderr "10_5gpu_hip_bitflip_rc3"
run_logged "10_assert_hip_bitflip" python3 -c '
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
failed = [r["backend"] for r in doc["results"] if r["status"] == "failed"]
passed = [r["backend"] for r in doc["results"] if r["status"] == "passed"]
assert failed == ["hip"] and len(passed) == 4, doc["results"]
print("OK isolated HIP bitflip (rc=3, 4 peers passed)")
' /logs/10_5gpu_hip_bitflip_rc3.stdout.log

run_cli_json_expect 4 "11_5gpu_l0_device_lost_rc4" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_GPU_FAULT=device_lost FAKE_GPU_FAULT_BACKEND=levelzero \
  python3 -m src.cli run --device all --test bandwidth-h2d --duration "${RUN_DURATION}" --json
assert_empty_stderr "11_5gpu_l0_device_lost_rc4"
run_logged "11_assert_l0_device_lost" python3 -c '
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
hw = [r["backend"] for r in doc["results"] if r["status"] == "hardware_error"]
assert hw == ["levelzero"], doc["results"]
print("OK isolated Level Zero device_lost (rc=4)")
' /logs/11_5gpu_l0_device_lost_rc4.stdout.log

run_cli_json_expect 3 "12_5gpu_opencl_bitflip_rc3" \
  env LD_LIBRARY_PATH=/tmp/shims CROSS_GPU_NO_CACHE=1 FAKE_GPU_LEAK_CHECK=1 FAKE_GPU_VRAM_MIB=64 \
      FAKE_GPU_FAULT=bitflip FAKE_GPU_FAULT_BACKEND=opencl \
  python3 -m src.cli run --device all --test bandwidth-d2h --duration "${RUN_DURATION}" --json
assert_empty_stderr "12_5gpu_opencl_bitflip_rc3"
run_logged "12_assert_opencl_bitflip" python3 -c '
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
failed = [r["backend"] for r in doc["results"] if r["status"] == "failed"]
passed = [r["backend"] for r in doc["results"] if r["status"] == "passed"]
assert failed == ["opencl"] and len(passed) == 4, doc["results"]
print("OK isolated OpenCL bitflip (rc=3, 4 peers passed)")
' /logs/12_5gpu_opencl_bitflip_rc3.stdout.log
EOF
  } > "${script_file}"

  local rc=0
  run_docker_script "backend-multigpu" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
    -e "RUN_DURATION=${DURATION}" \
    debian:trixie-slim bash -euo pipefail || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

run_target_backend_pocl() {
  local log_dir="$1"
  local script_file
  script_file="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
export DEBIAN_FRONTEND=noninteractive
run_logged "00_apt_install_pocl" bash -euo pipefail -c \
  'apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 ocl-icd-libopencl1 pocl-opencl-icd clinfo'

cp -a /repo /tmp/work
cd /tmp/work

run_logged "01_clinfo" clinfo

# 1. Verify production CrossGPU refuses silent CPU fallback when only PoCL CPU device exists
run_cli_json_expect 0 "02_list_no_gpu" python3 -m src.cli list --json
assert_empty_stderr "02_list_no_gpu"
run_logged "02_assert_zero_gpu" python3 -c '
import json, sys
devs = json.load(open(sys.argv[1], encoding="utf-8")).get("devices", [])
assert devs == [], f"Production mode must not silently expose PoCL CPU as GPU: {devs}"
print("OK: 0 GPU devices exposed on PoCL-only host")
' /logs/02_list_no_gpu.stdout.log

run_cli_json_expect 2 "03_run_no_gpu_init_error" \
  python3 -m src.cli run --backend opencl --test compute-fp32 --duration 100ms --json
assert_empty_stderr "03_run_no_gpu_init_error"
run_logged "03_assert_init_error_json" python3 -c '
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
assert doc.get("exit_code") == 2, doc
print("OK: single-doc JSON emitted with exit_code=2")
' /logs/03_run_no_gpu_init_error.stdout.log

# 2. Compile all 36 packaged .cl kernels through real PoCL LLVM and execute golden math checks
run_logged "04_compile_and_exec_opencl_kernels" python3 - <<'PY'
import ctypes
import pathlib
import struct
import sys

from src.workloads.common import golden_fp32_table, golden_int32

cl = ctypes.CDLL("libOpenCL.so.1")
c_int, c_uint, c_size_t, c_void_p, c_char_p = (
    ctypes.c_int, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_char_p
)

cl.clGetPlatformIDs.argtypes = [c_uint, ctypes.POINTER(c_void_p), ctypes.POINTER(c_uint)]
cl.clGetPlatformIDs.restype = c_int
cl.clGetDeviceIDs.argtypes = [c_void_p, ctypes.c_uint64, c_uint, ctypes.POINTER(c_void_p), ctypes.POINTER(c_uint)]
cl.clGetDeviceIDs.restype = c_int
cl.clGetDeviceInfo.argtypes = [c_void_p, c_uint, c_size_t, c_void_p, ctypes.POINTER(c_size_t)]
cl.clGetDeviceInfo.restype = c_int
cl.clCreateContext.argtypes = [c_void_p, c_uint, ctypes.POINTER(c_void_p), c_void_p, c_void_p, ctypes.POINTER(c_int)]
cl.clCreateContext.restype = c_void_p
cl.clCreateCommandQueue.argtypes = [c_void_p, c_void_p, ctypes.c_uint64, ctypes.POINTER(c_int)]
cl.clCreateCommandQueue.restype = c_void_p
cl.clReleaseCommandQueue.argtypes = [c_void_p]
cl.clReleaseCommandQueue.restype = c_int
cl.clCreateBuffer.argtypes = [c_void_p, ctypes.c_uint64, c_size_t, c_void_p, ctypes.POINTER(c_int)]
cl.clCreateBuffer.restype = c_void_p
cl.clReleaseMemObject.argtypes = [c_void_p]
cl.clReleaseMemObject.restype = c_int
cl.clCreateProgramWithSource.argtypes = [c_void_p, c_uint, ctypes.POINTER(c_char_p), ctypes.POINTER(c_size_t), ctypes.POINTER(c_int)]
cl.clCreateProgramWithSource.restype = c_void_p
cl.clBuildProgram.argtypes = [c_void_p, c_uint, ctypes.POINTER(c_void_p), c_char_p, c_void_p, c_void_p]
cl.clBuildProgram.restype = c_int
cl.clGetProgramBuildInfo.argtypes = [c_void_p, c_void_p, c_uint, c_size_t, c_void_p, ctypes.POINTER(c_size_t)]
cl.clGetProgramBuildInfo.restype = c_int
cl.clCreateKernel.argtypes = [c_void_p, c_char_p, ctypes.POINTER(c_int)]
cl.clCreateKernel.restype = c_void_p
cl.clReleaseKernel.argtypes = [c_void_p]
cl.clReleaseKernel.restype = c_int
cl.clSetKernelArg.argtypes = [c_void_p, c_uint, c_size_t, c_void_p]
cl.clSetKernelArg.restype = c_int
cl.clEnqueueNDRangeKernel.argtypes = [
    c_void_p, c_void_p, c_uint,
    ctypes.POINTER(c_size_t), ctypes.POINTER(c_size_t), ctypes.POINTER(c_size_t),
    c_uint, ctypes.POINTER(c_void_p), ctypes.POINTER(c_void_p),
]
cl.clEnqueueNDRangeKernel.restype = c_int
cl.clEnqueueReadBuffer.argtypes = [
    c_void_p, c_void_p, c_uint, c_size_t, c_size_t, c_void_p,
    c_uint, ctypes.POINTER(c_void_p), ctypes.POINTER(c_void_p),
]
cl.clEnqueueReadBuffer.restype = c_int
cl.clFinish.argtypes = [c_void_p]
cl.clFinish.restype = c_int
cl.clReleaseProgram.argtypes = [c_void_p]
cl.clReleaseProgram.restype = c_int
cl.clReleaseContext.argtypes = [c_void_p]
cl.clReleaseContext.restype = c_int

CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
CL_DEVICE_EXTENSIONS = 0x1030
CL_PROGRAM_BUILD_LOG = 0x1183
CL_MEM_READ_WRITE = 1 << 0
CL_TRUE = 1

n_plat = c_uint(0)
assert cl.clGetPlatformIDs(0, None, ctypes.byref(n_plat)) == 0 and n_plat.value > 0
plats = (c_void_p * n_plat.value)()
assert cl.clGetPlatformIDs(n_plat.value, plats, None) == 0

dev = c_void_p()
found = False
for p in plats:
    n_dev = c_uint(0)
    if cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 1, ctypes.byref(dev), ctypes.byref(n_dev)) == 0 and n_dev.value > 0:
        found = True
        break
assert found, "No PoCL OpenCL device found"

ext_sz = c_size_t(0)
cl.clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, None, ctypes.byref(ext_sz))
ext_buf = ctypes.create_string_buffer(max(1, ext_sz.value))
cl.clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, ext_sz.value, ext_buf, None)
ext_tokens = set(ext_buf.value.decode("utf-8", errors="replace").split())
print(f"PoCL extensions={len(ext_tokens)} has_fp64={'cl_khr_fp64' in ext_tokens} has_fp16={'cl_khr_fp16' in ext_tokens}")

err = c_int(0)
ctx = cl.clCreateContext(None, 1, ctypes.byref(dev), None, None, ctypes.byref(err))
assert ctx and err.value == 0, f"clCreateContext failed: {err.value}"
queue = cl.clCreateCommandQueue(ctx, dev, 0, ctypes.byref(err))
assert queue and err.value == 0, f"clCreateCommandQueue failed: {err.value}"

kernels = sorted(pathlib.Path("src/kernels").rglob("*.cl"))
assert len(kernels) == 36, f"Expected 36 .cl kernels, got {len(kernels)}"
compiled_ok = 0
gated_ok = 0
failures = []

try:
    for kpath in kernels:
        src_bytes = kpath.read_bytes()
        src_text = src_bytes.decode("utf-8")
        has_gate = '#error "UNSUPPORTED:' in src_text
        c_src = c_char_p(src_bytes)
        c_len = c_size_t(len(src_bytes))
        prog = cl.clCreateProgramWithSource(ctx, 1, ctypes.byref(c_src), ctypes.byref(c_len), ctypes.byref(err))
        assert prog and err.value == 0
        try:
            rc = cl.clBuildProgram(prog, 1, ctypes.byref(dev), b"", None, None)
            if rc == 0:
                compiled_ok += 1
                print(f"OK   {kpath}")
            else:
                log_sz = c_size_t(0)
                cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, None, ctypes.byref(log_sz))
                buf = ctypes.create_string_buffer(max(1, log_sz.value))
                cl.clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz.value, buf, None)
                msg = buf.value.decode("utf-8", errors="replace").strip()
                if has_gate and "UNSUPPORTED:" in msg:
                    gated_ok += 1
                    line = next((ln.strip() for ln in msg.splitlines() if "UNSUPPORTED:" in ln), msg[:200])
                    print(f"GATE {kpath} -> {line}")
                else:
                    print(f"FAIL [{rc}] {kpath}:\n{msg}", file=sys.stderr)
                    failures.append((str(kpath), rc, msg[:400]))
        finally:
            cl.clReleaseProgram(prog)

    if failures:
        raise SystemExit(f"OpenCL kernel compilation failures: {failures}")
    assert compiled_ok >= 16, f"Expected >= 16 compiled .cl kernels, got {compiled_ok}"
    print(f"Verified 36 .cl kernels on PoCL: compiled={compiled_ok}, expected_gates={gated_ok}, failures=0")

    N = 64
    for kname, ep, sa, sb, is_fp32 in (
        ("compute_int32", "kernel_compute_int32", 0x12345679, 0x9E3779B9, False),
        ("compute_fp32", "kernel_compute_fp32", 0.75, 0.5, True),
    ):
        src_bytes = pathlib.Path(f"src/kernels/opencl/{kname}.cl").read_bytes()
        c_src = c_char_p(src_bytes)
        c_len = c_size_t(len(src_bytes))
        prog = cl.clCreateProgramWithSource(ctx, 1, ctypes.byref(c_src), ctypes.byref(c_len), ctypes.byref(err))
        assert prog and err.value == 0
        assert cl.clBuildProgram(prog, 1, ctypes.byref(dev), b"", None, None) == 0
        kern = cl.clCreateKernel(prog, ep.encode("utf-8"), ctypes.byref(err))
        assert kern and err.value == 0
        buf_mem = cl.clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * 4, None, ctypes.byref(err))
        assert buf_mem and err.value == 0
        try:
            mem_arg = c_void_p(buf_mem)
            n_arg = ctypes.c_uint32(N)
            it_arg = ctypes.c_uint32(2)
            sa_arg = ctypes.c_float(sa) if is_fp32 else ctypes.c_uint32(sa)
            sb_arg = ctypes.c_float(sb) if is_fp32 else ctypes.c_uint32(sb)
            for idx, arg in enumerate((mem_arg, n_arg, it_arg, sa_arg, sb_arg)):
                assert cl.clSetKernelArg(kern, idx, ctypes.sizeof(arg), ctypes.byref(arg)) == 0
            gws = (c_size_t * 1)(N)
            assert cl.clEnqueueNDRangeKernel(queue, kern, 1, None, gws, None, 0, None, None) == 0
            assert cl.clFinish(queue) == 0
            raw = (ctypes.c_ubyte * (N * 4))()
            assert cl.clEnqueueReadBuffer(queue, buf_mem, CL_TRUE, 0, N * 4, raw, 0, None, None) == 0
            if is_fp32:
                got = struct.unpack(f"<{N}f", bytes(raw))
                tbl = golden_fp32_table(2, sa, sb)
                for gid, val in enumerate(got):
                    assert abs(val - tbl[gid & 15]) <= 1e-5, (gid, val, tbl[gid & 15])
            else:
                got = struct.unpack(f"<{N}I", bytes(raw))
                exp = tuple(golden_int32(gid, 2, sa, sb) for gid in range(N))
                assert got == exp, (got[:4], exp[:4])
            print(f"EXEC PASS {kname} on PoCL (N={N}, golden verified)")
        finally:
            cl.clReleaseMemObject(buf_mem)
            cl.clReleaseKernel(kern)
            cl.clReleaseProgram(prog)
finally:
    cl.clReleaseCommandQueue(queue)
    cl.clReleaseContext(ctx)
PY
EOF
  } > "${script_file}"

  local rc=0
  run_docker_script "backend-pocl" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
    debian:trixie-slim bash -euo pipefail || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

# ---------------------------------------------------------------------------
# Suite 3: Multi-Distro Libc / Python / FHS Packaging (distros)
# ---------------------------------------------------------------------------
run_target_distro() {
  local target="$1"
  local log_dir="$2"
  local image=""
  local install_cmd=""

  case "${target}" in
    distro-alpine)
      image="alpine:latest"
      install_cmd="apk add --no-cache python3 py3-pip py3-setuptools py3-wheel py3-build bash make"
      ;;
    distro-debian)
      image="debian:trixie-slim"
      install_cmd="apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 update -qq && DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 install -y --no-install-recommends python3 python3-pip python3-setuptools python3-wheel python3-build make"
      ;;
    distro-fedora)
      image="fedora:latest"
      install_cmd="dnf install -y -q python3 python3-pip python3-setuptools python3-wheel python3-build make"
      ;;
    distro-arch)
      image="archlinux:latest"
      install_cmd="pacman --disable-sandbox -Syu --noconfirm --needed python python-pip python-setuptools python-wheel python-build make"
      ;;
    *)
      printf 'Internal error: unknown distro target %s\n' "${target}" >&2
      return 1
      ;;
  esac

  local script_file
  script_file="$(mktemp)"
  cat <<'EOF' > "${script_file}"
trap 'chmod -R a+rX /logs 2>/dev/null || true' EXIT

run_logged() {
  _name="$1"; shift
  _out="/logs/${_name}.stdout.log"
  _err="/logs/${_name}.stderr.log"
  _rc=0
  printf '%s\n' "$*" > "/logs/${_name}.cmd"
  printf '0\n' > "/logs/${_name}.expected_rc"
  "$@" >"${_out}" 2>"${_err}" || _rc=$?
  printf '%s\n' "${_rc}" > "/logs/${_name}.rc"

  if [ "${_rc}" -eq 0 ]; then
    printf '  [PASS] %s (rc=0)\n' "${_name}"
    return 0
  fi

  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    printf '::error title=%s failed (rc=%s)::Command: %s\n' "${_name}" "${_rc}" "$*"
  fi
  printf '%s\n' "================================================================"
  printf '  [FAIL] %s\n' "${_name}"
  printf '  Command : %s\n' "$*"
  printf '  Exit    : %s (expected 0)\n' "${_rc}"
  printf '%s\n' "  --- stdout (${_name}.stdout.log) ---"
  if [ -s "${_out}" ]; then tail -n 200 "${_out}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "  --- stderr (${_name}.stderr.log) ---"
  if [ -s "${_err}" ]; then tail -n 200 "${_err}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "================================================================"
  return "${_rc}"
}

run_logged "00_install_deps" sh -eu -c "${INSTALL_CMD}"

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info

run_logged "01_python_version" python3 --version
run_logged "02_compileall" python3 -m compileall -q src tests
run_logged "03_unittest" python3 -m unittest discover -s tests -v
run_logged "04_embed_verify" python3 -m src.embed_binary verify
run_logged "05_completion_syntax" bash -n completions/cross-gpu.bash
run_logged "06_offline_wheel_sdist" python3 -m build --no-isolation --outdir /tmp/dist .
run_logged "07_verify_wheel" python3 -c '
import glob, zipfile
wheels = glob.glob("/tmp/dist/cross_gpu-*.whl")
assert len(wheels) == 1, f"Expected 1 wheel, found: {wheels}"
with zipfile.ZipFile(wheels[0]) as zf:
    names = zf.namelist()
kernels = [n for n in names if n.startswith("cross_gpu/kernels/") and n.endswith((".cu", ".hip", ".cl", ".metal"))]
assert len(kernels) == 90, f"Expected 90 kernels, got {len(kernels)}"
assert not any(n.startswith("src/") or "/telemetry/" in n for n in names), names
print("OK wheel:", wheels[0], "kernels:", len(kernels), "entries:", len(names))
'
run_logged "08_make_install_stage" make install DESTDIR=/tmp/stage PREFIX=/usr
run_logged "09_verify_staged_tree" sh -eu -c '
  test -x /tmp/stage/usr/bin/cross-gpu
  test -f /tmp/stage/usr/share/man/man1/cross-gpu.1
  test -f /tmp/stage/usr/share/bash-completion/completions/cross-gpu
  test -f /tmp/stage/usr/share/zsh/site-functions/_cross-gpu
  test -f /tmp/stage/usr/share/fish/vendor_completions.d/cross-gpu.fish
  echo "OK staged FHS tree verified under /tmp/stage/usr"
'
EOF

  local rc=0
  run_docker_script "${target}" "${log_dir}" "${script_file}" \
    run -i --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${log_dir}:/logs:rw" \
    -e "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
    -e "INSTALL_CMD=${install_cmd}" \
    "${image}" sh -eu || rc=$?

  rm -f "${script_file}"
  return "${rc}"
}

# ---------------------------------------------------------------------------
# Suite 4: Adversarial Container Constraints (adversarial)
# ---------------------------------------------------------------------------
run_target_adversarial() {
  local log_dir="$1"

  local wheel_vol="${RUN_ID}-wheels"
  "${DOCKER_BIN}" volume rm -f "${wheel_vol}" >/dev/null 2>&1 || true
  "${DOCKER_BIN}" volume create "${wheel_vol}" >/dev/null

  local build_script
  build_script="$(mktemp)"
  {
    emit_container_bash_prelude
    cat <<'EOF'
trap 'chmod -R a+rX /wheels /logs 2>/dev/null || true' EXIT
run_logged "00_builder_pip_tools" python3 -m pip install --quiet --no-cache-dir setuptools wheel build

cp -a /repo /tmp/work
cd /tmp/work
rm -rf build dist ./*.egg-info
run_logged "01_builder_wheel" python3 -m build --wheel --no-isolation --outdir /wheels .
EOF
  } > "${build_script}"

  local rc=0
  run_docker_script "adversarial-builder" "${log_dir}" "${build_script}" \
    run -i --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    -v "${REPO_ROOT}:/repo:ro" \
    -v "${wheel_vol}:/wheels:rw" \
    -v "${log_dir}:/logs:rw" \
    -e "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
    python:3.12-slim bash -euo pipefail || rc=$?
  rm -f "${build_script}"

  if [[ "${rc}" -ne 0 ]]; then
    "${DOCKER_BIN}" volume rm -f "${wheel_vol}" >/dev/null 2>&1 || true
    return "${rc}"
  fi

  local adv_script
  adv_script="$(mktemp)"
  cat <<'EOF' > "${adv_script}"
run_inner() {
  local name="$1"; shift
  local out="/tmp/${name}.stdout.log"
  local err="/tmp/${name}.stderr.log"
  local irc=0
  "$@" >"${out}" 2>"${err}" || irc=$?
  if [[ "${irc}" -eq 0 ]]; then
    printf '  [PASS] %s (rc=0)\n' "${name}"
    return 0
  fi
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    printf '::error title=adversarial/%s failed (rc=%s)::Command: %s\n' "${name}" "${irc}" "$*"
  fi
  printf '%s\n' "================================================================"
  printf '  [FAIL] %s\n' "${name}"
  printf '  Command : %s\n' "$*"
  printf '  Exit    : %s (expected 0)\n' "${irc}"
  printf '%s\n' "  --- stdout (${name}) ---"
  if [[ -s "${out}" ]]; then tail -n 200 "${out}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "  --- stderr (${name}) ---"
  if [[ -s "${err}" ]]; then tail -n 200 "${err}"; else printf '%s\n' "  (empty)"; fi
  printf '%s\n' "================================================================"
  return "${irc}"
}

assert_empty_stderr() {
  local name="$1"
  if [[ -s "/tmp/${name}.stderr.log" ]]; then
    printf '  [FAIL] %s polluted stderr in --json mode:\n' "${name}" >&2
    cat "/tmp/${name}.stderr.log" >&2
    return 1
  fi
}

run_inner "01_pip_install_target" python3 -m pip install --no-index --no-deps --no-cache-dir --target /tmp/site /wheels/*.whl
export PYTHONPATH=/tmp/site
cd /tmp

run_inner "02_version" python3 -m cross_gpu --version
run_inner "03_embed_verify" python3 -m cross_gpu.embed_binary verify

run_inner "04_list_json" python3 -m cross_gpu list --json
assert_empty_stderr "04_list_json"
run_inner "04_list_json_check" python3 -m json.tool /tmp/04_list_json.stdout.log

run_inner "05_info_json" python3 -m cross_gpu info --json
assert_empty_stderr "05_info_json"
run_inner "05_info_json_check" python3 -m json.tool /tmp/05_info_json.stdout.log

run_inner "06_no_cache_info_json" env CROSS_GPU_NO_CACHE=1 python3 -m cross_gpu info --json
assert_empty_stderr "06_no_cache_info_json"

run_inner "07_site_packages_check" python3 -c '
import pathlib
sp = pathlib.Path("/tmp/site")
assert (sp / "cross_gpu").is_dir()
assert not (sp / "src").exists()
assert not (sp / "cross_gpu" / "telemetry").exists()
print("OK site-packages layout verified")
'

run_inner "08_verify_noexec_tmp" bash -euo pipefail -c '
  printf "%s\n" "#!/bin/sh" "exit 0" > /tmp/noexec_probe.sh
  chmod +x /tmp/noexec_probe.sh
  if /tmp/noexec_probe.sh 2>/dev/null; then
    echo "FAIL: /tmp allowed direct script execution despite noexec mount" >&2
    exit 1
  fi
  echo "OK: /tmp rejected direct execution (noexec enforced)"
'
EOF

  local adv_out="${log_dir}/02_adversarial_container.stdout.log"
  local adv_err="${log_dir}/02_adversarial_container.stderr.log"
  printf '%s\n' "${DOCKER_BIN} run --platform ${DISTRO_PLATFORM} --network=none --read-only --tmpfs /tmp:rw,noexec,nosuid,size=256m --user 65534:65534 python:3.12-slim" \
    > "${log_dir}/02_adversarial_container.cmd"
  printf '0\n' > "${log_dir}/02_adversarial_container.expected_rc"

  rc=0
  "${DOCKER_BIN}" run -i --rm \
    --label "cross-gpu-emulate=${RUN_ID}" \
    --platform "${DISTRO_PLATFORM}" \
    --network=none \
    --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,size=256m \
    --user 65534:65534 \
    -e HOME=/tmp \
    -e XDG_CACHE_HOME=/tmp/cache \
    -e TMPDIR=/tmp \
    -e PIP_NO_CACHE_DIR=1 \
    -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
    -e "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
    -v "${wheel_vol}:/wheels:ro" \
    python:3.12-slim bash -euo pipefail < "${adv_script}" > "${adv_out}" 2> "${adv_err}" || rc=$?

  printf '%s\n' "${rc}" > "${log_dir}/02_adversarial_container.rc"
  cat "${adv_out}"
  if [[ -s "${adv_err}" ]]; then
    printf '%s\n' "--- adversarial stderr ---" >&2
    cat "${adv_err}" >&2
  fi

  rm -f "${adv_script}"
  "${DOCKER_BIN}" volume rm -f "${wheel_vol}" >/dev/null 2>&1 || true
  return "${rc}"
}

# ---------------------------------------------------------------------------
# Markdown Report Generator (Zero Host Python Dependency)
# ---------------------------------------------------------------------------
write_summary_markdown() {
  local report="${LOG_ROOT}/summary.md"
  {
    printf '%s\n' "# CrossGPU Docker Emulation Test Report"
    printf '\n'
    printf -- '- **Run ID:** `%s`\n' "${RUN_ID}"
    printf -- '- **Timestamp:** %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf -- '- **Docker Engine:** arch=`%s`, ncpu=`%s`\n' "${DOCKER_ARCH}" "${DOCKER_NCPU}"
    printf -- '- **Backend/Distro/Adversarial Platform:** `%s`\n' "${DISTRO_PLATFORM}"
    printf -- '- **Log Directory:** `%s`\n' "${LOG_ROOT}"
    printf '\n'
    printf '%s\n' "## Target Summary"
    printf '\n'
    printf '%s\n' "| Target | Status | Duration | Detail |"
    printf '%s\n' "|---|---|---:|---|"

    local idx=0
    local total="${#RESULTS_TARGET[@]}"
    local failed_targets=()
    while (( idx < total )); do
      local t="${RESULTS_TARGET[${idx}]}"
      local st="${RESULTS_STATUS[${idx}]}"
      local el="${RESULTS_ELAPSED[${idx}]}"
      local det="${RESULTS_DETAIL[${idx}]}"
      printf '| `%s` | **%s** | %ss | %s |\n' "${t}" "${st}" "${el}" "${det}"
      if [[ "${st}" == "FAIL" ]]; then
        failed_targets+=("${t}")
      fi
      idx=$((idx + 1))
    done

    if [[ "${#failed_targets[@]}" -gt 0 ]]; then
      printf '\n'
      printf '%s\n' "## Failure Details"
      local ft
      for ft in "${failed_targets[@]}"; do
        printf '\n'
        printf '### `%s`\n' "${ft}"
        local tdir="${LOG_ROOT}/${ft}"
        local found_step="0"
        local rc_file
        for rc_file in "${tdir}"/*.rc; do
          [[ -f "${rc_file}" ]] || continue
          local sname
          sname="$(basename "${rc_file}" .rc)"
          local src_rc
          src_rc="$(cat "${rc_file}" 2>/dev/null || echo "")"
          local exp_rc
          exp_rc="$(cat "${tdir}/${sname}.expected_rc" 2>/dev/null || echo 0)"
          if [[ -z "${src_rc}" || "${src_rc}" != "${exp_rc}" ]]; then
            found_step="1"
            local scmd
            scmd="$(cat "${tdir}/${sname}.cmd" 2>/dev/null || echo "(unknown)")"
            printf '\n'
            printf '<details open><summary><strong><code>%s</code> — exit <code>%s</code> (expected <code>%s</code>)</strong></summary>\n\n' \
              "${sname}" "$(decode_rc "${src_rc:-1}")" "${exp_rc}"
            printf '%s\n' '````text'
            printf 'Command: %s\n' "${scmd}"
            printf '%s\n' "--- stdout (tail -n 200) ---"
            if [[ -s "${tdir}/${sname}.stdout.log" ]]; then
              tail -n 200 "${tdir}/${sname}.stdout.log"
            else
              printf '%s\n' "(empty)"
            fi
            printf '%s\n' "--- stderr (tail -n 200) ---"
            if [[ -s "${tdir}/${sname}.stderr.log" ]]; then
              tail -n 200 "${tdir}/${sname}.stderr.log"
            else
              printf '%s\n' "(empty)"
            fi
            printf '%s\n\n</details>\n' '````'
          fi
        done
        if [[ "${found_step}" == "0" ]]; then
          printf -- '- Target exited non-zero without per-step failure record; check `%s/`.\n' "${tdir}"
        fi
      done
    fi
  } > "${report}"
  printf '%s' "${report}"
}

# ---------------------------------------------------------------------------
# Argument Parsing & Dispatch
# ---------------------------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --list)
      LIST_ONLY="1"
      shift
      ;;
    --suite)
      [[ "$#" -ge 2 ]] || { printf 'Error: --suite requires an argument\n' >&2; exit 1; }
      SUITES+=("$2")
      shift 2
      ;;
    --target)
      [[ "$#" -ge 2 ]] || { printf 'Error: --target requires an argument\n' >&2; exit 1; }
      TARGETS+=("$2")
      shift 2
      ;;
    --distro-platform)
      [[ "$#" -ge 2 ]] || { printf 'Error: --distro-platform requires an argument\n' >&2; exit 1; }
      DISTRO_PLATFORM="$2"
      shift 2
      ;;
    --duration)
      [[ "$#" -ge 2 ]] || { printf 'Error: --duration requires an argument\n' >&2; exit 1; }
      DURATION="$2"
      shift 2
      ;;
    --log-dir)
      [[ "$#" -ge 2 ]] || { printf 'Error: --log-dir requires an argument\n' >&2; exit 1; }
      LOG_ROOT="$2"
      shift 2
      ;;
    --skip-binfmt)
      SKIP_BINFMT="1"
      shift
      ;;
    --fail-fast)
      FAIL_FAST="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -v|--verbose)
      VERBOSE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Error: unknown option %s (use --help)\n' "$1" >&2
      exit 1
      ;;
  esac
done

if [[ "${LIST_ONLY}" == "1" ]]; then
  print_list
  exit 0
fi

if [[ "${#SUITES[@]}" -eq 0 && "${#TARGETS[@]}" -eq 0 ]]; then
  SUITES=("all")
fi

# Resolve suites -> targets (Bash 3.2 safe)
if [[ "${#SUITES[@]}" -gt 0 ]]; then
  for raw_suite in "${SUITES[@]}"; do
    IFS=',' read -r -a suite_parts <<< "${raw_suite}"
    for s in "${suite_parts[@]}"; do
      s="$(printf '%s' "${s}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
      [[ -n "${s}" ]] || continue
      case "${s}" in
        all)
          for t in "${ALL_TARGETS[@]}"; do add_target_unique "${t}"; done
          ;;
        arch)
          for t in arch-amd64 arch-arm64 arch-riscv64 arch-ppc64le arch-s390x; do add_target_unique "${t}"; done
          ;;
        backends)
          for t in backend-cuda backend-hip backend-levelzero backend-opencl backend-metal backend-multigpu backend-pocl; do add_target_unique "${t}"; done
          ;;
        distros)
          for t in distro-alpine distro-debian distro-fedora distro-arch; do add_target_unique "${t}"; done
          ;;
        adversarial)
          add_target_unique "adversarial"
          ;;
        *)
          printf 'Error: unknown --suite "%s" (valid: arch, backends, distros, adversarial, all)\n' "${s}" >&2
          exit 1
          ;;
      esac
    done
  done
fi

if [[ "${#TARGETS[@]}" -gt 0 ]]; then
  for raw_target in "${TARGETS[@]}"; do
    IFS=',' read -r -a target_parts <<< "${raw_target}"
    for t in "${target_parts[@]}"; do
      t="$(printf '%s' "${t}" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
      [[ -n "${t}" ]] || continue
      if ! is_valid_target "${t}"; then
        printf 'Error: unknown --target "%s" (run --list to see valid targets)\n' "${t}" >&2
        exit 1
      fi
      add_target_unique "${t}"
    done
  done
fi

if [[ "${#SELECTED_TARGETS[@]}" -eq 0 ]]; then
  printf 'Error: no targets selected.\n' >&2
  exit 1
fi

if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="${HOME}/.cache/cross-gpu-emulation/$(date +%Y%m%d-%H%M%S)"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Dry run (Docker-only emulation harness)\n'
  printf '  Log directory    : %s\n' "${LOG_ROOT}"
  printf '  Distro platform  : %s\n' "${DISTRO_PLATFORM}"
  printf '  Duration         : %s\n' "${DURATION}"
  printf '  Register binfmt  : %s\n' "$(if [[ "${SKIP_BINFMT}" == "1" ]]; then echo "no (--skip-binfmt)"; else echo "yes (tonistiigi/binfmt)"; fi)"
  printf '  Selected targets (%s):\n' "${#SELECTED_TARGETS[@]}"
  for t in "${SELECTED_TARGETS[@]}"; do
    printf '    - %-18s %s\n' "${t}" "$(target_desc "${t}")"
  done
  exit 0
fi

locate_and_verify_docker
mkdir -p "${LOG_ROOT}"

printf 'CrossGPU Docker-Only Emulation Runner (run_id=%s)\n' "${RUN_ID}"
printf 'Logs directory  : %s\n' "${LOG_ROOT}"
printf 'Selected targets: %s\n\n' "${SELECTED_TARGETS[*]}"

ensure_binfmt

ANY_FAILED="0"
for target in "${SELECTED_TARGETS[@]}"; do
  t_log_dir="${LOG_ROOT}/${target}"
  mkdir -p "${t_log_dir}"
  printf '%s Target: %s (%s)\n' "$(c_cyan '==>')" "${target}" "$(target_desc "${target}")"

  t_start="$(date +%s)"
  t_rc=0
  case "${target}" in
    arch-amd64|arch-arm64|arch-riscv64|arch-ppc64le|arch-s390x)
      run_target_arch "${target}" "${t_log_dir}" || t_rc=$?
      ;;
    backend-cuda)
      run_target_backend_cuda "${t_log_dir}" || t_rc=$?
      ;;
    backend-hip)
      run_target_backend_hip "${t_log_dir}" || t_rc=$?
      ;;
    backend-levelzero)
      run_target_backend_levelzero "${t_log_dir}" || t_rc=$?
      ;;
    backend-opencl)
      run_target_backend_opencl "${t_log_dir}" || t_rc=$?
      ;;
    backend-metal)
      run_target_backend_metal "${t_log_dir}" || t_rc=$?
      ;;
    backend-multigpu)
      run_target_backend_multigpu "${t_log_dir}" || t_rc=$?
      ;;
    backend-pocl)
      run_target_backend_pocl "${t_log_dir}" || t_rc=$?
      ;;
    distro-alpine|distro-debian|distro-fedora|distro-arch)
      run_target_distro "${target}" "${t_log_dir}" || t_rc=$?
      ;;
    adversarial)
      run_target_adversarial "${t_log_dir}" || t_rc=$?
      ;;
  esac
  t_end="$(date +%s)"
  t_elapsed=$((t_end - t_start))

  RESULTS_TARGET+=("${target}")
  RESULTS_ELAPSED+=("${t_elapsed}")
  if [[ "${t_rc}" -eq 0 ]]; then
    RESULTS_STATUS+=("PASS")
    RESULTS_DETAIL+=("$(target_desc "${target}")")
    printf '<== %s: %s (%ss)\n\n' "${target}" "$(c_green PASS)" "${t_elapsed}"
  else
    RESULTS_STATUS+=("FAIL")
    RESULTS_DETAIL+=("exited $(decode_rc "${t_rc}") — see ${t_log_dir}/")
    ANY_FAILED="1"
    printf '<== %s: %s (rc=%s, %ss)\n\n' "${target}" "$(c_red FAIL)" "$(decode_rc "${t_rc}")" "${t_elapsed}"
    if [[ "${FAIL_FAST}" == "1" ]]; then
      break
    fi
  fi
done

REPORT_PATH="$(write_summary_markdown)"

printf '%s\n' "========================================================================================"
printf '%-18s %-8s %9s  %s\n' "TARGET" "STATUS" "DURATION" "DETAIL"
printf '%s\n' "----------------------------------------------------------------------------------------"
idx=0
total="${#RESULTS_TARGET[@]}"
while (( idx < total )); do
  t="${RESULTS_TARGET[${idx}]}"
  st="${RESULTS_STATUS[${idx}]}"
  el="${RESULTS_ELAPSED[${idx}]}"
  det="${RESULTS_DETAIL[${idx}]}"
  if [[ "${st}" == "PASS" ]]; then
    badge="$(c_green PASS)"
  else
    badge="$(c_red FAIL)"
  fi
  pad=$(( ${#badge} - ${#st} ))
  printf "%-18s %-$((8 + pad))s %8ss  %s\n" "${t}" "${badge}" "${el}" "${det}"
  idx=$((idx + 1))
done
printf '%s\n' "----------------------------------------------------------------------------------------"
printf 'Full logs & report: %s\n' "${REPORT_PATH}"

if [[ "${ANY_FAILED}" != "0" ]]; then
  exit 1
fi
exit 0
