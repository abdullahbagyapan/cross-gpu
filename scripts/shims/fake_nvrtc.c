// SPDX-License-Identifier: MIT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "shim_common.h"

typedef int nvrtcResult;
typedef void *nvrtcProgram;

#define NVRTC_SUCCESS 0
#define NVRTC_ERROR_OUT_OF_MEMORY 1

static _Atomic int g_active_programs = 0;

static const char FAKE_PTX[] =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry kernel_fake() { ret; }\n";

__attribute__((destructor))
static void check_nvrtc_leaks(void) {
    if (!shim_leak_check_enabled()) return;
    int progs = atomic_load(&g_active_programs);
    if (progs != 0) {
        fprintf(stderr, "[fake_nvrtc] LEAK: programs=%d\n", progs);
        _exit(99);
    }
}

nvrtcResult nvrtcVersion(int *major, int *minor) {
    if (major) *major = 12;
    if (minor) *minor = 6;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog, const char *src, const char *name,
                               int numHeaders, const char *const *headers,
                               const char *const *includeNames) {
    (void)src; (void)name; (void)numHeaders; (void)headers; (void)includeNames;
    void *p = malloc(16);
    if (!p) return NVRTC_ERROR_OUT_OF_MEMORY;
    atomic_fetch_add(&g_active_programs, 1);
    if (prog) *prog = p;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char *const *options) {
    (void)prog; (void)numOptions; (void)options;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet) {
    (void)prog;
    if (logSizeRet) *logSizeRet = 1;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log) {
    (void)prog;
    if (log) log[0] = '\0';
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet) {
    (void)prog;
    if (ptxSizeRet) *ptxSizeRet = sizeof(FAKE_PTX);
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx) {
    (void)prog;
    if (ptx) memcpy(ptx, FAKE_PTX, sizeof(FAKE_PTX));
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t *cubinSizeRet) {
    (void)prog;
    /* Return 0 so embed_binary exercises the PTX fallback path cleanly. */
    if (cubinSizeRet) *cubinSizeRet = 0;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char *cubin) {
    (void)prog; (void)cubin;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog) {
    if (prog && *prog) {
        free(*prog);
        *prog = NULL;
        atomic_fetch_sub(&g_active_programs, 1);
    }
    return NVRTC_SUCCESS;
}

const char *nvrtcGetErrorString(nvrtcResult r) {
    return r == NVRTC_SUCCESS ? "NVRTC_SUCCESS" : "NVRTC_ERROR";
}
