// SPDX-License-Identifier: MIT
#include <stdlib.h>
#include <string.h>

typedef int hiprtcResult;
typedef void *hiprtcProgram;

#define HIPRTC_SUCCESS 0

static const unsigned char FAKE_HSACO[] = { 0x7f, 'E', 'L', 'F', 2, 1, 1, 0 };

hiprtcResult hiprtcVersion(int *major, int *minor) {
    if (major) *major = 6;
    if (minor) *minor = 2;
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcCreateProgram(hiprtcProgram *prog, const char *src, const char *name,
                                 int numHeaders, const char *const *headers,
                                 const char *const *includeNames) {
    (void)src; (void)name; (void)numHeaders; (void)headers; (void)includeNames;
    if (prog) *prog = malloc(16);
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOpts, const char *const *opts) {
    (void)prog; (void)numOpts; (void)opts;
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t *sz) {
    (void)prog;
    if (sz) *sz = sizeof(FAKE_HSACO);
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetCode(hiprtcProgram prog, char *code) {
    (void)prog;
    if (code) memcpy(code, FAKE_HSACO, sizeof(FAKE_HSACO));
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t *sz) {
    (void)prog;
    if (sz) *sz = 1;
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char *log) {
    (void)prog;
    if (log) log[0] = '\0';
    return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcDestroyProgram(hiprtcProgram *prog) {
    if (prog && *prog) { free(*prog); *prog = NULL; }
    return HIPRTC_SUCCESS;
}

const char *hiprtcGetErrorString(hiprtcResult r) {
    return r == HIPRTC_SUCCESS ? "HIPRTC_SUCCESS" : "HIPRTC_ERROR";
}