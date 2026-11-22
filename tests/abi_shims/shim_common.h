// SPDX-License-Identifier: MIT
#ifndef CROSS_GPU_SHIM_COMMON_H
#define CROSS_GPU_SHIM_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SHIM_MAX_GPUS 16

static inline int shim_env_int(const char *name, int fallback, int min_v, int max_v) {
    const char *v = getenv(name);
    if (!v || !*v) return fallback;
    char *end = NULL;
    long parsed = strtol(v, &end, 10);
    if (end == v || parsed < min_v || parsed > max_v) return fallback;
    return (int)parsed;
}

static inline int shim_fault_is(const char *mode) {
    const char *v = getenv("FAKE_GPU_FAULT");
    return v != NULL && strcmp(v, mode) == 0;
}

static inline int shim_fault_target_matches(const char *backend, int dev_idx) {
    const char *b = getenv("FAKE_GPU_FAULT_BACKEND");
    if (b && *b && strcmp(b, backend) != 0) return 0;
    int target_dev = shim_env_int("FAKE_GPU_FAULT_DEVICE", -1, -1, SHIM_MAX_GPUS - 1);
    return target_dev < 0 || target_dev == dev_idx;
}

static inline int shim_leak_check_enabled(void) {
    const char *v = getenv("FAKE_GPU_LEAK_CHECK");
    return v != NULL && strcmp(v, "1") == 0;
}

#endif