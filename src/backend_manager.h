#ifndef CROSSGPU_BACKEND_MANAGER_H
#define CROSSGPU_BACKEND_MANAGER_H

#include "crossgpu_backend.h"
#include "platform.h"

#include <stddef.h>

typedef struct loaded_backend {
    void *library_handle;
    const crossgpu_backend_plugin_t *api;
    char path[512];
} loaded_backend_t;

typedef struct crossgpu_device_candidate {
    crossgpu_device_info_t info;
    loaded_backend_t *backend;
    int preference_score;
} crossgpu_device_candidate_t;

typedef struct crossgpu_device {
    crossgpu_device_info_t info;          /* preferred candidate, used for display */
    loaded_backend_t *backend;            /* preferred candidate */
    crossgpu_device_candidate_t *candidates;
    size_t candidate_count;
    size_t candidate_capacity;
} crossgpu_device_t;

typedef struct backend_manager {
    loaded_backend_t *backends;
    size_t backend_count;
    size_t backend_capacity;
    crossgpu_device_t *devices;
    size_t device_count;
    size_t device_capacity;
    char diagnostics[CROSSGPU_ERROR_MAX];
} backend_manager_t;

int backend_manager_discover(backend_manager_t *manager,
                             const char *executable_path,
                             const char *backend_filter);
void backend_manager_destroy(backend_manager_t *manager);

#endif
