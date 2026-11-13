#include "backend_manager.h"
#include "registry.h"
#include "util.h"
#include "platform.h"

#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CROSSGPU_DEFAULT_PLUGIN_DIR
#define CROSSGPU_DEFAULT_PLUGIN_DIR "/usr/local/lib/cross-gpu/backends"
#endif

typedef struct candidate_device {
    crossgpu_device_info_t info;
    loaded_backend_t *backend;
} candidate_device_t;

static int grow_array(void **array, size_t *capacity, size_t needed, size_t element_size)
{
    size_t next;
    void *grown;

    if (needed <= *capacity)
        return 0;
    if (element_size == 0 || needed > SIZE_MAX / element_size)
        return -1;
    next = *capacity ? *capacity : 8;
    while (next < needed) {
        if (next > SIZE_MAX / 2) {
            next = needed;
            break;
        }
        next *= 2;
    }
    if (next > SIZE_MAX / element_size)
        return -1;
    grown = realloc(*array, next * element_size);
    if (grown == NULL)
        return -1;
    *array = grown;
    *capacity = next;
    return 0;
}

static int has_suffix(const char *text, const char *suffix)
{
    size_t a = strlen(text), b = strlen(suffix);
    return a >= b && strcmp(text + a - b, suffix) == 0;
}

static int is_backend_filename(const char *name)
{
    return strncmp(name, CROSSGPU_PLUGIN_PREFIX, strlen(CROSSGPU_PLUGIN_PREFIX)) == 0 &&
           has_suffix(name, CROSSGPU_PLUGIN_SUFFIX);
}

static int compare_names(const void *a, const void *b)
{
    const char * const *sa = (const char * const *)a;
    const char * const *sb = (const char * const *)b;
    return strcmp(*sa, *sb);
}

static void join_path(char *output, size_t output_size, const char *directory, const char *name)
{
    size_t length;
    crossgpu_copy_string(output, output_size, directory);
    length = strlen(output);
    if (length > 0 && output[length - 1] != CROSSGPU_PATH_SEPARATOR &&
        output[length - 1] != '/' && output[length - 1] != '\\') {
        const char separator[2] = { CROSSGPU_PATH_SEPARATOR, '\0' };
        (void)strncat(output, separator, output_size - strlen(output) - 1);
    }
    (void)strncat(output, name, output_size - strlen(output) - 1);
}

static int backend_path_loaded(const backend_manager_t *manager, const char *path)
{
    size_t i;
    for (i = 0; i < manager->backend_count; i++)
        if (crossgpu_platform_path_equal(manager->backends[i].path, path))
            return 1;
    return 0;
}

static int append_backend(backend_manager_t *manager, void *handle,
                          const crossgpu_backend_plugin_t *api, const char *path)
{
    loaded_backend_t *loaded;

    if (grow_array((void **)&manager->backends, &manager->backend_capacity,
                   manager->backend_count + 1, sizeof(*manager->backends)) != 0)
        return -1;
    loaded = &manager->backends[manager->backend_count++];
    memset(loaded, 0, sizeof(*loaded));
    loaded->library_handle = handle;
    loaded->api = api;
    crossgpu_copy_string(loaded->path, sizeof(loaded->path), path);
    return 0;
}

static void load_from_directory(backend_manager_t *manager, const char *directory,
                                const char *backend_filter)
{
    crossgpu_directory_entry_list_t entries;
    size_t i;

    if (directory == NULL || *directory == '\0')
        return;
    memset(&entries, 0, sizeof(entries));
    if (crossgpu_platform_list_directory(directory, &entries) != 0)
        return;
    if (entries.count > 1)
        qsort(entries.items, entries.count, sizeof(*entries.items), compare_names);

    for (i = 0; i < entries.count; i++) {
        char path[CROSSGPU_PATH_MAX];
        char canonical_path[CROSSGPU_PATH_MAX];
        char error[CROSSGPU_ERROR_MAX] = { 0 };
        void *handle;
        void *symbol;
        crossgpu_backend_get_api_fn get_api = NULL;
        const crossgpu_backend_plugin_t *api;

        if (!is_backend_filename(entries.items[i]))
            continue;
        join_path(path, sizeof(path), directory, entries.items[i]);
        if (crossgpu_platform_canonicalize_path(path, canonical_path,
                                                sizeof(canonical_path)) != 0)
            crossgpu_copy_string(canonical_path, sizeof(canonical_path), path);
        if (backend_path_loaded(manager, canonical_path))
            continue;

        handle = crossgpu_platform_library_open(canonical_path);
        if (handle == NULL) {
            fprintf(stderr, "cross-gpu: warning: cannot load %s: %s\n",
                    path, crossgpu_platform_library_error());
            continue;
        }

        symbol = crossgpu_platform_library_symbol(handle, CROSSGPU_BACKEND_ENTRYPOINT);
        if (symbol == NULL || sizeof(symbol) != sizeof(get_api)) {
            fprintf(stderr, "cross-gpu: warning: %s does not export %s: %s\n",
                    path, CROSSGPU_BACKEND_ENTRYPOINT, crossgpu_platform_library_error());
            crossgpu_platform_library_close(handle);
            continue;
        }
        memcpy(&get_api, &symbol, sizeof(get_api));

        api = get_api(CROSSGPU_BACKEND_ABI_VERSION);
        if (api == NULL || api->abi_version != CROSSGPU_BACKEND_ABI_VERSION ||
            api->struct_size < sizeof(crossgpu_backend_plugin_t) || api->name == NULL ||
            api->name[0] == '\0' || api->kind <= CROSSGPU_BACKEND_AUTO ||
            api->enumerate_devices == NULL || api->device_open == NULL ||
            api->device_close == NULL || api->workload_begin == NULL ||
            api->workload_step == NULL || api->workload_end == NULL) {
            fprintf(stderr, "cross-gpu: warning: incompatible backend plugin: %s\n", path);
            crossgpu_platform_library_close(handle);
            continue;
        }

        if (backend_filter != NULL && strcmp(backend_filter, "auto") != 0 &&
            strcmp(api->name, backend_filter) != 0) {
            crossgpu_platform_library_close(handle);
            continue;
        }

        if (api->initialize != NULL && api->initialize(error, sizeof(error)) != 0) {
            fprintf(stderr, "cross-gpu: %s backend unavailable: %s\n",
                    api->name, error[0] ? error : "initialization failed");
            crossgpu_platform_library_close(handle);
            continue;
        }

        if (append_backend(manager, handle, api, canonical_path) != 0) {
            fprintf(stderr, "cross-gpu: warning: out of memory while registering backend %s\n", api->name);
            if (api->shutdown != NULL)
                api->shutdown();
            crossgpu_platform_library_close(handle);
        }
    }

    crossgpu_platform_free_directory_list(&entries);
}

static void load_from_path_list(backend_manager_t *manager, const char *path_list,
                                const char *backend_filter)
{
    const char *part;
    const char *cursor;
    if (path_list == NULL || *path_list == '\0')
        return;
    cursor = path_list;
    while (*cursor != '\0') {
        size_t length;
        char *directory;
        part = cursor;
        while (*cursor != '\0' && *cursor != CROSSGPU_PATH_LIST_SEPARATOR)
            cursor++;
        length = (size_t)(cursor - part);
        if (length > 0) {
            directory = (char *)malloc(length + 1);
            if (directory == NULL)
                return;
            memcpy(directory, part, length);
            directory[length] = '\0';
            load_from_directory(manager, directory, backend_filter);
            free(directory);
        }
        if (*cursor == CROSSGPU_PATH_LIST_SEPARATOR)
            cursor++;
    }
}

static void load_relative_directory(backend_manager_t *manager, const char *executable_path,
                                    const char *backend_filter)
{
    char path[CROSSGPU_PATH_MAX];
    const char *slash;
    size_t prefix;

    if (executable_path == NULL || *executable_path == '\0')
        return;
    slash = strrchr(executable_path, CROSSGPU_PATH_SEPARATOR);
#if CROSSGPU_PLATFORM_WINDOWS
    {
        const char *alternate = strrchr(executable_path, '/');
        if (alternate != NULL && (slash == NULL || alternate > slash))
            slash = alternate;
    }
#endif
    if (slash == NULL)
        return;
    prefix = (size_t)(slash - executable_path);
    if (prefix + sizeof("/backends") >= sizeof(path))
        return;
    memcpy(path, executable_path, prefix);
    path[prefix] = '\0';
    {
        const char suffix[] = { CROSSGPU_PATH_SEPARATOR, 'b', 'a', 'c', 'k', 'e', 'n', 'd', 's', '\0' };
        (void)strcat(path, suffix);
    }
    load_from_directory(manager, path, backend_filter);
}

static int candidate_score(const candidate_device_t *candidate)
{
    /* All selection policy lives in the vendor/backend preference registry.
     * The manager deliberately knows nothing about individual API kinds. */
    return crossgpu_vendor_backend_score(candidate->info.vendor_id,
                                         candidate->backend->api->kind,
                                         candidate->backend->api->name);
}

static int fixed_string_length(const char *value, size_t capacity, size_t *length)
{
    const char *terminator;

    if (value == NULL || capacity == 0)
        return 0;
    terminator = (const char *)memchr(value, '\0', capacity);
    if (terminator == NULL)
        return 0;
    if (length != NULL)
        *length = (size_t)(terminator - value);
    return terminator != value;
}

static int fixed_string_equal(const char *a, size_t a_capacity,
                              const char *b, size_t b_capacity)
{
    size_t a_length, b_length;

    if (!fixed_string_length(a, a_capacity, &a_length) ||
        !fixed_string_length(b, b_capacity, &b_length))
        return 0;
    return a_length == b_length && memcmp(a, b, a_length) == 0;
}

static int has_physical_identity(const crossgpu_device_info_t *info)
{
    if (info == NULL || info->identity_kind <= CROSSGPU_IDENTITY_NONE)
        return 0;

    return fixed_string_length(info->identity_namespace,
                               sizeof(info->identity_namespace), NULL) &&
           fixed_string_length(info->identity_value,
                               sizeof(info->identity_value), NULL);
}

static int same_physical_device(const candidate_device_t *a, const candidate_device_t *b)
{
    if (!has_physical_identity(&a->info) || !has_physical_identity(&b->info))
        return 0;

    /* Identity kind, namespace, and value must all match exactly. */
    return a->info.identity_kind == b->info.identity_kind &&
           fixed_string_equal(a->info.identity_namespace,
                              sizeof(a->info.identity_namespace),
                              b->info.identity_namespace,
                              sizeof(b->info.identity_namespace)) &&
           fixed_string_equal(a->info.identity_value,
                              sizeof(a->info.identity_value),
                              b->info.identity_value,
                              sizeof(b->info.identity_value));
}

static void terminate_device_info_strings(crossgpu_device_info_t *info)
{
    info->identity_namespace[sizeof(info->identity_namespace) - 1] = '\0';
    info->identity_value[sizeof(info->identity_value) - 1] = '\0';
    info->backend_device_id[sizeof(info->backend_device_id) - 1] = '\0';
    info->name[sizeof(info->name) - 1] = '\0';
    info->vendor[sizeof(info->vendor) - 1] = '\0';
    info->driver_version[sizeof(info->driver_version) - 1] = '\0';
    info->api_version[sizeof(info->api_version) - 1] = '\0';
}

static crossgpu_device_t *find_device(backend_manager_t *manager,
                                      const candidate_device_t *candidate)
{
    size_t i;
    for (i = 0; i < manager->device_count; i++) {
        crossgpu_device_t *device = &manager->devices[i];
        if (device->candidate_count == 0)
            continue;
        {
            candidate_device_t existing;
            existing.info = device->candidates[0].info;
            existing.backend = device->candidates[0].backend;
            if (same_physical_device(candidate, &existing))
                return device;
        }
    }
    return NULL;
}

static int append_device_candidate(backend_manager_t *manager,
                                   const candidate_device_t *candidate)
{
    crossgpu_device_t *device;
    size_t i;
    const int preference_score = candidate_score(candidate);

    /* A zero score means this plugin is not valid for this vendor. */
    if (preference_score == 0)
        return 0;
    device = find_device(manager, candidate);

    if (device == NULL) {
        if (grow_array((void **)&manager->devices, &manager->device_capacity,
                       manager->device_count + 1, sizeof(*manager->devices)) != 0)
            return -1;
        device = &manager->devices[manager->device_count++];
        memset(device, 0, sizeof(*device));
    }

    for (i = 0; i < device->candidate_count; i++) {
        if (device->candidates[i].backend == candidate->backend &&
            device->candidates[i].info.backend_device_index == candidate->info.backend_device_index)
            return 0;
    }

    if (grow_array((void **)&device->candidates, &device->candidate_capacity,
                   device->candidate_count + 1, sizeof(*device->candidates)) != 0)
        return -1;
    device->candidates[device->candidate_count].info = candidate->info;
    device->candidates[device->candidate_count].backend = candidate->backend;
    device->candidates[device->candidate_count].preference_score = preference_score;
    device->candidate_count++;
    return 0;
}

static void sort_device_candidates(crossgpu_device_t *device)
{
    size_t i, j;
    for (i = 0; i < device->candidate_count; i++) {
        for (j = i + 1; j < device->candidate_count; j++) {
            if (device->candidates[j].preference_score > device->candidates[i].preference_score) {
                crossgpu_device_candidate_t temporary = device->candidates[i];
                device->candidates[i] = device->candidates[j];
                device->candidates[j] = temporary;
            }
        }
    }
    if (device->candidate_count > 0) {
        device->info = device->candidates[0].info;
        device->backend = device->candidates[0].backend;
    }
}

int backend_manager_discover(backend_manager_t *manager,
                             const char *executable_path,
                             const char *backend_filter)
{
    const char *environment_path;
    const char *path_only_env;
    int path_only;
    size_t i, j;

    if (manager == NULL)
        return -1;
    memset(manager, 0, sizeof(*manager));

    environment_path = getenv("CROSSGPU_PLUGIN_PATH");
    load_from_path_list(manager, environment_path, backend_filter);

    /* Tests and troubleshooting can request a fully isolated plugin search.
     * The explicit CROSSGPU_PLUGIN_PATH remains active, while executable- and
     * system-wide plugin directories are skipped. */
    path_only_env = getenv("CROSSGPU_PLUGIN_PATH_ONLY");
    path_only = path_only_env != NULL && strcmp(path_only_env, "1") == 0;
    if (!path_only) {
        load_relative_directory(manager, executable_path, backend_filter);
        load_from_directory(manager, "./backends", backend_filter);
        load_from_directory(manager, CROSSGPU_DEFAULT_PLUGIN_DIR, backend_filter);
#if CROSSGPU_PLATFORM_LINUX
        load_from_directory(manager, "/usr/lib/cross-gpu/backends", backend_filter);
#endif
    }

    for (i = 0; i < manager->backend_count; i++) {
        loaded_backend_t *backend = &manager->backends[i];
        crossgpu_device_info_t *temporary = NULL;
        size_t count = 0, actual_count = 0;
        char error[CROSSGPU_ERROR_MAX] = { 0 };

        if (backend_filter != NULL && strcmp(backend_filter, "auto") != 0 && strcmp(backend->api->name, backend_filter) != 0)
            continue;

        /* Two-pass enumeration removes the old global device-count ceiling. */
        if (backend->api->enumerate_devices(NULL, 0, &count, error, sizeof(error)) != 0) {
            fprintf(stderr, "cross-gpu: warning: %s device count query failed: %s\n",
                    backend->api->name, error[0] ? error : "unknown error");
            continue;
        }
        if (count == 0)
            continue;
        if (count > SIZE_MAX / sizeof(*temporary)) {
            fprintf(stderr, "cross-gpu: warning: %s reported an unreasonable device count\n", backend->api->name);
            continue;
        }
        temporary = calloc(count, sizeof(*temporary));
        if (temporary == NULL) {
            fprintf(stderr, "cross-gpu: warning: out of memory enumerating %s devices\n", backend->api->name);
            continue;
        }
        if (backend->api->enumerate_devices(temporary, count, &actual_count,
                                             error, sizeof(error)) != 0) {
            fprintf(stderr, "cross-gpu: warning: %s enumeration failed: %s\n",
                    backend->api->name, error[0] ? error : "unknown error");
            free(temporary);
            continue;
        }
        if (actual_count > count)
            actual_count = count;

        for (j = 0; j < actual_count; j++) {
            candidate_device_t candidate;
            candidate.info = temporary[j];
            candidate.info.struct_size = sizeof(candidate.info);
            terminate_device_info_strings(&candidate.info);
            candidate.backend = backend;
            if (append_device_candidate(manager, &candidate) != 0) {
                fprintf(stderr, "cross-gpu: warning: out of memory registering GPU candidates\n");
                free(temporary);
                backend_manager_destroy(manager);
                return -1;
            }
        }
        free(temporary);
    }

    for (i = 0; i < manager->device_count; i++)
        sort_device_candidates(&manager->devices[i]);

    return manager->device_count > (size_t)INT_MAX ? INT_MAX : (int)manager->device_count;
}

void backend_manager_destroy(backend_manager_t *manager)
{
    size_t i;
    if (manager == NULL)
        return;
    for (i = 0; i < manager->device_count; i++)
        free(manager->devices[i].candidates);
    free(manager->devices);
    for (i = manager->backend_count; i > 0; i--) {
        loaded_backend_t *backend = &manager->backends[i - 1];
        if (backend->api != NULL && backend->api->shutdown != NULL)
            backend->api->shutdown();
        if (backend->library_handle != NULL)
            crossgpu_platform_library_close(backend->library_handle);
    }
    free(manager->backends);
    memset(manager, 0, sizeof(*manager));
}
