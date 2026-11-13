#ifndef CROSSGPU_PLATFORM_POSIX_IMPL_H
#define CROSSGPU_PLATFORM_POSIX_IMPL_H

#include "platform.h"

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if CROSSGPU_PLATFORM_APPLE
#include <mach-o/dyld.h>
#endif

static char g_platform_library_error[512];

void *crossgpu_platform_library_open(const char *path)
{
    void *handle;
    (void)dlerror();
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        const char *error = dlerror();
        (void)snprintf(g_platform_library_error, sizeof(g_platform_library_error), "%s",
                       error != NULL ? error : "dynamic library load failed");
    } else {
        g_platform_library_error[0] = '\0';
    }
    return handle;
}

void *crossgpu_platform_library_symbol(void *library, const char *symbol)
{
    void *address;
    (void)dlerror();
    address = dlsym(library, symbol);
    if (address == NULL) {
        const char *error = dlerror();
        (void)snprintf(g_platform_library_error, sizeof(g_platform_library_error), "%s",
                       error != NULL ? error : "symbol not found");
    }
    return address;
}

void crossgpu_platform_library_close(void *library)
{
    if (library != NULL)
        (void)dlclose(library);
}

const char *crossgpu_platform_library_error(void)
{
    return g_platform_library_error[0] != '\0' ? g_platform_library_error : "unknown loader error";
}

static char *platform_strdup(const char *text)
{
    size_t length;
    char *copy;
    if (text == NULL)
        return NULL;
    length = strlen(text) + 1;
    copy = (char *)malloc(length);
    if (copy != NULL)
        memcpy(copy, text, length);
    return copy;
}

int crossgpu_platform_list_directory(const char *directory,
                                    crossgpu_directory_entry_list_t *entries)
{
    DIR *dir;
    struct dirent *entry;
    size_t capacity = 0;

    if (entries == NULL)
        return -1;
    entries->items = NULL;
    entries->count = 0;
    dir = opendir(directory);
    if (dir == NULL)
        return -1;

    while ((entry = readdir(dir)) != NULL) {
        char **grown;
        char *copy;
        size_t next;
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        if (entries->count == capacity) {
            next = capacity == 0 ? 16 : capacity * 2;
            if (next < capacity || next > SIZE_MAX / sizeof(*entries->items))
                break;
            grown = (char **)realloc(entries->items, next * sizeof(*entries->items));
            if (grown == NULL)
                break;
            entries->items = grown;
            capacity = next;
        }
        copy = platform_strdup(entry->d_name);
        if (copy == NULL)
            break;
        entries->items[entries->count++] = copy;
    }
    (void)closedir(dir);
    return 0;
}

void crossgpu_platform_free_directory_list(crossgpu_directory_entry_list_t *entries)
{
    size_t i;
    if (entries == NULL)
        return;
    for (i = 0; i < entries->count; i++)
        free(entries->items[i]);
    free(entries->items);
    entries->items = NULL;
    entries->count = 0;
}

int crossgpu_platform_canonicalize_path(const char *path, char *output, size_t output_size)
{
    char resolved[CROSSGPU_PATH_MAX];
    if (path == NULL || output == NULL || output_size == 0)
        return -1;
    if (realpath(path, resolved) == NULL)
        return -1;
    if (strlen(resolved) >= output_size)
        return -1;
    memcpy(output, resolved, strlen(resolved) + 1);
    return 0;
}

int crossgpu_platform_path_equal(const char *left, const char *right)
{
    return left != NULL && right != NULL && strcmp(left, right) == 0;
}

int crossgpu_platform_get_executable_path(char *output, size_t output_size)
{
    if (output == NULL || output_size == 0)
        return -1;
#if CROSSGPU_PLATFORM_APPLE
    {
        uint32_t required = 0;
        char temporary[CROSSGPU_PATH_MAX];
        if (_NSGetExecutablePath(NULL, &required) != -1 || required == 0 || required > sizeof(temporary))
            return -1;
        if (_NSGetExecutablePath(temporary, &required) != 0)
            return -1;
        if (crossgpu_platform_canonicalize_path(temporary, output, output_size) == 0)
            return 0;
        if (strlen(temporary) >= output_size)
            return -1;
        memcpy(output, temporary, strlen(temporary) + 1);
        return 0;
    }
#else
    {
        ssize_t length = readlink("/proc/self/exe", output, output_size - 1);
        if (length < 0 || (size_t)length >= output_size - 1)
            return -1;
        output[length] = '\0';
        return 0;
    }
#endif
}

uint64_t crossgpu_platform_now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

void crossgpu_platform_sleep_ms(unsigned int milliseconds)
{
    struct timespec request;
    request.tv_sec = (time_t)(milliseconds / 1000U);
    request.tv_nsec = (long)(milliseconds % 1000U) * 1000000L;
    while (nanosleep(&request, &request) != 0 && errno == EINTR)
        ;
}

int crossgpu_platform_thread_create(crossgpu_thread_t *thread,
                                    crossgpu_thread_fn function,
                                    void *argument)
{
    return pthread_create(thread, NULL, function, argument);
}

void crossgpu_platform_thread_join(crossgpu_thread_t thread)
{
    (void)pthread_join(thread, NULL);
}

void crossgpu_platform_mutex_lock(crossgpu_mutex_t *mutex)
{
    (void)pthread_mutex_lock(mutex);
}

void crossgpu_platform_mutex_unlock(crossgpu_mutex_t *mutex)
{
    (void)pthread_mutex_unlock(mutex);
}

#endif
