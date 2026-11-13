#define WIN32_LEAN_AND_MEAN
#include "platform.h"

#include <ctype.h>
#include <errno.h>
#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_platform_library_error[512];

typedef struct platform_thread_start {
    crossgpu_thread_fn function;
    void *argument;
} platform_thread_start_t;

void *crossgpu_platform_library_open(const char *path)
{
    HMODULE library = LoadLibraryA(path);
    if (library == NULL) {
        DWORD code = GetLastError();
        (void)FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                             NULL, code, 0, g_platform_library_error,
                             (DWORD)sizeof(g_platform_library_error), NULL);
        if (g_platform_library_error[0] == '\0')
            (void)snprintf(g_platform_library_error, sizeof(g_platform_library_error),
                           "Windows loader error %lu", (unsigned long)code);
    } else {
        g_platform_library_error[0] = '\0';
    }
    return (void *)library;
}

void *crossgpu_platform_library_symbol(void *library, const char *symbol)
{
    FARPROC address = GetProcAddress((HMODULE)library, symbol);
    void *result = NULL;
    if (address == NULL) {
        DWORD code = GetLastError();
        (void)snprintf(g_platform_library_error, sizeof(g_platform_library_error),
                       "GetProcAddress failed (error %lu)", (unsigned long)code);
        return NULL;
    }
    /* Windows uses the same pointer representation for FARPROC and data pointers. */
    memcpy(&result, &address, sizeof(result) < sizeof(address) ? sizeof(result) : sizeof(address));
    return result;
}

void crossgpu_platform_library_close(void *library)
{
    if (library != NULL)
        (void)FreeLibrary((HMODULE)library);
}

const char *crossgpu_platform_library_error(void)
{
    return g_platform_library_error[0] != '\0' ? g_platform_library_error : "unknown Windows loader error";
}

static int append_entry(crossgpu_directory_entry_list_t *entries,
                        size_t *capacity, const char *name)
{
    char **grown;
    char *copy;
    size_t length, next;
    if (entries->count == *capacity) {
        next = *capacity == 0 ? 16 : *capacity * 2;
        if (next < *capacity || next > SIZE_MAX / sizeof(*entries->items))
            return -1;
        grown = (char **)realloc(entries->items, next * sizeof(*entries->items));
        if (grown == NULL)
            return -1;
        entries->items = grown;
        *capacity = next;
    }
    length = strlen(name) + 1;
    copy = (char *)malloc(length);
    if (copy == NULL)
        return -1;
    memcpy(copy, name, length);
    entries->items[entries->count++] = copy;
    return 0;
}

int crossgpu_platform_list_directory(const char *directory,
                                    crossgpu_directory_entry_list_t *entries)
{
    WIN32_FIND_DATAA data;
    HANDLE search;
    char pattern[CROSSGPU_PATH_MAX];
    size_t length, capacity = 0;

    if (directory == NULL || entries == NULL)
        return -1;
    entries->items = NULL;
    entries->count = 0;
    length = strlen(directory);
    if (length + 3 >= sizeof(pattern))
        return -1;
    memcpy(pattern, directory, length);
    if (length > 0 && pattern[length - 1] != '\\' && pattern[length - 1] != '/')
        pattern[length++] = '\\';
    pattern[length++] = '*';
    pattern[length] = '\0';

    search = FindFirstFileA(pattern, &data);
    if (search == INVALID_HANDLE_VALUE)
        return -1;
    do {
        if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 &&
            strcmp(data.cFileName, ".") != 0 && strcmp(data.cFileName, "..") != 0) {
            if (append_entry(entries, &capacity, data.cFileName) != 0)
                break;
        }
    } while (FindNextFileA(search, &data));
    (void)FindClose(search);
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
    DWORD length;
    if (path == NULL || output == NULL || output_size == 0 || output_size > UINT32_MAX)
        return -1;
    length = GetFullPathNameA(path, (DWORD)output_size, output, NULL);
    return length > 0 && length < output_size ? 0 : -1;
}

int crossgpu_platform_path_equal(const char *left, const char *right)
{
    if (left == NULL || right == NULL)
        return 0;
    while (*left != '\0' && *right != '\0') {
        if (tolower((unsigned char)*left) != tolower((unsigned char)*right))
            return 0;
        left++;
        right++;
    }
    return *left == '\0' && *right == '\0';
}

int crossgpu_platform_get_executable_path(char *output, size_t output_size)
{
    DWORD length;
    if (output == NULL || output_size == 0 || output_size > UINT32_MAX)
        return -1;
    length = GetModuleFileNameA(NULL, output, (DWORD)output_size);
    return length > 0 && length < output_size ? 0 : -1;
}

uint64_t crossgpu_platform_now_ns(void)
{
    LARGE_INTEGER counter, frequency;
    uint64_t seconds, remainder;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&counter) || frequency.QuadPart <= 0)
        return 0;
    seconds = (uint64_t)(counter.QuadPart / frequency.QuadPart);
    remainder = (uint64_t)(counter.QuadPart % frequency.QuadPart);
    return seconds * UINT64_C(1000000000) +
           (remainder * UINT64_C(1000000000)) / (uint64_t)frequency.QuadPart;
}

void crossgpu_platform_sleep_ms(unsigned int milliseconds)
{
    Sleep((DWORD)milliseconds);
}

static unsigned __stdcall thread_trampoline(void *opaque)
{
    platform_thread_start_t *start = (platform_thread_start_t *)opaque;
    crossgpu_thread_fn function = start->function;
    void *argument = start->argument;
    free(start);
    (void)function(argument);
    return 0;
}

int crossgpu_platform_thread_create(crossgpu_thread_t *thread,
                                    crossgpu_thread_fn function,
                                    void *argument)
{
    platform_thread_start_t *start;
    uintptr_t handle;
    if (thread == NULL || function == NULL)
        return EINVAL;
    start = (platform_thread_start_t *)malloc(sizeof(*start));
    if (start == NULL)
        return ENOMEM;
    start->function = function;
    start->argument = argument;
    handle = _beginthreadex(NULL, 0, thread_trampoline, start, 0, NULL);
    if (handle == 0) {
        free(start);
        return EAGAIN;
    }
    *thread = (HANDLE)handle;
    return 0;
}

void crossgpu_platform_thread_join(crossgpu_thread_t thread)
{
    if (thread != NULL) {
        (void)WaitForSingleObject(thread, INFINITE);
        (void)CloseHandle(thread);
    }
}

void crossgpu_platform_mutex_lock(crossgpu_mutex_t *mutex)
{
    AcquireSRWLockExclusive(mutex);
}

void crossgpu_platform_mutex_unlock(crossgpu_mutex_t *mutex)
{
    ReleaseSRWLockExclusive(mutex);
}
