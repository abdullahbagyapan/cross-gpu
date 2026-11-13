#ifndef CROSSGPU_PLATFORM_H
#define CROSSGPU_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include "crossgpu_platform_config.h"

#if CROSSGPU_PLATFORM_WINDOWS
#  define CROSSGPU_PLATFORM_NAME "Windows"
#  define CROSSGPU_PLUGIN_PREFIX "crossgpu_backend_"
#  define CROSSGPU_PLUGIN_SUFFIX ".dll"
#  define CROSSGPU_PATH_SEPARATOR '\\'
#  define CROSSGPU_PATH_LIST_SEPARATOR ';'
#  define CROSSGPU_PATH_MAX 32768
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
typedef HANDLE crossgpu_thread_t;
typedef SRWLOCK crossgpu_mutex_t;
#  define CROSSGPU_MUTEX_INITIALIZER SRWLOCK_INIT
#elif CROSSGPU_PLATFORM_APPLE
#  define CROSSGPU_PLATFORM_NAME "Apple"
#  define CROSSGPU_PLUGIN_PREFIX "libcrossgpu_backend_"
#  define CROSSGPU_PLUGIN_SUFFIX ".dylib"
#  define CROSSGPU_PATH_SEPARATOR '/'
#  define CROSSGPU_PATH_LIST_SEPARATOR ':'
#  define CROSSGPU_PATH_MAX 4096
#  include <pthread.h>
typedef pthread_t crossgpu_thread_t;
typedef pthread_mutex_t crossgpu_mutex_t;
#  define CROSSGPU_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#elif CROSSGPU_PLATFORM_LINUX
#  define CROSSGPU_PLATFORM_NAME "Linux"
#  define CROSSGPU_PLUGIN_PREFIX "libcrossgpu_backend_"
#  define CROSSGPU_PLUGIN_SUFFIX ".so"
#  define CROSSGPU_PATH_SEPARATOR '/'
#  define CROSSGPU_PATH_LIST_SEPARATOR ':'
#  define CROSSGPU_PATH_MAX 4096
#  include <pthread.h>
typedef pthread_t crossgpu_thread_t;
typedef pthread_mutex_t crossgpu_mutex_t;
#  define CROSSGPU_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#endif

typedef void *(*crossgpu_thread_fn)(void *argument);

typedef struct crossgpu_directory_entry_list {
    char **items;
    size_t count;
} crossgpu_directory_entry_list_t;

/* Dynamic-library interface shared by the core and runtime-loaded plugins. */
void *crossgpu_platform_library_open(const char *path);
void *crossgpu_platform_library_symbol(void *library, const char *symbol);
void crossgpu_platform_library_close(void *library);
const char *crossgpu_platform_library_error(void);

/* Filesystem and executable-path operations. */
int crossgpu_platform_list_directory(const char *directory,
                                    crossgpu_directory_entry_list_t *entries);
void crossgpu_platform_free_directory_list(crossgpu_directory_entry_list_t *entries);
int crossgpu_platform_canonicalize_path(const char *path, char *output, size_t output_size);
int crossgpu_platform_path_equal(const char *left, const char *right);
int crossgpu_platform_get_executable_path(char *output, size_t output_size);

/* Monotonic timing and sleeping. */
uint64_t crossgpu_platform_now_ns(void);
void crossgpu_platform_sleep_ms(unsigned int milliseconds);

/* Thread and mutex operations. */
int crossgpu_platform_thread_create(crossgpu_thread_t *thread,
                                    crossgpu_thread_fn function,
                                    void *argument);
void crossgpu_platform_thread_join(crossgpu_thread_t thread);
void crossgpu_platform_mutex_lock(crossgpu_mutex_t *mutex);
void crossgpu_platform_mutex_unlock(crossgpu_mutex_t *mutex);

#endif
