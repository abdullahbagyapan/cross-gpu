#include "backend_manager.h"
#include "registry.h"
#include "util.h"
#include "platform.h"

#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_DURATION_NS (UINT64_C(60) * UINT64_C(1000000000))

#define ASCII_COLOR "\033[30m"
#define ASCII_COLOR_END "\033[0m"

static const char *const CROSSGPU_BANNER[] = {
    "                                                                                                                                                       ",
    "                                                                         #                                                                             ",
    "                                                             # ###      ## ##                                                                          ",
    "                                                             ##   ##### ###  ###                                                                       ",
    "                                                       ###########     ######   ##                                                                     ",
    "                                                    ##            ###      ####    ##                                                                  ",
    "                                                   ###############.  .### .    ###   ##                                                                ",
    "                                                 ###         .#####.##.   ##...  #     ##                                                              ",
    "                                              ###      ###.#         .#.######### ###    #                                                             ",
    "                                            ##    #######    ########     ######## ##### ###                                                           ",
    "                                           #  #####      ###### .# #######  ###.##.    ##  ## ##                                                       ",
    "                                          #######  ##  ####            ####..###       .##.  ### #                                                     ",
    "                                              ## .#.  ## ##          ###     .##.#####     ..  #  #                                                    ",
    "                                             ##  .  ### ##           # ######      ########         #                                                  ",
    "                                             #  ## ###  ###                  ###    ########## ##   #                                                  ",
    "                                              # .# ##    ##                     ###   ###. #.#  ## #                                                   ",
    "                                             ##  # ## .#  ###                      #   #####   ## #                                                    ",
    "                                         #######   ###  ##   ####                   ##    ###   #                                                      ",
    "                                    #####   ######  ###    #    ########            ##   ## #                                                          ",
    "                                 ###.  ##########     ####    ##         ###        ####   ##                                                          ",
    "                               ##  ########    #######  ######    ..#.##    ####     ### ##                                                            ",
    "                             ## #########  #####.#.####### ######    .#####.   ###   # #                                                               ",
    "                           ## ######### ########       #.#### #######   #.####   #####                                                                 ",
    "                     #####################################################################################################                             ",
    "                    ###        ######################################################          ####         ###   ####   #.                            ",
    "                   .##   #####   ###################################################   #######  ###  #####   ##   ####   #..                           ",
    "                   ###   ##########       #####        #####        ####         ###   ############  #####   ##   ####   ###                           ",
    "                   ###   ##########   ##   ###   ####   ###  ##########   ##########   ############  #####   ##   ####   ###                           ",
    "                  .###   ##########   ########   ####   ###        ####        #####            ###        ####   ####   ##.                           ",
    "                  ####   #####   ##   ########   ####   ##########  ###########  ###   #######  ###  ##########   ####   ##                            ",
    "                   ###          ###   ########         ####         ###         #####           ###  ###########        ##.                            ",
    "                   .#####################################################################################################.                             ",
    "                   ###################################################################################################.                                ",
    "                                                                                                          ####                                         ",
    "                     ##################################################################################     #####                                      ",
    "                                                                                                          ####   ########                              ",
    "                                                                                                                                                       ",                                                                                                                                                       
};

typedef struct run_result {
    crossgpu_test_type_t test;
    double elapsed_seconds;
    crossgpu_metrics_t metrics;
    int success;
    char error[CROSSGPU_ERROR_MAX];
} run_result_t;

typedef struct worker_context {
    size_t public_device_index;
    crossgpu_device_t *device;
    loaded_backend_t *active_backend;
    crossgpu_device_info_t active_info;
    uint64_t total_duration_ns;
    const crossgpu_test_type_t *tests;
    size_t test_count;
    run_result_t *results;
    int device_opened;
    char error[CROSSGPU_ERROR_MAX];
} worker_context_t;

static volatile sig_atomic_t g_stop_requested;
static volatile sig_atomic_t g_stop_signal;
static crossgpu_mutex_t g_output_mutex = CROSSGPU_MUTEX_INITIALIZER;

static void handle_signal(int signo)
{
    g_stop_signal = signo;
    g_stop_requested = 1;
}

static void print_banner(FILE *stream)
{
    size_t i;
    size_t line_count = sizeof(CROSSGPU_BANNER) / sizeof(CROSSGPU_BANNER[0]);

    fprintf(stream, "%s", ASCII_COLOR);
    for (i = 0; i < line_count; i++) {
        fprintf(stream, "%s\n", CROSSGPU_BANNER[i]);
    }
    fprintf(stream, "%s\n", ASCII_COLOR_END);
}

static void print_usage(FILE *stream)
{
    size_t i, test_count;
    const crossgpu_test_descriptor_t *tests = crossgpu_test_registry(&test_count);

    print_banner(stream);

    fprintf(stream,
        "CrossGPU\n"
        "Usage:\n"
        "  cross-gpu list\n"
        "  cross-gpu info\n"
        "  cross-gpu run [--device all|N] [--test TEST|all]\n"
        "                 [--duration TIME] [--backend auto|PLUGIN_NAME]\n"
        "  cross-gpu --help\n"
        "\n"
        "Commands:\n"
        "  list       List detected GPU devices\n"
        "  info       Show detailed information for detected GPUs\n"
        "  run        Run GPU stress workloads\n"
        "\n"
        "Options for run:\n"
        "  --device   Device index or 'all' (default: all)\n"
        "  --test     ");
    for (i = 0; i < test_count; i++)
        fprintf(stream, "%s%s", i ? ", " : "", tests[i].name);
    fprintf(stream,
        ", or all (default: all)\n"
        "  --duration Duration such as 60s, 5m, 1h (default: 60s)\n"
        "  --backend  auto or a loaded backend plugin name (default: auto)\n"
        "\n"
        "--duration is total elapsed workload time per selected device. With\n"
        "multiple tests, it is divided equally; each test receives total/test-count.\n"
        "\n"
        "Backend plugins are loaded from CROSSGPU_PLUGIN_PATH, the executable's\n"
        "backends/ directory, ./backends, or the installation plugin directory.\n"
        "Set CROSSGPU_PLUGIN_PATH_ONLY=1 to disable fallback plugin directories.\n");
}

static int parse_test(const char *text, crossgpu_test_type_t **tests, size_t *count)
{
    crossgpu_test_type_t parsed;
    crossgpu_test_type_t *result;
    size_t total;

    if (text == NULL || tests == NULL || count == NULL)
        return -1;
    *tests = NULL;
    *count = 0;

    if (strcmp(text, "all") == 0) {
        total = crossgpu_test_default_count();
        if (total == 0)
            return -1;
        result = calloc(total, sizeof(*result));
        if (result == NULL)
            return -1;
        if (crossgpu_test_copy_default(result, total) != total) {
            free(result);
            return -1;
        }
        *tests = result;
        *count = total;
        return 0;
    }

    if (crossgpu_test_type_parse(text, &parsed) != 0)
        return -1;
    result = malloc(sizeof(*result));
    if (result == NULL)
        return -1;
    result[0] = parsed;
    *tests = result;
    *count = 1;
    return 0;
}

static int parse_device(const char *text, int *all_devices, size_t *index)
{
    char *end = NULL;
    unsigned long long value;
    if (strcmp(text, "all") == 0) {
        *all_devices = 1;
        *index = 0;
        return 0;
    }
    if (*text == '\0' || *text == '-')
        return -1;
    value = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value > SIZE_MAX)
        return -1;
    *all_devices = 0;
    *index = (size_t)value;
    return 0;
}

static void print_device_list(const backend_manager_t *manager)
{
    size_t i;
    if (manager->device_count == 0) {
        puts("No supported GPU devices were detected.");
        puts("Check that a vendor driver and at least one backend plugin are installed.");
        return;
    }

    printf("%-4s %-9s %-18s %-10s %-12s %s\n",
           "ID", "PREFERRED", "VENDOR", "COMPUTE", "MEMORY", "DEVICE");
    printf("------------------------------------------------------------------------------------------------\n");
    for (i = 0; i < manager->device_count; i++) {
        const crossgpu_device_info_t *d = &manager->devices[i].info;
        char memory[32], compute[16];
        if (d->compute_units)
            (void)snprintf(compute, sizeof(compute), "%u", d->compute_units);
        else
            crossgpu_copy_string(compute, sizeof(compute), "n/a");
        if (d->physical_memory_bytes)
            crossgpu_format_bytes(d->physical_memory_bytes, memory, sizeof(memory));
        else
            crossgpu_copy_string(memory, sizeof(memory), "unknown");
        printf("%-4zu %-9s %-18.18s %-10s %-12s %s\n",
               i,
               manager->devices[i].backend->api->name,
               d->vendor,
               compute,
               memory,
               d->name);
    }
}

static void print_device_info(const backend_manager_t *manager)
{
    size_t i;
    if (manager->device_count == 0) {
        print_device_list(manager);
        return;
    }
    for (i = 0; i < manager->device_count; i++) {
        const crossgpu_device_info_t *d = &manager->devices[i].info;
        char physical[32], working_set[32], max_alloc[32];
        crossgpu_format_bytes(d->physical_memory_bytes, physical, sizeof(physical));
        crossgpu_format_bytes(d->working_set_bytes, working_set, sizeof(working_set));
        crossgpu_format_bytes(d->max_allocation_bytes, max_alloc, sizeof(max_alloc));
        printf("GPU %zu: %s\n", i, d->name);
        printf("  Preferred backend: %s\n", manager->devices[i].backend->api->name);
        printf("  Vendor:           %s (0x%04" PRIx32 ")\n",
               d->vendor[0] ? d->vendor : crossgpu_vendor_id_name(d->vendor_id),
               (uint32_t)d->vendor_id);
        if (d->device_id != 0)
            printf("  Device ID:        0x%04" PRIx32 "\n", d->device_id);
        else
            printf("  Device ID:        not exposed by this backend\n");
        printf("  Driver version:   %s\n", d->driver_version[0] ? d->driver_version : "unknown");
        printf("  API version:      %s\n", d->api_version[0] ? d->api_version : "unknown");
        if (d->compute_units)
            printf("  Compute units:    %u\n", d->compute_units);
        else
            printf("  Compute units:    not exposed by this backend\n");
        printf("  Clock:            %s", d->clock_mhz ? "" : "unknown\n");
        if (d->clock_mhz)
            printf("%u MHz\n", d->clock_mhz);
        printf("  Physical memory:  %s\n", d->physical_memory_bytes ? physical : "not exposed");
        printf("  Working set:      %s\n", d->working_set_bytes ? working_set : "not exposed");
        printf("  Max allocation:   %s\n", d->max_allocation_bytes ? max_alloc : "not exposed");
        printf("  Backend device ID: %s\n",
               d->backend_device_id[0] ? d->backend_device_id : "not provided");
        if (d->identity_kind > CROSSGPU_IDENTITY_NONE &&
            d->identity_namespace[0] != '\0' && d->identity_value[0] != '\0') {
            printf("  Physical identity: kind=%s namespace=%s value=%s\n",
                   crossgpu_identity_kind_name(d->identity_kind),
                   d->identity_namespace, d->identity_value);
        } else {
            printf("  Physical identity: not exposed\n");
        }
        printf("  Preferred plugin: %s\n", manager->devices[i].backend->api->name);
        if (manager->devices[i].candidate_count > 1) {
            size_t candidate;
            printf("  Fallbacks:        ");
            for (candidate = 1; candidate < manager->devices[i].candidate_count; candidate++) {
                printf("%s%s", candidate > 1 ? ", " : "",
                       manager->devices[i].candidates[candidate].backend->api->name);
            }
            putchar('\n');
        }
        if (i + 1 < manager->device_count)
            putchar('\n');
    }
}

static int run_one_test(worker_context_t *worker,
                        void *device_context,
                        crossgpu_test_type_t test,
                        uint64_t duration_ns,
                        run_result_t *result)
{
    const crossgpu_backend_plugin_t *api = worker->active_backend->api;
    crossgpu_workload_config_t config;
    void *workload_context = NULL;
    uint64_t start, deadline;
    char error[CROSSGPU_ERROR_MAX] = { 0 };
    int status = 0;

    memset(result, 0, sizeof(*result));
    result->test = test;
    config.struct_size = sizeof(config);
    config.memory_bytes = 0;
    config.batch_iterations = 1;

    if (api->workload_begin(device_context, test, &config, &workload_context,
                            error, sizeof(error)) != 0) {
        result->success = 0;
        crossgpu_copy_string(result->error, sizeof(result->error), error[0] ? error : "workload initialization failed");
        return -1;
    }

    start = crossgpu_now_ns();
    deadline = start + duration_ns;
    result->success = 1;

    do {
        crossgpu_metrics_t step_metrics;
        memset(&step_metrics, 0, sizeof(step_metrics));
        error[0] = '\0';
        if (api->workload_step(device_context, test, workload_context,
                               &step_metrics, error, sizeof(error)) != 0) {
            result->success = 0;
            crossgpu_copy_string(result->error, sizeof(result->error), error[0] ? error : "workload step failed");
            status = -1;
            break;
        }
        result->metrics.operations += step_metrics.operations;
        result->metrics.bytes_read += step_metrics.bytes_read;
        result->metrics.bytes_written += step_metrics.bytes_written;
        result->metrics.errors += step_metrics.errors;
        result->metrics.iterations += step_metrics.iterations;
    } while (!g_stop_requested && crossgpu_now_ns() < deadline);

    result->elapsed_seconds = (double)(crossgpu_now_ns() - start) / 1.0e9;
    api->workload_end(device_context, test, workload_context);
    if (g_stop_requested && result->metrics.iterations == 0)
        result->success = 0;
    return status;
}

static void *worker_main(void *argument)
{
    worker_context_t *worker = (worker_context_t *)argument;
    void *device_context = NULL;
    const crossgpu_backend_plugin_t *api = NULL;
    size_t i, candidate;
    char error[CROSSGPU_ERROR_MAX] = { 0 };
    uint64_t per_test_duration = worker->total_duration_ns / worker->test_count;
    int failed = 0;

    for (candidate = 0; candidate < worker->device->candidate_count; candidate++) {
        crossgpu_device_candidate_t *choice = &worker->device->candidates[candidate];
        api = choice->backend->api;
        error[0] = '\0';
        device_context = NULL;
        if (api->device_open(choice->info.backend_device_index, &device_context,
                             error, sizeof(error)) == 0) {
            worker->active_backend = choice->backend;
            worker->active_info = choice->info;
            worker->device_opened = 1;
            if (candidate > 0) {
                fprintf(stderr, "cross-gpu: preferred backend failed; using %s for %s\n",
                        api->name, choice->info.name);
            }
            break;
        }
        fprintf(stderr, "cross-gpu: %s could not initialize %s: %s\n",
                api->name, choice->info.name, error[0] ? error : "device initialization failed");
    }
    if (device_context == NULL || api == NULL) {
        crossgpu_copy_string(worker->error, sizeof(worker->error),
                        error[0] ? error : "all compatible backends failed to initialize this GPU");
        return NULL;
    }

    for (i = 0; i < worker->test_count && !g_stop_requested; i++) {
        crossgpu_platform_mutex_lock(&g_output_mutex);
        printf("GPU %zu: starting %s test with %s\n",
               worker->public_device_index,
               crossgpu_test_type_name(worker->tests[i]),
               api->name);
        crossgpu_platform_mutex_unlock(&g_output_mutex);
        run_one_test(worker, device_context, worker->tests[i],
                     per_test_duration, &worker->results[i]);
        if (!worker->results[i].success) {
            failed = 1;
            break;
        }
    }

    if (failed && worker->error[0] == '\0')
        crossgpu_copy_string(worker->error, sizeof(worker->error), "one or more tests failed");
    api->device_close(device_context);
    return NULL;
}

static void print_run_result(const worker_context_t *worker)
{
    size_t i;
    crossgpu_platform_mutex_lock(&g_output_mutex);
    if (worker->error[0]) {
        printf("GPU %zu (%s): FAILED: %s\n", worker->public_device_index,
               worker->device->info.name, worker->error);
    }
    for (i = 0; i < worker->test_count; i++) {
        const run_result_t *r = &worker->results[i];
        double transferred = (double)(r->metrics.bytes_read + r->metrics.bytes_written);
        double bandwidth = r->elapsed_seconds > 0.0 ? transferred / r->elapsed_seconds / 1.0e9 : 0.0;
        double gflops = r->elapsed_seconds > 0.0 ? (double)r->metrics.operations / r->elapsed_seconds / 1.0e9 : 0.0;
        if (r->elapsed_seconds == 0.0 && r->metrics.iterations == 0 && !r->error[0])
            continue;
        printf("GPU %zu [%s] | %-9s | %s | %.2f s | iterations: %" PRIu64,
               worker->public_device_index,
               worker->active_backend ? worker->active_backend->api->name : "unavailable",
               crossgpu_test_type_name(r->test),
               r->success ? "PASS" : "FAIL",
               r->elapsed_seconds,
               r->metrics.iterations);
        if (r->elapsed_seconds > 0.0) {
            const char *rate_unit = crossgpu_test_rate_unit(r->test);
            if (strcmp(rate_unit, "GFLOP/s") == 0)
                printf(" | %.2f %s", gflops, rate_unit);
            else if (rate_unit[0] != '\0')
                printf(" | %.2f %s", bandwidth, rate_unit);
        }
        printf(" | errors: %" PRIu64, r->metrics.errors);
        if (r->error[0])
            printf(" | %s", r->error);
        putchar('\n');
    }
    crossgpu_platform_mutex_unlock(&g_output_mutex);
}

static int command_run(backend_manager_t *manager,
                       int argc,
                       char **argv,
                       const char *executable_path)
{
    crossgpu_test_type_t *tests = NULL;
    size_t test_count = 0;
    uint64_t duration_ns = DEFAULT_DURATION_NS;
    const char *backend_filter = "auto";
    int all_devices = 1;
    size_t selected_index = 0;
    int argument_index;
    size_t selected_count = 0, i;
    worker_context_t *workers = NULL;
    crossgpu_thread_t *threads = NULL;
    int *thread_started = NULL;
    int overall_status = 0;

    if (parse_test("all", &tests, &test_count) != 0) {
        fprintf(stderr, "cross-gpu: unable to initialize default test registry\n");
        return 1;
    }

    /* Parse long options directly so the core does not depend on getopt_long,
     * which is not part of the Microsoft C runtime. */
    for (argument_index = 2; argument_index < argc; argument_index++) {
        const char *argument = argv[argument_index];
        const char *value = NULL;
        const char *option_name = argument;
        const char *equals = strchr(argument, '=');
        size_t option_length = equals != NULL ? (size_t)(equals - argument) : strlen(argument);

        if (equals != NULL)
            value = equals + 1;
        if (option_length == 8 && strncmp(option_name, "--device", option_length) == 0) {
            if (value == NULL && argument_index + 1 < argc)
                value = argv[++argument_index];
            if (value == NULL || parse_device(value, &all_devices, &selected_index) != 0) {
                fprintf(stderr, "cross-gpu: invalid or missing --device value\n");
                free(tests);
                return 2;
            }
        } else if (option_length == 6 && strncmp(option_name, "--test", option_length) == 0) {
            crossgpu_test_type_t *selected_tests = NULL;
            size_t selected_test_count = 0;
            if (value == NULL && argument_index + 1 < argc)
                value = argv[++argument_index];
            if (value == NULL || parse_test(value, &selected_tests, &selected_test_count) != 0) {
                fprintf(stderr, "cross-gpu: invalid or missing --test value\n");
                free(tests);
                return 2;
            }
            free(tests);
            tests = selected_tests;
            test_count = selected_test_count;
        } else if (option_length == 10 && strncmp(option_name, "--duration", option_length) == 0) {
            if (value == NULL && argument_index + 1 < argc)
                value = argv[++argument_index];
            if (value == NULL || crossgpu_parse_duration(value, &duration_ns) != 0) {
                fprintf(stderr, "cross-gpu: invalid or missing --duration value (examples: 60s, 5m, 1h)\n");
                free(tests);
                return 2;
            }
        } else if (option_length == 9 && strncmp(option_name, "--backend", option_length) == 0) {
            if (value == NULL && argument_index + 1 < argc)
                value = argv[++argument_index];
            if (value == NULL || value[0] == '\0') {
                fprintf(stderr, "cross-gpu: invalid or missing --backend value\n");
                free(tests);
                return 2;
            }
            backend_filter = value;
        } else if ((strcmp(argument, "--help") == 0) || (strcmp(argument, "-h") == 0)) {
            print_usage(stdout);
            free(tests);
            return 0;
        } else {
            fprintf(stderr, "cross-gpu: unknown option or unexpected argument: %s\n", argument);
            print_usage(stderr);
            free(tests);
            return 2;
        }
    }

    if (backend_manager_discover(manager, executable_path, backend_filter) < 0) {
        fprintf(stderr, "cross-gpu: backend discovery failed\n");
        free(tests);
        return 1;
    }
    if (manager->device_count == 0) {
        fprintf(stderr, "cross-gpu: no usable GPU devices found for backend '%s'\n",
                backend_filter);
        free(tests);
        return 2;
    }
    if (!all_devices && selected_index >= manager->device_count) {
        fprintf(stderr, "cross-gpu: device index %zu is out of range (detected %zu devices)\n",
                selected_index, manager->device_count);
        free(tests);
        return 2;
    }

    selected_count = all_devices ? manager->device_count : 1;
    workers = calloc(selected_count, sizeof(*workers));
    threads = calloc(selected_count, sizeof(*threads));
    thread_started = calloc(selected_count, sizeof(*thread_started));
    if (workers == NULL || threads == NULL || thread_started == NULL) {
        fprintf(stderr, "cross-gpu: out of memory while preparing workers\n");
        free(workers);
        free(threads);
        free(thread_started);
        free(tests);
        return 1;
    }
    for (i = 0; i < selected_count; i++) {
        workers[i].results = calloc(test_count, sizeof(*workers[i].results));
        if (workers[i].results == NULL) {
            size_t j;
            fprintf(stderr, "cross-gpu: out of memory allocating workload results\n");
            for (j = 0; j < i; j++)
                free(workers[j].results);
            free(workers);
            free(threads);
            free(thread_started);
            free(tests);
            return 1;
        }
    }

    printf("Starting GPU stress test: %zu device(s), %zu workload(s); "
           "total duration %.2f s; per-workload duration %.2f s; backend %s\n",
           selected_count, test_count,
           (double)duration_ns / 1.0e9,
           (double)(duration_ns / test_count) / 1.0e9, backend_filter);

    for (i = 0; i < selected_count; i++) {
        size_t device_index = all_devices ? i : selected_index;
        workers[i].public_device_index = device_index;
        workers[i].device = &manager->devices[device_index];
        workers[i].total_duration_ns = duration_ns;
        workers[i].test_count = test_count;
        workers[i].tests = tests;
        printf("  GPU %zu: %s [%s]\n", device_index,
               workers[i].device->info.name,
               workers[i].device->backend->api->name);
    }

    g_stop_requested = 0;
    g_stop_signal = 0;
    (void)signal(SIGINT, handle_signal);
#ifdef SIGTERM
    (void)signal(SIGTERM, handle_signal);
#endif

    for (i = 0; i < selected_count; i++) {
        int rc = crossgpu_platform_thread_create(&threads[i], worker_main, &workers[i]);
        if (rc == 0) {
            thread_started[i] = 1;
        } else {
            (void)snprintf(workers[i].error, sizeof(workers[i].error),
                           "worker thread creation failed (error %d)", rc);
            overall_status = 1;
        }
    }

    for (i = 0; i < selected_count; i++) {
        if (thread_started[i])
            crossgpu_platform_thread_join(threads[i]);
        print_run_result(&workers[i]);
        if (workers[i].error[0])
            overall_status = 1;
        {
            size_t j;
            for (j = 0; j < workers[i].test_count; j++) {
                if (!workers[i].results[j].success && workers[i].results[j].metrics.iterations > 0)
                    overall_status = 1;
                if (workers[i].results[j].metrics.errors > 0)
                    overall_status = 1;
            }
        }
    }

    if (g_stop_requested)
        puts("Stress run interrupted.");
    else if (overall_status == 0)
        puts("Stress run completed.");
    else
        puts("Stress run completed with errors.");

    if (g_stop_requested)
        overall_status = 128 + (int)g_stop_signal;

    for (i = 0; i < selected_count; i++)
        free(workers[i].results);
    free(workers);
    free(threads);
    free(thread_started);
    free(tests);
    (void)executable_path;
    return overall_status;
}

int main(int argc, char **argv)
{
    backend_manager_t manager;
    int result = 0;
    const char *command;
    char executable_path[CROSSGPU_PATH_MAX];

    executable_path[0] = '\0';
    if (crossgpu_platform_get_executable_path(executable_path, sizeof(executable_path)) != 0)
        crossgpu_copy_string(executable_path, sizeof(executable_path), argv[0]);

    if (argc < 2) {
        print_usage(stderr);
        return 2;
    }
    command = argv[1];
    memset(&manager, 0, sizeof(manager));

    if (strcmp(command, "--help") == 0 || strcmp(command, "-h") == 0 || strcmp(command, "help") == 0) {
        print_usage(stdout);
        return 0;
    }

    if (strcmp(command, "list") == 0 || strcmp(command, "info") == 0) {
        if (argc != 2) {
            fprintf(stderr, "cross-gpu: '%s' does not accept additional arguments\n", command);
            return 2;
        }
        (void)backend_manager_discover(&manager, executable_path, "auto");
        if (strcmp(command, "list") == 0)
            print_device_list(&manager);
        else
            print_device_info(&manager);
        backend_manager_destroy(&manager);
        return 0;
    }

    if (strcmp(command, "run") == 0) {
        result = command_run(&manager, argc, argv, executable_path);
        backend_manager_destroy(&manager);
        return result;
    }

    fprintf(stderr, "cross-gpu: unknown command: %s\n", command);
    print_usage(stderr);
    return 2;
}
