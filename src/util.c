#include "util.h"
#include "platform.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint64_t crossgpu_now_ns(void)
{
    return crossgpu_platform_now_ns();
}

void crossgpu_sleep_ms(unsigned int milliseconds)
{
    crossgpu_platform_sleep_ms(milliseconds);
}

void crossgpu_copy_string(char *destination, size_t destination_size, const char *source)
{
    if (destination == NULL || destination_size == 0)
        return;
    if (source == NULL)
        source = "";
    (void)snprintf(destination, destination_size, "%s", source);
}

int crossgpu_parse_duration(const char *text, uint64_t *nanoseconds)
{
    char *end = NULL;
    long double value;
    long double multiplier = 1.0L;
    long double result;

    if (text == NULL || nanoseconds == NULL || *text == '\0')
        return -1;

    errno = 0;
    value = strtold(text, &end);
    if (errno != 0 || end == text || !isfinite((double)value) || value <= 0.0L)
        return -1;

    if (*end == '\0' || strcmp(end, "s") == 0)
        multiplier = 1.0L;
    else if (strcmp(end, "ms") == 0)
        multiplier = 0.001L;
    else if (strcmp(end, "m") == 0)
        multiplier = 60.0L;
    else if (strcmp(end, "h") == 0)
        multiplier = 3600.0L;
    else
        return -1;

    result = value * multiplier * 1000000000.0L;
    if (result < 1000000.0L || result > 604800.0L * 1000000000.0L)
        return -1;

    *nanoseconds = (uint64_t)result;
    return 0;
}

void crossgpu_format_bytes(uint64_t bytes, char *buffer, size_t buffer_size)
{
    static const char *units[] = { "B", "KiB", "MiB", "GiB", "TiB" };
    double value = (double)bytes;
    size_t unit = 0;

    while (value >= 1024.0 && unit + 1 < sizeof(units) / sizeof(units[0])) {
        value /= 1024.0;
        unit++;
    }
    (void)snprintf(buffer, buffer_size, "%.2f %s", value, units[unit]);
}

