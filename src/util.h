#ifndef CROSSGPU_UTIL_H
#define CROSSGPU_UTIL_H

#include <stddef.h>
#include <stdint.h>

uint64_t crossgpu_now_ns(void);
void crossgpu_sleep_ms(unsigned int milliseconds);
void crossgpu_copy_string(char *destination, size_t destination_size, const char *source);
int crossgpu_parse_duration(const char *text, uint64_t *nanoseconds);
void crossgpu_format_bytes(uint64_t bytes, char *buffer, size_t buffer_size);

#endif
