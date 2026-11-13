#ifndef CROSSGPU_REGISTRY_H
#define CROSSGPU_REGISTRY_H

#include "crossgpu_backend.h"

#include <stddef.h>
#include <stdint.h>

typedef struct crossgpu_backend_descriptor {
    crossgpu_backend_kind_t kind;
    const char *name;
    const char *display_name;
} crossgpu_backend_descriptor_t;

typedef enum crossgpu_test_flags {
    CROSSGPU_TEST_FLAG_NONE = 0,
    CROSSGPU_TEST_FLAG_DEFAULT_ALL = 1u << 0,
    CROSSGPU_TEST_FLAG_COMPUTE_RATE = 1u << 1,
    CROSSGPU_TEST_FLAG_BANDWIDTH_RATE = 1u << 2
} crossgpu_test_flags_t;

typedef struct crossgpu_test_descriptor {
    crossgpu_test_type_t type;
    const char *name;
    const char *description;
    unsigned int flags;
} crossgpu_test_descriptor_t;

typedef struct crossgpu_vendor_descriptor {
    crossgpu_vendor_id_t vendor_id;
    const char *name;
} crossgpu_vendor_descriptor_t;

typedef struct crossgpu_backend_preference_descriptor {
    uint32_t vendor_id;
    crossgpu_backend_kind_t backend_kind;
    const char *backend_name;
    int score;
} crossgpu_backend_preference_descriptor_t;

/* Internal registry wildcard; never use this as a device-reported vendor ID. */
#define CROSSGPU_VENDOR_ANY UINT32_MAX

const crossgpu_backend_descriptor_t *crossgpu_backend_registry(size_t *count);
const crossgpu_test_descriptor_t *crossgpu_test_registry(size_t *count);
const crossgpu_vendor_descriptor_t *crossgpu_vendor_registry(size_t *count);
const crossgpu_backend_preference_descriptor_t *crossgpu_backend_preference_registry(size_t *count);
int crossgpu_vendor_backend_score(crossgpu_vendor_id_t vendor_id,
                                  crossgpu_backend_kind_t backend_kind,
                                  const char *backend_name);

int crossgpu_backend_kind_parse(const char *name, crossgpu_backend_kind_t *kind);
const char *crossgpu_identity_kind_name(crossgpu_identity_kind_t kind);
int crossgpu_test_type_parse(const char *name, crossgpu_test_type_t *test);
const crossgpu_test_descriptor_t *crossgpu_test_descriptor(crossgpu_test_type_t test);
size_t crossgpu_test_default_count(void);
size_t crossgpu_test_copy_default(crossgpu_test_type_t *tests, size_t capacity);
const char *crossgpu_test_rate_unit(crossgpu_test_type_t test);

#endif
