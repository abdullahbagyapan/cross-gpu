#include "registry.h"

#include <stdio.h>
#include <stdlib.h>

static int expect_score(const char *description,
                        crossgpu_vendor_id_t vendor,
                        crossgpu_backend_kind_t kind,
                        const char *backend,
                        int expected)
{
    const int actual = crossgpu_vendor_backend_score(vendor, kind, backend);
    if (actual != expected) {
        fprintf(stderr, "%s: expected score %d, got %d\n",
                description, expected, actual);
        return 1;
    }
    return 0;
}

int main(void)
{
    int failures = 0;

    failures += expect_score("NVIDIA CUDA preference", CROSSGPU_VENDOR_NVIDIA,
                             CROSSGPU_BACKEND_CUDA, "cuda", 110);
    failures += expect_score("NVIDIA OpenCL fallback", CROSSGPU_VENDOR_NVIDIA,
                             CROSSGPU_BACKEND_OPENCL, "opencl", 50);
    failures += expect_score("Intel Level Zero preference", CROSSGPU_VENDOR_INTEL,
                             CROSSGPU_BACKEND_LEVEL_ZERO, "level-zero", 120);
    failures += expect_score("Intel OpenCL fallback", CROSSGPU_VENDOR_INTEL,
                             CROSSGPU_BACKEND_OPENCL, "opencl", 50);
    failures += expect_score("AMD HIP preference", CROSSGPU_VENDOR_AMD,
                             CROSSGPU_BACKEND_HIP, "hip", 110);
    failures += expect_score("AMD OpenCL fallback", CROSSGPU_VENDOR_AMD,
                             CROSSGPU_BACKEND_OPENCL, "opencl", 50);
    failures += expect_score("Legacy AMD HIP preference", CROSSGPU_VENDOR_AMD_LEGACY,
                             CROSSGPU_BACKEND_HIP, "hip", 110);
    failures += expect_score("Legacy AMD OpenCL fallback", CROSSGPU_VENDOR_AMD_LEGACY,
                             CROSSGPU_BACKEND_OPENCL, "opencl", 50);
    failures += expect_score("Apple Metal preference", CROSSGPU_VENDOR_APPLE,
                             CROSSGPU_BACKEND_METAL, "metal", 110);
    failures += expect_score("Apple OpenCL fallback", CROSSGPU_VENDOR_APPLE,
                             CROSSGPU_BACKEND_OPENCL, "opencl", 50);
    failures += expect_score("Qualcomm OpenCL preference", CROSSGPU_VENDOR_QUALCOMM,
                             CROSSGPU_BACKEND_OPENCL, "opencl", 100);
    failures += expect_score("Arm OpenCL preference", CROSSGPU_VENDOR_ARM,
                             CROSSGPU_BACKEND_OPENCL, "opencl", 100);
    failures += expect_score("Broadcom OpenCL preference", CROSSGPU_VENDOR_BROADCOM,
                             CROSSGPU_BACKEND_OPENCL, "opencl", 100);
    failures += expect_score("Imagination OpenCL preference", CROSSGPU_VENDOR_IMAGINATION,
                             CROSSGPU_BACKEND_OPENCL, "opencl", 100);
    failures += expect_score("Generic vendor OpenCL compatibility",
                             (crossgpu_vendor_id_t)0x1234, CROSSGPU_BACKEND_OPENCL,
                             "opencl", 50);
    failures += expect_score("Unknown vendor OpenCL compatibility",
                             CROSSGPU_VENDOR_UNKNOWN, CROSSGPU_BACKEND_OPENCL,
                             "opencl", 50);
    failures += expect_score("Level Zero rejected for NVIDIA",
                             CROSSGPU_VENDOR_NVIDIA, CROSSGPU_BACKEND_LEVEL_ZERO,
                             "level-zero", 0);
    failures += expect_score("CUDA rejected for Intel",
                             CROSSGPU_VENDOR_INTEL, CROSSGPU_BACKEND_CUDA,
                             "cuda", 0);
    failures += expect_score("Unregistered plugin rejected",
                             CROSSGPU_VENDOR_NVIDIA, CROSSGPU_BACKEND_CUSTOM,
                             "vulkan", 0);
    failures += expect_score("Unknown plugin rejected",
                             (crossgpu_vendor_id_t)0x1234, CROSSGPU_BACKEND_CUSTOM,
                             "unknown-backend", 0);
    failures += expect_score("Mismatched kind cannot borrow OpenCL wildcard",
                             (crossgpu_vendor_id_t)0x1234, CROSSGPU_BACKEND_LEVEL_ZERO,
                             "opencl", 0);
    failures += expect_score("Empty backend name rejected",
                             CROSSGPU_VENDOR_NVIDIA, CROSSGPU_BACKEND_OPENCL,
                             "", 0);

    if (failures != 0) {
        fprintf(stderr, "%d backend preference registry test(s) failed\n", failures);
        return EXIT_FAILURE;
    }

    puts("Backend preference registry tests passed.");
    return EXIT_SUCCESS;
}
