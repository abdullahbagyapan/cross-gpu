#ifndef CROSSGPU_PLATFORM_CONFIG_H
#define CROSSGPU_PLATFORM_CONFIG_H

/*
 * Central compile-target detection shared by the public plugin ABI and the
 * internal platform adapter. Keep compiler/OS predefined macro checks here;
 * the rest of CrossGPU should use CROSSGPU_PLATFORM_* instead.
 */
#if defined(_WIN32) || defined(_WIN64)
#  define CROSSGPU_PLATFORM_WINDOWS 1
#  define CROSSGPU_PLATFORM_LINUX 0
#  define CROSSGPU_PLATFORM_APPLE 0
#elif defined(__APPLE__) && defined(__MACH__)
#  define CROSSGPU_PLATFORM_WINDOWS 0
#  define CROSSGPU_PLATFORM_LINUX 0
#  define CROSSGPU_PLATFORM_APPLE 1
#elif defined(__linux__)
#  define CROSSGPU_PLATFORM_WINDOWS 0
#  define CROSSGPU_PLATFORM_LINUX 1
#  define CROSSGPU_PLATFORM_APPLE 0
#else
#  error "CrossGPU supports Linux, Windows, and Apple platforms"
#endif

#endif
