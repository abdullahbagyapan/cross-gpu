CC ?= cc
DESTDIR ?=
BUILD_DIR ?= build

WITH_OPENCL ?= 1
WITH_CUDA ?= 0
WITH_HIP ?= 0
WITH_METAL ?= 0
WITH_LEVEL_ZERO ?= 0

NVCC ?= nvcc
HIPCC ?= hipcc
METAL_CC ?= clang
CXX ?= c++
OCLOC ?= ocloc
LEVEL_ZERO_CXXFLAGS ?= -O2 -g
LEVEL_ZERO_CPPFLAGS ?=
NVCCFLAGS ?= -O2 -std=c++14
HIPCCFLAGS ?= -O2 -std=c++14
METALFLAGS ?= -O2

# Select one explicit OS adapter and its native plugin naming convention.
ifeq ($(OS),Windows_NT)
    CROSSGPU_OS := Windows
else
    HOST_OS := $(shell uname -s 2>/dev/null)
    ifeq ($(HOST_OS),Darwin)
        CROSSGPU_OS := Apple
    else ifeq ($(HOST_OS),Linux)
        CROSSGPU_OS := Linux
    else
        $(error Unsupported host OS '$(HOST_OS)'; supported targets are Linux, Windows, and Apple)
    endif
endif

ifeq ($(CROSSGPU_OS),Windows)
    # GNU Make's built-in CC is often `cc` (MSYS runtime) on Windows;
    # force the native UCRT64 compiler unless explicitly overridden.
    ifeq ($(origin CC),default)
        CC := gcc
    endif
    ifeq ($(origin CXX),default)
        CXX := g++
    endif
    PREFIX ?= C:/CrossGPU
    PLATFORM_SOURCE := src/platform_windows.c
    PLUGIN_PREFIX := crossgpu_backend_
    PLUGIN_SUFFIX := dll
    EXE_SUFFIX := .exe
    SHARED_FLAGS := -shared
    PLUGIN_CFLAGS := -fvisibility=hidden
    DYNAMIC_LOADER_LIBS :=
    THREAD_LIBS :=
    OPENCL_PLATFORM_LIBS :=
    PATH_LIST_SEPARATOR := ;
    CUDA_PLUGIN_FLAGS := -Xcompiler=/MD -shared
    HIP_PLUGIN_FLAGS := -shared
    LEVEL_ZERO_LIBS ?= ze_loader.lib
else ifeq ($(CROSSGPU_OS),Apple)
    PREFIX ?= /usr/local
    PLATFORM_SOURCE := src/platform_apple.c
    PLUGIN_PREFIX := libcrossgpu_backend_
    PLUGIN_SUFFIX := dylib
    EXE_SUFFIX :=
    SHARED_FLAGS := -dynamiclib
    PLUGIN_CFLAGS := -fPIC -fvisibility=hidden
    DYNAMIC_LOADER_LIBS :=
    THREAD_LIBS := -pthread
    OPENCL_PLATFORM_LIBS := -framework OpenGL
    PATH_LIST_SEPARATOR := :
    CUDA_PLUGIN_FLAGS := -Xcompiler=-fPIC,-fvisibility=hidden -shared
    HIP_PLUGIN_FLAGS := -fPIC -fvisibility=hidden -shared
    LEVEL_ZERO_LIBS ?= -lze_loader
else
    PREFIX ?= /usr/local
    PLATFORM_SOURCE := src/platform_linux.c
    PLUGIN_PREFIX := libcrossgpu_backend_
    PLUGIN_SUFFIX := so
    EXE_SUFFIX :=
    SHARED_FLAGS := -shared
    PLUGIN_CFLAGS := -fPIC -fvisibility=hidden
    DYNAMIC_LOADER_LIBS := -ldl
    THREAD_LIBS := -pthread
    OPENCL_PLATFORM_LIBS :=
    PATH_LIST_SEPARATOR := :
    CUDA_PLUGIN_FLAGS := -Xcompiler=-fPIC,-fvisibility=hidden -shared
    HIP_PLUGIN_FLAGS := -fPIC -fvisibility=hidden -shared
    LEVEL_ZERO_LIBS ?= -lze_loader
endif

# Convert paths from the build shell to the target executable's format.
# Native Windows executables require Windows paths even when GNU Make runs
# inside the MSYS2 POSIX environment.
ifeq ($(CROSSGPU_OS),Windows)
    native_path = $(shell cygpath -m "$(1)")
else
    native_path = $(1)
endif

ifeq ($(WITH_METAL),1)
    ifneq ($(CROSSGPU_OS),Apple)
        $(error WITH_METAL=1 requires Apple macOS and the Metal SDK)
    endif
endif
ifeq ($(WITH_CUDA),1)
    ifneq ($(CROSSGPU_OS),Windows)
        ifneq ($(CROSSGPU_OS),Linux)
            $(error WITH_CUDA=1 requires Linux or Windows and the CUDA Toolkit)
        endif
    endif
endif
ifeq ($(WITH_HIP),1)
    ifneq ($(CROSSGPU_OS),Windows)
        ifneq ($(CROSSGPU_OS),Linux)
            $(error WITH_HIP=1 requires Linux or Windows and the HIP SDK)
        endif
    endif
endif
ifeq ($(WITH_LEVEL_ZERO),1)
    ifneq ($(CROSSGPU_OS),Windows)
        ifneq ($(CROSSGPU_OS),Linux)
            $(error WITH_LEVEL_ZERO=1 requires Linux or Windows and the Level Zero SDK)
        endif
    endif
endif
CPPFLAGS += -Iinclude -Isrc -Ibackends/opencl \
            -DCROSSGPU_DEFAULT_PLUGIN_DIR=\"$(PREFIX)/lib/cross-gpu/backends\"
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic -fno-common
LDFLAGS ?=
LDLIBS += $(DYNAMIC_LOADER_LIBS) $(THREAD_LIBS) -lm

PROGRAM := $(BUILD_DIR)/cross-gpu$(EXE_SUFFIX)
PROGRAM_INSTALL_NAME := cross-gpu$(EXE_SUFFIX)
OPENCL_PLUGIN := $(BUILD_DIR)/backends/$(PLUGIN_PREFIX)opencl.$(PLUGIN_SUFFIX)
CUDA_PLUGIN := $(BUILD_DIR)/backends/$(PLUGIN_PREFIX)cuda.$(PLUGIN_SUFFIX)
HIP_PLUGIN := $(BUILD_DIR)/backends/$(PLUGIN_PREFIX)hip.$(PLUGIN_SUFFIX)
METAL_PLUGIN := $(BUILD_DIR)/backends/$(PLUGIN_PREFIX)metal.$(PLUGIN_SUFFIX)
LEVEL_ZERO_PLUGIN := $(BUILD_DIR)/backends/$(PLUGIN_PREFIX)level_zero.$(PLUGIN_SUFFIX)
LEVEL_ZERO_SPV := $(BUILD_DIR)/backends/levelzero_kernels.spv
LEVEL_ZERO_SPV_SOURCE := $(BUILD_DIR)/backends/levelzero_kernels_spv.cpp
LEVEL_ZERO_EMBEDDER := $(BUILD_DIR)/tools/embed_binary$(EXE_SUFFIX)

MOCK_CUDA_PLUGIN := $(BUILD_DIR)/test-backends/$(PLUGIN_PREFIX)cuda_mock.$(PLUGIN_SUFFIX)
MOCK_OPENCL_PLUGIN := $(BUILD_DIR)/test-backends/$(PLUGIN_PREFIX)opencl_mock.$(PLUGIN_SUFFIX)
MOCK_CUSTOM_PLUGIN := $(BUILD_DIR)/test-backends/$(PLUGIN_PREFIX)vulkan_mock.$(PLUGIN_SUFFIX)
MOCK_OPENCL_OTHER_NAMESPACE_PLUGIN := $(BUILD_DIR)/test-backends-other-namespace/$(PLUGIN_PREFIX)opencl_other_namespace.$(PLUGIN_SUFFIX)
MOCK_INTEL_LEVEL_ZERO_PLUGIN := $(BUILD_DIR)/test-backends-intel/$(PLUGIN_PREFIX)level_zero_intel_mock.$(PLUGIN_SUFFIX)
MOCK_INTEL_OPENCL_PLUGIN := $(BUILD_DIR)/test-backends-intel/$(PLUGIN_PREFIX)opencl_intel_mock.$(PLUGIN_SUFFIX)
MOCK_WRONG_VENDOR_LEVEL_ZERO_PLUGIN := $(BUILD_DIR)/test-backends-wrong-vendor/$(PLUGIN_PREFIX)level_zero_wrong_vendor_mock.$(PLUGIN_SUFFIX)
REGISTRY_TEST := $(BUILD_DIR)/test-registry

CORE_SOURCES := \
    src/main.c \
    src/backend_manager.c \
    src/registry.c \
    src/util.c \
    $(PLATFORM_SOURCE)
CORE_OBJECTS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(CORE_SOURCES))

PLUGIN_TARGETS :=
ifeq ($(WITH_OPENCL),1)
    PLUGIN_TARGETS += $(OPENCL_PLUGIN)
endif
ifeq ($(WITH_CUDA),1)
    PLUGIN_TARGETS += $(CUDA_PLUGIN)
endif
ifeq ($(WITH_HIP),1)
    PLUGIN_TARGETS += $(HIP_PLUGIN)
endif
ifeq ($(WITH_METAL),1)
    PLUGIN_TARGETS += $(METAL_PLUGIN)
endif
ifeq ($(WITH_LEVEL_ZERO),1)
    PLUGIN_TARGETS += $(LEVEL_ZERO_PLUGIN)
endif

.PHONY: all clean install uninstall check print-config plugins

all: $(PROGRAM) $(PLUGIN_TARGETS)
plugins: $(PLUGIN_TARGETS)

$(PROGRAM): $(CORE_OBJECTS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $(CORE_OBJECTS) $(LDLIBS)

$(BUILD_DIR)/%.o: %.c include/crossgpu_backend.h include/crossgpu_platform_config.h src/backend_manager.h src/registry.h src/util.h src/platform.h src/platform_posix_impl.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OPENCL_PLUGIN): backends/opencl/opencl_backend.c \
                  backends/opencl/opencl_minimal.h \
                  include/crossgpu_backend.h include/crossgpu_platform_config.h src/platform.h src/platform_posix_impl.h $(PLATFORM_SOURCE)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(PLUGIN_CFLAGS) $(SHARED_FLAGS) \
		-o $@ backends/opencl/opencl_backend.c $(PLATFORM_SOURCE) $(DYNAMIC_LOADER_LIBS) $(THREAD_LIBS) $(OPENCL_PLATFORM_LIBS)

$(CUDA_PLUGIN): backends/cuda/cuda_backend.cu \
                backends/common/runtime_backend_impl.hpp \
                include/crossgpu_backend.h include/crossgpu_platform_config.h
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(CUDA_PLUGIN_FLAGS) \
		-Iinclude -o $@ backends/cuda/cuda_backend.cu

$(HIP_PLUGIN): backends/hip/hip_backend.hip \
               backends/common/runtime_backend_impl.hpp \
               include/crossgpu_backend.h include/crossgpu_platform_config.h
	@mkdir -p $(@D)
	$(HIPCC) $(HIPCCFLAGS) $(HIP_PLUGIN_FLAGS) \
		-Iinclude -o $@ backends/hip/hip_backend.hip

$(METAL_PLUGIN): backends/metal/metal_backend.m include/crossgpu_backend.h include/crossgpu_platform_config.h
	@mkdir -p $(@D)
	$(METAL_CC) $(METALFLAGS) -std=gnu11 -fobjc-arc -fPIC -fvisibility=hidden \
		-dynamiclib -Iinclude -framework Foundation -framework Metal \
		-o $@ backends/metal/metal_backend.m


# Level Zero runtime plugin. Kernels are compiled to generic SPIR-V at build
# time with Intel's OCLOC tool. A native build-time helper converts the binary
# to C++ source so the plugin can embed it on Linux and Windows without relying
# on GNU ld's ELF-specific binary-object feature.
$(LEVEL_ZERO_SPV): backends/levelzero/levelzero_kernels.cl
	@mkdir -p $(@D)
	@command -v "$(OCLOC)" >/dev/null 2>&1 || { echo "error: OCLOC not found; install Intel's OpenCL Offline Compiler or set OCLOC=/path/to/ocloc" >&2; exit 1; }
	$(OCLOC) -q -spv_only -file $< -out_dir "$(@D)" -output_no_suffix
	@test -s "$@" || { echo "error: OCLOC did not produce $@" >&2; exit 1; }

$(LEVEL_ZERO_EMBEDDER): tools/embed_binary.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $<

$(LEVEL_ZERO_SPV_SOURCE): $(LEVEL_ZERO_SPV) $(LEVEL_ZERO_EMBEDDER)
	$(LEVEL_ZERO_EMBEDDER) "$<" "$@"

$(LEVEL_ZERO_PLUGIN): backends/levelzero/levelzero_backend.cpp include/crossgpu_backend.h include/crossgpu_platform_config.h $(LEVEL_ZERO_SPV_SOURCE)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(LEVEL_ZERO_CPPFLAGS) $(LEVEL_ZERO_CXXFLAGS) \
		-I$(BUILD_DIR)/backends -std=c++14 -Wall -Wextra -Wpedantic $(PLUGIN_CFLAGS) \
		$(SHARED_FLAGS) -o $@ backends/levelzero/levelzero_backend.cpp $(LEVEL_ZERO_SPV_SOURCE) $(LEVEL_ZERO_LIBS)


$(REGISTRY_TEST): tests/test_registry.c src/registry.c src/registry.h include/crossgpu_backend.h include/crossgpu_platform_config.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tests/test_registry.c src/registry.c

$(MOCK_CUDA_PLUGIN): tests/mock_backend.c include/crossgpu_backend.h include/crossgpu_platform_config.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMOCK_BACKEND_KIND=CROSSGPU_BACKEND_CUDA \
		-DMOCK_BACKEND_NAME=\"cuda\" -DMOCK_FAIL_OPEN=1 \
		$(PLUGIN_CFLAGS) $(SHARED_FLAGS) -o $@ tests/mock_backend.c

$(MOCK_OPENCL_PLUGIN): tests/mock_backend.c include/crossgpu_backend.h include/crossgpu_platform_config.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMOCK_BACKEND_KIND=CROSSGPU_BACKEND_OPENCL \
		-DMOCK_BACKEND_NAME=\"opencl\" $(PLUGIN_CFLAGS) \
		$(SHARED_FLAGS) -o $@ tests/mock_backend.c

$(MOCK_CUSTOM_PLUGIN): tests/mock_backend.c include/crossgpu_backend.h include/crossgpu_platform_config.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMOCK_BACKEND_KIND=CROSSGPU_BACKEND_CUSTOM \
		-DMOCK_BACKEND_NAME=\"vulkan-mock-future-backend\" \
		$(PLUGIN_CFLAGS) $(SHARED_FLAGS) -o $@ tests/mock_backend.c

$(MOCK_OPENCL_OTHER_NAMESPACE_PLUGIN): tests/mock_backend.c include/crossgpu_backend.h include/crossgpu_platform_config.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMOCK_BACKEND_KIND=CROSSGPU_BACKEND_OPENCL \
		-DMOCK_BACKEND_NAME=\"opencl\" \
		-DMOCK_IDENTITY_NAMESPACE=\"another-pci-namespace\" \
		$(PLUGIN_CFLAGS) $(SHARED_FLAGS) -o $@ tests/mock_backend.c


$(MOCK_INTEL_LEVEL_ZERO_PLUGIN): tests/mock_backend.c include/crossgpu_backend.h include/crossgpu_platform_config.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMOCK_BACKEND_KIND=CROSSGPU_BACKEND_LEVEL_ZERO \
		-DMOCK_BACKEND_NAME=\"level-zero\" \
		-DMOCK_VENDOR_ID=CROSSGPU_VENDOR_INTEL -DMOCK_FAIL_OPEN=1 \
		$(PLUGIN_CFLAGS) $(SHARED_FLAGS) -o $@ tests/mock_backend.c

$(MOCK_INTEL_OPENCL_PLUGIN): tests/mock_backend.c include/crossgpu_backend.h include/crossgpu_platform_config.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMOCK_BACKEND_KIND=CROSSGPU_BACKEND_OPENCL \
		-DMOCK_BACKEND_NAME=\"opencl\" \
		-DMOCK_VENDOR_ID=CROSSGPU_VENDOR_INTEL \
		$(PLUGIN_CFLAGS) $(SHARED_FLAGS) -o $@ tests/mock_backend.c

$(MOCK_WRONG_VENDOR_LEVEL_ZERO_PLUGIN): tests/mock_backend.c include/crossgpu_backend.h include/crossgpu_platform_config.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMOCK_BACKEND_KIND=CROSSGPU_BACKEND_LEVEL_ZERO \
		-DMOCK_BACKEND_NAME=\"level-zero\" \
		-DMOCK_VENDOR_ID=CROSSGPU_VENDOR_NVIDIA \
		$(PLUGIN_CFLAGS) $(SHARED_FLAGS) -o $@ tests/mock_backend.c

check: all $(REGISTRY_TEST) $(MOCK_INTEL_LEVEL_ZERO_PLUGIN) $(MOCK_INTEL_OPENCL_PLUGIN) $(MOCK_WRONG_VENDOR_LEVEL_ZERO_PLUGIN) $(MOCK_CUDA_PLUGIN) $(MOCK_OPENCL_PLUGIN) $(MOCK_CUSTOM_PLUGIN) $(MOCK_OPENCL_OTHER_NAMESPACE_PLUGIN)
	set -eux; \
	./$(REGISTRY_TEST); \
	CROSSGPU_PLUGIN_PATH_ONLY=1; export CROSSGPU_PLUGIN_PATH_ONLY; \
	PLUGIN_PATH="$(call native_path,$(abspath $(BUILD_DIR)/test-backends))"; \
	OTHER_NAMESPACE_PLUGIN_PATH="$(call native_path,$(abspath $(BUILD_DIR)/test-backends-other-namespace))"; \
	INTEL_PLUGIN_PATH="$(call native_path,$(abspath $(BUILD_DIR)/test-backends-intel))"; \
	WRONG_VENDOR_PLUGIN_PATH="$(call native_path,$(abspath $(BUILD_DIR)/test-backends-wrong-vendor))"; \
	./$(PROGRAM) --help | grep -q "cross-gpu list"; \
	./$(PROGRAM) --help | grep -q "compute, memory, bandwidth, or all"; \
	CROSSGPU_PLUGIN_PATH="$$PLUGIN_PATH" ./$(PROGRAM) list | grep -q "Mock GPU"; \
	CROSSGPU_PLUGIN_PATH="$$PLUGIN_PATH" ./$(PROGRAM) list | awk 'BEGIN { n=0 } /^[[:space:]]*[0-9]+[[:space:]]/ { n++ } END { exit (n == 1 ? 0 : 1) }'; \
	CROSSGPU_PLUGIN_PATH="$$PLUGIN_PATH" ./$(PROGRAM) info | grep -q "cuda"; \
	if ! CROSSGPU_PLUGIN_PATH="$$PLUGIN_PATH" ./$(PROGRAM) run \
		--device 0 --test all --duration 300ms \
		> $(BUILD_DIR)/check-run.log 2>&1; then \
		cat $(BUILD_DIR)/check-run.log; exit 1; \
	fi; \
	grep -q "preferred backend failed; using opencl" $(BUILD_DIR)/check-run.log; \
	grep -q "Stress run completed." $(BUILD_DIR)/check-run.log; \
	CROSSGPU_PLUGIN_PATH="$$PLUGIN_PATH$(PATH_LIST_SEPARATOR)$$OTHER_NAMESPACE_PLUGIN_PATH" \
		./$(PROGRAM) list | awk 'BEGIN { n=0 } /^[[:space:]]*[0-9]+[[:space:]]/ { n++ } END { exit (n == 2 ? 0 : 1) }'; \
	CROSSGPU_PLUGIN_PATH="$$INTEL_PLUGIN_PATH" ./$(PROGRAM) list | awk 'BEGIN { n=0 } /^[[:space:]]*[0-9]+[[:space:]]/ { n++ } END { exit (n == 1 ? 0 : 1) }'; \
	CROSSGPU_PLUGIN_PATH="$$INTEL_PLUGIN_PATH" ./$(PROGRAM) info | grep -q "Preferred backend: level-zero"; \
	CROSSGPU_PLUGIN_PATH="$$INTEL_PLUGIN_PATH" ./$(PROGRAM) info | grep -q "Fallbacks:.*opencl"; \
	if ! CROSSGPU_PLUGIN_PATH="$$INTEL_PLUGIN_PATH" ./$(PROGRAM) run \
		--device 0 --test compute --duration 300ms \
		> $(BUILD_DIR)/check-level-zero-fallback.log 2>&1; then \
		cat $(BUILD_DIR)/check-level-zero-fallback.log; exit 1; \
	fi; \
	grep -q "preferred backend failed; using opencl" $(BUILD_DIR)/check-level-zero-fallback.log; \
	if CROSSGPU_PLUGIN_PATH="$$WRONG_VENDOR_PLUGIN_PATH" ./$(PROGRAM) list | grep -q "Mock GPU"; then \
		echo "Level Zero was incorrectly accepted for a non-Intel device" >&2; exit 1; \
	fi; \
	echo "All tests passed (registry, discovery, vendor preference, fallbacks, identity merge, namespace isolation, Level Zero vendor restriction)."

install: all
	install -d "$(DESTDIR)$(PREFIX)/bin"
	install -m 0755 $(PROGRAM) "$(DESTDIR)$(PREFIX)/bin/$(PROGRAM_INSTALL_NAME)"
ifneq ($(strip $(PLUGIN_TARGETS)),)
	install -d "$(DESTDIR)$(PREFIX)/lib/cross-gpu/backends"
	install -m 0755 $(PLUGIN_TARGETS) "$(DESTDIR)$(PREFIX)/lib/cross-gpu/backends/"
endif

uninstall:
	rm -f "$(DESTDIR)$(PREFIX)/bin/cross-gpu" "$(DESTDIR)$(PREFIX)/bin/cross-gpu.exe"
	rm -f "$(DESTDIR)$(PREFIX)/lib/cross-gpu/backends/$(PLUGIN_PREFIX)opencl.$(PLUGIN_SUFFIX)"
	rm -f "$(DESTDIR)$(PREFIX)/lib/cross-gpu/backends/$(PLUGIN_PREFIX)cuda.$(PLUGIN_SUFFIX)"
	rm -f "$(DESTDIR)$(PREFIX)/lib/cross-gpu/backends/$(PLUGIN_PREFIX)hip.$(PLUGIN_SUFFIX)"
	rm -f "$(DESTDIR)$(PREFIX)/lib/cross-gpu/backends/$(PLUGIN_PREFIX)metal.$(PLUGIN_SUFFIX)"
	rm -f "$(DESTDIR)$(PREFIX)/lib/cross-gpu/backends/$(PLUGIN_PREFIX)level_zero.$(PLUGIN_SUFFIX)"

print-config:
	@echo "CC=$(CC)"
	@echo "CXX=$(CXX)"
	@echo "NVCC=$(NVCC)"
	@echo "HIPCC=$(HIPCC)"
	@echo "METAL_CC=$(METAL_CC)"
	@echo "OCLOC=$(OCLOC)"
	@echo "PREFIX=$(PREFIX)"
	@echo "CROSSGPU_OS=$(CROSSGPU_OS)"
	@echo "PLATFORM_SOURCE=$(PLATFORM_SOURCE)"
	@echo "WITH_OPENCL=$(WITH_OPENCL)"
	@echo "WITH_CUDA=$(WITH_CUDA)"
	@echo "WITH_HIP=$(WITH_HIP)"
	@echo "WITH_METAL=$(WITH_METAL)"
	@echo "WITH_LEVEL_ZERO=$(WITH_LEVEL_ZERO)"
	@echo "PROGRAM=$(PROGRAM)"
	@echo "PLUGIN_TARGETS=$(PLUGIN_TARGETS)"

clean:
	rm -rf $(BUILD_DIR)
