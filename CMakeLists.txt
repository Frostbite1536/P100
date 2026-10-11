cmake_minimum_required(VERSION 3.24)
project(p100 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA is optional: without a CUDA compiler (dev laptop) only the CPU reference
# and tests are built. Kernels still get static sm_60 reports via bin/sm60-report.py.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Must be set before enable_language(CUDA), or CMake picks the compiler default.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60)
  endif()
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "60")
    message(FATAL_ERROR "P100 targets sm_60 only (INV-ARCH-001), got ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} dropped sm_60; use CUDA 12.x (INV-ARCH-003)")
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
                      $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
  set(P100_HAVE_CUDA ON)
else()
  message(STATUS "P100: no CUDA compiler found, building CPU reference and tests only")
  set(P100_HAVE_CUDA OFF)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

enable_testing()

# Default: the pip CUDA runtime wheel in this checkout's .venv (docs/TOOLCHAIN.md), if present.
file(GLOB _p100_wheel_cudart "${PROJECT_SOURCE_DIR}/.venv/lib/python3*/site-packages/nvidia/cuda_runtime")
set(P100_CUDART_ROOT "${_p100_wheel_cudart}" CACHE PATH "CUDA runtime root for host-only P2P targets")
if(P100_HAVE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(P100_CUDART_TARGET CUDA::cudart)
elseif(P100_CUDART_ROOT AND EXISTS "${P100_CUDART_ROOT}/include/cuda_runtime.h"
       AND EXISTS "${P100_CUDART_ROOT}/lib/libcudart.so.12")
  add_library(p100_cudart SHARED IMPORTED)
  set_target_properties(p100_cudart PROPERTIES
    IMPORTED_LOCATION "${P100_CUDART_ROOT}/lib/libcudart.so.12"
    INTERFACE_INCLUDE_DIRECTORIES "${P100_CUDART_ROOT}/include")
  # NVIDIA's pip wheels split the runtime's required crt headers into cuda_nvcc.
  if(NOT EXISTS "${P100_CUDART_ROOT}/include/crt/host_defines.h"
     AND EXISTS "${P100_CUDART_ROOT}/../cuda_nvcc/include/crt/host_defines.h")
    set_property(TARGET p100_cudart APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      "${P100_CUDART_ROOT}/../cuda_nvcc/include")
  endif()
  set(P100_CUDART_TARGET p100_cudart)
else()
  message(STATUS "P100: CUDA runtime unavailable; skipping P2P targets")
endif()
# Host-only CUDA runtime targets (P2P), added before tests/ so tests can see them.
if(P100_CUDART_TARGET)
  add_subdirectory(src/p2p)
  add_subdirectory(bench)
endif()
# Kernel targets (src/attention) will be added under if(P100_HAVE_CUDA) as they land.

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/reference/CMakeLists.txt)
  add_subdirectory(src/reference)
  target_sources(p100_ref PRIVATE src/reference/qmv_model.cpp)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  add_subdirectory(tests)
endif()
