cmake_minimum_required(VERSION 3.24)
project(hpcperf_lud LANGUAGES C CXX)

# Backend selection: CUDA (default). Upstream Rodinia (HPC-FAIR/rodinia_3.1)
# provides no HIP implementation -> HIP is pending for this benchmark.
set(BACKEND "CUDA" CACHE STRING "GPU backend: CUDA (HIP pending)")
set_property(CACHE BACKEND PROPERTY STRINGS CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

if(BACKEND STREQUAL "CUDA")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_executable(lud_cuda cuda/lud.cu cuda/lud_kernel.cu common/common.c)
    target_include_directories(lud_cuda PRIVATE common)
else()
    message(FATAL_ERROR "BACKEND '${BACKEND}' not available: upstream Rodinia has no HIP implementation (HIP pending)")
endif()

if(BUILD_TESTING)
    add_test(NAME lud_run COMMAND lud_cuda -s 256 -v)
    set_tests_properties(lud_run PROPERTIES TIMEOUT 600
        FAIL_REGULAR_EXPRESSION "dismatch")
endif()
