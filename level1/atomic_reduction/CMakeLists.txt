cmake_minimum_required(VERSION 3.24)
project(hpcperf_atomic_reduction LANGUAGES CXX)

# Backend selection: CUDA (default) or HIP
set(BACKEND "CUDA" CACHE STRING "GPU backend: CUDA or HIP")
set_property(CACHE BACKEND PROPERTY STRINGS CUDA HIP)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

if(BACKEND STREQUAL "CUDA")
    # Architecture comes from CMAKE_CUDA_ARCHITECTURES / CUDAARCHS (hpcperf_env.sh
    # sets CUDAARCHS=native); nothing machine-specific is hardcoded here.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_executable(atomic_reduction_cuda cuda/reduction.cu)
    target_include_directories(atomic_reduction_cuda PRIVATE common)
    set(_hpcperf_target atomic_reduction_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(atomic_reduction_hip hip/reduction.cu)
    set_source_files_properties(hip/reduction.cu PROPERTIES LANGUAGE HIP)
    target_include_directories(atomic_reduction_hip PRIVATE common)
    set(_hpcperf_target atomic_reduction_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

if(BUILD_TESTING)
    add_test(NAME atomic_reduction_run COMMAND ${_hpcperf_target} )
    set_tests_properties(atomic_reduction_run PROPERTIES TIMEOUT 600
        PASS_REGULAR_EXPRESSION "VERIFICATION: PASS"
        FAIL_REGULAR_EXPRESSION "FAIL")
endif()
