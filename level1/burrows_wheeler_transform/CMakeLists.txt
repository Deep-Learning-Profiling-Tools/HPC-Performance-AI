cmake_minimum_required(VERSION 3.24)
project(hpcperf_burrows_wheeler_transform LANGUAGES CXX)

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
    add_executable(burrows_wheeler_transform_cuda cuda/bwt.cu common/main.cpp)
    target_include_directories(burrows_wheeler_transform_cuda PRIVATE common)
    set(_hpcperf_target burrows_wheeler_transform_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(burrows_wheeler_transform_hip hip/bwt.cu common/main.cpp)
    set_source_files_properties(hip/bwt.cu PROPERTIES LANGUAGE HIP)
    target_include_directories(burrows_wheeler_transform_hip PRIVATE common)
    set(_hpcperf_target burrows_wheeler_transform_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

if(BUILD_TESTING)
    add_test(NAME burrows_wheeler_transform_run COMMAND ${_hpcperf_target} 10000000)
    set_tests_properties(burrows_wheeler_transform_run PROPERTIES TIMEOUT 600
        PASS_REGULAR_EXPRESSION "PASS"
        FAIL_REGULAR_EXPRESSION "FAIL")
endif()
