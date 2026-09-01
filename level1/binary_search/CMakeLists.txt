cmake_minimum_required(VERSION 3.24)
project(hpcperf_binary_search LANGUAGES CXX)

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
    add_executable(binary_search_cuda cuda/main.cu)
    set(_hpcperf_target binary_search_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(binary_search_hip hip/main.cu)
    set_source_files_properties(hip/main.cu PROPERTIES LANGUAGE HIP)
    set(_hpcperf_target binary_search_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

if(BUILD_TESTING)
    add_test(NAME binary_search_run COMMAND ${_hpcperf_target} 8388608 100)
    set_tests_properties(binary_search_run PROPERTIES TIMEOUT 600
        FAIL_REGULAR_EXPRESSION "incorrect result")
endif()
