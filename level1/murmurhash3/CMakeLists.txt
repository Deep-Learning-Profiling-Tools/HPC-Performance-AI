cmake_minimum_required(VERSION 3.24)
project(hpcperf_murmurhash3 LANGUAGES CXX)

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
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_executable(murmurhash3_cuda cuda/murmurhash3.cu)
    set(_hpcperf_target murmurhash3_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(murmurhash3_hip hip/murmurhash3.cu)
    set_source_files_properties(hip/murmurhash3.cu PROPERTIES LANGUAGE HIP)
    set(_hpcperf_target murmurhash3_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

if(BUILD_TESTING)
    add_test(NAME murmurhash3_run COMMAND ${_hpcperf_target} 100000 100)
    set_tests_properties(murmurhash3_run PROPERTIES TIMEOUT 600
        PASS_REGULAR_EXPRESSION "SUCCESS"
        FAIL_REGULAR_EXPRESSION "FAIL")
endif()
