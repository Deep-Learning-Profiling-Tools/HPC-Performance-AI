cmake_minimum_required(VERSION 3.24)
project(hpcperf_bezier_surface LANGUAGES CXX)

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
    add_executable(bezier_surface_cuda cuda/main.cu)
    set(_hpcperf_target bezier_surface_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(bezier_surface_hip hip/main.cu)
    set_source_files_properties(hip/main.cu PROPERTIES LANGUAGE HIP)
    set(_hpcperf_target bezier_surface_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

if(BUILD_TESTING)
    add_test(NAME bezier_surface_run COMMAND ${_hpcperf_target} -f ${CMAKE_CURRENT_SOURCE_DIR}/data/control.txt -n 8192)
    set_tests_properties(bezier_surface_run PROPERTIES TIMEOUT 600
        PASS_REGULAR_EXPRESSION "PASS"
        FAIL_REGULAR_EXPRESSION "FAIL")
endif()
