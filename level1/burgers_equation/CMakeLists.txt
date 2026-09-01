cmake_minimum_required(VERSION 3.24)
project(hpcperf_burgers_equation LANGUAGES CXX)

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
    add_executable(burgers_equation_cuda cuda/main.cu)
    target_include_directories(burgers_equation_cuda PRIVATE common)
    set(_hpcperf_target burgers_equation_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(burgers_equation_hip hip/main.cu)
    set_source_files_properties(hip/main.cu PROPERTIES LANGUAGE HIP)
    target_include_directories(burgers_equation_hip PRIVATE common)
    set(_hpcperf_target burgers_equation_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

if(BUILD_TESTING)
    add_test(NAME burgers_equation_run COMMAND ${_hpcperf_target} 8200 8100 100)
    set_tests_properties(burgers_equation_run PROPERTIES TIMEOUT 600
        PASS_REGULAR_EXPRESSION "PASS"
        FAIL_REGULAR_EXPRESSION "FAIL")
endif()
