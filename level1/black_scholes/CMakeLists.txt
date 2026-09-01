cmake_minimum_required(VERSION 3.24)
project(hpcperf_black_scholes LANGUAGES CXX)

# Backend selection: CUDA (default) or HIP
set(BACKEND "CUDA" CACHE STRING "GPU backend: CUDA or HIP")
set_property(CACHE BACKEND PROPERTY STRINGS CUDA HIP)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

set(COMMON_SRCS
    common/bs_benchmark.cc
    common/benchmark_runner.cc
    common/time_measurement_impl.cc
    common/time_keeper_impl.cc
    common/time_keeper_summary_printer.cc
    common/timer_impl.cc)

if(BACKEND STREQUAL "CUDA")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_executable(black_scholes_cuda cuda/main.cc cuda/bs_cuda_benchmark.cu ${COMMON_SRCS})
    target_include_directories(black_scholes_cuda PRIVATE common cuda ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(_hpcperf_target black_scholes_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(black_scholes_hip hip/main.cc hip/bs_hip_benchmark.cpp ${COMMON_SRCS})
    target_include_directories(black_scholes_hip PRIVATE common hip)
    set_source_files_properties(hip/bs_hip_benchmark.cpp PROPERTIES LANGUAGE HIP)
    set(_hpcperf_target black_scholes_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

if(BUILD_TESTING)
    add_test(NAME black_scholes_run COMMAND ${_hpcperf_target} -x 1048576)
    set_tests_properties(black_scholes_run PROPERTIES TIMEOUT 600
        FAIL_REGULAR_EXPRESSION "Verification failed")
endif()
