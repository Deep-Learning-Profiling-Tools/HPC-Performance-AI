cmake_minimum_required(VERSION 3.24)
project(hpcperf_color_histogram LANGUAGES CXX)

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
    common/hist_benchmark.cc
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
    add_executable(color_histogram_cuda cuda/main.cc cuda/hist_cuda_benchmark.cu ${COMMON_SRCS})
    target_include_directories(color_histogram_cuda PRIVATE common cuda ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(_hpcperf_target color_histogram_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(color_histogram_hip hip/main.cc hip/hist_hip_benchmark.hip ${COMMON_SRCS})
    target_include_directories(color_histogram_hip PRIVATE common hip)
    set_source_files_properties(hip/hist_hip_benchmark.hip PROPERTIES LANGUAGE HIP)
    set(_hpcperf_target color_histogram_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

if(BUILD_TESTING)
    add_test(NAME color_histogram_run COMMAND ${_hpcperf_target} -x 1048576)
    set_tests_properties(color_histogram_run PROPERTIES TIMEOUT 600
        PASS_REGULAR_EXPRESSION "Passed")
endif()
