cmake_minimum_required(VERSION 3.24)
project(hpcperf_aes LANGUAGES CXX)

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
    common/aes_benchmark.cc
    common/benchmark_runner.cc
    common/time_measurement_impl.cc
    common/time_keeper_impl.cc
    common/time_keeper_summary_printer.cc
    common/timer_impl.cc)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
# Deterministic input generation (upstream data server is offline; input
# content does not affect correctness semantics -- GPU output is verified
# against the upstream CPU AES reference).
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/data/gen_input.py
                        ${CMAKE_CURRENT_BINARY_DIR}/data RESULT_VARIABLE _gen_rc)
if(NOT _gen_rc EQUAL 0)
    message(FATAL_ERROR "input generation failed")
endif()

if(BACKEND STREQUAL "CUDA")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_executable(aes_cuda cuda/main.cc cuda/aes_cuda_benchmark.cu ${COMMON_SRCS})
    target_include_directories(aes_cuda PRIVATE common cuda ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(_hpcperf_target aes_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(aes_hip hip/main.cc hip/aes_hip_benchmark.hip ${COMMON_SRCS})
    target_include_directories(aes_hip PRIVATE common hip)
    set_source_files_properties(hip/aes_hip_benchmark.hip PROPERTIES LANGUAGE HIP)
    set(_hpcperf_target aes_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

if(BUILD_TESTING)
    add_test(NAME aes_run COMMAND ${_hpcperf_target} -i ${CMAKE_CURRENT_BINARY_DIR}/data/input_1MB.hex -k ${CMAKE_CURRENT_BINARY_DIR}/data/key.hex)
    set_tests_properties(aes_run PROPERTIES TIMEOUT 600
        PASS_REGULAR_EXPRESSION "Passed"
        FAIL_REGULAR_EXPRESSION "expected to be")
endif()
