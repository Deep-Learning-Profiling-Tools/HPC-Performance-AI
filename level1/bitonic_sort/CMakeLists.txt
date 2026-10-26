cmake_minimum_required(VERSION 3.24)
project(hpcperf_bitonic_sort LANGUAGES CXX)

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
    add_executable(bitonic_sort_cuda cuda/main.cu)
    set(_hpcperf_target bitonic_sort_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(bitonic_sort_hip hip/main.cu)
    set_source_files_properties(hip/main.cu PROPERTIES LANGUAGE HIP)
    set(_hpcperf_target bitonic_sort_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

# tools/timing ROI markers: header-only and a no-op unless measured (tools/timing/roi/README.md)
target_include_directories(${_hpcperf_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/timing/roi)
# tools/golden: CPU-reference gate (off by default) and output capture (tools/golden/README.md)
target_include_directories(${_hpcperf_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden)

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    # one plain run of the registry's default input (no CPU reference), its output compared with
    # the frozen B200 golden reference (reference/<input>.json)
    add_test(NAME bitonic_sort_run COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden/hpcperf_golden.py
             run ${CMAKE_CURRENT_SOURCE_DIR} --exe $<TARGET_FILE:${_hpcperf_target}> --workdir ${CMAKE_CURRENT_BINARY_DIR}/golden-ctest)
    set_tests_properties(bitonic_sort_run PROPERTIES TIMEOUT 600
        PASS_REGULAR_EXPRESSION "PASS: golden check"
        FAIL_REGULAR_EXPRESSION "FAIL: golden check")
endif()
