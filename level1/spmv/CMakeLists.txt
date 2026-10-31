cmake_minimum_required(VERSION 3.24)
project(hpcperf_spmv LANGUAGES CXX)

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
    add_executable(spmv_cuda cuda/main.cu)
    set(_hpcperf_target spmv_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(spmv_hip hip/main.cpp)
    set_source_files_properties(hip/main.cpp PROPERTIES LANGUAGE HIP)
    set(_hpcperf_target spmv_hip)
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
    add_test(NAME spmv_run COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden/hpcperf_golden.py
             run ${CMAKE_CURRENT_SOURCE_DIR} --exe $<TARGET_FILE:${_hpcperf_target}> --workdir ${CMAKE_CURRENT_BINARY_DIR}/golden-ctest)
    set_tests_properties(spmv_run PROPERTIES TIMEOUT 900
        PASS_REGULAR_EXPRESSION "PASS: golden check"
        FAIL_REGULAR_EXPRESSION "FAIL: golden check")
    # correctness probe (inputs.yaml purpose: correctness-probe): the same kernel on real-valued set-up data on which
    # the evaluation precision and single skipped rows are visible (`--probe`, cuda/main.cu probe_uniform); checked
    # against its own frozen reference with its own per-element rule, never timed
    add_test(NAME spmv_probe COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden/hpcperf_golden.py
             run ${CMAKE_CURRENT_SOURCE_DIR} probe-real --exe $<TARGET_FILE:${_hpcperf_target}> --workdir ${CMAKE_CURRENT_BINARY_DIR}/golden-ctest-probe-real)
    set_tests_properties(spmv_probe PROPERTIES TIMEOUT 300
        PASS_REGULAR_EXPRESSION "PASS: golden check"
        FAIL_REGULAR_EXPRESSION "FAIL: golden check")
endif()
