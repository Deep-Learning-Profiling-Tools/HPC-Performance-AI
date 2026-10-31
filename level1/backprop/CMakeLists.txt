cmake_minimum_required(VERSION 3.24)
project(hpcperf_backprop LANGUAGES C CXX)

# Backend selection: CUDA (default). Upstream Rodinia (HPC-FAIR/rodinia_3.1)
# provides no HIP implementation -> HIP is pending for this benchmark.
set(BACKEND "CUDA" CACHE STRING "GPU backend: CUDA (HIP pending)")
set_property(CACHE BACKEND PROPERTY STRINGS CUDA)

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
    add_executable(backprop_cuda cuda/backprop.c cuda/backprop_cuda.cu cuda/facetrain.c cuda/imagenet.c)
else()
    message(FATAL_ERROR "BACKEND '${BACKEND}' not available: upstream Rodinia has no HIP implementation (HIP pending)")
endif()

# tools/timing ROI markers: header-only and a no-op unless measured (tools/timing/roi/README.md)
target_include_directories(backprop_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/timing/roi)
# tools/golden: CPU-reference gate (off by default) and output capture (tools/golden/README.md)
target_include_directories(backprop_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden)

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    # one plain run of the registry's default input (no CPU reference), its output compared with
    # the frozen B200 golden reference (reference/<input>.json)
    add_test(NAME backprop_run COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden/hpcperf_golden.py
             run ${CMAKE_CURRENT_SOURCE_DIR} --exe $<TARGET_FILE:backprop_cuda> --workdir ${CMAKE_CURRENT_BINARY_DIR}/golden-ctest)
    set_tests_properties(backprop_run PROPERTIES TIMEOUT 600
        PASS_REGULAR_EXPRESSION "PASS: golden check"
        FAIL_REGULAR_EXPRESSION "FAIL: golden check")
    # correctness probes (inputs.yaml purpose: correctness-probe): the same kernels on set-up data on which a
    # constant fill or a skipped kernel cannot pass (`--probe`, cuda/facetrain.c, cuda/backprop.c); checked against their own
    # frozen reference, never timed
    add_test(NAME backprop_probe_n32 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden/hpcperf_golden.py
             run ${CMAKE_CURRENT_SOURCE_DIR} probe-dpn1-n32 --exe $<TARGET_FILE:backprop_cuda> --workdir ${CMAKE_CURRENT_BINARY_DIR}/golden-ctest-probe-dpn1-n32)
    set_tests_properties(backprop_probe_n32 PROPERTIES TIMEOUT 300
        PASS_REGULAR_EXPRESSION "PASS: golden check"
        FAIL_REGULAR_EXPRESSION "FAIL: golden check")
    add_test(NAME backprop_probe_n1024 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden/hpcperf_golden.py
             run ${CMAKE_CURRENT_SOURCE_DIR} probe-dpn1-n1024 --exe $<TARGET_FILE:backprop_cuda> --workdir ${CMAKE_CURRENT_BINARY_DIR}/golden-ctest-probe-dpn1-n1024)
    set_tests_properties(backprop_probe_n1024 PROPERTIES TIMEOUT 300
        PASS_REGULAR_EXPRESSION "PASS: golden check"
        FAIL_REGULAR_EXPRESSION "FAIL: golden check")
endif()
