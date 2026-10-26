cmake_minimum_required(VERSION 3.24)
project(hpcperf_hotspot LANGUAGES CXX)

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
    add_executable(hotspot_cuda cuda/hotspot.cu)
else()
    message(FATAL_ERROR "BACKEND '${BACKEND}' not available: upstream Rodinia has no HIP implementation (HIP pending)")
endif()

# tools/timing ROI markers: header-only and a no-op unless measured (tools/timing/roi/README.md)
target_include_directories(hotspot_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/timing/roi)
# tools/golden: CPU-reference gate (off by default) and output capture (tools/golden/README.md)
target_include_directories(hotspot_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden)

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    # one plain run of the registry's default input (no CPU reference: verify.py is used only once, offline,
    # by hpcperf_golden.py make), its output compared with the frozen B200 golden reference (reference/<input>.json)
    add_test(NAME hotspot_run COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden/hpcperf_golden.py
             run ${CMAKE_CURRENT_SOURCE_DIR} --exe $<TARGET_FILE:hotspot_cuda> --workdir ${CMAKE_CURRENT_BINARY_DIR}/golden-ctest)
    set_tests_properties(hotspot_run PROPERTIES TIMEOUT 900
        PASS_REGULAR_EXPRESSION "PASS: golden check"
        FAIL_REGULAR_EXPRESSION "FAIL: golden check")
endif()
