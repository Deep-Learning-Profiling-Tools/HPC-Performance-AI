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

# correctness-probe input (inputs.yaml g512-p2-t21-pnoise9, purpose: correctness-probe; never timed): the vendored
# power_512 plus a deterministic random 0..9 W per cell, generated at configure time into <build>/data (data/gen_probe.py);
# the performance inputs read only the vendored files
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/data/gen_probe.py
                        ${CMAKE_CURRENT_SOURCE_DIR}/data/power_512 ${CMAKE_CURRENT_BINARY_DIR}/data/power_512_pnoise9
                RESULT_VARIABLE _probe_rc)
if(NOT _probe_rc EQUAL 0)
    message(FATAL_ERROR "hotspot correctness-probe input generation (data/gen_probe.py) failed")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             ${CMAKE_CURRENT_SOURCE_DIR}/data/gen_probe.py ${CMAKE_CURRENT_SOURCE_DIR}/data/power_512)

if(BUILD_TESTING)
    # one plain run of the registry's default input (no CPU reference: verify.py is used only once, offline,
    # by hpcperf_golden.py make), its output compared with the frozen B200 golden reference (reference/<input>.json)
    add_test(NAME hotspot_run COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden/hpcperf_golden.py
             run ${CMAKE_CURRENT_SOURCE_DIR} --exe $<TARGET_FILE:hotspot_cuda> --workdir ${CMAKE_CURRENT_BINARY_DIR}/golden-ctest)
    set_tests_properties(hotspot_run PROPERTIES TIMEOUT 900
        PASS_REGULAR_EXPRESSION "PASS: golden check"
        FAIL_REGULAR_EXPRESSION "FAIL: golden check")
    # correctness probe (inputs.yaml purpose: correctness-probe): 21 steps with a strong random power map, on which
    # no-ops, skipped iterations, stale blocks/halos, frozen boundaries, dropped terms and f16 state fail; checked
    # against its own frozen reference, never timed
    add_test(NAME hotspot_probe COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden/hpcperf_golden.py
             run ${CMAKE_CURRENT_SOURCE_DIR} g512-p2-t21-pnoise9 --exe $<TARGET_FILE:hotspot_cuda> --workdir ${CMAKE_CURRENT_BINARY_DIR}/golden-ctest-probe)
    set_tests_properties(hotspot_probe PROPERTIES TIMEOUT 300
        PASS_REGULAR_EXPRESSION "PASS: golden check"
        FAIL_REGULAR_EXPRESSION "FAIL: golden check")
endif()
