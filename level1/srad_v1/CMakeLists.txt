cmake_minimum_required(VERSION 3.24)
project(hpcperf_srad_v1 LANGUAGES CXX)

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
    add_executable(srad_v1_cuda cuda/main.cu)
else()
    message(FATAL_ERROR "BACKEND '${BACKEND}' not available: upstream Rodinia has no HIP implementation (HIP pending)")
endif()

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    # verify.py runs the benchmark and validates its output against a
    # CPU reference (see verify.py header for details)
    add_test(NAME srad_v1_run COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/verify.py $<TARGET_FILE:srad_v1_cuda> 100 0.5 502 458 ${CMAKE_CURRENT_SOURCE_DIR}/data/image.pgm)
    set_tests_properties(srad_v1_run PROPERTIES TIMEOUT 900
        PASS_REGULAR_EXPRESSION "PASS"
        FAIL_REGULAR_EXPRESSION "FAIL")
endif()
