cmake_minimum_required(VERSION 3.24)
project(hpcperf_del_dot_vec_2d LANGUAGES CXX)

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
    add_executable(del_dot_vec_2d_cuda cuda/main.cu)
    target_include_directories(del_dot_vec_2d_cuda PRIVATE common)
    set(_hpcperf_target del_dot_vec_2d_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(del_dot_vec_2d_hip hip/main.cpp)
    target_include_directories(del_dot_vec_2d_hip PRIVATE common)
    set_source_files_properties(hip/main.cpp PROPERTIES LANGUAGE HIP)
    set(_hpcperf_target del_dot_vec_2d_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

if(BUILD_TESTING)
    add_test(NAME del_dot_vec_2d_run COMMAND ${_hpcperf_target})
    set_tests_properties(del_dot_vec_2d_run PROPERTIES TIMEOUT 900
        PASS_REGULAR_EXPRESSION "PASS"
        FAIL_REGULAR_EXPRESSION "FAIL")
endif()
