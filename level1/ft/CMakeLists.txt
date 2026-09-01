cmake_minimum_required(VERSION 3.24)
project(hpcperf_ft LANGUAGES CXX)

# Backend selection: CUDA (default) or HIP
set(BACKEND "CUDA" CACHE STRING "GPU backend: CUDA or HIP")
set_property(CACHE BACKEND PROPERTY STRINGS CUDA HIP)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)

# npbparams.hpp is pre-generated with upstream sys/setparams for CLASS=B
# (see README to change the problem class).

if(BACKEND STREQUAL "CUDA")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_executable(ft_cuda cuda/ft.cu common/c_print_results.cpp
                   common/c_randdp.cpp common/c_timers.cu common/wtime.cpp)
    target_include_directories(ft_cuda PRIVATE cuda common)
    # upstream ft.cu uses OpenMP on the host side
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(ft_cuda PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(ft_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    set(_hpcperf_target ft_cuda)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    add_executable(ft_hip hip/ft.cpp)
    target_include_directories(ft_hip PRIVATE hip)
    set_source_files_properties(hip/ft.cpp PROPERTIES LANGUAGE HIP)
    set(_hpcperf_target ft_hip)
else()
    message(FATAL_ERROR "Unknown BACKEND '${BACKEND}'; expected CUDA or HIP")
endif()

if(BUILD_TESTING)
    add_test(NAME ft_run COMMAND ${_hpcperf_target})
    set_tests_properties(ft_run PROPERTIES TIMEOUT 1200
        PASS_REGULAR_EXPRESSION "Verification    =               SUCCESSFUL"
        FAIL_REGULAR_EXPRESSION "UNSUCCESSFUL;NOT PERFORMED")
endif()
