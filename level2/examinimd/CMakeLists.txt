cmake_minimum_required(VERSION 3.10)

project(ExaMiniMD LANGUAGES CXX)

include(GNUInstallDirs)

find_package(Kokkos 3.0 REQUIRED)
# Kokkos >= 5.0 no longer ships Kokkos_StaticCrsGraph.hpp (used by
# src/neighbor_types/neighbor_csr.h); it lives in Kokkos Kernels as
# KokkosSparse::StaticCrsGraph.
find_package(KokkosKernels REQUIRED)
option(USE_MPI "Build with MPI" ON)

if (USE_MPI)
  find_package(MPI REQUIRED CXX)
else()
  kokkos_check(DEVICES Cuda RETURN_VALUE USE_CUDA)
  if (USING_CUDA)
    if (CMAKE_CXX_COMPILER_ID MATCHES "nvcc")
      enable_language(CUDA)
    else()
      message(FATAL_ERROR "Kokkos uses CUDA but compiler is not nvcc")
    endif()
  endif()
endif()

# force target_source to resolve relative paths immediately
if(POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

add_subdirectory(src)
add_subdirectory(input)
