
cmake_minimum_required(VERSION 3.16.0)
project(shaw CXX)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed.
Remove CMakeCache.txt. Make a separate build directory and run CMake from there.")
endif()

# default to release if build type is empty
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# we need c++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#==================================

find_package(yaml-cpp REQUIRED HINTS ${YAMLCPP_DIR})
include_directories(${YAMLCPP_DIR}/include)

if(EXISTS "${YAMLCPP_DIR}/lib64/libyaml-cpp.a")
  set(YAML_CPP_LIBRARIES ${YAMLCPP_DIR}/lib64/libyaml-cpp.a)
endif()
if(EXISTS "${YAMLCPP_DIR}/lib/libyaml-cpp.a")
  set(YAML_CPP_LIBRARIES ${YAMLCPP_DIR}/lib/libyaml-cpp.a)
endif()

find_package(KokkosKernels REQUIRED HINTS ${KOKKOSKERNELS_DIR})

# executables
add_executable(
  shawExe
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kokkos/main.cc)
target_link_libraries(shawExe dl ${YAML_CPP_LIBRARIES} Kokkos::kokkoskernels)

# HPC-Performance-AI: the upstream post-processing/ROM tools (extractStateFromSnaps,
# reconstructFomState, reconstructSeismogram, computeThinSVD -- the last one needs
# the bundled Eigen copy) are not part of the FOM benchmark and are not built here.

# tests
enable_testing()
add_subdirectory(tests)
