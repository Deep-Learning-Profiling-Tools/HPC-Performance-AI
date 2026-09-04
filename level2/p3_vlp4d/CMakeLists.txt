cmake_minimum_required(VERSION 3.16)
project(P3miniapps LANGUAGES CXX)

# [HPC-Performance-AI] Trimmed copy of the upstream P3-miniapps top-level
# CMakeLists.txt for the single mini-app "vlp4d" (CUDA / HIP models only).
# Removed relative to upstream: the KOKKOS block (ext_lib/kokkos is not
# shipped) and the unit-test block (ext_lib/googletest and tests/ are not
# shipped). include(CTest) is kept so that -DBUILD_TESTING=OFF from the
# upstream build instructions remains a recognised option. The APPLICATION
# default is "vlp4d" instead of AUTO (only this mini-app is shipped here).

# Add cmake helpers
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

# Find libraries 
find_package(mdspan CONFIG)
if(NOT mdspan_FOUND)
    add_subdirectory(ext_lib/mdspan)
endif()

set(PROGRAMMING_MODEL AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
set(APPLICATION vlp4d CACHE STRING "CHOICE OF APPLICATION")

add_subdirectory(lib)

include(CTest)

# Mini applications
add_subdirectory(miniapps)
