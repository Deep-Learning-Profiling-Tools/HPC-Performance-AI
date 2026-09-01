cmake_minimum_required(VERSION 3.24)
project(hpcperf-level1 LANGUAGES NONE)

# Add every benchmark directory that already has a CMakeLists.txt.
file(GLOB _bench_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
list(SORT _bench_dirs)
foreach(_b ${_bench_dirs})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_b}/CMakeLists.txt)
        add_subdirectory(${_b})
    endif()
endforeach()
