
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
link_libraries(dl ${YAML_CPP_LIBRARIES} Kokkos::kokkoskernels)

# HPC-Performance-AI: of the upstream test suite only the end-to-end FOM
# regression test fomInnerDomain is kept (see README.md, "Changes from upstream").
add_subdirectory(fomInnerDomain)
