# HPC-Performance-AI: INPUT_DECK_PATH is the absolute deck path resolved in the
# top-level CMakeLists.txt (upstream used the raw INPUT_DECK here, which only
# resolves for an in-source build -- see there).
add_executable(cbnpic example.cpp ${INPUT_DECK_PATH})
target_link_libraries(cbnpic PUBLIC CabanaPIC)
target_include_directories( cbnpic PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
install(TARGETS cbnpic DESTINATION ${CMAKE_INSTALL_BINDIR})
