# HPC-Performance-AI: when INPUT_DECK is set the top-level CMakeLists.txt adds
# -DUSER_INPUT_DECK=<deck> directory-wide, but every test below compiles its
# own deck and sets USER_INPUT_DECK itself, so drop the inherited one here.
# Otherwise each test TU is compiled with both and the preprocessor warns
# '"USER_INPUT_DECK" redefined' (harmless: the macro is only ever #ifdef'd).
if(INPUT_DECK_PATH)
  remove_definitions(-DUSER_INPUT_DECK=${INPUT_DECK_PATH})
endif()

add_subdirectory(decks)
if (${SOLVER_TYPE} STREQUAL "EM")
  add_subdirectory(energy_comparison)
endif()

#LIST(APPEND TESTS example)

foreach (test ${TESTS})
    add_executable(${test} ./${test}.cpp)
    target_link_libraries(${test} CabanaPIC)
    add_test(NAME ${test} COMMAND ./${test})
endforeach(test)
