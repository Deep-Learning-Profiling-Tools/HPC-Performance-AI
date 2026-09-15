# MiniEM driver (Trilinos/packages/panzer/mini-em/example/BlockPrec at the commit pinned in
# setup_level2_deps.sh). Upstream builds this executable only inside a Trilinos source build through TriBITS
# (TRIBITS_ADD_EXECUTABLE); there is no standalone build system, so this file provides one against the exported
# targets of the installed Trilinos (.deps/install/trilinos). main.cpp / MiniEM_helpers.* / decks/*.xml are
# byte-identical to upstream (UPSTREAM_SHA256SUMS). Compiler: the same nvcc_wrapper-through-mpicxx recipe the
# dependency was built with (build.sh exports OMPI_CXX), because the Panzer/Phalanx headers contain device code.
cmake_minimum_required(VERSION 3.23)
project(hpcperf_miniem CXX)

find_package(Trilinos REQUIRED COMPONENTS PanzerMiniEM PanzerAdaptersSTK PanzerDiscFE MueLu Teko Belos Ifpack2
                                          Stratimikos Piro NOX Tpetra Teuchos)
message(STATUS "Trilinos ${Trilinos_VERSION} from ${Trilinos_DIR}")

add_executable(PanzerMiniEM_BlockPrec main.cpp MiniEM_helpers.cpp)
target_include_directories(PanzerMiniEM_BlockPrec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(PanzerMiniEM_BlockPrec PRIVATE Trilinos::all_selected_libs)
target_compile_features(PanzerMiniEM_BlockPrec PRIVATE cxx_std_20)

# the input decks travel with the binary: run.sh executes from <build>/decks so the relative solver-file
# names inside the decks (solverMueLu.xml, ...) resolve exactly as in upstream's test directory
file(GLOB HPCPERF_MINIEM_DECKS ${CMAKE_CURRENT_SOURCE_DIR}/decks/*.xml)
add_custom_target(hpcperf_miniem_decks ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/decks
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HPCPERF_MINIEM_DECKS} ${CMAKE_BINARY_DIR}/decks
  COMMENT "copying MiniEM input decks next to the executable")
