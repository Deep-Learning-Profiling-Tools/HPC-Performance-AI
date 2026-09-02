cmake_minimum_required(VERSION 3.22)
project(pmkokkos VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Options
# ============================================================
# Inner SIMD width of the Cabana AoSoA.  16 matches the SYCL/PVC default
# (CABANA_VERIFICATION_v07_05c_local.md A5/B5).
set(PMKOKKOS_VECTOR_LENGTH 16 CACHE STRING "AoSoA inner vector length")
option(PMKOKKOS_ENABLE_TESTS "Build GoogleTest test suite" ON)
# FP-precise mode for the cross-code P(k) bias FP-environment recon
# (analysis/prompts/recon_pk_bias_phase1_then_test.md, 2026-05-30).
# When ON, suppresses FMA fusion and reassociation across the whole code
# base — single-flag change, no other build settings touched.  Default
# OFF preserves the production -O3 / icpx-default behavior.
option(PMKOKKOS_FP_PRECISE "Compile with -fp-model=precise -ffp-contract=off" OFF)
if(PMKOKKOS_FP_PRECISE)
    add_compile_options(-fp-model=precise -ffp-contract=off)
    message(STATUS "PMKOKKOS_FP_PRECISE=ON  →  adding -fp-model=precise -ffp-contract=off")
endif()

# ============================================================
# Dependencies
# ============================================================
# Cabana::Grid pulls in Kokkos::kokkos, MPI::MPI_CXX, and Heffte::Heffte
# transitively (see Cabana_Targets.cmake), so a single component request
# covers the FFT stack, Halo, Array, and core AoSoA APIs.
find_package(Kokkos REQUIRED)
find_package(Cabana REQUIRED COMPONENTS Core Grid)
# Parallel HDF5 (Aurora module: hdf5/1.14.6 — verified Parallel HDF5: ON
# via h5pcc -showconfig).  C component is enough for the H5* C API used
# by SnapshotWriter/SnapshotReader; HL is included for future utilities.
find_package(HDF5 COMPONENTS C HL REQUIRED)

# ============================================================
# Core library
# ============================================================
add_library(pmkokkos_core
    src/CodeUnits.cpp
    src/Cosmology.cpp
    src/Grid.cpp
    src/Particles.cpp
    src/pm/PackUnpack.cpp
    src/pm/GreensFunction.cpp
    src/pm/MassAssign.cpp
    src/pm/ForceInterp.cpp
    src/pm/PMStep.cpp
    src/time/Drift.cpp
    src/time/Integrator.cpp
    src/time/PMDriver.cpp
    src/comm/Migrator.cpp
    src/ic/WhiteNoise.cpp
    src/ic/TransferFunction.cpp
    src/ic/PowerSpectrum.cpp
    src/ic/ICGreensFunction.cpp
    src/ic/InitialConditions.cpp
    src/ic/CoordConvert.cpp
    src/io/IndatParser.cpp
    src/io/SnapshotWriter.cpp
    src/io/SnapshotReader.cpp
    # Vendored CBRNG (mirrored verbatim from HACC; do not modify in-tree).
    # See external/cbrng/README.md for the mirror policy.
    external/cbrng/CBRNG_Random.cxx
)

# HPC-Performance-AI: with a CUDA-enabled Kokkos every CXX source -- including
# the host-only vendored CBRNG -- is compiled through nvcc_wrapper, and
# Random123's features/nvccfeatures.h then tags its functions __device__ only,
# so the host functions in CBRNG_Random.cxx cannot call them. Random123's
# documented override hook is R123_CUDA_DEVICE; define it empty for that one
# translation unit (plain host code, exactly what gcc/icpx produce). No-op for
# non-CUDA compilers (gccfeatures.h already defines it empty). The vendored
# files themselves stay verbatim (see external/cbrng/README.md).
set_source_files_properties(external/cbrng/CBRNG_Random.cxx PROPERTIES
    COMPILE_DEFINITIONS "R123_CUDA_DEVICE=")

target_include_directories(pmkokkos_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/cbrng>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(pmkokkos_core PUBLIC
    PMKOKKOS_VECTOR_LENGTH=${PMKOKKOS_VECTOR_LENGTH}
)

target_link_libraries(pmkokkos_core PUBLIC
    Kokkos::kokkos
    Cabana::Core
    Cabana::Grid
    HDF5::HDF5
)

# ============================================================
# Driver applications
# ============================================================
add_executable(pm_ic apps/pm_ic.cpp)
target_link_libraries(pm_ic PRIVATE pmkokkos_core)

# pm_run pipeline as a small library so test_pm_run_indat_driven can call
# pm_run_main() directly (avoids nested mpiexec inside a ctest test).
add_library(pm_run_lib STATIC apps/pm_run_lib.cpp)
target_include_directories(pm_run_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/apps>)
target_link_libraries(pm_run_lib PUBLIC pmkokkos_core)

add_executable(pm_run apps/pm_run.cpp)
target_link_libraries(pm_run PRIVATE pm_run_lib)

add_executable(snapshot_inspect apps/snapshot_inspect.cpp)
target_link_libraries(snapshot_inspect PRIVATE pmkokkos_core)

# HPC-Performance-AI addition: standalone correctness checker for pm_run output
# used by validate.sh (serial HDF5 C API only; see apps/snapshot_check.cpp).
add_executable(snapshot_check apps/snapshot_check.cpp)
target_link_libraries(snapshot_check PRIVATE HDF5::HDF5)

# Diagnostic single-kick driver — not a permanent test, see
# apps/diag/pm_single_kick.cpp and the comparison script
# analysis/scripts/compare_single_kick.py.
add_executable(pm_single_kick apps/diag/pm_single_kick.cpp)
target_link_libraries(pm_single_kick PRIVATE pm_run_lib)

# ============================================================
# Tests
# ============================================================
if(PMKOKKOS_ENABLE_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()

    # Host-only tests (no Kokkos::initialize, no GPU).
    foreach(test_name
        test_types
        test_units
        test_cosmology
        test_white_noise_against_hacc
        test_transfer_function
        test_power_spectrum_normalization
        test_growth_factor_extension
        test_growth_factor_hacc_parity
        test_indat_parser
    )
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE
            pmkokkos_core
            GTest::gtest
            GTest::gtest_main
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()

    # GPU tests (default exec space — SYCL on Aurora).  Run via mpiexec on a
    # compute node; tag with "gpu" so a later `ctest -L gpu` selects them.
    foreach(test_name
        test_grid
        test_fft_roundtrip
        test_greens_function
        test_mass_assign
        test_cic_convention_hacc_aligned
        test_force_interp
        test_pm_step_two_body
        test_pm_step_two_body_spectral
        test_force_kernel_shape
        test_pm_step_kick
        test_drift
        test_integrator_scalars
        test_migrator
        test_migration_protects_scatter
        test_orbit
        test_white_noise_reproducibility
        test_white_noise_statistics
        test_white_noise_kspace
        test_ic_kernel
        test_ic_coord_convert
        test_zeldovich_pk_recovery
        test_ic_end_to_end
        test_snapshot_roundtrip
        test_snapshot_partial_particles
        test_schema_v1_compat
        test_pm_ic_end_to_end
        test_pm_run_end_to_end
    )
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE
            pmkokkos_core
            GTest::gtest
        )
        add_test(NAME ${test_name}
                 COMMAND mpiexec -n 1 $<TARGET_FILE:${test_name}>)
        set_tests_properties(${test_name} PROPERTIES LABELS "gpu")
    endforeach()

    # Indat-driven pm_run test (prompt 13).  Links against pm_run_lib so it
    # can call pm_run_main() directly without a nested mpiexec subprocess.
    add_executable(test_pm_run_indat_driven tests/test_pm_run_indat_driven.cpp)
    target_link_libraries(test_pm_run_indat_driven PRIVATE
        pm_run_lib
        GTest::gtest
    )
    add_test(NAME test_pm_run_indat_driven
             COMMAND mpiexec -n 1 $<TARGET_FILE:test_pm_run_indat_driven>)
    set_tests_properties(test_pm_run_indat_driven PROPERTIES LABELS "gpu")

    # Multi-rank runs of test_mass_assign exercise the halo.scatter step in
    # the deposit (the ScatterReduce::Sum that closes Cabana::Grid::p2g).
    foreach(nranks 2 4)
        add_test(NAME test_mass_assign_n${nranks}
                 COMMAND mpiexec -n ${nranks}
                         $<TARGET_FILE:test_mass_assign>)
        set_tests_properties(test_mass_assign_n${nranks}
                             PROPERTIES LABELS "gpu")
    endforeach()

    # Multi-rank runs of test_migrator exercise the Cabana::Distributor wrapper.
    foreach(nranks 2 4)
        add_test(NAME test_migrator_n${nranks}
                 COMMAND mpiexec -n ${nranks}
                         $<TARGET_FILE:test_migrator>)
        set_tests_properties(test_migrator_n${nranks}
                             PROPERTIES LABELS "gpu")
    endforeach()

    # Multi-rank runs of test_migration_protects_scatter — THE regression
    # guard for the missing pre-kick migrate in Integrator::full_step
    # (see analysis/references/pmkokkos_migration_audit.md and
    # analysis/prompts/fix_dkd_migration.md).  The 2-rank run is the
    # decisive one (rank boundary at z = ng/2).
    foreach(nranks 2 4)
        add_test(NAME test_migration_protects_scatter_n${nranks}
                 COMMAND mpiexec -n ${nranks}
                         $<TARGET_FILE:test_migration_protects_scatter>)
        set_tests_properties(test_migration_protects_scatter_n${nranks}
                             PROPERTIES LABELS "gpu")
    endforeach()

    # Multi-rank runs of test_white_noise_reproducibility — THE acceptance
    # test for prompt 10a.
    foreach(nranks 2 4 8)
        add_test(NAME test_white_noise_reproducibility_n${nranks}
                 COMMAND mpiexec -n ${nranks}
                         $<TARGET_FILE:test_white_noise_reproducibility>)
        set_tests_properties(test_white_noise_reproducibility_n${nranks}
                             PROPERTIES LABELS "gpu")
    endforeach()

    # Multi-rank snapshot round-trip — THE acceptance test for prompt 11.
    foreach(nranks 2 4)
        add_test(NAME test_snapshot_roundtrip_n${nranks}
                 COMMAND mpiexec -n ${nranks}
                         $<TARGET_FILE:test_snapshot_roundtrip>)
        set_tests_properties(test_snapshot_roundtrip_n${nranks}
                             PROPERTIES LABELS "gpu")
    endforeach()

    foreach(nranks 2 4)
        add_test(NAME test_snapshot_partial_particles_n${nranks}
                 COMMAND mpiexec -n ${nranks}
                         $<TARGET_FILE:test_snapshot_partial_particles>)
        set_tests_properties(test_snapshot_partial_particles_n${nranks}
                             PROPERTIES LABELS "gpu")
    endforeach()
endif()
