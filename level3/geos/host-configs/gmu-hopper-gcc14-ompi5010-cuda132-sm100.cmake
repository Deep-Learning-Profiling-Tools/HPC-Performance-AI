# GEOS / thirdPartyLibs host-config for GMU Hopper dgx003 (4x B200, RHEL 10):
# system GCC 14.2.1, conda Open MPI 5.0.10 (wrappers redirected to that GCC via
# OMPI_CC/OMPI_CXX in build.sh), CUDA 13.2 sm_100, hypre on the device, no Trilinos,
# no PETSc, no OpenMP (as upstream's CUDA CI rows: ENABLE_HYPRE_DEVICE=CUDA,
# ENABLE_TRILINOS=OFF), no Caliper/MathPresso/Doxygen (not needed by the cases run).
# Modelled on thirdPartyLibs docker/Stanford/sherlock-gcc10-cuda12-sm70.cmake and
# GEOS host-configs/environment.cmake (which hard-codes CUDA_ARCH sm_86 for CI).
# Paths under HPCPERF_L3_* are injected by build.sh through -D on the command line.
set(CONFIG_NAME "gmu-hopper-gcc14-ompi5010-cuda132-sm100" CACHE PATH "" FORCE)

set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE PATH "" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE PATH "" FORCE)
set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran" CACHE PATH "" FORCE)
set(ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)

set(ENABLE_MPI ON CACHE BOOL "" FORCE)
set(MPI_C_COMPILER "${HPCPERF_MPI_BIN}/mpicc" CACHE PATH "" FORCE)
set(MPI_CXX_COMPILER "${HPCPERF_MPI_BIN}/mpicxx" CACHE PATH "" FORCE)
set(MPI_Fortran_COMPILER "${HPCPERF_MPI_BIN}/mpifort" CACHE PATH "" FORCE)
set(MPIEXEC_EXECUTABLE "${HPCPERF_MPI_BIN}/mpirun" CACHE PATH "" FORCE)
set(MPIEXEC_NUMPROC_FLAG "-np" CACHE STRING "" FORCE)
set(ENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC ON CACHE BOOL "")
set(ENABLE_GTEST_DEATH_TESTS ON CACHE BOOL "" FORCE)

set(ENABLE_OPENMP OFF CACHE BOOL "" FORCE)

# BLAS/LAPACK: private OpenBLAS built by build.sh (single-threaded)
set(BLAS_LIBRARIES "${HPCPERF_OPENBLAS_LIB}" CACHE PATH "" FORCE)
set(LAPACK_LIBRARIES "${HPCPERF_OPENBLAS_LIB}" CACHE PATH "" FORCE)

# CUDA
set(ENABLE_CUDA ON CACHE BOOL "" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR "${HPCPERF_CUDA_ROOT}" CACHE PATH "" FORCE)
set(CMAKE_CUDA_COMPILER "${HPCPERF_CUDA_ROOT}/bin/nvcc" CACHE STRING "" FORCE)
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++" CACHE STRING "" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "100" CACHE STRING "" FORCE)
set(CUDA_ARCH "sm_100" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS "-restrict -arch ${CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr -Werror cross-execution-space-call,reorder,deprecated-declarations" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler -DNDEBUG -Xcompiler -O3" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo ${CMAKE_CUDA_FLAGS_RELEASE}" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -Xcompiler -O0" CACHE STRING "" FORCE)

# Linear algebra interface: hypre on the GPU; Trilinos/PETSc off (upstream CUDA CI configuration)
set(GEOS_LA_INTERFACE "Hypre" CACHE STRING "" FORCE)
set(ENABLE_HYPRE ON CACHE BOOL "" FORCE)
set(ENABLE_HYPRE_DEVICE "CUDA" CACHE STRING "" FORCE)
set(ENABLE_TRILINOS OFF CACHE BOOL "" FORCE)
# hypredrive (hypre driver library) is GEOS develop's default companion of hypre, but the TPL superbuild's hypredrive
# step does not compile against the Umpire-enabled hypre here (missing Umpire include path) and the beam workflow
# does not use it; GEOS documents "-DENABLE_HYPREDRV=OFF" for exactly this (SetupGeosxThirdParty.cmake).
set(ENABLE_HYPREDRV OFF CACHE BOOL "" FORCE)
set(ENABLE_PETSC OFF CACHE BOOL "" FORCE)
set(ENABLE_SUPERLU_DIST ON CACHE BOOL "" FORCE)
set(ENABLE_SUITESPARSE ON CACHE BOOL "" FORCE)
set(ENABLE_SCOTCH ON CACHE BOOL "" FORCE)
set(ENABLE_VTK ON CACHE BOOL "" FORCE)

# Tests stay at upstream's default (ENABLE_TESTS ON, gtest/gbenchmark built); BLT 0.6.2's CUDA runtime smoke test needs
# the back-port level3/geos/patches/geos-blt-0001-cuda13-memoryClockRate.patch (CUDA 13 removed cudaDeviceProp::memoryClockRate).
set(ENABLE_CALIPER OFF CACHE BOOL "" FORCE)
set(ENABLE_MATHPRESSO OFF CACHE BOOL "" FORCE)
set(ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)
set(ENABLE_UNCRUSTIFY OFF CACHE BOOL "" FORCE)
set(ENABLE_XML_UPDATES OFF CACHE BOOL "" FORCE)
set(ENABLE_PYGEOSX OFF CACHE BOOL "" FORCE)

# GEOS build: point at the private TPL install when provided
if(DEFINED HPCPERF_GEOS_TPL_DIR)
  set(GEOS_TPL_DIR "${HPCPERF_GEOS_TPL_DIR}" CACHE PATH "" FORCE)
  # GEOS adds the MPI include directory (= the conda environment's include/) as -isystem AHEAD of the TPL include
  # directories, and that directory carries an unrelated 32-bit metis.h (a 2013 conda package): ParMETISInterface.cpp
  # then fails its "ParMETIS must be built with 64-bit indices" static assertion although the TPL METIS/ParMETIS are
  # 64-bit. -I directories are searched before -isystem ones, so the TPL METIS/ParMETIS headers are put first here
  # (host and CUDA compile lines); no source change.
  set(CMAKE_CXX_FLAGS "-I${HPCPERF_GEOS_TPL_DIR}/parmetis/include -I${HPCPERF_GEOS_TPL_DIR}/metis/include" CACHE STRING "" FORCE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${HPCPERF_GEOS_TPL_DIR}/parmetis/include -I${HPCPERF_GEOS_TPL_DIR}/metis/include" CACHE STRING "" FORCE)
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../../../_upstream/level3/GEOS/host-configs/tpls.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/../../../_upstream/level3/GEOS/host-configs/tpls.cmake")
  endif()
endif()
