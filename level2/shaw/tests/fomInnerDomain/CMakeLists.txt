
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../compare.py compare.py COPYONLY)

configure_file(input.yaml input.yaml COPYONLY)

# HPC-Performance-AI: the gold snapshot matrices are stored xz-compressed in the
# repository (9.4 MB -> 3.5 MB) and are decompressed into the build tree here.
find_program(XZ_EXECUTABLE xz REQUIRED)
foreach(GOLD snaps_vp_0_gold snaps_sp_0_gold)
  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${GOLD} OR
     ${CMAKE_CURRENT_SOURCE_DIR}/${GOLD}.xz IS_NEWER_THAN ${CMAKE_CURRENT_BINARY_DIR}/${GOLD})
    execute_process(
      COMMAND ${XZ_EXECUTABLE} -dkc ${CMAKE_CURRENT_SOURCE_DIR}/${GOLD}.xz
      OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${GOLD}
      RESULT_VARIABLE XZ_RESULT)
    if(XZ_RESULT)
      message(FATAL_ERROR "Failed to decompress ${GOLD}.xz")
    endif()
  endif()
endforeach()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../fullMesh21x51 DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME fomInnerDomain
COMMAND ${CMAKE_COMMAND}
-DCMD_FOM=$<TARGET_FILE:shawExe>
-DINPUT_FNAME=input.yaml
-P ${CMAKE_CURRENT_SOURCE_DIR}/test.cmake
)
