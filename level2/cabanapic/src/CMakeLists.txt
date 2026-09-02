file(GLOB HEADERS "*.h")
file(GLOB SOURCES "*.cpp")

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(CabanaPIC ${SOURCES})

target_include_directories(CabanaPIC PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# HPC-Performance-AI: Cabana >= 0.6 exports the core library as Cabana::Core
# (upstream was written against the pre-0.6 name Cabana::cabanacore).
target_link_libraries(CabanaPIC PUBLIC Cabana::Core )

install(TARGETS CabanaPIC DESTINATION lib)
