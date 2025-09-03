function(set_global_target_properties target)
  target_compile_features(${target} PRIVATE cxx_std_20)
  target_compile_options(${target} PRIVATE -fdiagnostics-color=always -Wall
                                           -Wextra -pedantic)
  set(INCLUDE_DIRS ${PROJECT_SOURCE_DIR})
  get_filename_component(INCLUDE_DIRS ${INCLUDE_DIRS} PATH)
  target_include_directories(
    ${target}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC $<BUILD_INTERFACE:${INCLUDE_DIRS}>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endfunction()
