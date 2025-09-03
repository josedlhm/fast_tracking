pybind11_add_module(st_mapping_pybind MODULE st_mapping_pybind.cpp)
target_compile_features(st_mapping_pybind PUBLIC cxx_std_17)
target_link_libraries(st_mapping_pybind PRIVATE st_mapping_core)
install(TARGETS st_mapping_pybind DESTINATION .)


pybind11_add_module(track_accel_pybind MODULE track_accel_pybind.cpp)
target_compile_features(track_accel_pybind PUBLIC cxx_std_17)
target_link_libraries(track_accel_pybind PRIVATE
  st_mapping_core         # if TrackAccel.cpp is compiled into this
)
install(TARGETS track_accel_pybind DESTINATION .)