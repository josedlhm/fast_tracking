add_library(st_mapping_core STATIC)

target_sources(st_mapping_core PRIVATE
  Mapping.cpp
  Registration.cpp
  Threshold.cpp
  LocalMap.cpp
  TrackAccel.cpp
)

# Link only what these files actually use
target_link_libraries(st_mapping_core
  PUBLIC
    Eigen3::Eigen
    TBB::tbb
    Sophus::Sophus
    ${OpenCV_LIBS}
    tsl::robin_map        # LocalMap uses robin-map for voxel/hash maps
)

set_global_target_properties(st_mapping_core)