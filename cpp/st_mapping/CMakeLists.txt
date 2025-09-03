
cmake_minimum_required(VERSION 3.16...3.26)
project(
  st_mapping_cpp
  VERSION 0.1.0
  LANGUAGES CXX)

# Setup build options
option(USE_SYSTEM_EIGEN3 "Use system pre-installed Eigen" ON)
option(USE_SYSTEM_SOPHUS "Use system pre-installed Sophus" OFF)
option(USE_SYSTEM_TBB "Use system pre-installed oneAPI/tbb" ON)
option(USE_SYSTEM_TSL-ROBIN-MAP "Use system pre-installed tsl_robin" OFF)
option(USE_SYSTEM_OPENCV "Use system pre-installed OpenCV" ON)
option(USE_SYSTEM_CERES "Use system pre-installed ceres-solver" OFF)

# Set build type
set(CMAKE_BUILD_TYPE Release)

# cmake options and functions
include(cmake/StandardProjectOptions.cmake)
include(cmake/GlobalTargetProperty.cmake)

# Find dependencies
include(3rdparty/find_dependencies.cmake)

# Add all the libraries (and executables)
add_subdirectory(core)
