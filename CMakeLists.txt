cmake_minimum_required(VERSION 3.1)
project (LATTICE)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message("Found Eigen3 in: ${EIGEN3_INCLUDE_DIR}")

# Catch unit testing library
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

include_directories(inc)

# Make unit test executable
add_executable(tests test/main.cpp src/hmc.cpp src/dirac_op.cpp src/4d.cpp)
target_link_libraries(tests Catch)

# Make hmc executable
add_executable(hmc main.cpp src/hmc.cpp src/dirac_op.cpp src/4d.cpp)

#enable_testing()
add_test(NAME Addition COMMAND tests)

#Make running the unit tests part of the build, so failed test = failed build
add_custom_command(TARGET tests POST_BUILD COMMAND tests -d yes)
