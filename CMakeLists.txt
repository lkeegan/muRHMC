cmake_minimum_required(VERSION 3.1)
project (muRHMC)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -Wno-int-in-bool-context")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -g") #remove g if not profiling release code

#Comment out following line to disable openMP
FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Eigen3 REQUIRED)
# for anubis/euler comment out above line and instead use:
#set(EIGEN3_INCLUDE_DIR "/home/keeganl/include/eigen3")
include_directories(${EIGEN3_INCLUDE_DIR})
message("Found Eigen3 in: ${EIGEN3_INCLUDE_DIR}")

# Catch unit testing library
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

#set header files
include_directories(inc)

#set common source files
set (SRCS ${SRCS} src/hmc.cpp src/dirac_op.cpp src/4d.cpp src/io.cpp)

# Make unit test executable
add_executable(tests test/main.cpp ${SRCS})
target_link_libraries(tests Catch)

# Make rhmc executable
add_executable(rhmc rhmc.cpp ${SRCS})

# Make measurements executables
add_executable(measure_exact measure_exact.cpp ${SRCS})
add_executable(measure_noise measure_noise.cpp ${SRCS})
add_executable(eigenvalues eigenvalues.cpp ${SRCS})
add_executable(measure measure.cpp ${SRCS})
add_executable(CG CG.cpp ${SRCS})

#enable_testing()
add_test(NAME Addition COMMAND tests)

#Make running the unit tests part of the build, so failed test = failed build
add_custom_command(TARGET tests POST_BUILD COMMAND tests -d yes)
