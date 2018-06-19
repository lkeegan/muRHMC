cmake_minimum_required(VERSION 3.1)
project (muRHMC)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE")

# if the environment variable $MKLROOT is defined, then link with MKL
# and define eigen macro EIGEN_USE_MKL_ALL
if(DEFINED ENV{MKLROOT})
	message("MKLROOT is defined: linking with MKL")
	#Note: including MKL headers generates huge number of compiler warnings
	#SYSTEM keyword here means include headers as system headers and therefore suppresses compiler warnings
	include_directories(SYSTEM $ENV{MKLROOT}/include)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DEIGEN_USE_MKL_ALL -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")
endif()

FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
	message("OPENMP FOUND")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Catch2 unit testing library
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

#set header files
include_directories(inc)

#set common source files
set (SRCS ${SRCS} src/4d.cpp src/su3.cpp src/dirac_op.cpp src/inverters.cpp src/io.cpp src/hmc.cpp src/rhmc.cpp src/rational_approx.cpp)

# Make unit test executable
set (TEST_SRCS ${SRCS} test/4d.cpp test/su3.cpp test/io.cpp test/inverters.cpp test/hmc.cpp test/rhmc.cpp test/rational_approx.cpp test/dirac_op.cpp)

add_executable(tests test/main.cpp ${TEST_SRCS})
target_link_libraries(tests Catch)

# Make rhmc executable
add_executable(rhmc rhmc.cpp ${SRCS})
add_executable(hmc hmc.cpp ${SRCS})

# Make measurements executables
add_executable(measure_exact measure_exact.cpp ${SRCS})
add_executable(measure_noise measure_noise.cpp ${SRCS})
add_executable(eigenvalues eigenvalues.cpp ${SRCS})
add_executable(measure measure.cpp ${SRCS})
add_executable(CG CG.cpp ${SRCS})
add_executable(polyakov polyakov.cpp ${SRCS})

add_executable(benchmark benchmark.cpp src/4d.cpp src/su3.cpp src/dirac_op.cpp)

#enable_testing()
add_test(NAME Addition COMMAND tests)

#Make running the unit tests part of the build, so failed test = failed build
add_custom_command(TARGET tests POST_BUILD COMMAND tests -d yes)
