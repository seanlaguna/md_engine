# CMake makefile.
# To build the program, do
# mkdir -p build/
# cd build/
# cmake ..
# make

cmake_minimum_required (VERSION 2.6)
cmake_policy (SET CMP0064 OLD)

project (MDengine)

# Set the version number
set (VERSION_MAJOR 0)
set (VERSION_MINOR 4)

# Activate debug output of called with -DDEBUG=1
if (DEBUG)
    set (CMAKE_BUILD_TYPE "DEBUG")
#    set (CUDA_VERBOSE_BUILD ON)
#    set (CMAKE_VERBOSE_MAKEFILE ON)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -Wall")
    set (CUDA_NVCC_DEBUG_FLAGS -Xcompiler -Wall;)
endif (DEBUG)

if (TEST)
    find_package(GTest REQUIRED)
    enable_testing()
endif (TEST)

if (CMAKE_BUILD_TYPE MATCHES "RELEASE")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif ()

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpic")

message (STATUS ${CMAKE_CXX_FLAGS})

# Find CUDA
find_package(CUDA REQUIRED)
set (CUDA_PROPAGATE_HOST_FLAGS OFF)
set (custom_cuda_flags -std=c++11; -Xcompiler -fpic; --use_fast_math; --prec-div=true;)
list (APPEND CUDA_NVCC_FLAGS ${custom_cuda_flags})
list (APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_DEBUG_FLAGS})
get_filename_component (CUDA_CUFFT_LIBRARY_PATH ${CUDA_CUFFT_LIBRARIES} DIRECTORY)

# Find Python libraries
find_package (PythonLibs 2.7 REQUIRED)
include_directories (${PYTHON_INCLUDE_DIRS})
get_filename_component (PYTHON_LIBRARY_PATH ${PYTHON_LIBRARIES} DIRECTORY)

# Find boost
find_package (Boost COMPONENTS python REQUIRED)

# Find Pugi XML
find_package (PugiXML REQUIRED)

# Finding CUDA-aware MPI
find_package (MPI REQUIRED)
message (STATUS ${MPI_CXX_FOUND})
message (STATUS ${MPI_CXX_COMPILER})
message (STATUS ${MPI_CXX_COMPILE_FLAGS})
message (STATUS ${MPI_CXX_INCLUDE_PATH})
message (STATUS ${MPI_CXX_LINK_FLAGS})
message (STATUS ${MPI_CXX_LIBRARIES})
if (MPI_CXX_FOUND)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    find_program (OMPI_INFO NAMES ompi_info)
    if (OMPI_INFO)
        execute_process (COMMAND ${OMPI_INFO} --parseable --all
                         OUTPUT_VARIABLE _ompi_info_out
        )
        if (_ompi_info_out MATCHES "mpi_built_with_cuda_support:value:true")
            message (STATUS "Found CUDA-aware MPI")
            set (MPI_CUDA true)
        else ()
            message (STATUS "Found MPI, but no CUDA-aware MPI")
        endif ()
    else (OMPI_INFO)
        message (STATUS "Could not run ompi_info")
    endif (OMPI_INFO)
endif (MPI_CXX_FOUND)

message (STATUS "Building MD_engine Version ${VERSION_MAJOR}.${VERSION_MINOR}")

# Set the source files
#file (GLOB SOURCES "*.cpp" "*.cu")
#list (REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

# Create configuration file
configure_file (
    "${PROJECT_SOURCE_DIR}/Config.h.in"
    "${PROJECT_BINARY_DIR}/Config.h"
)
include_directories ("${PROJECT_BINARY_DIR}")

set (MD_ENGINE_LIB_NAME Sim)

# Add src directory
add_subdirectory (src)

# Install Python library
if (PYTHON)
    add_subdirectory (python)
endif (PYTHON)

# Add tests
if (TEST)
    add_subdirectory (unit_test)
endif (TEST)
