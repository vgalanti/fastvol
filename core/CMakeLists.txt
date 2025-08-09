cmake_minimum_required(VERSION 3.18)
project(fastvol CXX)


# default to release for optim. ------------------------------------------------------------------*/
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Default to Release build" FORCE)
endif()


# verbose vectorization --------------------------------------------------------------------------*/
option(FASTVOL_VERBOSE "Enable verbose loop vectorization analysis" OFF)


# compiler standards -----------------------------------------------------------------------------*/
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# include paths ----------------------------------------------------------------------------------*/
include(GNUInstallDirs)


# base C flags -----------------------------------------------------------------------------------*/
set(CMAKE_CXX_FLAGS "-O3 -march=native -ffast-math ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ffp-contract=fast -fno-math-errno")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${PROJECT_SOURCE_DIR}/include")

if (FASTVOL_VERBOSE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-optimized -fopt-info-vec-missed")
    endif()
endif()


# OpenMP detection -------------------------------------------------------------------------------*/
if (APPLE)
    # allow python setup.py to override the OMP paths
    if (NOT DEFINED FASTVOL_OMP_INCLUDE_DIR)
        execute_process(COMMAND brew --prefix libomp
            OUTPUT_VARIABLE FASTVOL_OMP_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(FASTVOL_OMP_INCLUDE_DIR "${FASTVOL_OMP_INCLUDE_DIR}/include")
    endif()

    if (NOT DEFINED FASTVOL_OMP_LIB_DIR)
        execute_process(COMMAND brew --prefix libomp
            OUTPUT_VARIABLE FASTVOL_OMP_LIB_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(FASTVOL_OMP_LIB_DIR "${FASTVOL_OMP_LIB_DIR}/lib")
    endif()

    message(STATUS "OpenMP include: ${FASTVOL_OMP_INCLUDE_DIR}")
    message(STATUS "OpenMP lib    : ${FASTVOL_OMP_LIB_DIR}")

    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${FASTVOL_OMP_INCLUDE_DIR}")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${FASTVOL_OMP_LIB_DIR}/libomp.dylib")
endif()

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    list(APPEND EXTRA_LIBS ${OpenMP_omp_LIBRARY})
endif()


# CUDA detection ---------------------------------------------------------------------------------*/
set(CUDA_ENABLED OFF)
if (NOT DEFINED ENV{FASTVOL_CUDA_ENABLED})
    find_program(NVCC_EXECUTABLE nvcc)
    if (NVCC_EXECUTABLE)
        message(STATUS "CUDA detected via nvcc: ${NVCC_EXECUTABLE}")
        enable_language(CUDA)
        set(CUDA_ENABLED ON)
    else()
        message(STATUS "nvcc not found: building CPU-only")
    endif()
elseif ("$ENV{FASTVOL_CUDA_ENABLED}" STREQUAL "1")
    find_program(NVCC_EXECUTABLE nvcc)
    if (NVCC_EXECUTABLE)
        message(STATUS "CUDA forced ON via FASTVOL_CUDA_ENABLED")
        enable_language(CUDA)
        set(CUDA_ENABLED ON)
    else()
        message(FATAL_ERROR "FASTVOL_CUDA_ENABLED=1 but nvcc not found")
    endif()
endif()


# source discovery -------------------------------------------------------------------------------*/
file(GLOB_RECURSE SRC_CXX ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(SRC_ALL ${SRC_CXX})

if (CUDA_ENABLED)
    file(GLOB_RECURSE SRC_CU ${PROJECT_SOURCE_DIR}/src/*.cu)
    list(APPEND SRC_ALL ${SRC_CU})
endif()


# static library build ---------------------------------------------------------------------------*/
add_library(fastvol STATIC ${SRC_ALL})
target_include_directories(fastvol
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
set_target_properties(fastvol PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if (CUDA_ENABLED)
    set_source_files_properties(${SRC_CU} PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(fastvol PUBLIC FASTVOL_CUDA_ENABLED)
    target_compile_options(fastvol PRIVATE -O3)
    target_link_libraries(fastvol PRIVATE cudart cublas)
endif()

if (OpenMP_CXX_FOUND)
    target_link_libraries(fastvol PRIVATE ${OpenMP_omp_LIBRARY})
endif()


# benchmark build(s) -----------------------------------------------------------------------------*/
add_executable(bench bench/bench.cpp)

set(BENCH_TARGETS
    bench
)

foreach(target IN LISTS BENCH_TARGETS)
    target_link_libraries(${target} PRIVATE fastvol)
    if (CUDA_ENABLED)
        target_compile_definitions(${target} PRIVATE FASTVOL_CUDA_ENABLED)
    endif()
endforeach()


# install public headers only --------------------------------------------------------------------*/
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/fastvol
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fastvol
        FILES_MATCHING PATTERN "*.hpp")


# install static library -------------------------------------------------------------------------*/
install(TARGETS fastvol
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# verbose build target ---------------------------------------------------------------------------*/
add_custom_target(verbose
    COMMAND ${CMAKE_COMMAND} -E echo "Enabling full verbose vectorization + build output"
    COMMAND ${CMAKE_COMMAND} -B ${CMAKE_BINARY_DIR} -DFASTVOL_VERBOSE=ON -DCMAKE_VERBOSE_MAKEFILE=ON
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --verbose
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
