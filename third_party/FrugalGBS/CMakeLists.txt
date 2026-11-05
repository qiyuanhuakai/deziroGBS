cmake_minimum_required(VERSION 3.18)

project(FrugalGBS VERSION 1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(FGBS_ENABLE_CUDA "Enable CUDA backend." ON)
option(BUILD_TESTING "Build and enable tests." ON)

set(FGBS_REAL_PRECISION 1 CACHE STRING "Real precision (1=single, 2=double).")
set(FGBS_MAX_CUTOFF_ORDER 5 CACHE STRING "Maximum cumulant cutoff order.")

set(FGBS_EIGEN3_DIR "" CACHE PATH "Optional hint path for Eigen3 package config/prefix.")
set(FGBS_CATCH2_DIR "" CACHE PATH "Optional hint path for Catch2 package config/prefix.")

if(FGBS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(WARNING "CUDA compiler not found. Disabling CUDA backend.")
        set(FGBS_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA backend." FORCE)
    endif()
endif()

set(FGBS_CXX_COMPILE_OPTIONS "-Wall" "-Wextra")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(FGBS_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(FGBS_SOURCE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(FGBS_GENERATED_INCLUDE_DIR ${CMAKE_BINARY_DIR}/generated)
set(FGBS_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/FrugalGBS")
set(FGBS_PUBLIC_HEADER_DIR "${FGBS_SOURCE_INCLUDE_DIR}/fgbs")

set(FGBS_RUNTIME_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)
set(FGBS_LIBRARY_OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${FGBS_LIBRARY_OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${FGBS_LIBRARY_OUTPUT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${FGBS_RUNTIME_OUTPUT_DIR})

file(MAKE_DIRECTORY ${FGBS_GENERATED_INCLUDE_DIR}/fgbs)

configure_file(${FGBS_PUBLIC_HEADER_DIR}/global.hpp.in ${FGBS_GENERATED_INCLUDE_DIR}/fgbs/global.hpp @ONLY)

find_package(OpenMP REQUIRED)

if(FGBS_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(Eigen3 3.4 REQUIRED CONFIG HINTS "${FGBS_EIGEN3_DIR}")
set(FGBS_EIGEN3_TARGET Eigen3::Eigen)
message(STATUS "Using external Eigen3 (${Eigen3_DIR})")

add_subdirectory(src)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FrugalGBSHelpers.cmake)

set(FGBS_INSTALL_TARGETS
    FrugalGBS_CPU
    fgbs_frontend
    fgbs_cpu
)
set(FGBS_COMMON_EXPORT_TARGETS ${FGBS_INSTALL_TARGETS})
if(FGBS_ENABLE_CUDA)
    list(APPEND FGBS_INSTALL_TARGETS FrugalGBS_GPU fgbs_gpu)
    set(FGBS_GPU_EXPORT_TARGETS FrugalGBS_GPU fgbs_gpu)
endif()

set(FGBS_PUBLIC_HEADER_FILES
    ${FGBS_PUBLIC_HEADER_DIR}/cumulants.hpp
    ${FGBS_PUBLIC_HEADER_DIR}/data.hpp
    ${FGBS_PUBLIC_HEADER_DIR}/layout.hpp
    ${FGBS_PUBLIC_HEADER_DIR}/sampler.hpp
    ${FGBS_PUBLIC_HEADER_DIR}/stats.hpp
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

install(
    TARGETS ${FGBS_COMMON_EXPORT_TARGETS}
    EXPORT FrugalGBSTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(FGBS_ENABLE_CUDA)
    install(
        TARGETS ${FGBS_GPU_EXPORT_TARGETS}
        EXPORT FrugalGBSGPUTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

install(
    DIRECTORY ${FGBS_PUBLIC_HEADER_DIR}/core/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fgbs/core
    FILES_MATCHING
    PATTERN "*.hpp"
)

install(
    FILES ${FGBS_PUBLIC_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fgbs
)

install(
    FILES ${FGBS_SOURCE_INCLUDE_DIR}/fgbs.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY ${FGBS_PUBLIC_HEADER_DIR}/cpu/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fgbs/cpu
    FILES_MATCHING
    PATTERN "*.hpp"
)

if(FGBS_ENABLE_CUDA)
    install(
        DIRECTORY ${FGBS_PUBLIC_HEADER_DIR}/gpu/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fgbs/gpu
        FILES_MATCHING
        PATTERN "*.cuh"
    )
endif()

install(
    FILES ${FGBS_GENERATED_INCLUDE_DIR}/fgbs/global.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fgbs
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FrugalGBSConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/FrugalGBSConfig.cmake
    INSTALL_DESTINATION ${FGBS_INSTALL_CMAKE_DIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/FrugalGBSConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/FrugalGBSConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/FrugalGBSConfigVersion.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FrugalGBSHelpers.cmake
    DESTINATION ${FGBS_INSTALL_CMAKE_DIR}
)

install(
    EXPORT FrugalGBSTargets
    FILE FrugalGBSTargets.cmake
    NAMESPACE FrugalGBS::
    DESTINATION ${FGBS_INSTALL_CMAKE_DIR}
)

if(FGBS_ENABLE_CUDA)
    install(
        EXPORT FrugalGBSGPUTargets
        FILE FrugalGBSGPUTargets.cmake
        NAMESPACE FrugalGBS::
        DESTINATION ${FGBS_INSTALL_CMAKE_DIR}
    )
endif()
