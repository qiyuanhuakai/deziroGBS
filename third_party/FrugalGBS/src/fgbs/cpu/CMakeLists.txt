set(FGBS_XSIMD_DIR "" CACHE PATH "Optional hint path for xsimd package config/prefix.")
find_package(xsimd 14 REQUIRED CONFIG HINTS "${FGBS_XSIMD_DIR}")
message(STATUS "Using external xsimd (${xsimd_DIR})")

add_library(fgbs_cpu STATIC
    cpu_sampler.cpp
)

target_include_directories(fgbs_cpu PUBLIC ${FGBS_PUBLIC_INCLUDE_DIRS})

target_link_libraries(fgbs_cpu
    PUBLIC
        fgbs_frontend
        xsimd
    PRIVATE
        OpenMP::OpenMP_CXX
        ${FGBS_EIGEN3_TARGET}
)

fgbs_apply_target_build_settings(fgbs_cpu)
