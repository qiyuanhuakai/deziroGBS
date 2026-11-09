set(FGBS_PUBLIC_INCLUDE_DIRS
    $<BUILD_INTERFACE:${FGBS_SOURCE_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${FGBS_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

function(fgbs_apply_target_build_settings TARGET_NAME)
    foreach(CXX_COMPILE_OPTION IN LISTS FGBS_CXX_COMPILE_OPTIONS)
        target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CXX_COMPILE_OPTION}>")
        if(FGBS_ENABLE_CUDA)
            target_compile_options(${TARGET_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=${CXX_COMPILE_OPTION}>")
        endif()
    endforeach()

    target_compile_options(${TARGET_NAME} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>")
    if(FGBS_ENABLE_CUDA)
        target_compile_options(${TARGET_NAME} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>")
    endif()
endfunction()

add_library(fgbs_frontend STATIC
    layout.cpp
    core/correlator_utils.cpp
    core/cumulant_utils.cpp
    core/file_utils.cpp
    core/matrix_utils.cpp
    core/mmpy_format.cpp
    core/result.cpp
    core/string_utils.cpp
    core/thread_utils.cpp
    core/time_utils.cpp
)
target_include_directories(fgbs_frontend PUBLIC ${FGBS_PUBLIC_INCLUDE_DIRS})
target_compile_features(fgbs_frontend PUBLIC cxx_std_20)
target_link_libraries(fgbs_frontend
    PRIVATE
        OpenMP::OpenMP_CXX
        ${FGBS_EIGEN3_TARGET}
)
fgbs_apply_target_build_settings(fgbs_frontend)

add_subdirectory(cpu)
if(FGBS_ENABLE_CUDA)
    add_subdirectory(gpu)
endif()

add_library(FrugalGBS_CPU INTERFACE)
add_library(FrugalGBS::FrugalGBS_CPU ALIAS FrugalGBS_CPU)
target_include_directories(FrugalGBS_CPU INTERFACE ${FGBS_PUBLIC_INCLUDE_DIRS})
target_link_libraries(FrugalGBS_CPU INTERFACE
    fgbs_cpu
    OpenMP::OpenMP_CXX
    ${FGBS_EIGEN3_TARGET}
)
target_compile_definitions(FrugalGBS_CPU INTERFACE FGBS_GPU_ENABLED=0)

if(FGBS_ENABLE_CUDA)
    add_library(FrugalGBS_GPU INTERFACE)
    add_library(FrugalGBS::FrugalGBS_GPU ALIAS FrugalGBS_GPU)
    target_include_directories(FrugalGBS_GPU INTERFACE ${FGBS_PUBLIC_INCLUDE_DIRS})
    target_link_libraries(FrugalGBS_GPU INTERFACE
        fgbs_gpu
        OpenMP::OpenMP_CXX
        ${FGBS_EIGEN3_TARGET}
    )
    target_compile_definitions(FrugalGBS_GPU INTERFACE FGBS_GPU_ENABLED=1)
endif()
