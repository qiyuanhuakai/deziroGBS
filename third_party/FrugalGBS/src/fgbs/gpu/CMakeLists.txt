add_library(fgbs_gpu STATIC
    gpu_utils.cu
)

target_include_directories(fgbs_gpu PUBLIC ${FGBS_PUBLIC_INCLUDE_DIRS})

target_link_libraries(fgbs_gpu
    PUBLIC
        fgbs_frontend
        CUDA::cudart
        CUDA::curand
    PRIVATE
        OpenMP::OpenMP_CXX
        ${FGBS_EIGEN3_TARGET}
)

fgbs_apply_target_build_settings(fgbs_gpu)
