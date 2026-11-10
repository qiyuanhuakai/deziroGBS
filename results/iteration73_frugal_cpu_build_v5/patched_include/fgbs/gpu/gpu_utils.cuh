#ifndef FGBS_GPU_UTILS_CUH
#define FGBS_GPU_UTILS_CUH

#include "fgbs/core/macro_utils.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/string_utils.hpp"
#include "fgbs/core/type_utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace fgbs {

namespace gpu {

Result<void> cuda_result(cudaError_t code, char const* context);

#define FGBS_INTERNAL_RUN_KERNEL(NAME, BLOCK_SIZE, THREAD_COUNT, ...)                                          \
([&]() {                                                                                                       \
    kernel_##NAME<<<get_grid_size(FGBS_INTERNAL_EXPAND(BLOCK_SIZE, THREAD_COUNT)), BLOCK_SIZE>>>(__VA_ARGS__); \
    return cuda_result(cudaGetLastError(), "kernel_"#NAME);                                                    \
}())

#define FGBS_INTERNAL_RUN_KERNEL_WITH_STREAM(NAME, BLOCK_SIZE, THREAD_COUNT, STREAM, ...)                                 \
([&]() {                                                                                                                  \
    kernel_##NAME<<<get_grid_size(FGBS_INTERNAL_EXPAND(BLOCK_SIZE, THREAD_COUNT)), BLOCK_SIZE, 0, STREAM>>>(__VA_ARGS__); \
    return cuda_result(cudaGetLastError(), "kernel_"#NAME);                                                               \
}())

template<typename T, typename V>
usize get_grid_size(T block_size, V thread_count) {
    return static_cast<usize>(std::ceil(static_cast<f64>(thread_count) / static_cast<f64>(block_size)));
}

std::vector<usize> get_chunk_indices(usize data_count, usize chunk_count);

Result<void> set_device_ids(std::vector<int> device_ids);

Result<void> set_default_device_ids();

std::vector<int> const& get_device_ids();

Result<int> get_current_device_id();

Result<usize> get_free_memory();

Result<usize> get_max_threads();

}

}

#endif
