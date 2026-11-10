#include "fgbs/gpu/gpu_utils.cuh"

#include <cuda_runtime.h>

#include <vector>

namespace fgbs {

namespace gpu {

Result<void> cuda_result(cudaError_t code, char const* context) {
    if (code == cudaSuccess) {
        return Ok<void>{};
    }
    return Err{build_string(context, ": ", cudaGetErrorString(code))};
}

std::vector<usize> get_chunk_indices(usize data_count, usize chunk_count) {
    if (chunk_count == 0) {
        return {0};
    }

    std::vector<usize> chunk_indices(chunk_count + 1);

    usize base_size = data_count / chunk_count;
    usize remainder_size = data_count % chunk_count;

    for (usize i = 0, offset = 0; i < chunk_count; ++i) {
        chunk_indices[i] = offset;
        offset += base_size + (i < remainder_size);
    }
    chunk_indices[chunk_count] = data_count;

    return chunk_indices;
}

namespace {

bool _device_ids_initialized = false;
std::vector<int> _device_ids{};

}

Result<void> set_device_ids(std::vector<int> device_ids) {
    if (device_ids.empty()) {
        return Err{"Device ID vector is empty."};
    }

    for (usize i = 0; i < device_ids.size(); ++i) {
        for (usize j = i + 1; j < device_ids.size(); ++j) {
            if (device_ids[i] == device_ids[j]) {
                return Err{build_string("Device ID ", device_ids[i], " is duplicated.")};
            }
        }
    }

    int device_count = 0;
    auto device_count_result = cuda_result(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (!device_count_result.ok()) {
        return Err{device_count_result.error()};
    }
    if (device_count <= 0) {
        return Err{"No CUDA devices are available."};
    }

    for (const auto id : device_ids) {
        if (id < 0 || id >= device_count) {
            return Err{build_string("Device ID ", id, " is invalid for ", device_count, " available device(s).")};
        }
    }

    _device_ids_initialized = true;
    _device_ids = std::move(device_ids);
    return Ok<void>{};
}

Result<void> set_default_device_ids() {
    // Modified from QuEST
    int device_count = 0;
    auto device_count_result = cuda_result(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (!device_count_result.ok()) {
        return Err{device_count_result.error()};
    }

    std::vector<int> device_ids;
    device_ids.reserve(device_count);

    cudaDeviceProp props;
    for (int i = 0; i < device_count; ++i) {
        auto props_result = cuda_result(cudaGetDeviceProperties(&props, i), "cudaGetDeviceProperties");
        if (!props_result.ok()) {
            return Err{build_string("Failed to query device ", i, ": ", props_result.error())};
        }
        if (props.major != 9999) {
            device_ids.push_back(i);
        }
    }

    if (device_ids.empty()) {
        return Err{"No usable CUDA devices were found."};
    }

    return set_device_ids(std::move(device_ids));
}

std::vector<int> const& get_device_ids() {
    if (_device_ids_initialized) {
        return _device_ids;
    }

    _device_ids_initialized = true;

    auto init_result = set_default_device_ids();
    if (!init_result.ok()) {
        _device_ids.clear();
    }

    return _device_ids;
}

Result<int> get_current_device_id() {
    int id;
    auto current_device_id_result = cuda_result(cudaGetDevice(&id), "cudaGetDevice");
    if (!current_device_id_result.ok()) {
        return Err{current_device_id_result.error()};
    }
    return Ok<int>{id};
}

Result<usize> get_free_memory() {
    usize free_memory, total_memory;
    auto free_memory_result = cuda_result(cudaMemGetInfo(&free_memory, &total_memory), "cudaMemGetInfo");
    if (!free_memory_result.ok()) {
        return Err{free_memory_result.error()};
    }
    return Ok<usize>{free_memory};
}

Result<usize> get_max_threads() {
    auto current_id_result = get_current_device_id();
    if (!current_id_result.ok()) {
        return Err{current_id_result.error()};
    }

    cudaDeviceProp props;
    auto device_properties_result = cuda_result(cudaGetDeviceProperties(&props, current_id_result.value()), "cudaGetDeviceProperties");
    if (!device_properties_result.ok()) {
        return Err{device_properties_result.error()};
    }

    if (props.multiProcessorCount <= 0 || props.maxThreadsPerMultiProcessor <= 0) {
        return Err{"cudaGetDeviceProperties returned invalid thread capacity."};
    }

    return Ok<usize>{static_cast<usize>(props.multiProcessorCount) * static_cast<usize>(props.maxThreadsPerMultiProcessor)};
}

}

}
