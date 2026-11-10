#ifndef FGBS_GPU_CUMULANTS_IMPL_CUH
#define FGBS_GPU_CUMULANTS_IMPL_CUH

#include "fgbs/gpu/gpu_common_impl.cuh"

#include "fgbs/core/cumulant_utils.hpp"
#include "fgbs/core/consts.hpp"
#include "fgbs/core/thread_utils.hpp"

#include <thread>
#include <vector>

namespace fgbs {

namespace gpu {

template<usize N>
__global__ void kernel_compute_cumulants(real* __restrict__ cumulants, usize cumulant_count, real const* __restrict__ data, upto<N> const* __restrict__ order_vec, upto<N> const* __restrict__ ones_vec, usize const* __restrict__ cumulant_term_range_vec, usize const* __restrict__ cumulant_order_range_vec, usize const* __restrict__ cumulant_index_range_vec, usize const* __restrict__ cumulant_term_vec, usize const* __restrict__ cumulant_order_vec, usize const* __restrict__ cumulant_index_vec, usize const* __restrict__ index_arrays, real const* __restrict__ coeff_vec, usize cutoff_order) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= cumulant_count) {
        return;
    }

    const auto order = order_vec[idx];
    const auto ones = &ones_vec[cutoff_order * idx];

    real sum = 0.0;

    const auto term_range_begin = cumulant_term_range_vec[order], term_range_end = cumulant_term_range_vec[order + 1];

    for (usize i = term_range_begin, j = cumulant_order_range_vec[order], k = cumulant_index_range_vec[order]; i < term_range_end; ++i) {
        const auto t = cumulant_term_vec[i], r = j + t;

        real term = 1.0;

        for (; j < r; ++j) {
            usize index = 0;
            for (usize m = 0, s = k + cumulant_order_vec[j]; k < s; ++k, ++m) {
                index += index_arrays[(N + 1) * (cutoff_order - m) + ones[cumulant_index_vec[s - m - 1]]];
            }
            term *= data[index];
        }

        sum += coeff_vec[i] * term;
    }

    cumulants[idx] = sum;
}

template<usize N>
Result<std::vector<real>> compute_cumulants(BitstringLayout<N> const& bitstring_layout, CumulantLayout<N> const& cumulant_layout, std::vector<real> const& data, usize sample_count, bool from_cumulants) {
    const auto cutoff_order = bitstring_layout.cutoff_order;
    if (cutoff_order == 0) {
        return Ok<std::vector<real>>{std::vector<real>{0.0}};
    }
    if (from_cumulants && sample_count != 0) {
        return Err{"Finite-sample bias correction for cumulant input is not yet implemented."};
    }

    auto const& device_ids = get_device_ids();
    if (device_ids.empty()) {
        return Err{"GPU cumulant computation failed: no configured GPU devices are available."};
    }
    const auto device_count = device_ids.size();

    const auto layout_size = bitstring_layout.len;
    std::vector<real> cumulants(layout_size);
    std::vector<usize> cumulant_chunk_indices = get_chunk_indices(layout_size, device_count);
    std::vector<std::thread> threads;
    ThreadResultWrapper thread_state;

    for (usize i = 0; i < device_count; ++i) {
        threads.push_back(std::thread([&, from_cumulants, cutoff_order, i] {
            if (!thread_state.ok()) {
                return;
            }

            FGBS_THREAD_VALIDATE_OR_RETURN(thread_state, cuda_result(cudaSetDevice(device_ids[i]), "GPU cumulant computation failed at cudaSetDevice"));

            const auto partial_cumulant_begin = cumulant_chunk_indices[i];
            const auto partial_cumulant_count = cumulant_chunk_indices[i + 1] - partial_cumulant_begin;
            if (partial_cumulant_count == 0) {
                return;
            }

            DeviceData<real> d_cumulants(partial_cumulant_count);
            DeviceData<real> d_data(data);
            DeviceData<upto<N>> d_order_vec(&bitstring_layout.order_vec[partial_cumulant_begin], partial_cumulant_count);
            DeviceData<upto<N>> d_ones_vec(bitstring_layout.get_ones_ptr(partial_cumulant_begin), cutoff_order * partial_cumulant_count);
            DeviceData<usize> d_cumulant_term_range_vec(cumulant_layout.cumulant_term_range_vec);
            DeviceData<usize> d_cumulant_order_range_vec(cumulant_layout.cumulant_order_range_vec);
            DeviceData<usize> d_cumulant_index_range_vec(cumulant_layout.cumulant_index_range_vec);
            DeviceData<usize> d_cumulant_term_vec(cumulant_layout.cumulant_term_vec);
            DeviceData<usize> d_cumulant_order_vec(cumulant_layout.cumulant_order_vec);
            DeviceData<usize> d_cumulant_index_vec(cumulant_layout.cumulant_index_vec);
            DeviceData<usize> d_index_arrays(FGBS_INTERNAL_PTR_CAST(usize const*, gbs_index_arrays<N>.data()), (N + 1) * (N + 1));
            DeviceData<real> d_coeff_vec(compute_cumulant_coeff_vec(cumulant_layout, sample_count, from_cumulants));
            FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, validate_device_data(d_cumulants, d_data, d_order_vec, d_ones_vec, d_cumulant_term_range_vec, d_cumulant_order_range_vec, d_cumulant_index_range_vec, d_cumulant_term_vec, d_cumulant_order_vec, d_cumulant_index_vec, d_index_arrays, d_coeff_vec), "GPU cumulant computation failed: ");

            FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, FGBS_INTERNAL_RUN_KERNEL(compute_cumulants<N>, 256, partial_cumulant_count, d_cumulants.ptr(), partial_cumulant_count, d_data.ptr(), d_order_vec.ptr(), d_ones_vec.ptr(), d_cumulant_term_range_vec.ptr(), d_cumulant_order_range_vec.ptr(), d_cumulant_index_range_vec.ptr(), d_cumulant_term_vec.ptr(), d_cumulant_order_vec.ptr(), d_cumulant_index_vec.ptr(), d_index_arrays.ptr(), d_coeff_vec.ptr(), cutoff_order), "GPU cumulant computation failed: ");

            FGBS_THREAD_VALIDATE_OR_RETURN(thread_state, cuda_result(cudaDeviceSynchronize(), "GPU cumulant computation failed at cudaDeviceSynchronize"));

            FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, d_cumulants.copy_to_host(&cumulants[partial_cumulant_begin]), "GPU cumulant computation failed: ");
        }));
    }

    join_threads(threads);

    if (!thread_state.ok()) {
        return Err{thread_state.get().error()};
    }

    return Ok<std::vector<real>>{std::move(cumulants)};
}

}

}

#endif
