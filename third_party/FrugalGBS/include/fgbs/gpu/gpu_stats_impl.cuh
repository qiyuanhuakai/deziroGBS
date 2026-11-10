#ifndef FGBS_GPU_STATS_IMPL_CUH
#define FGBS_GPU_STATS_IMPL_CUH

#include "fgbs/gpu/gpu_common_impl.cuh"

#include "fgbs/core/stat_utils.hpp"
#include "fgbs/core/thread_utils.hpp"

#include <thread>
#include <vector>

namespace fgbs {

namespace gpu {

template<usize N, typename T>
__global__ void kernel_estimate_sample_parity_sums(T* __restrict__ parity_sums, usize parity_sum_count, Bitstring<N> const* __restrict__ samples, usize sample_count, Bitstring<N> const* __restrict__ correlator_bitstrings, T parity_sum_divisor) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= parity_sum_count) {
        return;
    }

    auto correlator_bitstring = &correlator_bitstrings[idx];

    i64 parity_sum = 0;
    for (usize i = 0; i < sample_count; ++i) {
        parity_sum += static_cast<i64>(device_characteristic<N>(correlator_bitstring, &samples[i]));
    }

    parity_sums[idx] = static_cast<T>(parity_sum) / parity_sum_divisor;
}

template<usize N>
Result<std::vector<real>> estimate_sample_correlator_sublists(SublistLayout<N> const& sublist_layout, std::vector<Bitstring<N>> const& samples) {
    const auto order_count = sublist_layout.cutoff_order + 1;
    std::vector<real> sample_correlator_sublists(sublist_layout.sublist_offset_vec[order_count], real{0.0});

    if (samples.empty()) {
        return Ok<std::vector<real>>{std::move(sample_correlator_sublists)};
    }

    auto const& device_ids = get_device_ids();
    if (device_ids.empty()) {
        return Err{"GPU correlator estimation failed: no configured GPU devices are available."};
    }
    const auto device_count = device_ids.size();
    const auto sample_count = samples.size();
    const auto sample_count_fp = static_cast<real>(sample_count);

    if (device_count == 1) {
        FGBS_VALIDATE_OR_RETURN(cuda_result(cudaSetDevice(device_ids[0]), "GPU correlator estimation failed at cudaSetDevice"));

        DeviceData<Bitstring<N>> d_samples(samples);
        FGBS_VALIDATE_OR_RETURN_CTX(validate_device_data(d_samples), "GPU correlator estimation failed: ");

        for (usize order = 0; order < order_count; ++order) {
            const auto sublist_size = sublist_layout.sublist_size_vec[order];
            if (sublist_size == 0) {
                continue;
            }

            DeviceData<real> d_correlators(sublist_size);
            DeviceData<Bitstring<N>> d_correlator_bitstrings(sublist_layout.get_sublist(sublist_layout.layout_bitstring_sublists, order));
            FGBS_VALIDATE_OR_RETURN_CTX(validate_device_data(d_correlators, d_correlator_bitstrings), "GPU correlator estimation failed: ");

            FGBS_VALIDATE_OR_RETURN_CTX(FGBS_INTERNAL_RUN_KERNEL(estimate_sample_parity_sums<FGBS_INTERNAL_EXPAND(N, real)>, 256, sublist_size, d_correlators.ptr(), sublist_size, d_samples.ptr(), sample_count, d_correlator_bitstrings.ptr(), sample_count_fp), "GPU correlator estimation failed: ");

            FGBS_VALIDATE_OR_RETURN(cuda_result(cudaDeviceSynchronize(), "GPU correlator estimation failed at cudaDeviceSynchronize"));

            auto correlators = sublist_layout.get_sublist_mut(sample_correlator_sublists, order);
            FGBS_VALIDATE_OR_RETURN_CTX(d_correlators.copy_to_host(correlators.data()), "GPU correlator estimation failed: ");
        }

        return Ok<std::vector<real>>{std::move(sample_correlator_sublists)};
    }

    std::vector<usize> sample_chunk_indices = get_chunk_indices(sample_count, device_count);
    std::vector<std::vector<i64>> parity_sum_sublists(device_count, std::vector<i64>(sublist_layout.sublist_offset_vec[order_count], i64{0}));
    std::vector<std::thread> threads;
    ThreadResultWrapper thread_state;

    for (usize i = 0; i < device_count; ++i) {
        threads.push_back(std::thread([&, i] {
            if (!thread_state.ok()) {
                return;
            }

            FGBS_THREAD_VALIDATE_OR_RETURN(thread_state, cuda_result(cudaSetDevice(device_ids[i]), "GPU correlator estimation failed at cudaSetDevice"));

            const auto sample_chunk_begin = sample_chunk_indices[i];
            const auto sample_chunk_size = sample_chunk_indices[i + 1] - sample_chunk_begin;
            if (sample_chunk_size == 0) {
                return;
            }

            DeviceData<Bitstring<N>> d_samples(&samples[sample_chunk_begin], sample_chunk_size);
            FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, validate_device_data(d_samples), "GPU correlator estimation failed: ");

            for (usize order = 0; order < order_count; ++order) {
                const auto sublist_size = sublist_layout.sublist_size_vec[order];
                if (sublist_size == 0) {
                    continue;
                }

                DeviceData<i64> d_parity_sums(sublist_size);
                DeviceData<Bitstring<N>> d_correlator_bitstrings(sublist_layout.get_sublist(sublist_layout.layout_bitstring_sublists, order));
                FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, validate_device_data(d_parity_sums, d_correlator_bitstrings), "GPU correlator estimation failed: ");

                FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, FGBS_INTERNAL_RUN_KERNEL(estimate_sample_parity_sums<FGBS_INTERNAL_EXPAND(N, i64)>, 256, sublist_size, d_parity_sums.ptr(), sublist_size, d_samples.ptr(), sample_chunk_size, d_correlator_bitstrings.ptr(), i64{1}), "GPU correlator estimation failed: ");

                FGBS_THREAD_VALIDATE_OR_RETURN(thread_state, cuda_result(cudaDeviceSynchronize(), "GPU correlator estimation failed at cudaDeviceSynchronize"));

                auto parity_sums = sublist_layout.get_sublist_mut(parity_sum_sublists[i], order);
                FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, d_parity_sums.copy_to_host(parity_sums.data()), "GPU correlator estimation failed: ");
            }
        }));
    }

    join_threads(threads);

    if (!thread_state.ok()) {
        return Err{thread_state.get().error()};
    }

    sample_correlator_sublists = compute_sums<real, i64>(parity_sum_sublists, sample_count_fp);

    return Ok<std::vector<real>>{std::move(sample_correlator_sublists)};
}

}

}

#endif
