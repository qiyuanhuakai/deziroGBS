#ifndef FGBS_GPU_SAMPLER_IMPL_CUH
#define FGBS_GPU_SAMPLER_IMPL_CUH

#include "fgbs/gpu/gpu_common_impl.cuh"

#include "fgbs/core/consts.hpp"
#include "fgbs/core/validation.hpp"
#include "fgbs/core/random_utils.hpp"
#include "fgbs/core/thread_utils.hpp"

#include <curand_kernel.h>

#include <climits>
#include <cmath>
#include <thread>
#include <vector>

namespace fgbs {

namespace gpu {

#if FGBS_REAL_PRECISION < 2
#define FGBS_INTERNAL_CURAND_UNIFORM_REAL curand_uniform
#else
#define FGBS_INTERNAL_CURAND_UNIFORM_REAL curand_uniform_double
#endif
template<usize N>
using cache_idx = upto<N <= 2 ? 1 : N*(N - 1) * (N - 2)>;

template<usize N>
__host__ __device__ constexpr usize device_partial_marginal_arr_size() {
    return N <= 1 ? 1 : N * (N - 1) / 2;
}

template<usize N>
__host__ __device__ constexpr usize device_elided_marginal_arr_size() {
    return N <= 1 ? 1 : N * (N - 1) / 2;
}

template<usize N>
__host__ __device__ constexpr usize device_partial_elided_marginal_arr_size() {
    return N <= 2 ? 1 : N * (N - 1) * (N - 2) / 6;
}

template<usize N>
__host__ __device__ constexpr usize device_double_elided_marginal_arr_size() {
    return N <= 2 ? 1 : N * (N - 1) * (N - 2) / 6;
}

template<usize N>
__forceinline__ __device__ real device_get_partial_marginal(real const* __restrict__ marginal_arr, cache_idx<N> begin, cache_idx<N> end) {
    if (begin == end) {
        return real{1.0};
    }
    else {
        return marginal_arr[begin + end * (end - 1) / 2];
    }
}

template<usize N>
__forceinline__ __device__ void device_set_partial_marginal(real* __restrict__ marginal_arr, cache_idx<N> begin, cache_idx<N> end, real marginal) {
    marginal_arr[begin + end * (end - 1) / 2] = marginal;
}

template<usize N>
__forceinline__ __device__ real device_get_elided_marginal(real const* __restrict__ marginal_arr, cache_idx<N> elided, cache_idx<N> end) {
    return marginal_arr[elided + end * (end - 1) / 2];
}

template<usize N>
__forceinline__ __device__ void device_set_elided_marginal(real* __restrict__ marginal_arr, cache_idx<N> elided, cache_idx<N> end, real marginal) {
    marginal_arr[elided + end * (end - 1) / 2] = marginal;
}

template<usize N>
__forceinline__ __device__ real device_get_partial_elided_marginal(real const* __restrict__ partial_marginal_arr, real const* __restrict__ partial_elided_marginal_arr, cache_idx<N> begin, cache_idx<N> elided, cache_idx<N> end) {
    if (begin == elided) {
        return device_get_partial_marginal<N>(partial_marginal_arr, begin + 1, end);
    }
    else {
        return partial_elided_marginal_arr[begin + elided * (elided - 1) / 2 + end * (end - 1) * (end - 2) / 6];
    }
}

template<usize N>
__forceinline__ __device__ void device_set_partial_elided_marginal(real* __restrict__ marginal_arr, cache_idx<N> begin, cache_idx<N> elided, cache_idx<N> end, real marginal) {
    marginal_arr[begin + elided * (elided - 1) / 2 + end * (end - 1) * (end - 2) / 6] = marginal;
}

template<usize N>
__forceinline__ __device__ real device_get_double_elided_marginal(real const* __restrict__ marginal_arr, cache_idx<N> first_elided, cache_idx<N> second_elided, cache_idx<N> end) {
    return marginal_arr[first_elided + second_elided * (second_elided - 1) / 2 + end * (end - 1) * (end - 2) / 6];
}

template<usize N>
__forceinline__ __device__ void device_set_double_elided_marginal(real* __restrict__ marginal_arr, cache_idx<N> first_elided, cache_idx<N> second_elided, cache_idx<N> end, real marginal) {
    marginal_arr[first_elided + second_elided * (second_elided - 1) / 2 + end * (end - 1) * (end - 2) / 6] = marginal;
}

#define FGBS_INTERNAL_DEVICE_MARGINAL_END_ZERO_DIFF \
end_zero_diff : {}                                  \
                                                    \
const auto parity_sign = 2 * parity_arr[i];         \
prev_prob = device_clamped_prob(prev_prob * (real{0.5} + real{0.5} * parity_sign * first_correlator_cumulants[i]) + parity_sign * zero_diff);

template<usize N>
__forceinline__ __device__ void device_compute_partial_marginal(real const* __restrict__ correlator_cumulants, real const* __restrict__ first_correlator_cumulants, usize const* __restrict__ index_arrays, usize cutoff_order, real* __restrict__ partial_marginal_arr, real const* __restrict__ partial_elided_marginal_arr, real const* __restrict__ parity_arr, cache_idx<N> begin, cache_idx<N> end, usize marginal_order) {
    auto i = end - 1;

    auto prev_prob = device_get_partial_marginal<N>(partial_marginal_arr, begin, i);

    real zero_diff = 0.0, c = 0.0;

    auto first_cumulant_offset = index_arrays[(N + 1) * cutoff_order + i];

    if (marginal_order <= 1) {
        goto end_zero_diff;
    }

    for (cache_idx<N> j = begin; j < i; ++j) {
        auto second_mult = real{0.5} * parity_arr[j];

        auto second_cumulant_offset = first_cumulant_offset + index_arrays[(N + 1) * (cutoff_order - 1) + j];

        device_kahan_add(zero_diff, c, second_mult * correlator_cumulants[second_cumulant_offset] * device_get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, begin, j, i));

        if (marginal_order <= 2) {
            continue;
        }

        second_mult *= device_get_partial_marginal<N>(partial_marginal_arr, j + 1, i);

        for (cache_idx<N> k = begin; k < j; ++k) {
            auto third_cumulant_offset = second_cumulant_offset + index_arrays[(N + 1) * (cutoff_order - 2) + k];

            device_kahan_add(zero_diff, c, second_mult * parity_arr[k] * correlator_cumulants[third_cumulant_offset] * device_get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, begin, k, j));
        }
    }

    FGBS_INTERNAL_DEVICE_MARGINAL_END_ZERO_DIFF

    device_set_partial_marginal<N>(partial_marginal_arr, begin, end, prev_prob);
}

#define FGBS_INTERNAL_DEVICE_COMPUTE_ELIDED_MARGINAL_SECOND(K, J)                                                                                                      \
device_kahan_add(zero_diff, c, second_mult* correlator_cumulants[second_cumulant_offset] * device_get_double_elided_marginal<N>(double_elided_marginal_arr, K, J, i)); \
                                                                                                                                                                       \
if (marginal_order <= 2) {                                                                                                                                             \
    continue;                                                                                                                                                          \
}                                                                                                                                                                      \
                                                                                                                                                                       \
second_mult *= device_get_partial_marginal<N>(partial_marginal_arr, J + 1, i);

#define FGBS_INTERNAL_DEVICE_COMPUTE_ELIDED_MARGINAL_THIRD(L, K, J) device_kahan_add(zero_diff, c, second_mult* parity_arr[k] * correlator_cumulants[third_cumulant_offset] * device_get_double_elided_marginal<N>(double_elided_marginal_arr, L, K, J));

template<usize N>
__forceinline__ __device__ void device_compute_elided_marginal(real const* __restrict__ correlator_cumulants, real const* __restrict__ first_correlator_cumulants, usize const* __restrict__ index_arrays, usize cutoff_order, real const* __restrict__ partial_marginal_arr, real* __restrict__ elided_marginal_arr, real const* __restrict__ double_elided_marginal_arr, real const* __restrict__ parity_arr, cache_idx<N> elided, cache_idx<N> end, usize marginal_order) {
    auto i = end - 1;

    auto prev_prob = device_get_elided_marginal<N>(elided_marginal_arr, elided, i);

    real zero_diff = 0.0, c = 0.0;

    auto first_cumulant_offset = index_arrays[(N + 1) * cutoff_order + i];

    if (marginal_order <= 1) {
        goto end_zero_diff;
    }

    for (cache_idx<N> j = 0; j < i; ++j) {
        if (j == elided) {
            continue;
        }

        auto second_mult = real{0.5} * parity_arr[j];

        auto second_cumulant_offset = first_cumulant_offset + index_arrays[(N + 1) * (cutoff_order - 1) + j];

        if (j < elided) {
            // j[]e[]i
            FGBS_INTERNAL_DEVICE_COMPUTE_ELIDED_MARGINAL_SECOND(j, elided)

            for (cache_idx<N> k = 0; k < j; ++k) {
                auto third_cumulant_offset = second_cumulant_offset + index_arrays[(N + 1) * (cutoff_order - 2) + k];

                // k[]j[]e[]i
                FGBS_INTERNAL_DEVICE_COMPUTE_ELIDED_MARGINAL_THIRD(k, j, elided)
            }
        }
        else {
            // e[]j[]i
            FGBS_INTERNAL_DEVICE_COMPUTE_ELIDED_MARGINAL_SECOND(elided, j)

            for (cache_idx<N> k = 0; k < j; ++k) {
                if (k == elided) {
                    continue;
                }

                auto third_cumulant_offset = second_cumulant_offset + index_arrays[(N + 1) * (cutoff_order - 2) + k];

                if (k < elided) {
                    // k[]e[]j[]i
                    FGBS_INTERNAL_DEVICE_COMPUTE_ELIDED_MARGINAL_THIRD(k, elided, j)
                }
                else {
                    // e[]k[]j[]i
                    FGBS_INTERNAL_DEVICE_COMPUTE_ELIDED_MARGINAL_THIRD(elided, k, j)
                }
            }
        }
    }

    FGBS_INTERNAL_DEVICE_MARGINAL_END_ZERO_DIFF

    device_set_elided_marginal<N>(elided_marginal_arr, elided, end, prev_prob);
}

#define FGBS_INTERNAL_DEVICE_COMPUTE_PARTIAL_ELIDED_MARGINAL_SECOND(K, J) device_kahan_add(zero_diff, c, second_parity* correlator_cumulants[second_cumulant_offset] * device_get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, begin, K, J) * device_get_partial_marginal<N>(partial_marginal_arr, J + 1, i));

template<usize N>
__forceinline__ __device__ void device_compute_partial_elided_marginal(real const* __restrict__ correlator_cumulants, real const* __restrict__ first_correlator_cumulants, usize const* __restrict__ index_arrays, usize cutoff_order, real const* __restrict__ partial_marginal_arr, real* __restrict__ partial_elided_marginal_arr, real const* __restrict__ parity_arr, cache_idx<N> begin, cache_idx<N> elided, cache_idx<N> end, usize marginal_order) {
    auto i = end - 1;

    auto prev_prob = device_get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, begin, elided, i);

    real zero_diff = 0.0, c = 0.0;

    auto first_cumulant_offset = index_arrays[(N + 1) * cutoff_order + i];

    if (marginal_order <= 1) {
        goto end_zero_diff;
    }

    for (cache_idx<N> j = begin; j < i; ++j) {
        if (j == elided) {
            continue;
        }

        auto second_parity = real{0.5} * parity_arr[j];

        auto second_cumulant_offset = first_cumulant_offset + index_arrays[(N + 1) * (cutoff_order - 1) + j];

        if (j < elided) {
            // j[]e[]i
            FGBS_INTERNAL_DEVICE_COMPUTE_PARTIAL_ELIDED_MARGINAL_SECOND(j, elided)
        }
        else {
            // e[]j[]i
            FGBS_INTERNAL_DEVICE_COMPUTE_PARTIAL_ELIDED_MARGINAL_SECOND(elided, j)
        }
    }

    FGBS_INTERNAL_DEVICE_MARGINAL_END_ZERO_DIFF

    device_set_partial_elided_marginal<N>(partial_elided_marginal_arr, begin, elided, end, prev_prob);
}

#define FGBS_INTERNAL_DEVICE_COMPUTE_DOUBLE_ELIDED_MARGINAL_SECOND(L, K, J) device_kahan_add(zero_diff, c, second_parity* correlator_cumulants[second_cumulant_offset] * device_get_double_elided_marginal<N>(double_elided_marginal_arr, L, K, J) * device_get_partial_marginal<N>(partial_marginal_arr, J + 1, i));

template<usize N>
__forceinline__ __device__ void device_compute_double_elided_marginal(real const* __restrict__ correlator_cumulants, real const* __restrict__ first_correlator_cumulants, usize const* __restrict__ index_arrays, usize cutoff_order, real const* __restrict__ partial_marginal_arr, real* __restrict__ double_elided_marginal_arr, real const* __restrict__ parity_arr, cache_idx<N> first_elided, cache_idx<N> second_elided, cache_idx<N> end, usize marginal_order) {
    auto i = end - 1;

    auto prev_prob = device_get_double_elided_marginal<N>(double_elided_marginal_arr, first_elided, second_elided, i);

    real zero_diff = 0.0, c = 0.0;

    auto first_cumulant_offset = index_arrays[(N + 1) * cutoff_order + i];

    if (marginal_order <= 1) {
        goto end_zero_diff;
    }

    for (cache_idx<N> j = 0; j < i; ++j) {
        if (j == first_elided || j == second_elided) {
            continue;
        }

        auto second_parity = real{0.5} * parity_arr[j];

        auto second_cumulant_offset = first_cumulant_offset + index_arrays[(N + 1) * (cutoff_order - 1) + j];

        if (j < first_elided) {
            // j[]e1[]e2[]i
            FGBS_INTERNAL_DEVICE_COMPUTE_DOUBLE_ELIDED_MARGINAL_SECOND(j, first_elided, second_elided)
        }
        else if (j < second_elided) {
            // e1[]j[]e2[]i
            FGBS_INTERNAL_DEVICE_COMPUTE_DOUBLE_ELIDED_MARGINAL_SECOND(first_elided, j, second_elided)
        }
        else {
            // e1[]e2[]j[]i
            FGBS_INTERNAL_DEVICE_COMPUTE_DOUBLE_ELIDED_MARGINAL_SECOND(first_elided, second_elided, j)
        }
    }

    FGBS_INTERNAL_DEVICE_MARGINAL_END_ZERO_DIFF

    device_set_double_elided_marginal<N>(double_elided_marginal_arr, first_elided, second_elided, end, prev_prob);
}

__forceinline__ __device__ usize device_partial_marginal_order(usize cutoff_order) {
    if (cutoff_order >= 5) {
        return cutoff_order - 2;
    }
    else if (cutoff_order >= 3) {
        return cutoff_order - 1;
    }
    else {
        return 2;
    }
}

__forceinline__ __device__ usize device_elided_marginal_order(usize cutoff_order) {
    if (cutoff_order >= 5) {
        return cutoff_order - 2;
    }
    else if (cutoff_order >= 3) {
        return cutoff_order - 1;
    }
    else {
        return 2;
    }
}

__forceinline__ __device__ usize device_partial_elided_marginal_order(usize cutoff_order) {
    if (cutoff_order >= 5) {
        return cutoff_order - 3;
    }
    else {
        return 2;
    }
}

__forceinline__ __device__ usize device_double_elided_marginal_order(usize cutoff_order) {
    if (cutoff_order >= 5) {
        return cutoff_order - 3;
    }
    else {
        return 2;
    }
}

__global__ void kernel_init_rand_states(curandState* rand_states, usize rand_state_count, ull seed, ull offset) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= rand_state_count) {
        return;
    }

    curand_init(seed, static_cast<ull>(idx) + offset, 0, &rand_states[idx]);
}

template<usize N>
__global__ void kernel_generate_samples(Bitstring<N>* __restrict__ samples, usize sample_count, curandState* rand_states, usize const* __restrict__ sum_indices, real const* __restrict__ correlator_cumulants, real const* __restrict__ first_correlator_cumulants, usize const* __restrict__ index_arrays, usize cutoff_order, real* __restrict__ partial_marginal_cache, real* __restrict__ elided_marginal_cache, real* __restrict__ partial_elided_marginal_cache, real* __restrict__ double_elided_marginal_cache) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= sample_count) {
        return;
    }

    auto sample = &samples[idx];

    auto rand_state = rand_states[idx];

    real prev_prob = 1.0;

    auto partial_marginal_arr = &partial_marginal_cache[idx * device_partial_marginal_arr_size<N>()];
    auto elided_marginal_arr = &elided_marginal_cache[idx * device_elided_marginal_arr_size<N>()];
    auto partial_elided_marginal_arr = &partial_elided_marginal_cache[idx * device_partial_elided_marginal_arr_size<N>()];
    auto double_elided_marginal_arr = &double_elided_marginal_cache[idx * device_double_elided_marginal_arr_size<N>()];

    real parity_arr[N];

    auto partial_order = device_partial_marginal_order(cutoff_order);
    auto elided_order = device_elided_marginal_order(cutoff_order);
    auto partial_elided_order = device_partial_elided_marginal_order(cutoff_order);
    auto double_elided_order = device_double_elided_marginal_order(cutoff_order);

    for (cache_idx<N> i = 0; i < N; ++i) {
        real zero_diff = 0.0, c = 0.0;

        auto correlator_cumulant_ptr = &correlator_cumulants[sum_indices[i]];

        if (cutoff_order <= 1) {
            goto end_zero_diff;
        }

        for (cache_idx<N> j = 0; j < i; ++j) {
            if (j == 0) {
                device_set_partial_marginal<N>(partial_marginal_arr, j, i, prev_prob);
            }
            else {
                device_compute_partial_marginal<N>(correlator_cumulants, first_correlator_cumulants, index_arrays, cutoff_order, partial_marginal_arr, partial_elided_marginal_arr, parity_arr, j, i, partial_order);
            }
        }

        for (cache_idx<N> j = 0; j < i; ++j) {
            if (j == 0) {
                device_set_elided_marginal<N>(elided_marginal_arr, j, i, device_get_partial_marginal<N>(partial_marginal_arr, 1, i));
            }
            else if (j == i - 1) {
                device_set_elided_marginal<N>(elided_marginal_arr, j, i, device_get_partial_marginal<N>(partial_marginal_arr, 0, i - 1));
            }
            else {
                device_compute_elided_marginal<N>(correlator_cumulants, first_correlator_cumulants, index_arrays, cutoff_order, partial_marginal_arr, elided_marginal_arr, double_elided_marginal_arr, parity_arr, j, i, elided_order);
            }
        }

        for (cache_idx<N> j = 0; j < i; ++j) {
            for (cache_idx<N> k = 0; k < j; ++k) {
                if (k == 0) {
                    device_set_partial_elided_marginal<N>(partial_elided_marginal_arr, k, j, i, device_get_elided_marginal<N>(elided_marginal_arr, j, i));
                }
                else if (j == i - 1) {
                    device_set_partial_elided_marginal<N>(partial_elided_marginal_arr, k, j, i, device_get_partial_marginal<N>(partial_marginal_arr, k, i - 1));
                }
                else {
                    device_compute_partial_elided_marginal<N>(correlator_cumulants, first_correlator_cumulants, index_arrays, cutoff_order, partial_marginal_arr, partial_elided_marginal_arr, parity_arr, k, j, i, partial_elided_order);
                }
            }
        }

        for (cache_idx<N> j = 0; j < i; ++j) {
            for (cache_idx<N> k = 0; k < j; ++k) {
                if (k == 0 && j == 1) {
                    device_set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, device_get_partial_marginal<N>(partial_marginal_arr, 2, i));
                }
                else if (k == 0 && j == i - 1) {
                    device_set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, device_get_partial_marginal<N>(partial_marginal_arr, 1, i - 1));
                }
                else if (k == i - 2 && j == i - 1) {
                    device_set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, device_get_partial_marginal<N>(partial_marginal_arr, 0, i - 2));
                }
                else if (j == i - 1) {
                    device_set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, device_get_elided_marginal<N>(elided_marginal_arr, k, i - 1));
                }
                else {
                    device_compute_double_elided_marginal<N>(correlator_cumulants, first_correlator_cumulants, index_arrays, cutoff_order, partial_marginal_arr, double_elided_marginal_arr, parity_arr, k, j, i, double_elided_order);
                }
            }
        }

        for (cache_idx<N> j = 0; j < i; ++j) {
            auto second_parity = real{0.5} * parity_arr[j];

            device_kahan_add(zero_diff, c, second_parity * (*(++correlator_cumulant_ptr)) * device_get_elided_marginal<N>(elided_marginal_arr, j, i));

            if (cutoff_order <= 2) {
                continue;
            }

            auto second_marginal = device_get_partial_marginal<N>(partial_marginal_arr, j + 1, i);

            for (cache_idx<N> k = 0; k < j; ++k) {
                auto third_parity = second_parity * parity_arr[k];

                device_kahan_add(zero_diff, c, third_parity * (*(++correlator_cumulant_ptr)) * device_get_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i));

                if (cutoff_order <= 3) {
                    continue;
                }

                auto third_marginal = device_get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, k + 1, j, i);

                for (cache_idx<N> l = 0; l < k; ++l) {
                    auto fourth_parity = third_parity * parity_arr[l];

                    device_kahan_add(zero_diff, c, fourth_parity * (*(++correlator_cumulant_ptr)) * device_get_double_elided_marginal<N>(double_elided_marginal_arr, l, k, j) * second_marginal);

                    if (cutoff_order <= 4) {
                        continue;
                    }

                    for (cache_idx<N> m = 0; m < l; ++m) {
                        auto fifth_parity = fourth_parity * parity_arr[m];

                        device_kahan_add(zero_diff, c, fifth_parity * (*(++correlator_cumulant_ptr)) * device_get_double_elided_marginal<N>(double_elided_marginal_arr, m, l, k) * third_marginal);
                    }
                }
            }
        }

    end_zero_diff : {}

        const auto zero_next_prob = device_clamped_prob(prev_prob * (real{0.5} + real{0.5} * first_correlator_cumulants[i]) + zero_diff);
        const auto zero_prob = zero_next_prob / prev_prob;

        if (FGBS_INTERNAL_CURAND_UNIFORM_REAL(&rand_state) > zero_prob) {
            device_flip_bit<N>(sample, i);
            parity_arr[i] = real{-0.5};
            prev_prob -= zero_next_prob;
        }
        else {
            parity_arr[i] = real{0.5};
            prev_prob = zero_next_prob;
        }
    }

    rand_states[idx] = rand_state;
}

template<usize N>
Result<std::vector<Bitstring<N>>> generate_samples(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, usize sample_count, uint seed) {
    if (sample_count == 0) {
        return Ok<std::vector<Bitstring<N>>>{std::vector<Bitstring<N>>{}};
    }

    auto const& device_ids = get_device_ids();
    if (device_ids.empty()) {
        return Err{"GPU sample generation failed: no configured GPU devices are available."};
    }
    const auto device_count = device_ids.size();
    const auto cutoff_order = bitstring_layout.cutoff_order;
    std::vector<usize> sample_chunk_indices = get_chunk_indices(sample_count, device_count);
    std::vector<Bitstring<N>> samples(sample_count);
    std::vector<std::thread> threads;
    ThreadResultWrapper thread_state;

    auto seed_generator = UniformRandomInt<usize>(seed, 0, ULLONG_MAX);
    const auto seeds = get_random_vector<uint>(seed_generator, device_count);

    for (usize i = 0; i < device_count; ++i) {
        threads.push_back(std::thread([&, cutoff_order, i] {
            if (!thread_state.ok()) {
                return;
            }

            FGBS_THREAD_VALIDATE_OR_RETURN(thread_state, cuda_result(cudaSetDevice(device_ids[i]), "GPU sample generation failed at cudaSetDevice"));

            const auto sample_chunk_begin = sample_chunk_indices[i];
            const auto sample_chunk_size = sample_chunk_indices[i + 1] - sample_chunk_begin;
            if (sample_chunk_size == 0) {
                return;
            }

            DeviceData<usize> d_sum_indices(bitstring_layout.sum_index_arr);
            DeviceData<real> d_correlator_cumulants(sampling_target.correlator_cumulant_vec);
            DeviceData<real> d_first_correlator_cumulants(sampling_target.first_correlator_cumulant_vec);
            DeviceData<usize> d_index_arrays(FGBS_INTERNAL_PTR_CAST(usize const*, gbs_index_arrays<N>.data()), (N + 1) * (N + 1));
            FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, validate_device_data(d_sum_indices, d_correlator_cumulants, d_first_correlator_cumulants, d_index_arrays), "GPU sample generation failed: ");

            constexpr auto PARTIAL_MARGINAL_COUNT = device_partial_marginal_arr_size<N>();
            constexpr auto ELIDED_MARGINAL_COUNT = device_elided_marginal_arr_size<N>();
            constexpr auto PARTIAL_ELIDED_MARGINAL_COUNT = device_partial_elided_marginal_arr_size<N>();
            constexpr auto DOUBLE_ELIDED_MARGINAL_COUNT = device_double_elided_marginal_arr_size<N>();

            auto free_memory_result = get_free_memory();
            FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, free_memory_result, "GPU sample generation failed: ");

            auto max_threads_result = get_max_threads();
            FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, max_threads_result, "GPU sample generation failed: ");

            const auto free_memory = free_memory_result.value();
            const auto max_threads = max_threads_result.value();
            if (max_threads == 0) {
                thread_state.set(Result<void>{Err{"GPU sample generation failed: max thread count is zero."}});
                return;
            }

            const auto sample_stack_memory = usize{128} + sizeof(real) * N;
            const auto sample_global_memory = sizeof(Bitstring<N>) + sizeof(real) * (PARTIAL_MARGINAL_COUNT + ELIDED_MARGINAL_COUNT + PARTIAL_ELIDED_MARGINAL_COUNT + DOUBLE_ELIDED_MARGINAL_COUNT);
            if (sample_global_memory == 0) {
                thread_state.set(Result<void>{Err{"GPU sample generation failed: sample global memory footprint is zero."}});
                return;
            }

            const auto free_memory_fp = 0.95 * static_cast<f64>(free_memory);
            const auto max_threads_fp = static_cast<f64>(max_threads);
            const auto sample_stack_memory_fp = static_cast<f64>(sample_stack_memory);
            const auto sample_global_memory_fp = static_cast<f64>(sample_global_memory);

            const auto sub_chunk_max_size_fp = (free_memory_fp - sample_stack_memory_fp * max_threads_fp) / sample_global_memory_fp;
            if (!std::isfinite(sub_chunk_max_size_fp) || sub_chunk_max_size_fp < 1.0) {
                thread_state.set(Result<void>{Err{"GPU sample generation failed: invalid sub-chunk capacity from memory budget."}});
                return;
            }

            const auto sub_chunk_count_fp = std::ceil(static_cast<f64>(sample_chunk_size) / sub_chunk_max_size_fp);
            if (!std::isfinite(sub_chunk_count_fp) || sub_chunk_count_fp < 1.0) {
                thread_state.set(Result<void>{Err{"GPU sample generation failed: invalid sub-chunk count."}});
                return;
            }

            const auto sub_chunk_count = static_cast<usize>(sub_chunk_count_fp);
            std::vector<usize> sub_chunk_indices = get_chunk_indices(sample_chunk_size, sub_chunk_count);
            if (sub_chunk_indices.size() < 2 || sub_chunk_indices[1] == 0) {
                thread_state.set(Result<void>{Err{"GPU sample generation failed: computed sub-chunk indices are invalid."}});
                return;
            }
            const auto sub_chunk_max_size = sub_chunk_indices[1];

            DeviceData<Bitstring<N>> d_samples(sub_chunk_max_size);
            DeviceData<curandState> d_rand_states(sub_chunk_max_size);
            DeviceData<real> d_partial_marginal_cache(PARTIAL_MARGINAL_COUNT * sub_chunk_max_size);
            DeviceData<real> d_elided_marginal_cache(ELIDED_MARGINAL_COUNT * sub_chunk_max_size);
            DeviceData<real> d_partial_elided_marginal_cache(PARTIAL_ELIDED_MARGINAL_COUNT * sub_chunk_max_size);
            DeviceData<real> d_double_elided_marginal_cache(DOUBLE_ELIDED_MARGINAL_COUNT * sub_chunk_max_size);
            FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, validate_device_data(d_samples, d_rand_states, d_partial_marginal_cache, d_elided_marginal_cache, d_partial_elided_marginal_cache, d_double_elided_marginal_cache), "GPU sample generation failed: ");

            FGBS_THREAD_VALIDATE_OR_RETURN_CTX(thread_state, FGBS_INTERNAL_RUN_KERNEL(init_rand_states, 256, sub_chunk_max_size, d_rand_states.ptr(), sub_chunk_max_size, seeds[i], static_cast<ull>(sample_chunk_begin)), "GPU sample generation failed: ");

            cudaStream_t stream{};
            bool has_stream = false;
            bool host_registered = false;

            FGBS_THREAD_VALIDATE_OR_RETURN(thread_state, cuda_result(cudaStreamCreate(&stream), "GPU sample generation failed at cudaStreamCreate"));
            has_stream = true;

            auto* samples_chunk = &samples[sample_chunk_begin];
            FGBS_THREAD_VALIDATE_OR_RETURN(thread_state, cuda_result(cudaHostRegister(samples_chunk, sizeof(Bitstring<N>) * sample_chunk_size, cudaHostRegisterDefault), "GPU sample generation failed at cudaHostRegister"));
            host_registered = true;

            for (usize j = 0; j < sub_chunk_count && thread_state.ok(); ++j) {
                const auto sub_chunk_begin = sub_chunk_indices[j];
                const auto sub_chunk_size = sub_chunk_indices[j + 1] - sub_chunk_begin;

                FGBS_THREAD_VALIDATE_OR_BREAK(thread_state, cuda_result(cudaMemset(d_samples.ptr(), 0, sizeof(Bitstring<N>) * sub_chunk_size), "GPU sample generation failed at cudaMemset"));

                FGBS_THREAD_VALIDATE_OR_BREAK_CTX(thread_state, FGBS_INTERNAL_RUN_KERNEL_WITH_STREAM(generate_samples<N>, 128, sub_chunk_size, stream, d_samples.ptr(), sub_chunk_size, d_rand_states.ptr(), d_sum_indices.ptr(), d_correlator_cumulants.ptr(), d_first_correlator_cumulants.ptr(), d_index_arrays.ptr(), cutoff_order, d_partial_marginal_cache.ptr(), d_elided_marginal_cache.ptr(), d_partial_elided_marginal_cache.ptr(), d_double_elided_marginal_cache.ptr()), "GPU sample generation failed: ");

                FGBS_THREAD_VALIDATE_OR_BREAK(thread_state, cuda_result(cudaMemcpyAsync(&samples[sample_chunk_begin + sub_chunk_begin], d_samples.ptr(), sizeof(Bitstring<N>) * sub_chunk_size, cudaMemcpyDeviceToHost, stream), "GPU sample generation failed at cudaMemcpyAsync(DeviceToHost)"));
            }

            if (has_stream) {
                thread_state.set(cuda_result(cudaStreamSynchronize(stream), "GPU sample generation failed at cudaStreamSynchronize"));
            }

            if (host_registered) {
                thread_state.set(cuda_result(cudaHostUnregister(samples_chunk), "GPU sample generation failed at cudaHostUnregister"));
            }

            if (has_stream) {
                thread_state.set(cuda_result(cudaStreamDestroy(stream), "GPU sample generation failed at cudaStreamDestroy"));
            }
        }));
    }

    join_threads(threads);

    if (!thread_state.ok()) {
        return Err{thread_state.get().error()};
    }

    return Ok<std::vector<Bitstring<N>>>{std::move(samples)};
}

}

}

#undef FGBS_INTERNAL_CURAND_UNIFORM_REAL
#undef FGBS_INTERNAL_DEVICE_COMPUTE_DOUBLE_ELIDED_MARGINAL_SECOND
#undef FGBS_INTERNAL_DEVICE_COMPUTE_ELIDED_MARGINAL_SECOND
#undef FGBS_INTERNAL_DEVICE_COMPUTE_ELIDED_MARGINAL_THIRD
#undef FGBS_INTERNAL_DEVICE_COMPUTE_PARTIAL_ELIDED_MARGINAL_SECOND
#undef FGBS_INTERNAL_DEVICE_MARGINAL_END_ZERO_DIFF

#endif
