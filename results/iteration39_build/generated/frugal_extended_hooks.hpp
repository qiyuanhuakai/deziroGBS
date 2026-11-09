// Generated audit hooks; NOT an official FrugalGBS API.
/*
# The MIT License (MIT)

Copyright (c) 2026 "tomdodd4598"

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#pragma once
#include "fgbs/cpu/cpu_sampler.hpp"
#include <string>
#include <stdexcept>
#define GBSLAB39_RESTRICT __restrict__
namespace fgbs::cpu {
struct AuditRecord { double uniform, threshold, previous, proposed; };
struct PrefixRecord { double threshold, previous, proposed; };
template<usize N, typename T>
void update_result_audited(UniformRandom<T>& rand, SampleResult<N, T>& result, T& prev_prob, T* GBSLAB39_RESTRICT parity_arr, usize i, T zero_next_prob, AuditRecord* records) {
    const auto zero_prob = zero_next_prob / prev_prob;

    if constexpr (std::is_same_v<T, ProbSimd>) {
        const auto random_value = rand();
        const auto mask = random_value > zero_prob;
        for (usize j = 0; j < ProbSimd::size; ++j)
            records[j*N+i] = {random_value.get(j), zero_prob.get(j), prev_prob.get(j), zero_next_prob.get(j)};
        for (usize j = 0; j < ProbSimd::size; ++j) {
            if (mask.get(j)) {
                flip_bit<N>(result[j], i);
            }
        }
        parity_arr[i] = xsimd::select(mask, -get_half<ProbSimd>(), get_half<ProbSimd>());
        prev_prob = xsimd::select(mask, prev_prob - zero_next_prob, zero_next_prob);
    }
    else {
        if (rand() > zero_prob) {
            flip_bit<N>(result, i);
            parity_arr[i] = -get_half<Prob>();
            prev_prob -= zero_next_prob;
        }
        else {
            parity_arr[i] = get_half<Prob>();
            prev_prob = zero_next_prob;
        }
    }
}
template<usize N, typename T>
SampleResult<N, T> generate_sample_audited(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, UniformRandom<T>& rand, T* GBSLAB39_RESTRICT partial_marginal_arr, T* GBSLAB39_RESTRICT elided_marginal_arr, T* GBSLAB39_RESTRICT partial_elided_marginal_arr, T* GBSLAB39_RESTRICT double_elided_marginal_arr, AuditRecord* records) {
    SampleResult<N, T> result{};
    T prev_prob = get_one<T>();
    T parity_arr[N];

    const auto partial_order = partial_marginal_order(bitstring_layout);
    const auto elided_order = elided_marginal_order(bitstring_layout);
    const auto partial_elided_order = partial_elided_marginal_order(bitstring_layout);
    const auto double_elided_order = double_elided_marginal_order(bitstring_layout);

    for (usize i = 0; i < N; ++i) {
        ProbSum<T> zero_diff{};
        auto correlator_cumulant_ptr = &sampling_target.correlator_cumulant_vec[bitstring_layout.sum_index_arr[i]];

        if (bitstring_layout.cutoff_order <= 1) {
            goto end_zero_diff;
        }

        for (usize j = 0; j < i; ++j) {
            if (j == 0) {
                set_partial_marginal<N>(partial_marginal_arr, j, i, prev_prob);
            }
            else {
                compute_partial_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, partial_elided_marginal_arr, parity_arr, j, i, partial_order);
            }
        }

        for (usize j = 0; j < i; ++j) {
            if (j == 0) {
                set_elided_marginal<N>(elided_marginal_arr, j, i, get_partial_marginal<N>(partial_marginal_arr, 1, i));
            }
            else if (j == i - 1) {
                set_elided_marginal<N>(elided_marginal_arr, j, i, get_partial_marginal<N>(partial_marginal_arr, 0, i - 1));
            }
            else {
                compute_elided_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, elided_marginal_arr, double_elided_marginal_arr, parity_arr, j, i, elided_order);
            }
        }

        for (usize j = 0; j < i; ++j) {
            for (usize k = 0; k < j; ++k) {
                if (k == 0) {
                    set_partial_elided_marginal<N>(partial_elided_marginal_arr, k, j, i, get_elided_marginal<N>(elided_marginal_arr, j, i));
                }
                else if (j == i - 1) {
                    set_partial_elided_marginal<N>(partial_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, k, i - 1));
                }
                else {
                    compute_partial_elided_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, partial_elided_marginal_arr, parity_arr, k, j, i, partial_elided_order);
                }
            }
        }

        for (usize j = 0; j < i; ++j) {
            for (usize k = 0; k < j; ++k) {
                if (k == 0 && j == 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, 2, i));
                }
                else if (k == 0 && j == i - 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, 1, i - 1));
                }
                else if (k == i - 2 && j == i - 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, 0, i - 2));
                }
                else if (j == i - 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_elided_marginal<N>(elided_marginal_arr, k, i - 1));
                }
                else {
                    compute_double_elided_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, double_elided_marginal_arr, parity_arr, k, j, i, double_elided_order);
                }
            }
        }

        for (usize j = 0; j < i; ++j) {
            auto second_parity = get_half<Prob>() * parity_arr[j];
            zero_diff += second_parity * (*(++correlator_cumulant_ptr)) * get_elided_marginal<N>(elided_marginal_arr, j, i);

            if (bitstring_layout.cutoff_order <= 2) {
                continue;
            }

            auto second_marginal = get_partial_marginal<N>(partial_marginal_arr, j + 1, i);
            for (usize k = 0; k < j; ++k) {
                auto third_parity = second_parity * parity_arr[k];
                zero_diff += third_parity * (*(++correlator_cumulant_ptr)) * get_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i);

                if (bitstring_layout.cutoff_order <= 3) {
                    continue;
                }

                auto third_marginal = get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, k + 1, j, i);
                for (usize l = 0; l < k; ++l) {
                    auto fourth_parity = third_parity * parity_arr[l];
                    zero_diff += fourth_parity * (*(++correlator_cumulant_ptr)) * get_double_elided_marginal<N>(double_elided_marginal_arr, l, k, j) * second_marginal;

                    if (bitstring_layout.cutoff_order <= 4) {
                        continue;
                    }

                    for (usize m = 0; m < l; ++m) {
                        auto fifth_parity = fourth_parity * parity_arr[m];
                        zero_diff += fifth_parity * (*(++correlator_cumulant_ptr)) * get_double_elided_marginal<N>(double_elided_marginal_arr, m, l, k) * third_marginal;
                    }
                }
            }
        }

    end_zero_diff : {}

        const auto zero_next_prob = clamped_prob(prev_prob * (real{0.5} + real{0.5} * sampling_target.first_correlator_cumulant_vec[i]) + zero_diff.get());
        update_result_audited<N>(rand, result, prev_prob, parity_arr, i, zero_next_prob, records);
    }

    return result;
}

template<usize N>
Result<std::vector<Bitstring<N>>> generate_samples_audited(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, usize sample_count, uint seed, std::vector<AuditRecord>& records) {
    if (sample_count == 0) {
        return Ok<std::vector<Bitstring<N>>>{std::vector<Bitstring<N>>{}};
    }

    std::vector<Bitstring<N>> samples(sample_count);
    UniformRandomInt<usize> seed_generator{seed, 0, ULLONG_MAX};
    const auto seeds = get_random_vector<uint>(seed_generator, ::omp_get_max_threads());

    const auto batch_size = xsimd::batch<Prob>::size;
    const auto tail_size = sample_count % batch_size;
    const auto batch_count = (sample_count - 1 + batch_size) / batch_size;

    records.resize(batch_count * batch_size * N);
    #pragma omp parallel default(none) shared(samples, seeds, bitstring_layout, sampling_target, records) firstprivate(batch_size, tail_size, batch_count)
    {
        UniformRandom<ProbSimd> rand(seeds[::omp_get_thread_num()], 0.0, 1.0);

        std::vector<ProbSimd> partial_marginal_arr(partial_marginal_arr_size<N>());
        std::vector<ProbSimd> elided_marginal_arr(elided_marginal_arr_size<N>());
        std::vector<ProbSimd> partial_elided_marginal_arr(partial_elided_marginal_arr_size<N>());
        std::vector<ProbSimd> double_elided_marginal_arr(double_elided_marginal_arr_size<N>());

        #pragma omp for
        for (usize i = 0; i < batch_count; ++i) {
            const auto iter_length = i == batch_count - 1 && tail_size > 0 ? tail_size : batch_size;
            const auto bitstring_batch = &samples[i * batch_size];
            const auto sample_batch = generate_sample_audited<N, ProbSimd>(bitstring_layout, sampling_target, rand, partial_marginal_arr.data(), elided_marginal_arr.data(), partial_elided_marginal_arr.data(), double_elided_marginal_arr.data(), records.data() + i*batch_size*N);
            for (usize j = 0; j < iter_length; ++j) {
                bitstring_batch[j] = sample_batch[j];
            }
        }
    }

    return Ok<std::vector<Bitstring<N>>>{std::move(samples)};
}
template<usize N>
PrefixRecord trace_prefix(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target,
    std::string const& forced_prefix, ProbSimd* partial_marginal_arr, ProbSimd* elided_marginal_arr,
    ProbSimd* partial_elided_marginal_arr, ProbSimd* double_elided_marginal_arr) {
    using T = ProbSimd;
    T prev_prob = get_one<T>();
    T parity_arr[N];

    const auto partial_order = partial_marginal_order(bitstring_layout);
    const auto elided_order = elided_marginal_order(bitstring_layout);
    const auto partial_elided_order = partial_elided_marginal_order(bitstring_layout);
    const auto double_elided_order = double_elided_marginal_order(bitstring_layout);

    for (usize i = 0; i < N; ++i) {
        ProbSum<T> zero_diff{};
        auto correlator_cumulant_ptr = &sampling_target.correlator_cumulant_vec[bitstring_layout.sum_index_arr[i]];

        if (bitstring_layout.cutoff_order <= 1) {
            goto end_zero_diff;
        }

        for (usize j = 0; j < i; ++j) {
            if (j == 0) {
                set_partial_marginal<N>(partial_marginal_arr, j, i, prev_prob);
            }
            else {
                compute_partial_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, partial_elided_marginal_arr, parity_arr, j, i, partial_order);
            }
        }

        for (usize j = 0; j < i; ++j) {
            if (j == 0) {
                set_elided_marginal<N>(elided_marginal_arr, j, i, get_partial_marginal<N>(partial_marginal_arr, 1, i));
            }
            else if (j == i - 1) {
                set_elided_marginal<N>(elided_marginal_arr, j, i, get_partial_marginal<N>(partial_marginal_arr, 0, i - 1));
            }
            else {
                compute_elided_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, elided_marginal_arr, double_elided_marginal_arr, parity_arr, j, i, elided_order);
            }
        }

        for (usize j = 0; j < i; ++j) {
            for (usize k = 0; k < j; ++k) {
                if (k == 0) {
                    set_partial_elided_marginal<N>(partial_elided_marginal_arr, k, j, i, get_elided_marginal<N>(elided_marginal_arr, j, i));
                }
                else if (j == i - 1) {
                    set_partial_elided_marginal<N>(partial_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, k, i - 1));
                }
                else {
                    compute_partial_elided_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, partial_elided_marginal_arr, parity_arr, k, j, i, partial_elided_order);
                }
            }
        }

        for (usize j = 0; j < i; ++j) {
            for (usize k = 0; k < j; ++k) {
                if (k == 0 && j == 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, 2, i));
                }
                else if (k == 0 && j == i - 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, 1, i - 1));
                }
                else if (k == i - 2 && j == i - 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, 0, i - 2));
                }
                else if (j == i - 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_elided_marginal<N>(elided_marginal_arr, k, i - 1));
                }
                else {
                    compute_double_elided_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, double_elided_marginal_arr, parity_arr, k, j, i, double_elided_order);
                }
            }
        }

        for (usize j = 0; j < i; ++j) {
            auto second_parity = get_half<Prob>() * parity_arr[j];
            zero_diff += second_parity * (*(++correlator_cumulant_ptr)) * get_elided_marginal<N>(elided_marginal_arr, j, i);

            if (bitstring_layout.cutoff_order <= 2) {
                continue;
            }

            auto second_marginal = get_partial_marginal<N>(partial_marginal_arr, j + 1, i);
            for (usize k = 0; k < j; ++k) {
                auto third_parity = second_parity * parity_arr[k];
                zero_diff += third_parity * (*(++correlator_cumulant_ptr)) * get_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i);

                if (bitstring_layout.cutoff_order <= 3) {
                    continue;
                }

                auto third_marginal = get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, k + 1, j, i);
                for (usize l = 0; l < k; ++l) {
                    auto fourth_parity = third_parity * parity_arr[l];
                    zero_diff += fourth_parity * (*(++correlator_cumulant_ptr)) * get_double_elided_marginal<N>(double_elided_marginal_arr, l, k, j) * second_marginal;

                    if (bitstring_layout.cutoff_order <= 4) {
                        continue;
                    }

                    for (usize m = 0; m < l; ++m) {
                        auto fifth_parity = fourth_parity * parity_arr[m];
                        zero_diff += fifth_parity * (*(++correlator_cumulant_ptr)) * get_double_elided_marginal<N>(double_elided_marginal_arr, m, l, k) * third_marginal;
                    }
                }
            }
        }

    end_zero_diff : {}

        const auto zero_next_prob = clamped_prob(prev_prob * (real{0.5} + real{0.5} * sampling_target.first_correlator_cumulant_vec[i]) + zero_diff.get());
        const auto threshold = zero_next_prob / prev_prob;
        if (i == forced_prefix.size()) return {threshold.get(0), prev_prob.get(0), zero_next_prob.get(0)};
        const auto mask = forced_prefix[i] == '1' ? get_one<T>() > get_zero<T>() : get_one<T>() < get_zero<T>();
        parity_arr[i] = xsimd::select(mask, -get_half<T>(), get_half<T>());
        prev_prob = xsimd::select(mask, prev_prob - zero_next_prob, zero_next_prob);
    }

    throw std::runtime_error("Prefix must be shorter than the mode count");
}

}
#undef GBSLAB39_RESTRICT
