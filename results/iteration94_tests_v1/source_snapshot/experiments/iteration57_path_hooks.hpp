#pragma once
#include "frugal_extended_hooks.hpp"
namespace fgbs::cpu {
template<usize N>
std::vector<PrefixRecord> trace_path(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target,
    std::string const& forced_prefix, ProbSimd* partial_marginal_arr, ProbSimd* elided_marginal_arr,
    ProbSimd* partial_elided_marginal_arr, ProbSimd* double_elided_marginal_arr) {
    using T = ProbSimd;
    if (forced_prefix.size() != N || forced_prefix.find_first_not_of("01") != std::string::npos) throw std::runtime_error("Complete binary path required");
    std::vector<PrefixRecord> records;
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
        records.push_back({threshold.get(0), prev_prob.get(0), zero_next_prob.get(0)});
        const auto mask = forced_prefix[i] == '1' ? get_one<T>() > get_zero<T>() : get_one<T>() < get_zero<T>();
        parity_arr[i] = xsimd::select(mask, -get_half<T>(), get_half<T>());
        prev_prob = xsimd::select(mask, prev_prob - zero_next_prob, zero_next_prob);
    }

    return records;
}

}
