// Generated instrumentation, NOT an official FrugalGBS API.
// Upstream commit: fa8b796f2afebd4845a2e335e457735952617ef8
// Source function SHA256: 9c14a870f2875e808025d9e3535a71ba86144e5639ab14f0b7be5b594dabb3fb
/*
# The MIT License (MIT)

Copyright (c) 2026 "tomdodd4598"

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#ifndef GBSLAB_FRUGAL_LOG_TRACE_HPP
#define GBSLAB_FRUGAL_LOG_TRACE_HPP
#include "fgbs/cpu/cpu_sampler.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>
#if defined(_MSC_VER)
#define GBSLAB_RESTRICT __restrict
#elif defined(__clang__) || defined(__GNUC__)
#define GBSLAB_RESTRICT __restrict__
#else
#define GBSLAB_RESTRICT
#endif
namespace fgbs::cpu {
struct TraceLogPathResult {
    long double probability = 1.0L;
    long double log_probability = 0.0L;
    std::uint64_t saturated_threshold_path_steps = 0;
    std::uint64_t nonfinite_threshold_path_steps = 0;
};

// The official decision is ONE iff U > threshold. Its continuous-uniform
// zero-branch mass is the threshold clipped to [0,1], not proposed_zero itself.
// NaN compares false and therefore selects ZERO; it is still an invalid trace
// for evidence and the adapter reports it explicitly rather than hiding it.
inline bool record_forced_threshold(TraceLogPathResult& trace, double threshold, bool one) {
    if (!std::isfinite(threshold)) ++trace.nonfinite_threshold_path_steps;
    if (threshold < 0.0 || threshold > 1.0) ++trace.saturated_threshold_path_steps;
    const double p_zero = std::isnan(threshold) ? 1.0
        : std::max(0.0, std::min(1.0, threshold));
    const long double chosen = one ? 1.0L - p_zero : p_zero;
    trace.probability *= chosen;
    if (chosen == 0.0L) {
        trace.log_probability = -std::numeric_limits<long double>::infinity();
        return false;
    }
    trace.log_probability += one ? std::log1p(-static_cast<long double>(p_zero))
        : std::log(static_cast<long double>(p_zero));
    return true; // A product underflow does not terminate the log-density trace.
}

template<usize N>
TraceLogPathResult trace_sample_log_probability(
    BitstringLayout<N> const& bitstring_layout,
    SamplingTarget<N> const& sampling_target,
    Bitstring<N> const& forced_bits,
    ProbSimd* GBSLAB_RESTRICT partial_marginal_arr,
    ProbSimd* GBSLAB_RESTRICT elided_marginal_arr,
    ProbSimd* GBSLAB_RESTRICT partial_elided_marginal_arr,
    ProbSimd* GBSLAB_RESTRICT double_elided_marginal_arr) {
    using T = ProbSimd;
    TraceLogPathResult trace{};
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
        // All SIMD lanes follow the same forced physical-bit prefix.
        // This changes branch choice only; upstream marginal arithmetic above
        // and the state updates below retain their original SIMD precision.
        const auto previous = prev_prob.get(0);
        const auto proposed_zero = zero_next_prob.get(0);
        if (!(previous > 0.0) || !std::isfinite(previous)) {
            throw std::runtime_error("Nonpositive/nonfinite reachable prefix mass");
        }
        const auto threshold = proposed_zero / previous;
        const bool one = get_bit<N>(forced_bits, i);
        if (!record_forced_threshold(trace, threshold, one)) return trace;
        parity_arr[i] = one ? -get_half<T>() : get_half<T>();
        prev_prob = one ? prev_prob - zero_next_prob : zero_next_prob;
    }

    return trace;
}

} // namespace fgbs::cpu
#undef GBSLAB_RESTRICT
#endif
