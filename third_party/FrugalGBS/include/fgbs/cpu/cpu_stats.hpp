#ifndef FGBS_CPU_STATS_HPP
#define FGBS_CPU_STATS_HPP

#include "fgbs/core/bitstring_utils.hpp"
#include "fgbs/core/structs.hpp"
#include "fgbs/core/result.hpp"

#include <vector>

namespace fgbs {

namespace cpu {

template<usize N>
Result<std::vector<real>> estimate_sample_correlator_sublists(SublistLayout<N> const& sublist_layout, std::vector<Bitstring<N>> const& samples) {
    const auto order_count = sublist_layout.cutoff_order + 1;
    const auto sample_count = samples.size();
    std::vector<real> sample_correlator_sublists(sublist_layout.sublist_offset_vec[order_count], real{0.0});

    if (sample_count == 0) {
        return Ok<std::vector<real>>{std::move(sample_correlator_sublists)};
    }

    const auto sample_count_inv = real{1.0} / static_cast<real>(sample_count);

    for (usize order = 0; order < order_count; ++order) {
        const auto correlator_bitstrings = sublist_layout.get_sublist(sublist_layout.layout_bitstring_sublists, order);
        const auto sublist_size = correlator_bitstrings.size();
        auto correlators = sublist_layout.get_sublist_mut(sample_correlator_sublists, order);

        #pragma omp parallel for default(none) shared(correlators, correlator_bitstrings, samples) firstprivate(sublist_size, sample_count, sample_count_inv)
        for (usize i = 0; i < sublist_size; ++i) {
            i64 parity_sum = 0;
            for (usize j = 0; j < sample_count; ++j) {
                parity_sum += static_cast<i64>(characteristic<N>(correlator_bitstrings[i], samples[j]));
            }
            correlators[i] = static_cast<real>(parity_sum) * sample_count_inv;
        }
    }

    return Ok<std::vector<real>>{std::move(sample_correlator_sublists)};
}

}

}

#endif
