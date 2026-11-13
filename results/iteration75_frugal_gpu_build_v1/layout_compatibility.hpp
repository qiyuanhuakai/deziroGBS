#ifndef FGBS_LAYOUT_HPP
#define FGBS_LAYOUT_HPP

#include "fgbs/core/bitstring_utils.hpp"
#include "fgbs/core/consts.hpp"
#include "fgbs/core/structs.hpp"
#include "fgbs/core/validation.hpp"
#include "fgbs/core/iter_utils.hpp"
#include "fgbs/core/math_utils.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/string_utils.hpp"
#include "fgbs/core/type_utils.hpp"

#include <omp.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace fgbs {

usize get_parallel_fork_depth(int thread_count);

template<usize N, typename BITSTRING>
void generate_bitstring_info(BitstringLayout<N>& bitstring_layout, usize parallel_fork_depth, BITSTRING bitstring, usize order = 0, isize bitstring_index = N - 1, usize layout_index = 0) {
    if (bitstring_index < 0) {
        bitstring_layout.layout_bitstring_vec[layout_index] = bitstring;
        bitstring_layout.order_vec[layout_index] = get_weight<N>(bitstring);
        const auto ones = get_ones<N>(bitstring);
        std::copy(ones.cbegin(), ones.cend(), bitstring_layout.get_ones_slice_mut(layout_index).data());
        return;
    }

    const auto fork = N - bitstring_index - 1 < parallel_fork_depth;

    // Host compatibility: recurse serially; OpenMP correlator loop is retained.
    generate_bitstring_info<N>(bitstring_layout, parallel_fork_depth, bitstring, order, bitstring_index - 1, layout_index);

    if (order < bitstring_layout.cutoff_order) {
        // Host compatibility: recurse serially; OpenMP correlator loop is retained.
        {
            flip_bit<N>(bitstring, bitstring_index);
            generate_bitstring_info<N>(bitstring_layout, parallel_fork_depth, std::move(bitstring), order + 1, bitstring_index - 1, layout_index + gbs_index_arrays<N>[bitstring_layout.cutoff_order - order][bitstring_index]);
        }
    }
}

template<usize N>
Result<BitstringLayout<N>> generate_bitstring_layout(usize cutoff_order) {
    FGBS_VALIDATE_OR_RETURN(validate_cutoff_order<N>(cutoff_order));

    BitstringLayout<N> bitstring_layout(cutoff_order);
    bitstring_layout.layout_bitstring_vec.resize(bitstring_layout.len);
    bitstring_layout.order_vec.resize(bitstring_layout.len);
    bitstring_layout.ones_vec.resize(bitstring_layout.cutoff_order * bitstring_layout.len);

    const auto parallel_fork_depth = get_parallel_fork_depth(::omp_get_max_threads());
    #pragma omp parallel default(none) firstprivate(parallel_fork_depth) shared(bitstring_layout)
    {
        #pragma omp single
        generate_bitstring_info<N>(bitstring_layout, parallel_fork_depth, Bitstring<N>{});
    }

    return Ok<BitstringLayout<N>>{std::move(bitstring_layout)};
}

template<usize N>
Result<CumulantLayout<N>> generate_cumulant_layout(usize cutoff_order) {
    FGBS_VALIDATE_OR_RETURN(validate_cutoff_order<N>(cutoff_order));

    CumulantLayout<N> cumulant_layout(cutoff_order);
    cumulant_layout.cumulant_term_range_vec = {0};
    cumulant_layout.cumulant_order_range_vec = {0};
    cumulant_layout.cumulant_index_range_vec = {0};

    for (usize i = 0; i <= cumulant_layout.cutoff_order; ++i) {
        std::vector<usize> range;
        for (usize j = 0; j < i; ++j) {
            range.push_back(j);
        }

        for_each_partition(range.cbegin(), range.cend(), 0, [&](auto&& p) {
            if (p.empty()) {
                return;
            }

            cumulant_layout.cumulant_term_vec.push_back(p.size());
            for (auto const& v : p) {
                cumulant_layout.cumulant_order_vec.push_back(v.size());
                cumulant_layout.cumulant_index_vec.insert(cumulant_layout.cumulant_index_vec.end(), v.cbegin(), v.cend());
            }
        });

        cumulant_layout.cumulant_term_range_vec.push_back(cumulant_layout.cumulant_term_vec.size());
        cumulant_layout.cumulant_order_range_vec.push_back(cumulant_layout.cumulant_order_vec.size());
        cumulant_layout.cumulant_index_range_vec.push_back(cumulant_layout.cumulant_index_vec.size());
    }

    return Ok<CumulantLayout<N>>{std::move(cumulant_layout)};
}

template<usize N>
Result<SublistLayout<N>> generate_sublist_layout(BitstringLayout<N> const& bitstring_layout, usize last_stride) {
    FGBS_VALIDATE_OR_RETURN(validate_bitstring_layout(bitstring_layout));

    SublistLayout<N> sublist_layout(bitstring_layout.cutoff_order);
    const auto order_count = bitstring_layout.cutoff_order + 1;

    if (last_stride == 0) {
        return Err{"last_stride must be positive."};
    }

    sublist_layout.sublist_index_vec.resize(bitstring_layout.len, INVALID_INDEX);
    sublist_layout.sublist_size_vec.resize(order_count);
    sublist_layout.sublist_offset_vec.resize(order_count + 1);

    const auto thread_count = static_cast<usize>(::omp_get_max_threads());
    std::vector<std::vector<usize>> visit_counts(thread_count, std::vector<usize>(order_count, 0));

    #pragma omp parallel default(none) shared(visit_counts, bitstring_layout) firstprivate(thread_count)
    {
        const auto thread_index = static_cast<usize>(::omp_get_thread_num());
        auto& thread_visit_counts = visit_counts[thread_index];
        const auto range_begin = (bitstring_layout.len * thread_index) / thread_count;
        const auto range_end = (bitstring_layout.len * (thread_index + 1)) / thread_count;

        for (usize i = range_begin; i < range_end; ++i) {
            ++thread_visit_counts[static_cast<usize>(bitstring_layout.order_vec[i])];
        }
    }

    std::vector<std::vector<usize>> visit_offsets(thread_count, std::vector<usize>(order_count, 0));

    usize sublist_offset = 0;
    for (usize i = 0; i < order_count; ++i) {
        usize visit_count = 0;
        for (usize j = 0; j < thread_count; ++j) {
            visit_offsets[j][i] = visit_count;
            visit_count += visit_counts[j][i];
        }

        const auto stride = i == sublist_layout.cutoff_order ? last_stride : usize{1};
        const auto count = visit_count == 0 ? usize{0} : 1 + (visit_count - 1) / stride;
        sublist_layout.sublist_size_vec[i] = count;
        sublist_layout.sublist_offset_vec[i] = sublist_offset;
        sublist_offset += count;
    }

    sublist_layout.sublist_offset_vec[order_count] = sublist_offset;
    sublist_layout.layout_bitstring_sublists.resize(sublist_offset);
    sublist_layout.ones_sublists.resize(bitstring_layout.cutoff_order * sublist_offset);

    #pragma omp parallel default(none) shared(sublist_layout, bitstring_layout, visit_offsets) firstprivate(last_stride, thread_count)
    {
        const auto thread_index = static_cast<usize>(::omp_get_thread_num());
        auto visit_indices = visit_offsets[thread_index];
        const auto range_begin = (bitstring_layout.len * thread_index) / thread_count;
        const auto range_end = (bitstring_layout.len * (thread_index + 1)) / thread_count;

        for (usize i = range_begin; i < range_end; ++i) {
            const auto order = static_cast<usize>(bitstring_layout.order_vec[i]);
            const auto stride = order == sublist_layout.cutoff_order ? last_stride : usize{1};
            const auto order_visit_index = visit_indices[order]++;
            if (order_visit_index % stride == 0) {
                sublist_layout.sublist_index_vec[i] = order_visit_index / stride;
            }
        }
    }

    #pragma omp parallel for default(none) shared(sublist_layout, bitstring_layout)
    for (usize i = 0; i < bitstring_layout.len; ++i) {
        const auto sublist_index = sublist_layout.sublist_index_vec[i];
        if (sublist_index != INVALID_INDEX) {
            const auto order = bitstring_layout.order_vec[i];
            const auto ones = bitstring_layout.get_ones_slice(i);

            sublist_layout.get_sublist_mut(sublist_layout.layout_bitstring_sublists, order)[sublist_index] = bitstring_layout.layout_bitstring_vec[i];
            std::copy(ones.data(), ones.data() + bitstring_layout.cutoff_order, sublist_layout.get_ones_sublist_mut(order, sublist_index).data());
        }
    }

    return Ok<SublistLayout<N>>{std::move(sublist_layout)};
}

}

#endif
