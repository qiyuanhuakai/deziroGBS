#ifndef FGBS_CORE_SUBLIST_UTILS_HPP
#define FGBS_CORE_SUBLIST_UTILS_HPP

#include "fgbs/core/structs.hpp"

#include <vector>

namespace fgbs {

// FUNCTION = T(T, usize)
template<usize N, typename T, typename FUNCTION>
std::vector<T> to_sublists(BitstringLayout<N> const& bitstring_layout, SublistLayout<N> const& sublist_layout, std::vector<T> const& values, FUNCTION&& f) {
    const auto order_count = bitstring_layout.cutoff_order + 1;
    std::vector<T> sublists(sublist_layout.sublist_offset_vec[order_count]);

    #pragma omp parallel for default(none) shared(sublists, values, bitstring_layout, sublist_layout, f)
    for (usize i = 0; i < bitstring_layout.len; ++i) {
        const auto sublist_index = sublist_layout.sublist_index_vec[i];
        if (sublist_index != INVALID_INDEX) {
            const auto order = bitstring_layout.order_vec[i];
            sublist_layout.get_sublist_mut(sublists, order)[sublist_index] = f(values[i], order);
        }
    }

    return sublists;
}

// FUNCTION = T(T, usize)
template<usize N, typename T, typename FUNCTION>
std::vector<T> from_sublists(BitstringLayout<N> const& bitstring_layout, SublistLayout<N> const& sublist_layout, std::vector<T> const& sublists, FUNCTION&& f) {
    std::vector<T> values(bitstring_layout.len);

    #pragma omp parallel for default(none) shared(sublists, values, bitstring_layout, sublist_layout, f)
    for (usize i = 0; i < bitstring_layout.len; ++i) {
        const auto sublist_index = sublist_layout.sublist_index_vec[i];
        if (sublist_index != INVALID_INDEX) {
            const auto order = bitstring_layout.order_vec[i];
            values[i] = f(sublist_layout.get_sublist(sublists, order)[sublist_index], order);
        }
    }

    return values;
}

}

#endif
