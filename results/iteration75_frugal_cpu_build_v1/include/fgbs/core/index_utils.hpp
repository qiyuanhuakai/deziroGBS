#ifndef FGBS_CORE_INDEX_UTILS_HPP
#define FGBS_CORE_INDEX_UTILS_HPP

#include "fgbs/core/consts.hpp"
#include "fgbs/core/type_utils.hpp"

namespace fgbs {

template<usize N, typename VECTOR>
usize ones_index(VECTOR&& ones, usize cutoff_order) {
    usize index = 0;
    const auto order = ones.size();
    for (usize i = 0; i < order; ++i) {
        index += gbs_index_arrays<N>[cutoff_order - i][ones[order - i - 1]];
    }
    return index;
}

}

#endif
