#ifndef FGBS_CORE_CONSTS_HPP
#define FGBS_CORE_CONSTS_HPP

#include "fgbs/core/type_utils.hpp"

#include <array>

namespace fgbs {

constexpr usize INVALID_INDEX = static_cast<usize>(-1);

template<usize N>
constexpr std::array<std::array<usize, N + 1>, N + 1> get_index_arrays() {
    std::array<std::array<usize, N + 1>, N + 1> arr{};
    for (usize i = 0; i <= N; ++i) {
        arr[0][i] = 1;
        usize binom = 1;
        for (usize j = 1; j <= N; ++j) {
            if (j <= i) {
                binom = (binom * (i - (j - 1))) / j;
            }
            else {
                binom = 0;
            }
            arr[j][i] = arr[j - 1][i] + binom;
        }
    }

    return arr;
}

template<usize N>
inline constexpr std::array<std::array<usize, N + 1>, N + 1> gbs_index_arrays = get_index_arrays<N>();

}

#endif
