#ifndef FGBS_CORE_BITSTRING_UTILS_HPP
#define FGBS_CORE_BITSTRING_UTILS_HPP

#include "fgbs/core/macro_utils.hpp"
#include "fgbs/core/math_utils.hpp"
#include "fgbs/core/type_utils.hpp"

#include <string>
#include <vector>

namespace fgbs {

template<usize N>
bool get_bit(Bitstring<N> const& bitstring, usize index) {
    return (FGBS_INTERNAL_PTR_CAST(u8 const*, &bitstring)[index / 8] >> (index % 8)) & u8{1};
}

template<usize N>
void set_bit(Bitstring<N>& bitstring, usize index) {
    FGBS_INTERNAL_PTR_CAST(u8*, &bitstring)[index / 8] |= u8{1} << (index % 8);
}

template<usize N>
void flip_bit(Bitstring<N>& bitstring, usize index) {
    FGBS_INTERNAL_PTR_CAST(u8*, &bitstring)[index / 8] ^= u8{1} << (index % 8);
}

template<usize N>
upto<N> get_weight(Bitstring<N> const& bitstring) {
    constexpr auto BITSTRING_SIZE = bitstring_size(N);
    i32 weight = 0;
    for (usize i = 0; i < BITSTRING_SIZE; ++i) {
        weight += popcount32(FGBS_INTERNAL_PTR_CAST(u32 const*, &bitstring)[i]);
    }
    return static_cast<upto<N>>(weight);
}

template<usize N>
i32 partial_parity(Bitstring<N> const& bitstring, usize end) {
    usize bulk_size = end / 32;
    usize tail_size = end % 32;
    i32 weight = 0;
    for (usize i = 0; i < bulk_size; ++i) {
        weight += popcount32(FGBS_INTERNAL_PTR_CAST(u32 const*, &bitstring)[i]);
    }
    if (tail_size > 0) {
        weight += popcount32(FGBS_INTERNAL_PTR_CAST(u32 const*, &bitstring)[bulk_size] & ((1 << tail_size) - 1));
    }
    return 1 - 2 * (weight & 1);
}

template<usize N>
i32 characteristic(Bitstring<N> const& x, Bitstring<N> const& y) {
    constexpr auto BITSTRING_SIZE = bitstring_size(N);
    i32 weight = 0;
    for (usize i = 0; i < BITSTRING_SIZE; ++i) {
        weight += popcount32(FGBS_INTERNAL_PTR_CAST(u32 const*, &x)[i] & FGBS_INTERNAL_PTR_CAST(u32 const*, &y)[i]);
    }
    return 1 - 2 * (weight & 1);
}

template<usize N>
std::vector<upto<N>> get_zeros(Bitstring<N> const& bitstring, usize max_index = N) {
    std::vector<upto<N>> v{};
    for (usize i = 0; i < max_index; ++i) {
        if (!get_bit<N>(bitstring, i)) {
            v.push_back(static_cast<upto<N>>(i));
        }
    }
    return v;
}

template<usize N>
std::vector<upto<N>> get_ones(Bitstring<N> const& bitstring, usize max_index = N) {
    std::vector<upto<N>> v{};
    for (usize i = 0; i < max_index; ++i) {
        if (get_bit<N>(bitstring, i)) {
            v.push_back(static_cast<upto<N>>(i));
        }
    }
    return v;
}

template<usize N>
void to_str(Bitstring<N> const& bitstring, char* str) {
    for (usize i = 0; i < N; ++i) {
        str[i] = '0' + get_bit<N>(bitstring, i);
    }
}

template<usize N>
std::string to_str(Bitstring<N> const& bitstring) {
    std::string str(N, '\0');
    to_str<N>(bitstring, str.data());
    return str;
}

}

#endif
