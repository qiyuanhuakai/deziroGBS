#ifndef FGBS_CORE_MATH_UTILS_HPP
#define FGBS_CORE_MATH_UTILS_HPP

#include "fgbs/core/type_utils.hpp"

#include <cstddef>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

namespace fgbs {

#ifdef __GNUC__
inline int popcount32(u32 x) {
    return __builtin_popcount(x);
}

inline int popcount64(u64 x) {
    return __builtin_popcountll(x);
}
#else

/**
 * \brief Based on function by Peter Cordes: https://stackoverflow.com/a/109025
 **/
inline int popcount32(u32 x) {
    x -= (x >> 1) & 0x55555555;
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x *= 0x01010101;
    return x >> 24;
}

inline int popcount64(u64 x) {
    x -= (x >> 1) & 0x5555555555555555;
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
    x *= 0x0101010101010101;
    return x >> 56;
}
#endif

template<typename T>
constexpr std::enable_if_t<std::is_integral_v<T>, real> parity(T x) {
    return static_cast<real>(static_cast<std::make_signed_t<T>>(1 - 2 * (x & 1)));
}

template<typename T>
constexpr T pow2(T n) {
    return T{1} << n;
}

/**
 * \brief Based on function by paxdiablo: https://stackoverflow.com/a/994647
 **/
template<typename T>
constexpr T log2(T x) {
    if (x == 0) {
        return static_cast<T>(std::numeric_limits<std::make_signed_t<T>>::min());
    }

    T y = 0;
    while (x > 1) {
        x >>= 1;
        ++y;
    }
    return y;
}

/**
 * \brief Based on function by alexeykuzmin0: https://stackoverflow.com/a/44719219
 **/
template<typename T>
constexpr T binomial(T n, T k) {
    return k > n ? T{0} : (k == 0 || k == n) ? T{1} : (k == 1 || k == n - 1) ? n : binomial(n - 1, k - 1) * n / k;
}

template<typename T>
constexpr T factorial(T n) {
    T result = 1;
    for (T i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

template<typename T>
constexpr std::enable_if_t<std::is_integral_v<T>, real> parity_factorial(T n) {
    using V = std::make_signed_t<T>;
    V result = 1;
    for (V i = 1, m = static_cast<V>(n); i <= m; ++i) {
        result *= -i;
    }
    return static_cast<real>(result);
}

#define FGBS_INTERNAL_GET_INT_BITS(NAME, CONDITION) \
template<usize N, typename T>                       \
std::vector<upto<N>> NAME(T x) {                    \
    std::vector<upto<N>> v;                         \
    for (usize i = 0; i < N; ++i) {                 \
        if (CONDITION) {                            \
            v.push_back(static_cast<upto<N>>(i));   \
        }                                           \
    }                                               \
    return v;                                       \
}

FGBS_INTERNAL_GET_INT_BITS(get_zeros, !((x >> i) & 1))
FGBS_INTERNAL_GET_INT_BITS(get_ones, (x >> i) & 1)

template<typename VECTOR>
usize ones_to_usize(VECTOR&& ones) {
    return std::accumulate(ones.cbegin(), ones.cend(), usize{0}, [](usize acc, usize i) { return acc + pow2(i); });
}

}

#undef FGBS_INTERNAL_GET_INT_BITS

#endif
