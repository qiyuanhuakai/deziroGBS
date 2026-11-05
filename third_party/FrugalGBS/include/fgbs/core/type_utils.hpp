#ifndef FGBS_CORE_TYPE_UTILS_HPP
#define FGBS_CORE_TYPE_UTILS_HPP

#include "fgbs/global.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace fgbs {

using usize = std::size_t;
using isize = std::make_signed_t<usize>;

using iptrdiff = std::ptrdiff_t;
using uptrdiff = std::make_unsigned_t<iptrdiff>;

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using uint = unsigned int;
using ull = unsigned long long int;

using f32 = float;
using f64 = double;

using real = typename std::conditional_t < FGBS_REAL_PRECISION<2, f32, f64>;
using complex = std::complex<real>;

template<usize N>
using upto = typename std::conditional_t<N <= UINT8_MAX, u8, typename std::conditional_t<N <= UINT16_MAX, u16, typename std::conditional_t<N <= UINT32_MAX, u32, u64>>>;

template<typename T>
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template<typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

template<typename T>
using collection_value_t = typename remove_cvref_t<T>::value_type;

constexpr usize bitstring_size(usize n) {
    return (n + usize{31}) / usize{32};
}

template<usize N>
using Bitstring = std::array<u32, bitstring_size(N)>;

}

#endif
