#ifndef FGBS_CORE_TUPLE_UTILS_HPP
#define FGBS_CORE_TUPLE_UTILS_HPP

#include <tuple>

namespace fgbs {

// FUNCTION = void(T const&)
template<typename FUNCTION, typename TUPLE>
constexpr void apply_ordered(FUNCTION&& f, TUPLE&& t) {
    std::apply([&](auto const&... x) { (..., f(x)); }, t);
}

}

#endif
