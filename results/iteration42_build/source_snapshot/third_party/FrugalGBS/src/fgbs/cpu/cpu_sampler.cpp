#include "fgbs/cpu/cpu_sampler.hpp"

#include <xsimd/xsimd.hpp>

namespace fgbs {

namespace cpu {

#define FGBS_INTERNAL_DEFINE_NAIVE_PLUS(T)      \
template<>                                      \
NaiveSum<T>& NaiveSum<T>::operator+=(T value) { \
    _sum += value;                              \
    return *this;                               \
}

FGBS_INTERNAL_DEFINE_NAIVE_PLUS(real)
FGBS_INTERNAL_DEFINE_NAIVE_PLUS(xsimd::batch<real>)

#define FGBS_INTERNAL_DEFINE_KAHAN_PLUS(T)      \
template<>                                      \
KahanSum<T>& KahanSum<T>::operator+=(T value) { \
    const auto y = value - _compensation;       \
    const auto t = _sum + y;                    \
    _compensation = (t - _sum) - y;             \
    _sum = t;                                   \
    return *this;                               \
}

FGBS_INTERNAL_DEFINE_KAHAN_PLUS(real)
FGBS_INTERNAL_DEFINE_KAHAN_PLUS(xsimd::batch<real>)

#undef FGBS_INTERNAL_DEFINE_KAHAN_PLUS
#undef FGBS_INTERNAL_DEFINE_NAIVE_PLUS

}

}
