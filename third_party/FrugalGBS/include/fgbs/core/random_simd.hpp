#ifndef FGBS_CORE_RANDOM_SIMD_HPP
#define FGBS_CORE_RANDOM_SIMD_HPP

#include "fgbs/core/random_utils.hpp"
#include "fgbs/core/type_utils.hpp"

#include <xsimd/xsimd.hpp>

#include <utility>

namespace fgbs {

template<typename T>
struct UniformRandomIntSimd {
public:
    template<typename... ARGS>
    UniformRandomIntSimd(uint seed, ARGS&&... args) : _rand{seed, std::forward<ARGS>(args)...} {}

    xsimd::batch<T> operator()() {
        alignas(xsimd::default_arch::alignment()) T data[xsimd::batch<T>::size];
        for (usize i = 0; i < xsimd::batch<T>::size; ++i) {
            data[i] = _rand();
        }
        return xsimd::load_aligned(data);
    }

private:
    UniformRandomInt<T> _rand;
};

template<typename T>
struct UniformRandomRealSimd {
public:
    template<typename... ARGS>
    UniformRandomRealSimd(uint seed, ARGS&&... args) : _rand{seed, std::forward<ARGS>(args)...} {}

    xsimd::batch<T> operator()() {
        alignas(xsimd::default_arch::alignment()) T data[xsimd::batch<T>::size];
        for (usize i = 0; i < xsimd::batch<T>::size; ++i) {
            data[i] = _rand();
        }
        return xsimd::load_aligned(data);
    }

private:
    UniformRandomReal<T> _rand;
};

template<typename T>
struct NormalRandomRealSimd {
public:
    template<typename... ARGS>
    NormalRandomRealSimd(uint seed, ARGS&&... args) : _rand{seed, std::forward<ARGS>(args)...} {}

    xsimd::batch<T> operator()() {
        alignas(xsimd::default_arch::alignment()) T data[xsimd::batch<T>::size];
        for (usize i = 0; i < xsimd::batch<T>::size; ++i) {
            data[i] = _rand();
        }
        return xsimd::load_aligned(data);
    }

private:
    NormalRandomReal<T> _rand;
};

}

#endif
