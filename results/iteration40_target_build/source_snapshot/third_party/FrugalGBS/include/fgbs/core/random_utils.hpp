#ifndef FGBS_CORE_RANDOM_UTILS_HPP
#define FGBS_CORE_RANDOM_UTILS_HPP

#include "fgbs/core/type_utils.hpp"

#include <cstdint>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace fgbs {

using mt19937 = typename std::conditional_t<sizeof(std::intmax_t) <= 4, std::mt19937, std::mt19937_64>;

template<typename T>
struct UniformRandomInt {
public:
    template<typename... ARGS>
    UniformRandomInt(uint seed, ARGS&&... args) : _gen{seed}, _dist{std::uniform_int_distribution<T>(std::forward<ARGS>(args)...)} {}

    T operator()() {
        return _dist(_gen);
    }

private:
    mt19937 _gen;
    std::uniform_int_distribution<T> _dist;
};

template<typename T>
struct UniformRandomReal {
public:
    template<typename... ARGS>
    UniformRandomReal(uint seed, ARGS&&... args) : _gen{seed}, _dist{std::uniform_real_distribution<T>(std::forward<ARGS>(args)...)} {}

    T operator()() {
        return _dist(_gen);
    }

private:
    mt19937 _gen;
    std::uniform_real_distribution<T> _dist;
};

template<typename T>
struct NormalRandomReal {
public:
    template<typename... ARGS>
    NormalRandomReal(uint seed, ARGS&&... args) : _gen{seed}, _dist{std::normal_distribution<T>(std::forward<ARGS>(args)...)} {}

    T operator()() {
        return _dist(_gen);
    }

private:
    mt19937 _gen;
    std::normal_distribution<T> _dist;
};

template<typename T, typename RANDOM_GEN>
std::vector<T> get_random_vector(RANDOM_GEN& random, usize size) {
    std::vector<T> vec(size);
    for (usize i = 0; i < size; ++i) {
        vec[i] = static_cast<T>(random());
    }
    return vec;
}

}

#endif
