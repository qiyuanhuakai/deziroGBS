#ifndef FGBS_CPU_SAMPLER_HPP
#define FGBS_CPU_SAMPLER_HPP

#include "fgbs/core/bitstring_utils.hpp"
#include "fgbs/core/consts.hpp"
#include "fgbs/core/structs.hpp"
#include "fgbs/core/random_simd.hpp"
#include "fgbs/core/random_utils.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/type_utils.hpp"

#include <xsimd/xsimd.hpp>

#include <omp.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#define FGBS_INTERNAL_RESTRICT __restrict
#elif defined(__clang__) || defined(__GNUC__)
#define FGBS_INTERNAL_RESTRICT __restrict__
#else
#define FGBS_INTERNAL_RESTRICT
#endif

namespace fgbs {

namespace cpu {

template<typename T>
struct NaiveSum {
public:
    T get() const {
        return _sum;
    }

    NaiveSum<T>& operator+=(T value);

private:
    T _sum{};
};

using Prob = real;
using ProbSimd = xsimd::batch<Prob>;

template<typename T>
struct Randomization;

template<>
struct Randomization<Prob> {
    using type = UniformRandomReal<real>;
};

template<>
struct Randomization<ProbSimd> {
    using type = UniformRandomRealSimd<real>;
};

template<typename T>
using UniformRandom = typename Randomization<T>::type;

template<typename T>
struct KahanCompensation;

template<>
struct KahanCompensation<Prob> {
    using type = real;
};

template<>
struct KahanCompensation<ProbSimd> {
    using type = xsimd::batch<real>;
};

template<typename T>
struct KahanSum {
public:
    T get() const {
        return _sum;
    }

    KahanSum<T>& operator+=(T value);

private:
    T _sum{};
    typename KahanCompensation<T>::type _compensation{};
};

template<typename T>
using ProbSum = KahanSum<T>;

template<typename T>
T get_zero();

template<>
inline real get_zero<real>() {
    return 0.0;
}

template<>
inline xsimd::batch<real> get_zero<xsimd::batch<real>>() {
    return xsimd::batch<real>(0.0);
}

template<typename T>
T get_one();

template<>
inline real get_one<real>() {
    return 1.0;
}

template<>
inline xsimd::batch<real> get_one<xsimd::batch<real>>() {
    return xsimd::batch<real>(1.0);
}

template<typename T>
T get_half();

template<>
inline real get_half<real>() {
    return 0.5;
}

template<>
inline xsimd::batch<real> get_half<xsimd::batch<real>>() {
    return xsimd::batch<real>(0.5);
}

inline real clamped_prob(real prob) {
    return prob < 0.0 ? real{0.0} : (prob > 1.0 ? real{1.0} : prob);
}

inline xsimd::batch<real> clamped_prob(xsimd::batch<real> prob) {
    return xsimd::max(xsimd::batch<real>(0.0), xsimd::min(prob, xsimd::batch<real>(1.0)));
}

template<usize N>
constexpr usize partial_marginal_arr_size() {
    return N <= 1 ? 1 : N * (N - 1) / 2;
}

template<usize N>
constexpr usize elided_marginal_arr_size() {
    return N <= 1 ? 1 : N * (N - 1) / 2;
}

template<usize N>
constexpr usize partial_elided_marginal_arr_size() {
    return N <= 2 ? 1 : N * (N - 1) * (N - 2) / 6;
}

template<usize N>
constexpr usize double_elided_marginal_arr_size() {
    return N <= 2 ? 1 : N * (N - 1) * (N - 2) / 6;
}

template<usize N>
using SampleBatch = std::array<Bitstring<N>, xsimd::batch<Prob>::size>;

template<usize N, typename T>
using SampleResult = std::conditional_t<std::is_same_v<T, ProbSimd>, SampleBatch<N>, Bitstring<N>>;

template<usize N, typename T>
T get_partial_marginal(T const* FGBS_INTERNAL_RESTRICT marginal_arr, usize begin, usize end) {
    if (begin == end) {
        return get_one<T>();
    }
    return marginal_arr[begin + end * (end - 1) / 2];
}

template<usize N, typename T>
void set_partial_marginal(T* FGBS_INTERNAL_RESTRICT marginal_arr, usize begin, usize end, T marginal) {
    marginal_arr[begin + end * (end - 1) / 2] = marginal;
}

template<usize N, typename T>
T get_elided_marginal(T const* FGBS_INTERNAL_RESTRICT marginal_arr, usize elided, usize end) {
    return marginal_arr[elided + end * (end - 1) / 2];
}

template<usize N, typename T>
void set_elided_marginal(T* FGBS_INTERNAL_RESTRICT marginal_arr, usize elided, usize end, T marginal) {
    marginal_arr[elided + end * (end - 1) / 2] = marginal;
}

template<usize N, typename T>
T get_partial_elided_marginal(T const* FGBS_INTERNAL_RESTRICT partial_marginal_arr, T const* FGBS_INTERNAL_RESTRICT partial_elided_marginal_arr, usize begin, usize elided, usize end) {
    if (begin == elided) {
        return get_partial_marginal<N>(partial_marginal_arr, begin + 1, end);
    }
    return partial_elided_marginal_arr[begin + elided * (elided - 1) / 2 + end * (end - 1) * (end - 2) / 6];
}

template<usize N, typename T>
void set_partial_elided_marginal(T* FGBS_INTERNAL_RESTRICT marginal_arr, usize begin, usize elided, usize end, T marginal) {
    marginal_arr[begin + elided * (elided - 1) / 2 + end * (end - 1) * (end - 2) / 6] = marginal;
}

template<usize N, typename T>
T get_double_elided_marginal(T const* FGBS_INTERNAL_RESTRICT marginal_arr, usize first_elided, usize second_elided, usize end) {
    return marginal_arr[first_elided + second_elided * (second_elided - 1) / 2 + end * (end - 1) * (end - 2) / 6];
}

template<usize N, typename T>
void set_double_elided_marginal(T* FGBS_INTERNAL_RESTRICT marginal_arr, usize first_elided, usize second_elided, usize end, T marginal) {
    marginal_arr[first_elided + second_elided * (second_elided - 1) / 2 + end * (end - 1) * (end - 2) / 6] = marginal;
}

template<usize N>
usize partial_marginal_order(BitstringLayout<N> const& bitstring_layout) {
    if (bitstring_layout.cutoff_order >= 5) {
        return bitstring_layout.cutoff_order - 2;
    }
    if (bitstring_layout.cutoff_order >= 3) {
        return bitstring_layout.cutoff_order - 1;
    }
    return 2;
}

template<usize N>
usize elided_marginal_order(BitstringLayout<N> const& bitstring_layout) {
    if (bitstring_layout.cutoff_order >= 5) {
        return bitstring_layout.cutoff_order - 2;
    }
    if (bitstring_layout.cutoff_order >= 3) {
        return bitstring_layout.cutoff_order - 1;
    }
    return 2;
}

template<usize N>
usize partial_elided_marginal_order(BitstringLayout<N> const& bitstring_layout) {
    if (bitstring_layout.cutoff_order >= 5) {
        return bitstring_layout.cutoff_order - 3;
    }
    return 2;
}

template<usize N>
usize double_elided_marginal_order(BitstringLayout<N> const& bitstring_layout) {
    if (bitstring_layout.cutoff_order >= 5) {
        return bitstring_layout.cutoff_order - 3;
    }
    return 2;
}

#define MARGINAL_END_ZERO_DIFF              \
end_zero_diff : {}                          \
                                            \
const auto parity_sign = 2 * parity_arr[i]; \
prev_prob = clamped_prob(prev_prob * (real{0.5} + real{0.5} * parity_sign * sampling_target.first_correlator_cumulant_vec[i]) + parity_sign * zero_diff.get());

template<usize N, typename T>
void compute_partial_marginal(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, T* FGBS_INTERNAL_RESTRICT partial_marginal_arr, T const* FGBS_INTERNAL_RESTRICT partial_elided_marginal_arr, T const* FGBS_INTERNAL_RESTRICT parity_arr, usize begin, usize end, usize marginal_order) {
    auto i = end - 1;

    auto prev_prob = get_partial_marginal<N>(partial_marginal_arr, begin, i);
    ProbSum<T> zero_diff{};
    auto first_cumulant_offset = gbs_index_arrays<N>[bitstring_layout.cutoff_order][i];

    if (marginal_order <= 1) {
        goto end_zero_diff;
    }

    for (usize j = begin; j < i; ++j) {
        auto second_mult = get_half<T>() * parity_arr[j];
        auto second_cumulant_offset = first_cumulant_offset + gbs_index_arrays<N>[bitstring_layout.cutoff_order - 1][j];

        zero_diff += second_mult * sampling_target.correlator_cumulant_vec[second_cumulant_offset] * get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, begin, j, i);

        if (marginal_order <= 2) {
            continue;
        }

        second_mult *= get_partial_marginal<N>(partial_marginal_arr, j + 1, i);
        for (usize k = begin; k < j; ++k) {
            auto third_cumulant_offset = second_cumulant_offset + gbs_index_arrays<N>[bitstring_layout.cutoff_order - 2][k];
            zero_diff += second_mult * parity_arr[k] * sampling_target.correlator_cumulant_vec[third_cumulant_offset] * get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, begin, k, j);
        }
    }

    MARGINAL_END_ZERO_DIFF
    set_partial_marginal<N>(partial_marginal_arr, begin, end, prev_prob);
}

#define COMPUTE_ELIDED_MARGINAL_SECOND(K, J)                                                                                                                     \
zero_diff += second_mult * sampling_target.correlator_cumulant_vec[second_cumulant_offset] * get_double_elided_marginal<N>(double_elided_marginal_arr, K, J, i); \
                                                                                                                                                                 \
if (marginal_order <= 2) {                                                                                                                                       \
    continue;                                                                                                                                                    \
}                                                                                                                                                                \
                                                                                                                                                                 \
second_mult *= get_partial_marginal<N>(partial_marginal_arr, J + 1, i);

#define COMPUTE_ELIDED_MARGINAL_THIRD(L, K, J) zero_diff += second_mult * parity_arr[k] * sampling_target.correlator_cumulant_vec[third_cumulant_offset] * get_double_elided_marginal<N>(double_elided_marginal_arr, L, K, J);

template<usize N, typename T>
void compute_elided_marginal(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, T const* FGBS_INTERNAL_RESTRICT partial_marginal_arr, T* FGBS_INTERNAL_RESTRICT elided_marginal_arr, T const* FGBS_INTERNAL_RESTRICT double_elided_marginal_arr, T const* FGBS_INTERNAL_RESTRICT parity_arr, usize elided, usize end, usize marginal_order) {
    auto i = end - 1;
    auto prev_prob = get_elided_marginal<N>(elided_marginal_arr, elided, i);
    ProbSum<T> zero_diff{};
    auto first_cumulant_offset = gbs_index_arrays<N>[bitstring_layout.cutoff_order][i];

    if (marginal_order <= 1) {
        goto end_zero_diff;
    }

    for (usize j = 0; j < i; ++j) {
        if (j == elided) {
            continue;
        }
        auto second_mult = get_half<T>() * parity_arr[j];
        auto second_cumulant_offset = first_cumulant_offset + gbs_index_arrays<N>[bitstring_layout.cutoff_order - 1][j];

        if (j < elided) {
            COMPUTE_ELIDED_MARGINAL_SECOND(j, elided)
            for (usize k = 0; k < j; ++k) {
                auto third_cumulant_offset = second_cumulant_offset + gbs_index_arrays<N>[bitstring_layout.cutoff_order - 2][k];
                COMPUTE_ELIDED_MARGINAL_THIRD(k, j, elided)
            }
        }
        else {
            COMPUTE_ELIDED_MARGINAL_SECOND(elided, j)
            for (usize k = 0; k < j; ++k) {
                if (k == elided) {
                    continue;
                }
                auto third_cumulant_offset = second_cumulant_offset + gbs_index_arrays<N>[bitstring_layout.cutoff_order - 2][k];
                if (k < elided) {
                    COMPUTE_ELIDED_MARGINAL_THIRD(k, elided, j)
                }
                else {
                    COMPUTE_ELIDED_MARGINAL_THIRD(elided, k, j)
                }
            }
        }
    }

    MARGINAL_END_ZERO_DIFF
    set_elided_marginal<N>(elided_marginal_arr, elided, end, prev_prob);
}

#define COMPUTE_PARTIAL_ELIDED_MARGINAL_SECOND(K, J) zero_diff += second_parity * sampling_target.correlator_cumulant_vec[second_cumulant_offset] * get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, begin, K, J) * get_partial_marginal<N>(partial_marginal_arr, J + 1, i);

template<usize N, typename T>
void compute_partial_elided_marginal(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, T const* FGBS_INTERNAL_RESTRICT partial_marginal_arr, T* FGBS_INTERNAL_RESTRICT partial_elided_marginal_arr, T const* FGBS_INTERNAL_RESTRICT parity_arr, usize begin, usize elided, usize end, usize marginal_order) {
    auto i = end - 1;
    auto prev_prob = get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, begin, elided, i);
    ProbSum<T> zero_diff{};
    auto first_cumulant_offset = gbs_index_arrays<N>[bitstring_layout.cutoff_order][i];

    if (marginal_order <= 1) {
        goto end_zero_diff;
    }

    for (usize j = begin; j < i; ++j) {
        if (j == elided) {
            continue;
        }
        auto second_parity = get_half<T>() * parity_arr[j];
        auto second_cumulant_offset = first_cumulant_offset + gbs_index_arrays<N>[bitstring_layout.cutoff_order - 1][j];
        if (j < elided) {
            COMPUTE_PARTIAL_ELIDED_MARGINAL_SECOND(j, elided)
        }
        else {
            COMPUTE_PARTIAL_ELIDED_MARGINAL_SECOND(elided, j)
        }
    }

    MARGINAL_END_ZERO_DIFF
    set_partial_elided_marginal<N>(partial_elided_marginal_arr, begin, elided, end, prev_prob);
}

#define COMPUTE_DOUBLE_ELIDED_MARGINAL_SECOND(L, K, J) zero_diff += second_parity * sampling_target.correlator_cumulant_vec[second_cumulant_offset] * get_double_elided_marginal<N>(double_elided_marginal_arr, L, K, J) * get_partial_marginal<N>(partial_marginal_arr, J + 1, i);

template<usize N, typename T>
void compute_double_elided_marginal(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, T const* FGBS_INTERNAL_RESTRICT partial_marginal_arr, T* FGBS_INTERNAL_RESTRICT double_elided_marginal_arr, T const* FGBS_INTERNAL_RESTRICT parity_arr, usize first_elided, usize second_elided, usize end, usize marginal_order) {
    auto i = end - 1;
    auto prev_prob = get_double_elided_marginal<N>(double_elided_marginal_arr, first_elided, second_elided, i);
    ProbSum<T> zero_diff{};
    auto first_cumulant_offset = gbs_index_arrays<N>[bitstring_layout.cutoff_order][i];

    if (marginal_order <= 1) {
        goto end_zero_diff;
    }

    for (usize j = 0; j < i; ++j) {
        if (j == first_elided || j == second_elided) {
            continue;
        }
        auto second_parity = get_half<T>() * parity_arr[j];
        auto second_cumulant_offset = first_cumulant_offset + gbs_index_arrays<N>[bitstring_layout.cutoff_order - 1][j];

        if (j < first_elided) {
            COMPUTE_DOUBLE_ELIDED_MARGINAL_SECOND(j, first_elided, second_elided)
        }
        else if (j < second_elided) {
            COMPUTE_DOUBLE_ELIDED_MARGINAL_SECOND(first_elided, j, second_elided)
        }
        else {
            COMPUTE_DOUBLE_ELIDED_MARGINAL_SECOND(first_elided, second_elided, j)
        }
    }

    MARGINAL_END_ZERO_DIFF
    set_double_elided_marginal<N>(double_elided_marginal_arr, first_elided, second_elided, end, prev_prob);
}

template<usize N, typename T>
void update_result(UniformRandom<T>& rand, SampleResult<N, T>& result, T& prev_prob, T* FGBS_INTERNAL_RESTRICT parity_arr, usize i, T zero_next_prob) {
    const auto zero_prob = zero_next_prob / prev_prob;

    if constexpr (std::is_same_v<T, ProbSimd>) {
        const auto mask = rand() > zero_prob;
        for (usize j = 0; j < ProbSimd::size; ++j) {
            if (mask.get(j)) {
                flip_bit<N>(result[j], i);
            }
        }
        parity_arr[i] = xsimd::select(mask, -get_half<ProbSimd>(), get_half<ProbSimd>());
        prev_prob = xsimd::select(mask, prev_prob - zero_next_prob, zero_next_prob);
    }
    else {
        if (rand() > zero_prob) {
            flip_bit<N>(result, i);
            parity_arr[i] = -get_half<Prob>();
            prev_prob -= zero_next_prob;
        }
        else {
            parity_arr[i] = get_half<Prob>();
            prev_prob = zero_next_prob;
        }
    }
}

template<usize N, typename T>
SampleResult<N, T> generate_sample(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, UniformRandom<T>& rand, T* FGBS_INTERNAL_RESTRICT partial_marginal_arr, T* FGBS_INTERNAL_RESTRICT elided_marginal_arr, T* FGBS_INTERNAL_RESTRICT partial_elided_marginal_arr, T* FGBS_INTERNAL_RESTRICT double_elided_marginal_arr) {
    SampleResult<N, T> result{};
    T prev_prob = get_one<T>();
    T parity_arr[N];

    const auto partial_order = partial_marginal_order(bitstring_layout);
    const auto elided_order = elided_marginal_order(bitstring_layout);
    const auto partial_elided_order = partial_elided_marginal_order(bitstring_layout);
    const auto double_elided_order = double_elided_marginal_order(bitstring_layout);

    for (usize i = 0; i < N; ++i) {
        ProbSum<T> zero_diff{};
        auto correlator_cumulant_ptr = &sampling_target.correlator_cumulant_vec[bitstring_layout.sum_index_arr[i]];

        if (bitstring_layout.cutoff_order <= 1) {
            goto end_zero_diff;
        }

        for (usize j = 0; j < i; ++j) {
            if (j == 0) {
                set_partial_marginal<N>(partial_marginal_arr, j, i, prev_prob);
            }
            else {
                compute_partial_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, partial_elided_marginal_arr, parity_arr, j, i, partial_order);
            }
        }

        for (usize j = 0; j < i; ++j) {
            if (j == 0) {
                set_elided_marginal<N>(elided_marginal_arr, j, i, get_partial_marginal<N>(partial_marginal_arr, 1, i));
            }
            else if (j == i - 1) {
                set_elided_marginal<N>(elided_marginal_arr, j, i, get_partial_marginal<N>(partial_marginal_arr, 0, i - 1));
            }
            else {
                compute_elided_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, elided_marginal_arr, double_elided_marginal_arr, parity_arr, j, i, elided_order);
            }
        }

        for (usize j = 0; j < i; ++j) {
            for (usize k = 0; k < j; ++k) {
                if (k == 0) {
                    set_partial_elided_marginal<N>(partial_elided_marginal_arr, k, j, i, get_elided_marginal<N>(elided_marginal_arr, j, i));
                }
                else if (j == i - 1) {
                    set_partial_elided_marginal<N>(partial_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, k, i - 1));
                }
                else {
                    compute_partial_elided_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, partial_elided_marginal_arr, parity_arr, k, j, i, partial_elided_order);
                }
            }
        }

        for (usize j = 0; j < i; ++j) {
            for (usize k = 0; k < j; ++k) {
                if (k == 0 && j == 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, 2, i));
                }
                else if (k == 0 && j == i - 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, 1, i - 1));
                }
                else if (k == i - 2 && j == i - 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_partial_marginal<N>(partial_marginal_arr, 0, i - 2));
                }
                else if (j == i - 1) {
                    set_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i, get_elided_marginal<N>(elided_marginal_arr, k, i - 1));
                }
                else {
                    compute_double_elided_marginal<N>(bitstring_layout, sampling_target, partial_marginal_arr, double_elided_marginal_arr, parity_arr, k, j, i, double_elided_order);
                }
            }
        }

        for (usize j = 0; j < i; ++j) {
            auto second_parity = get_half<Prob>() * parity_arr[j];
            zero_diff += second_parity * (*(++correlator_cumulant_ptr)) * get_elided_marginal<N>(elided_marginal_arr, j, i);

            if (bitstring_layout.cutoff_order <= 2) {
                continue;
            }

            auto second_marginal = get_partial_marginal<N>(partial_marginal_arr, j + 1, i);
            for (usize k = 0; k < j; ++k) {
                auto third_parity = second_parity * parity_arr[k];
                zero_diff += third_parity * (*(++correlator_cumulant_ptr)) * get_double_elided_marginal<N>(double_elided_marginal_arr, k, j, i);

                if (bitstring_layout.cutoff_order <= 3) {
                    continue;
                }

                auto third_marginal = get_partial_elided_marginal<N>(partial_marginal_arr, partial_elided_marginal_arr, k + 1, j, i);
                for (usize l = 0; l < k; ++l) {
                    auto fourth_parity = third_parity * parity_arr[l];
                    zero_diff += fourth_parity * (*(++correlator_cumulant_ptr)) * get_double_elided_marginal<N>(double_elided_marginal_arr, l, k, j) * second_marginal;

                    if (bitstring_layout.cutoff_order <= 4) {
                        continue;
                    }

                    for (usize m = 0; m < l; ++m) {
                        auto fifth_parity = fourth_parity * parity_arr[m];
                        zero_diff += fifth_parity * (*(++correlator_cumulant_ptr)) * get_double_elided_marginal<N>(double_elided_marginal_arr, m, l, k) * third_marginal;
                    }
                }
            }
        }

    end_zero_diff : {}

        const auto zero_next_prob = clamped_prob(prev_prob * (real{0.5} + real{0.5} * sampling_target.first_correlator_cumulant_vec[i]) + zero_diff.get());
        update_result<N>(rand, result, prev_prob, parity_arr, i, zero_next_prob);
    }

    return result;
}

template<usize N>
Result<std::vector<Bitstring<N>>> generate_samples(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, usize sample_count, uint seed) {
    if (sample_count == 0) {
        return Ok<std::vector<Bitstring<N>>>{std::vector<Bitstring<N>>{}};
    }

    std::vector<Bitstring<N>> samples(sample_count);
    UniformRandomInt<usize> seed_generator{seed, 0, ULLONG_MAX};
    const auto seeds = get_random_vector<uint>(seed_generator, ::omp_get_max_threads());

    const auto batch_size = xsimd::batch<Prob>::size;
    const auto tail_size = sample_count % batch_size;
    const auto batch_count = (sample_count - 1 + batch_size) / batch_size;

    #pragma omp parallel default(none) shared(samples, seeds, bitstring_layout, sampling_target) firstprivate(batch_size, tail_size, batch_count)
    {
        UniformRandom<ProbSimd> rand(seeds[::omp_get_thread_num()], 0.0, 1.0);

        std::vector<ProbSimd> partial_marginal_arr(partial_marginal_arr_size<N>());
        std::vector<ProbSimd> elided_marginal_arr(elided_marginal_arr_size<N>());
        std::vector<ProbSimd> partial_elided_marginal_arr(partial_elided_marginal_arr_size<N>());
        std::vector<ProbSimd> double_elided_marginal_arr(double_elided_marginal_arr_size<N>());

        #pragma omp for
        for (usize i = 0; i < batch_count; ++i) {
            const auto iter_length = i == batch_count - 1 && tail_size > 0 ? tail_size : batch_size;
            const auto bitstring_batch = &samples[i * batch_size];
            const auto sample_batch = generate_sample<N, ProbSimd>(bitstring_layout, sampling_target, rand, partial_marginal_arr.data(), elided_marginal_arr.data(), partial_elided_marginal_arr.data(), double_elided_marginal_arr.data());
            for (usize j = 0; j < iter_length; ++j) {
                bitstring_batch[j] = sample_batch[j];
            }
        }
    }

    return Ok<std::vector<Bitstring<N>>>{std::move(samples)};
}

#undef MARGINAL_END_ZERO_DIFF
#undef COMPUTE_ELIDED_MARGINAL_SECOND
#undef COMPUTE_ELIDED_MARGINAL_THIRD
#undef COMPUTE_PARTIAL_ELIDED_MARGINAL_SECOND
#undef COMPUTE_DOUBLE_ELIDED_MARGINAL_SECOND
#undef FGBS_INTERNAL_RESTRICT

}

}

#endif
