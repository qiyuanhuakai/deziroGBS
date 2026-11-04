#ifndef FGBS_CORE_STAT_UTILS_HPP
#define FGBS_CORE_STAT_UTILS_HPP

#include "fgbs/core/random_utils.hpp"
#include "fgbs/core/stat_structs.hpp"
#include "fgbs/core/type_utils.hpp"

#include <omp.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace fgbs {

template<typename T, typename V>
std::vector<T> compute_sums(std::vector<std::vector<V>> const& vecs, T sum_divisor) {
    const auto vec_count = vecs.size();
    if (vec_count == 0) {
        return std::vector<T>{};
    }

    const auto result_count = vecs[0].size();
    std::vector<T> sums(result_count);

    #pragma omp parallel for default(none) shared(sums, vecs, sum_divisor) firstprivate(result_count, vec_count)
    for (usize i = 0; i < result_count; ++i) {
        V total_sum = 0;
        for (usize j = 0; j < vec_count; ++j) {
            total_sum += vecs[j][i];
        }

        sums[i] = static_cast<T>(total_sum) / sum_divisor;
    }

    return sums;
}

template<typename XS>
f64 compute_mean(XS&& xs) {
    const auto count = xs.size();
    if (count == 0) {
        return 0.0;
    }

    f64 sum = 0.0;

    #pragma omp parallel for default(none) firstprivate(count) shared(xs) reduction(+ : sum)
    for (usize i = 0; i < count; ++i) {
        sum += xs[i];
    }

    return sum / count;
}

template<typename XS>
f64 compute_variance(XS&& xs) {
    const auto count = xs.size();
    if (count <= 1) {
        return 0.0;
    }

    const auto mean = compute_mean(xs);

    f64 sum_sq_diff = 0.0;

    #pragma omp parallel for default(none) firstprivate(count, mean) shared(xs) reduction(+ : sum_sq_diff)
    for (usize i = 0; i < count; ++i) {
        const f64 diff = xs[i] - mean;
        sum_sq_diff += diff * diff;
    }

    return sum_sq_diff / count;
}

template<typename XS, typename YS>
std::pair<std::vector<collection_value_t<XS>>, std::vector<collection_value_t<YS>>> resample(XS&& xs, YS&& ys, uint seed) {
    const usize count = xs.size();

    std::vector<collection_value_t<XS>> xs_resample(count);
    std::vector<collection_value_t<YS>> ys_resample(count);

    auto seed_generator = UniformRandomInt<usize>(seed, 0, ULLONG_MAX);
    const auto seeds = get_random_vector<uint>(seed_generator, ::omp_get_max_threads());

    #pragma omp parallel default(none) firstprivate(count) shared(xs, ys, xs_resample, ys_resample, seeds)
    {
        UniformRandomInt<usize> rand(seeds[::omp_get_thread_num()], 0, count - 1);

        #pragma omp for
        for (usize i = 0; i < count; ++i) {
            const auto j = rand();
            xs_resample[i] = xs[j];
            ys_resample[i] = ys[j];
        }
    }

    return {xs_resample, ys_resample};
}

template<typename XS, typename YS>
std::tuple<f64, f64, f64> compute_linear_regression_stats(XS&& xs, YS&& ys) {
    const auto count = xs.size();

    const auto x_mean = compute_mean(xs), y_mean = compute_mean(ys);

    f64 covariance = 0.0, x_variance = 0.0, y_variance = 0.0;

    #pragma omp parallel for default(none) firstprivate(count, x_mean, y_mean) shared(xs, ys) reduction(+ : covariance, x_variance, y_variance)
    for (usize i = 0; i < count; ++i) {
        const f64 x_diff = xs[i] - x_mean, y_diff = ys[i] - y_mean;
        covariance += x_diff * y_diff;
        x_variance += x_diff * x_diff;
        y_variance += y_diff * y_diff;
    }

    const f64 correlation = covariance / std::sqrt(x_variance * y_variance);
    const f64 gradient = covariance / x_variance;
    const f64 intercept = y_mean - gradient * x_mean;

    return {correlation, gradient, intercept};
}

template<typename XS, typename YS, typename FUNCTION>
LinregressInfo compute_linear_regression_info(XS&& xs, YS&& ys, FUNCTION&& transform, usize bootstrap_count, uint resample_seed) {
    auto seed_generator = UniformRandomInt<usize>(resample_seed, 0, ULLONG_MAX);

    std::vector<f64> correlation_vec(bootstrap_count), gradient_vec(bootstrap_count), intercept_vec(bootstrap_count);

    for (usize i = 0; i < bootstrap_count; ++i) {
        const auto [xs_resample, ys_resample] = resample(xs, ys, static_cast<uint>(seed_generator()));
        const auto bootstrap_stats = compute_linear_regression_stats(transform(std::move(xs_resample)), transform(std::move(ys_resample)));

        correlation_vec[i] = std::get<0>(bootstrap_stats);
        gradient_vec[i] = std::get<1>(bootstrap_stats);
        intercept_vec[i] = std::get<2>(bootstrap_stats);
    }

    const auto stats = compute_linear_regression_stats(transform(std::forward<XS>(xs)), transform(std::forward<YS>(ys)));

    const auto correlation_error = std::sqrt(compute_variance(std::move(correlation_vec)));
    const auto gradient_error = std::sqrt(compute_variance(std::move(gradient_vec)));
    const auto intercept_error = std::sqrt(compute_variance(std::move(intercept_vec)));

    return LinregressInfo{std::get<0>(stats), correlation_error, std::get<1>(stats), gradient_error, std::get<2>(stats), intercept_error};
}

template<typename XS>
std::vector<f64> compute_ranks(XS&& xs) {
    const usize count = xs.size();

    std::vector<usize> temp(count);

    std::iota(temp.begin(), temp.end(), 0);
    std::sort(temp.begin(), temp.end(), [&](usize i, usize j) { return xs[i] < xs[j]; });

    std::vector<f64> ranks(count);

    for (usize i = 0; i < count;) {
        usize j = i + 1;

        while (j < count && xs[temp[i]] == xs[temp[j]]) {
            ++j;
        }

        const f64 mean_rank = 0.5 * (i + j - 1) + 1.0;

        for (usize k = i; k < j; ++k) {
            ranks[temp[k]] = mean_rank;
        }

        i = j;
    }

    return ranks;
}

template<typename XS, typename YS>
LinregressInfo compute_pearson_regression_info(XS&& xs, YS&& ys, usize bootstrap_count, uint resample_seed) {
    return compute_linear_regression_info(std::forward<XS>(xs), std::forward<YS>(ys), [](auto&& v) -> decltype(auto) { return std::forward<decltype(v)>(v); }, bootstrap_count, resample_seed);
}

template<typename XS, typename YS>
LinregressInfo compute_spearman_regression_info(XS&& xs, YS&& ys, usize bootstrap_count, uint resample_seed) {
    return compute_linear_regression_info(std::forward<XS>(xs), std::forward<YS>(ys), [](auto&& v) { return compute_ranks(std::forward<decltype(v)>(v)); }, bootstrap_count, resample_seed);
}

}

#endif
