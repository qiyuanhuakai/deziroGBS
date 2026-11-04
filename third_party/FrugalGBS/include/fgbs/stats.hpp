#ifndef FGBS_STATS_HPP
#define FGBS_STATS_HPP

#if FGBS_GPU_ENABLED
#include "fgbs/gpu/gpu_stats.cuh"
#else
#include "fgbs/cpu/cpu_stats.hpp"
#endif

#include "fgbs/core/bitstring_utils.hpp"
#include "fgbs/core/cumulant_utils.hpp"
#include "fgbs/core/structs.hpp"
#include "fgbs/core/validation.hpp"
#include "fgbs/core/random_utils.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/stat_structs.hpp"
#include "fgbs/core/stat_utils.hpp"
#include "fgbs/core/string_utils.hpp"
#include "fgbs/core/sublist_utils.hpp"
#include "fgbs/core/type_utils.hpp"

#include "fgbs/cumulants.hpp"

#include <array>
#include <climits>
#include <cmath>
#include <utility>
#include <vector>

namespace fgbs {

template<usize N>
struct SampleClickStats {
    std::array<f64, N + 1> total_click_probs{};
    std::array<f64, N + 1> total_click_prob_errors{};
};

struct SampleCumulantStats {
    std::vector<LinregressInfo> pearson_info_vec;
    std::vector<LinregressInfo> spearman_info_vec;

    std::vector<real> sample_moment_cumulant_sublists;
};

template<usize N>
SampleClickStats<N> compute_sample_click_stats(std::vector<Bitstring<N>> const& samples) {
    const auto sample_count = samples.size();
    SampleClickStats<N> analysis;
    if (sample_count == 0) {
        return analysis;
    }

    #pragma omp parallel default(none) firstprivate(sample_count) shared(samples, analysis)
    {
        std::array<usize, N + 1> local_total_click_counts{};

        #pragma omp for
        for (usize i = 0; i < sample_count; ++i) {
            ++local_total_click_counts[get_weight<N>(samples[i])];
        }

        #pragma omp critical
        {
            for (usize i = 0; i <= N; ++i) {
                analysis.total_click_probs[i] += local_total_click_counts[i];
            }
        }
    }

    for (usize i = 0; i <= N; ++i) {
        const auto click_prob = analysis.total_click_probs[i] / sample_count;
        analysis.total_click_probs[i] = click_prob;
        analysis.total_click_prob_errors[i] = std::sqrt(click_prob * (1.0 - click_prob) / sample_count);
    }

    return analysis;
}

template<usize N>
Result<std::vector<real>> estimate_sample_correlator_sublists(SublistLayout<N> const& sublist_layout, std::vector<Bitstring<N>> const& samples) {
    #if FGBS_GPU_ENABLED
    auto sample_correlator_sublists_result = gpu::estimate_sample_correlator_sublists<N>(sublist_layout, samples);
    #else
    auto sample_correlator_sublists_result = cpu::estimate_sample_correlator_sublists<N>(sublist_layout, samples);
    #endif

    FGBS_VALIDATE_OR_RETURN_CTX(sample_correlator_sublists_result, "Failed to estimate sample correlator sublists: ");
    return sample_correlator_sublists_result;
}

template<usize N>
Result<std::vector<real>> compute_sample_correlator_cumulants(BitstringLayout<N> const& bitstring_layout, CumulantLayout<N> const& cumulant_layout, std::vector<real> const& sample_correlators, usize sample_count) {
    auto sample_correlator_cumulants_result = compute_cumulants<N>(bitstring_layout, cumulant_layout, sample_correlators, sample_count, false);
    FGBS_VALIDATE_OR_RETURN_CTX(sample_correlator_cumulants_result, "Failed to compute sample cumulants: ");
    if (sample_correlator_cumulants_result.value().size() != bitstring_layout.len) {
        return Err{"Failed to compute sample cumulants."};
    }
    return Ok<std::vector<real>>{std::move(sample_correlator_cumulants_result.value())};
}

template<usize N>
std::pair<std::vector<LinregressInfo>, std::vector<LinregressInfo>> compute_regression_info(SublistLayout<N> const& sublist_layout, StatsTarget<N> const& stats_target, std::vector<real> const& sample_cumulant_sublists, usize bootstrap_count, UniformRandomInt<usize>& seed_generator) {
    const auto cutoff_order = sublist_layout.cutoff_order;
    const auto order_count = cutoff_order + 1;

    std::vector<LinregressInfo> pearson_info_vec(cutoff_order);
    std::vector<LinregressInfo> spearman_info_vec(cutoff_order);

    for (usize i = 1; i < order_count; ++i) {
        const auto theory_cumulant_sublist = sublist_layout.get_sublist(stats_target.moment_cumulant_sublists, i);
        const auto sample_cumulant_sublist = sublist_layout.get_sublist(sample_cumulant_sublists, i);

        pearson_info_vec[i - 1] = compute_pearson_regression_info(theory_cumulant_sublist, sample_cumulant_sublist, bootstrap_count, static_cast<uint>(seed_generator()));
        spearman_info_vec[i - 1] = compute_spearman_regression_info(theory_cumulant_sublist, sample_cumulant_sublist, bootstrap_count, static_cast<uint>(seed_generator()));
    }

    return std::pair<std::vector<LinregressInfo>, std::vector<LinregressInfo>>{std::move(pearson_info_vec), std::move(spearman_info_vec)};
}

template<usize N>
Result<SampleCumulantStats> compute_sample_cumulant_stats(BitstringLayout<N> const& bitstring_layout, CumulantLayout<N> const& cumulant_layout, SublistLayout<N> const& sublist_layout, StatsTarget<N> const& stats_target, std::vector<Bitstring<N>> const& samples, usize bootstrap_count, uint bootstrap_seed) {
    FGBS_VALIDATE_OR_RETURN(validate_bitstring_layout(bitstring_layout));
    FGBS_VALIDATE_OR_RETURN(validate_cumulant_layout<N>(bitstring_layout, cumulant_layout));
    FGBS_VALIDATE_OR_RETURN(validate_sublist_layout(bitstring_layout, sublist_layout));
    FGBS_VALIDATE_OR_RETURN(validate_stats_target(sublist_layout, stats_target));

    const auto sample_count = samples.size();
    if (sample_count == 0) {
        return Ok<SampleCumulantStats>{SampleCumulantStats{}};
    }

    if (bootstrap_count == 0) {
        return Err{"Bootstrap count must be positive."};
    }

    const auto expected_size = stats_target.moment_cumulant_sublists.size();
    if (expected_size == 0 || bitstring_layout.order_vec.empty() || bitstring_layout.ones_vec.empty() || sublist_layout.sublist_index_vec.empty()) {
        return Err{"Target data does not include statistics input data."};
    }

    auto sample_correlator_sublists_result = estimate_sample_correlator_sublists(sublist_layout, samples);
    FGBS_VALIDATE_OR_RETURN(sample_correlator_sublists_result);
    auto sample_correlator_sublists = std::move(sample_correlator_sublists_result.value());

    auto sample_correlators = from_sublists(bitstring_layout, sublist_layout, sample_correlator_sublists, [](real value, usize) { return value; });

    auto sample_correlator_cumulants_result = compute_sample_correlator_cumulants(bitstring_layout, cumulant_layout, sample_correlators, sample_count);
    FGBS_VALIDATE_OR_RETURN(sample_correlator_cumulants_result);

    auto const& sample_correlator_cumulants = sample_correlator_cumulants_result.value();
    auto sample_moment_cumulant_sublists = to_sublists(bitstring_layout, sublist_layout, sample_correlator_cumulants, [](real value, usize order) {
        return to_moment_cumulant(value, order);
    });

    UniformRandomInt<usize> seed_generator{bootstrap_seed, 0, ULLONG_MAX};
    auto [pearson_info_vec, spearman_info_vec] = compute_regression_info(sublist_layout, stats_target, sample_moment_cumulant_sublists, bootstrap_count, seed_generator);

    return Ok<SampleCumulantStats>{SampleCumulantStats{std::move(pearson_info_vec), std::move(spearman_info_vec), std::move(sample_moment_cumulant_sublists)}};
}

}

#endif
