#ifndef FGBS_DATA_HPP
#define FGBS_DATA_HPP

#include "fgbs/core/correlator_utils.hpp"
#include "fgbs/core/cumulant_utils.hpp"
#include "fgbs/core/consts.hpp"
#include "fgbs/core/validation.hpp"
#include "fgbs/core/matrix_utils.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/sublist_utils.hpp"
#include "fgbs/core/type_utils.hpp"

#include "fgbs/cumulants.hpp"
#include "fgbs/layout.hpp"

#include <utility>
#include <vector>

namespace fgbs {

template<usize N>
Result<SamplingTarget<N>> generate_sampling_target(BitstringLayout<N> const& bitstring_layout, CumulantLayout<N> const& cumulant_layout, Matrix<f64> const& cov_matrix) {
    FGBS_VALIDATE_OR_RETURN(validate_bitstring_layout(bitstring_layout));
    FGBS_VALIDATE_OR_RETURN(validate_cumulant_layout<N>(bitstring_layout, cumulant_layout));
    FGBS_VALIDATE_OR_RETURN(validate_cov_matrix(cov_matrix, N));

    auto correlator_vec = compute_correlator_vec<N>(bitstring_layout, cov_matrix);

    auto correlator_cumulants_result = compute_cumulants<N>(bitstring_layout, cumulant_layout, correlator_vec, 0, false);
    FGBS_VALIDATE_OR_RETURN_CTX(correlator_cumulants_result, "Failed to compute correlator cumulants: ");

    auto correlator_cumulant_vec = std::move(correlator_cumulants_result.value());
    if (correlator_cumulant_vec.size() != bitstring_layout.len) {
        return Err{"Failed to compute correlator cumulants."};
    }

    std::vector<real> first_correlator_cumulant_vec(N);
    for (usize i = 0; i < N; ++i) {
        first_correlator_cumulant_vec[i] = correlator_cumulant_vec[bitstring_layout.sum_index_arr[i]];
    }

    return Ok<SamplingTarget<N>>{SamplingTarget<N>{std::move(correlator_cumulant_vec), std::move(first_correlator_cumulant_vec)}};
}

template<usize N>
Result<std::vector<real>> generate_correlator_vec(BitstringLayout<N> const& bitstring_layout, Matrix<f64> const& cov_matrix) {
    FGBS_VALIDATE_OR_RETURN(validate_bitstring_layout(bitstring_layout));
    FGBS_VALIDATE_OR_RETURN(validate_cov_matrix(cov_matrix, N));
    return Ok<std::vector<real>>{compute_correlator_vec<N>(bitstring_layout, cov_matrix)};
}

template<usize N>
Result<std::vector<real>> generate_moment_cumulant_vec(BitstringLayout<N> const& bitstring_layout, CumulantLayout<N> const& cumulant_layout, std::vector<real> const& correlator_vec) {
    FGBS_VALIDATE_OR_RETURN(validate_bitstring_layout(bitstring_layout));
    FGBS_VALIDATE_OR_RETURN(validate_cumulant_layout<N>(bitstring_layout, cumulant_layout));

    if (correlator_vec.size() != bitstring_layout.len) {
        return Err{"Correlator vector has invalid length."};
    }

    auto correlator_cumulants_result = compute_cumulants<N>(bitstring_layout, cumulant_layout, correlator_vec, 0, false);
    FGBS_VALIDATE_OR_RETURN_CTX(correlator_cumulants_result, "Failed to compute correlator cumulants: ");

    auto correlator_cumulants = std::move(correlator_cumulants_result.value());
    if (correlator_cumulants.size() != bitstring_layout.len) {
        return Err{"Failed to compute correlator cumulants."};
    }

    std::vector<real> moment_cumulant_vec(bitstring_layout.len);

    #pragma omp parallel for default(none) shared(moment_cumulant_vec, bitstring_layout, correlator_cumulants)
    for (usize i = 0; i < bitstring_layout.len; ++i) {
        moment_cumulant_vec[i] = to_moment_cumulant(correlator_cumulants[i], bitstring_layout.order_vec[i]);
    }

    return Ok<std::vector<real>>{std::move(moment_cumulant_vec)};
}

template<usize N>
Result<std::vector<real>> generate_moment_vec(BitstringLayout<N> const& bitstring_layout, CumulantLayout<N> const& cumulant_layout, std::vector<real> const& moment_cumulant_vec) {
    FGBS_VALIDATE_OR_RETURN(validate_bitstring_layout(bitstring_layout));
    FGBS_VALIDATE_OR_RETURN(validate_cumulant_layout<N>(bitstring_layout, cumulant_layout));

    if (moment_cumulant_vec.size() != bitstring_layout.len) {
        return Err{"Moment-cumulant vector has invalid length."};
    }

    auto moments_result = compute_cumulants<N>(bitstring_layout, cumulant_layout, moment_cumulant_vec, 0, true);
    FGBS_VALIDATE_OR_RETURN_CTX(moments_result, "Failed to compute moments: ");

    if (moments_result.value().size() != bitstring_layout.len) {
        return Err{"Failed to compute moments."};
    }

    return Ok<std::vector<real>>{std::move(moments_result.value())};
}

template<usize N>
Result<StatsTarget<N>> generate_stats_target(BitstringLayout<N> const& bitstring_layout, SublistLayout<N> const& sublist_layout, std::vector<real> const& moment_cumulant_vec) {
    FGBS_VALIDATE_OR_RETURN(validate_bitstring_layout(bitstring_layout));
    FGBS_VALIDATE_OR_RETURN(validate_sublist_layout(bitstring_layout, sublist_layout));

    if (moment_cumulant_vec.size() != bitstring_layout.len) {
        return Err{"Moment-cumulant vector has invalid length."};
    }

    auto moment_cumulant_sublists = to_sublists(bitstring_layout, sublist_layout, moment_cumulant_vec, [](real value, usize) { return value; });

    return Ok<StatsTarget<N>>{StatsTarget<N>{std::move(moment_cumulant_sublists)}};
}

}

#endif
