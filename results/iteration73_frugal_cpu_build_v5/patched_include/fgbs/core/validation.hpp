#ifndef FGBS_CORE_VALIDATION_HPP
#define FGBS_CORE_VALIDATION_HPP

#include "fgbs/core/structs.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/string_utils.hpp"

#include "fgbs/global.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#define FGBS_VALIDATE_OR_RETURN(RESULT)          \
do {                                             \
    auto&& _fgbs_result = (RESULT);              \
    if (!_fgbs_result.ok()) {                    \
        return fgbs::Err{_fgbs_result.error()};  \
    }                                            \
}                                                \
while (false)

#define FGBS_VALIDATE_OR_RETURN_CTX(RESULT, ...)                                 \
do {                                                                             \
    auto&& _fgbs_result = (RESULT);                                              \
    if (!_fgbs_result.ok()) {                                                    \
        return fgbs::Err{fgbs::build_string(__VA_ARGS__, _fgbs_result.error())}; \
    }                                                                            \
}                                                                                \
while (false)

namespace fgbs {

template<usize N>
Result<void> validate_cutoff_order(usize cutoff_order) {
    if (cutoff_order == 0) {
        return Err{"Cutoff order must be positive!"};
    }
    if (cutoff_order > FGBS_MAX_CUTOFF_ORDER) {
        return Err{build_string("Cutoff order ", cutoff_order, " can not be larger than maximum value of ", FGBS_MAX_CUTOFF_ORDER, "!")};
    }
    if (cutoff_order > N) {
        return Err{build_string("Cutoff order ", cutoff_order, " is not supported for sampler size ", N, "!")};
    }
    return Ok<void>{};
}

template<typename MATRIX>
Result<void> validate_cov_matrix(MATRIX&& cov_matrix, usize mode_count) {
    const auto dim = static_cast<usize>(cov_matrix.rows());
    if (cov_matrix.cols() != static_cast<isize>(dim) || dim != 2 * mode_count) {
        return Err{build_string("Covariance matrix must be square with dimension ", 2 * mode_count, " for sampler size ", mode_count, "!")};
    }

    constexpr auto symmetry_tol = 1e-10;
    for (usize i = 0; i < dim; ++i) {
        const auto diag = static_cast<f64>(cov_matrix(i, i));
        if (!std::isfinite(diag)) {
            return Err{build_string("Covariance matrix has non-finite diagonal entry at (", i, ", ", i, ").")};
        }

        for (usize j = i + 1; j < dim; ++j) {
            const auto x = static_cast<f64>(cov_matrix(i, j));
            const auto y = static_cast<f64>(cov_matrix(j, i));
            if (!std::isfinite(x) || !std::isfinite(y)) {
                return Err{build_string("Covariance matrix has non-finite entry at (", i, ", ", j, ") or (", j, ", ", i, ").")};
            }

            const auto scale = std::max(std::fabs(x), std::fabs(y));
            const auto tol = symmetry_tol * (scale > 1.0 ? scale : 1.0);
            if (std::fabs(x - y) > tol) {
                return Err{build_string("Covariance matrix is not symmetric at (", i, ", ", j, ") and (", j, ", ", i, ").")};
            }
        }
    }

    auto shifted_cov_matrix = cov_matrix;
    for (usize i = 0; i < dim; ++i) {
        shifted_cov_matrix(i, i) += 1.0;
    }

    const auto llt = shifted_cov_matrix.llt();
    if (llt.info() != decltype(llt.info())(0)) {
        return Err{"Covariance matrix plus identity must be positive definite."};
    }

    return Ok<void>{};
}

template<usize N>
Result<void> validate_bitstring_layout(BitstringLayout<N> const& bitstring_layout) {
    FGBS_VALIDATE_OR_RETURN(validate_cutoff_order<N>(bitstring_layout.cutoff_order));
    if (bitstring_layout.order_vec.size() != bitstring_layout.len || bitstring_layout.ones_vec.size() != bitstring_layout.cutoff_order * bitstring_layout.len) {
        return Err{"Bitstring layout is missing required bitstring metadata."};
    }
    if (bitstring_layout.layout_bitstring_vec.size() != bitstring_layout.len) {
        return Err{"Bitstring layout is missing required layout-bitstring data."};
    }

    const auto len = bitstring_layout.len;
    const auto cutoff_order = bitstring_layout.cutoff_order;
    const auto ones_size = bitstring_layout.ones_vec.size();

    usize first_invalid_order = len;
    usize first_invalid_ones_index = ones_size;

    #pragma omp parallel for default(none) firstprivate(len, cutoff_order) shared(bitstring_layout) reduction(min : first_invalid_order, first_invalid_ones_index)
    for (usize i = 0; i < len; ++i) {
        if (bitstring_layout.order_vec[i] > cutoff_order && i < first_invalid_order) {
            first_invalid_order = i;
        }

        const auto ones_begin = cutoff_order * i;
        for (usize j = 0; j < cutoff_order; ++j) {
            const auto ones_index = ones_begin + j;
            if (bitstring_layout.ones_vec[ones_index] >= N && ones_index < first_invalid_ones_index) {
                first_invalid_ones_index = ones_index;
            }
        }
    }

    if (first_invalid_order != len) {
        return Err{build_string("Bitstring layout order at index ", first_invalid_order, " exceeds cutoff order.")};
    }
    if (first_invalid_ones_index != ones_size) {
        return Err{build_string("Bitstring layout one-index value at index ", first_invalid_ones_index, " exceeds sampler size.")};
    }
    return Ok<void>{};
}

template<usize N>
Result<void> validate_cumulant_layout(BitstringLayout<N> const& bitstring_layout, CumulantLayout<N> const& cumulant_layout) {
    FGBS_VALIDATE_OR_RETURN(validate_cutoff_order<N>(cumulant_layout.cutoff_order));
    if (cumulant_layout.cutoff_order != bitstring_layout.cutoff_order) {
        return Err{"Bitstring layout and cumulant layout cutoff orders do not match."};
    }

    const auto order_count = cumulant_layout.cutoff_order + 1;
    if (cumulant_layout.cumulant_term_range_vec.size() != order_count + 1 || cumulant_layout.cumulant_order_range_vec.size() != order_count + 1 || cumulant_layout.cumulant_index_range_vec.size() != order_count + 1) {
        return Err{"Cumulant layout is missing required cumulant metadata."};
    }

    if (cumulant_layout.cumulant_term_range_vec[0] != 0 || cumulant_layout.cumulant_order_range_vec[0] != 0 || cumulant_layout.cumulant_index_range_vec[0] != 0) {
        return Err{"Cumulant layout range metadata must start at zero."};
    }

    for (usize i = 0; i < order_count; ++i) {
        const auto term_begin = cumulant_layout.cumulant_term_range_vec[i];
        const auto term_end = cumulant_layout.cumulant_term_range_vec[i + 1];
        const auto order_begin = cumulant_layout.cumulant_order_range_vec[i];
        const auto order_end = cumulant_layout.cumulant_order_range_vec[i + 1];
        const auto index_begin = cumulant_layout.cumulant_index_range_vec[i];
        const auto index_end = cumulant_layout.cumulant_index_range_vec[i + 1];

        if (term_end < term_begin || order_end < order_begin || index_end < index_begin) {
            return Err{"Cumulant layout range metadata must be non-decreasing."};
        }

        if (term_end > cumulant_layout.cumulant_term_vec.size() || order_end > cumulant_layout.cumulant_order_vec.size() || index_end > cumulant_layout.cumulant_index_vec.size()) {
            return Err{"Cumulant layout range metadata exceeds cumulant vector sizes."};
        }

        usize expected_order_count = 0;
        for (usize j = term_begin; j < term_end; ++j) {
            if (cumulant_layout.cumulant_term_vec[j] == 0) {
                return Err{"Cumulant layout contains a zero-arity cumulant term."};
            }
            expected_order_count += cumulant_layout.cumulant_term_vec[j];
        }
        if (expected_order_count != order_end - order_begin) {
            return Err{"Cumulant layout term-range and order-range metadata are inconsistent."};
        }

        usize expected_index_count = 0;
        for (usize j = order_begin; j < order_end; ++j) {
            expected_index_count += cumulant_layout.cumulant_order_vec[j];
        }
        if (expected_index_count != index_end - index_begin) {
            return Err{"Cumulant layout order-range and index-range metadata are inconsistent."};
        }
    }

    if (cumulant_layout.cumulant_term_range_vec[order_count] != cumulant_layout.cumulant_term_vec.size() || cumulant_layout.cumulant_order_range_vec[order_count] != cumulant_layout.cumulant_order_vec.size() || cumulant_layout.cumulant_index_range_vec[order_count] != cumulant_layout.cumulant_index_vec.size()) {
        return Err{"Cumulant layout range metadata does not cover cumulant vectors fully."};
    }

    return Ok<void>{};
}

template<usize N>
Result<void> validate_sublist_layout(BitstringLayout<N> const& bitstring_layout, SublistLayout<N> const& sublist_layout) {
    FGBS_VALIDATE_OR_RETURN(validate_cutoff_order<N>(sublist_layout.cutoff_order));
    if (sublist_layout.cutoff_order != bitstring_layout.cutoff_order) {
        return Err{"Bitstring layout and sublist layout cutoff orders do not match."};
    }
    if (sublist_layout.sublist_index_vec.size() != bitstring_layout.len) {
        return Err{"Sublist layout is missing required bitstring metadata."};
    }

    const auto order_count = sublist_layout.cutoff_order + 1;
    if (sublist_layout.sublist_size_vec.size() != order_count || sublist_layout.sublist_offset_vec.size() != order_count + 1) {
        return Err{"Sublist layout is missing required sublist metadata."};
    }

    if (sublist_layout.sublist_offset_vec[0] != 0) {
        return Err{"Sublist layout has invalid sublist offsets."};
    }
    for (usize i = 0; i < order_count; ++i) {
        const auto range_begin = sublist_layout.sublist_offset_vec[i];
        const auto range_end = sublist_layout.sublist_offset_vec[i + 1];
        if (range_end < range_begin) {
            return Err{"Sublist layout has invalid sublist offsets."};
        }
        if (range_end - range_begin != sublist_layout.sublist_size_vec[i]) {
            return Err{"Sublist layout has inconsistent sublist sizes and offsets."};
        }
    }

    const auto expected_sublist_total = sublist_layout.sublist_offset_vec[order_count];
    if (sublist_layout.layout_bitstring_sublists.size() != expected_sublist_total) {
        return Err{"Sublist layout is missing required layout-bitstring sublists."};
    }
    if (sublist_layout.ones_sublists.size() != sublist_layout.cutoff_order * expected_sublist_total) {
        return Err{"Sublist layout is missing required ones sublists."};
    }

    const auto len = bitstring_layout.len;

    usize first_invalid_order = len;
    usize first_invalid_sublist_index = len;

    #pragma omp parallel for default(none) firstprivate(len, order_count) shared(bitstring_layout, sublist_layout) reduction(min : first_invalid_order, first_invalid_sublist_index)
    for (usize i = 0; i < len; ++i) {
        const auto sublist_index = sublist_layout.sublist_index_vec[i];
        if (sublist_index == INVALID_INDEX) {
            continue;
        }
        const auto order = static_cast<usize>(bitstring_layout.order_vec[i]);
        if (order >= order_count) {
            if (i < first_invalid_order) {
                first_invalid_order = i;
            }
            continue;
        }
        if (sublist_index >= sublist_layout.sublist_size_vec[order]) {
            if (i < first_invalid_sublist_index) {
                first_invalid_sublist_index = i;
            }
        }
    }

    if (first_invalid_order != len) {
        return Err{"Sublist layout references an invalid bitstring order."};
    }
    if (first_invalid_sublist_index != len) {
        return Err{"Sublist layout index metadata contains out-of-range sublist indices."};
    }

    return Ok<void>{};
}

template<usize N>
Result<void> validate_sampling_target(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target) {
    FGBS_VALIDATE_OR_RETURN(validate_bitstring_layout(bitstring_layout));
    if (sampling_target.correlator_cumulant_vec.size() != bitstring_layout.len) {
        return Err{"Target data has invalid correlator-cumulant vector length."};
    }
    if (sampling_target.first_correlator_cumulant_vec.size() != N) {
        return Err{"Target data has invalid first-correlator-cumulant vector length."};
    }
    return Ok<void>{};
}

template<usize N>
Result<void> validate_stats_target(SublistLayout<N> const& sublist_layout, StatsTarget<N> const& stats_target) {
    const auto order_count = sublist_layout.cutoff_order + 1;
    if (sublist_layout.sublist_offset_vec.size() != order_count + 1) {
        return Err{"Cannot validate statistics target without valid sublist layout metadata."};
    }
    if (stats_target.moment_cumulant_sublists.size() != sublist_layout.sublist_offset_vec[order_count]) {
        return Err{"Statistics target has invalid moment-cumulant sublist vector length."};
    }
    return Ok<void>{};
}

}

#endif
