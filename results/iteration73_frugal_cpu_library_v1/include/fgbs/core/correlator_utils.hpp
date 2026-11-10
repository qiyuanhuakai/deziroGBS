#ifndef FGBS_CORE_CORRELATOR_UTILS_HPP
#define FGBS_CORE_CORRELATOR_UTILS_HPP

#include "fgbs/core/structs.hpp"
#include "fgbs/core/math_utils.hpp"
#include "fgbs/core/matrix_utils.hpp"
#include "fgbs/core/slice.hpp"
#include "fgbs/core/type_utils.hpp"

#include <cmath>
#include <vector>

namespace fgbs {

Matrix<f64> quad_cholesky(Matrix<f64> const& prev_sub_mat_l, std::vector<usize> const& sub_indices, Matrix<f64> const& mat, usize lower_right_quad_start_index);

f64 compute_inner_correlator(Matrix<f64> const& sub_cov_l, Matrix<f64> const& sub_cov, usize original_sub_n, std::vector<usize>& modes);

template<usize N>
f64 compute_correlator(Matrix<f64> const& cov_matrix, usize order, Slice<upto<N>> ones) {
    std::vector<usize> cov_indices;
    for (usize i = 0; i < order; ++i) {
        const auto j = ones[i];
        cov_indices.push_back(j);
        cov_indices.push_back(j + N);
    }

    const auto sub_dim = cov_indices.size();
    const Matrix<f64> sub_cov = cov_matrix(cov_indices, cov_indices);
    const auto sub_n = sub_dim / 2;

    const Matrix<f64> sub_cov_l((sub_cov + identity_matrix<f64>(sub_dim)).llt().matrixL());
    const auto sqrt_det_sub_cov = std::fabs(triangular_determinant(sub_cov_l));

    std::vector<usize> modes;
    return std::pow(2.0, sub_dim) / sqrt_det_sub_cov + compute_inner_correlator(sub_cov_l, sub_cov, sub_n, modes);
}

template<usize N>
std::vector<real> compute_correlator_vec(BitstringLayout<N> const& bitstring_layout, Matrix<f64> const& cov_matrix) {
    std::vector<real> correlator_vec(bitstring_layout.len);

    #pragma omp parallel for default(none) shared(correlator_vec, cov_matrix, bitstring_layout)
    for (usize i = 0; i < bitstring_layout.len; ++i) {
        correlator_vec[i] = static_cast<real>(compute_correlator<N>(cov_matrix, bitstring_layout.order_vec[i], bitstring_layout.get_ones_slice(i)));
    }

    return correlator_vec;
}

}

#endif
