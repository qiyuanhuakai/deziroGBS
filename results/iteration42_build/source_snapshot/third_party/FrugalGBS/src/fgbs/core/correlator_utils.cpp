#include "fgbs/core/correlator_utils.hpp"

#include <cmath>
#include <vector>

namespace fgbs {

Matrix<f64> quad_cholesky(Matrix<f64> const& prev_sub_mat_l, std::vector<usize> const& sub_indices, Matrix<f64> const& mat, usize lower_right_quad_start_index) {
    Matrix<f64> inner_sub_mat_l = prev_sub_mat_l(sub_indices, sub_indices);
    const auto mat_size = static_cast<usize>(mat.rows());

    for (usize i = lower_right_quad_start_index; i < mat_size; ++i) {
        for (usize j = lower_right_quad_start_index; j < i; ++j) {
            f64 y = 0.0;
            for (usize k = 0; k < j; ++k) {
                y += inner_sub_mat_l(i, k) * inner_sub_mat_l(j, k);
            }
            inner_sub_mat_l(i, j) = (mat(i, j) - y) / inner_sub_mat_l(j, j);
        }

        f64 y = 0.0;
        for (usize k = 0; k < i; ++k) {
            const f64 z = inner_sub_mat_l(i, k);
            y += z * z;
        }
        inner_sub_mat_l(i, i) = std::sqrt(mat(i, i) - y);
    }

    return inner_sub_mat_l;
}

f64 compute_inner_correlator(Matrix<f64> const& sub_cov_l, Matrix<f64> const& sub_cov, usize original_sub_n, std::vector<usize>& modes) {
    f64 inner_correlator = 0.0;

    const auto mode_count = modes.size();
    const auto start = mode_count == 0 ? 0 : modes[mode_count - 1] + 1;

    const auto sub_dim = static_cast<usize>(sub_cov.rows());

    for (usize i = start; i < original_sub_n; ++i) {
        const auto inner_dim = sub_dim - 2;
        const auto lower_right_quad_start_index = 2 * (i - mode_count);

        std::vector<usize> sub_indices(inner_dim);
        for (usize j = 0; j < lower_right_quad_start_index; ++j) {
            sub_indices[j] = j;
        }
        for (usize j = lower_right_quad_start_index; j < inner_dim; ++j) {
            sub_indices[j] = j + 2;
        }

        const Matrix<f64> inner_cov = sub_cov(sub_indices, sub_indices);
        const auto inner_n = inner_dim / 2;

        const Matrix<f64> inner_cov_shifted = inner_cov + identity_matrix<f64>(inner_dim);
        const Matrix<f64> inner_cov_l = quad_cholesky(sub_cov_l, sub_indices, inner_cov_shifted, lower_right_quad_start_index);
        const auto sqrt_det_inner_cov = std::fabs(triangular_determinant(inner_cov_l));

        modes.push_back(i);
        inner_correlator += parity(original_sub_n - inner_n) * std::pow(2.0, inner_dim) / sqrt_det_inner_cov + compute_inner_correlator(inner_cov_l, inner_cov, original_sub_n, modes);
        modes.pop_back();
    }

    return inner_correlator;
}

}
