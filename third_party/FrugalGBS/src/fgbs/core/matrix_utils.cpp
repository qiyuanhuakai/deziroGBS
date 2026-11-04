#include "fgbs/core/matrix_utils.hpp"

namespace fgbs {

void reverse_cov_matrix(Matrix<f64>& cov_matrix) {
    const usize dim = cov_matrix.rows();
    const usize n = dim / 2;

    for (usize i = 0; i < n / 2; ++i) {
        const auto x = n - i - 1;
        const auto y = x + n;
        cov_matrix.row(i).head(dim).swap(cov_matrix.row(x).head(dim));
        cov_matrix.col(i).head(dim).swap(cov_matrix.col(x).head(dim));
        cov_matrix.row(n + i).head(dim).swap(cov_matrix.row(y).head(dim));
        cov_matrix.col(n + i).head(dim).swap(cov_matrix.col(y).head(dim));
    }
}

}
