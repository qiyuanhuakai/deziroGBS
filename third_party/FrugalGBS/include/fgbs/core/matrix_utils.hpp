#ifndef FGBS_CORE_MATRIX_UTILS_HPP
#define FGBS_CORE_MATRIX_UTILS_HPP

#include "fgbs/core/type_utils.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>

namespace fgbs {

template<typename T>
using Matrix = Eigen::Matrix<T, -1, -1>;

template<typename T>
Matrix<T> identity_matrix(usize dim) {
    return Matrix<T>::Identity(dim, dim);
}

template<typename MATRIX>
auto triangular_determinant(MATRIX&& mat) {
    return mat.template triangularView<Eigen::Lower>().determinant();
}

void reverse_cov_matrix(Matrix<f64>& cov_matrix);

}

#endif
