#ifndef FGBS_CORE_COV_MATRIX_IO_HPP
#define FGBS_CORE_COV_MATRIX_IO_HPP

#include "fgbs/core/file_utils.hpp"
#include "fgbs/core/matrix_utils.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/string_utils.hpp"

#include <exception>
#include <string>
#include <utility>

namespace fgbs {

template<typename FILENAME>
Result<Matrix<f64>> get_cov_matrix(FILENAME&& file_name) {
    auto lines_result = get_file_lines(std::forward<FILENAME>(file_name));
    if (!lines_result.ok()) {
        return Err{lines_result.error()};
    }

    auto const& lines = lines_result.value();
    if (lines.empty()) {
        return Err{"Covariance matrix input is empty."};
    }

    const auto dim = lines.size();
    Matrix<f64> cov_matrix(dim, dim);

    for (usize i = 0; i < dim; ++i) {
        const auto line_split = split(lines[i], ',');
        if (line_split.size() != dim) {
            return Err{build_string("Covariance matrix must be square: row ", i, " has ", line_split.size(), " values, expected ", dim, '.')};
        }

        for (usize j = 0; j < dim; ++j) {
            try {
                cov_matrix(i, j) = std::stod(to_str(line_split[j]));
            } catch (std::exception const& e) {
                return Err{build_string("Invalid covariance value at row ", i, ", column ", j, ": ", e.what())};
            }
        }
    }

    return Ok<Matrix<f64>>{std::move(cov_matrix)};
}

}

#endif
