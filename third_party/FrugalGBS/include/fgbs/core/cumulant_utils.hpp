#ifndef FGBS_CORE_CUMULANT_UTILS_HPP
#define FGBS_CORE_CUMULANT_UTILS_HPP

#include "fgbs/core/structs.hpp"
#include "fgbs/core/math_utils.hpp"
#include "fgbs/core/slice.hpp"
#include "fgbs/core/type_utils.hpp"

#include <vector>

namespace fgbs {

inline real to_moment_cumulant(real correlator_cumulant, usize order) {
    return order == 1 ? real{0.5} - real{0.5} * correlator_cumulant : std::pow(real{-0.5}, order) * correlator_cumulant;
}

real compute_cumulant_coeff(usize sample_count, Slice<usize> order_slice, std::vector<real> const& ratio_vec, std::vector<std::vector<real>> const& mult_vec);

template<usize N>
std::vector<real> compute_cumulant_coeff_vec(CumulantLayout<N> const& cumulant_layout, usize sample_count, bool from_cumulants) {
    auto const& term_vec = cumulant_layout.cumulant_term_vec;
    const auto term_count = term_vec.size();

    std::vector<real> coeff_vec(term_count, 1.0);

    if (from_cumulants) {
        return coeff_vec;
    }

    if (sample_count == usize{0}) {
        for (usize i = 0; i < term_count; ++i) {
            coeff_vec[i] = parity_factorial(term_vec[i] - 1);
        }
    }
    else {
        const auto cutoff_order = cumulant_layout.cutoff_order;
        std::vector<real> ratio_vec(cutoff_order + 1);

        for (usize i = 1; i <= cutoff_order; ++i) {
            real ratio = 1.0;
            for (usize j = 1; j <= i - 1; ++j) {
                ratio *= j;
                ratio /= sample_count - i + j;
            }
            ratio_vec[i] = ratio / sample_count;
        }

        std::vector<real> fact_vec(cutoff_order + 1);
        fact_vec[0] = 1.0;

        std::vector<std::vector<real>> mult_vec(cutoff_order + 1, std::vector<real>(cutoff_order + 1));
        mult_vec[0][0] = 1.0;

        for (usize i = 1; i <= cutoff_order; ++i) {
            fact_vec[i] = i * fact_vec[i - 1];

            for (usize j = 1; j <= i; ++j) {
                mult_vec[i][j] = fact_vec[j - 1] * (j * mult_vec[i - 1][j] + mult_vec[i - 1][j - 1]);
            }
        }

        for (usize i = 0, j = 0; i < term_count; ++i) {
            const auto order_count = term_vec[i];
            coeff_vec[i] = compute_cumulant_coeff(sample_count, Slice<usize>{&cumulant_layout.cumulant_order_vec[j], order_count}, ratio_vec, mult_vec);
            j += order_count;
        }
    }

    return coeff_vec;
}

}

#endif
