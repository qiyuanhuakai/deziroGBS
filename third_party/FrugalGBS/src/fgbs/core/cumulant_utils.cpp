#include "fgbs/core/cumulant_utils.hpp"

#include <vector>

namespace fgbs {

real compute_cumulant_coeff(usize sample_count, Slice<usize> order_slice, std::vector<real> const& ratio_vec, std::vector<std::vector<real>> const& mult_vec) {
    const auto order_count = order_slice.size();
    std::vector<usize> index_vec(order_count, 1);
    bool complete = false;
    real coeff = 0.0;

    while (!complete) {
        usize ratio_index = 0;
        for (usize i = 0; i < order_count; ++i) {
            ratio_index += index_vec[i];
        }

        auto term = ratio_vec[ratio_index];
        for (usize i = 0; i < order_count; ++i) {
            term *= mult_vec[order_slice[i]][index_vec[i]];
        }
        coeff += term;

        for (usize i = order_count; i-- > 0;) {
            if (index_vec[i] < order_slice[i]) {
                ++index_vec[i];
                for (usize j = i + 1; j < order_count; ++j) {
                    index_vec[j] = 1;
                }
                break;
            }
            else if (i == 0) {
                complete = true;
            }
        }
    }

    for (usize i = 0; i < order_count; ++i) {
        coeff *= sample_count;
    }

    return parity(order_count - 1) * coeff;
}

}
