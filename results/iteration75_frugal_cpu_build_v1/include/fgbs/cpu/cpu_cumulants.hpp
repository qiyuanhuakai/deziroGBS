#ifndef FGBS_CPU_CUMULANTS_HPP
#define FGBS_CPU_CUMULANTS_HPP

#include "fgbs/core/cumulant_utils.hpp"
#include "fgbs/core/consts.hpp"
#include "fgbs/core/structs.hpp"
#include "fgbs/core/result.hpp"

#include <vector>

namespace fgbs {

namespace cpu {

template<usize N>
Result<std::vector<real>> compute_cumulants(BitstringLayout<N> const& bitstring_layout, CumulantLayout<N> const& cumulant_layout, std::vector<real> const& data, usize sample_count, bool from_cumulants) {
    const auto cutoff_order = cumulant_layout.cutoff_order;
    if (cutoff_order == 0) {
        return Ok<std::vector<real>>{std::vector<real>{0.0}};
    }
    if (from_cumulants && sample_count != 0) {
        return Err{"Finite-sample bias correction for cumulant input is not yet implemented."};
    }

    const auto layout_size = bitstring_layout.len;
    std::vector<real> cumulants(layout_size);

    const auto coeff_vec = compute_cumulant_coeff_vec(cumulant_layout, sample_count, from_cumulants);

    #pragma omp parallel for default(shared) firstprivate(cutoff_order, layout_size)
    for (usize i = 0; i < layout_size; ++i) {
        const auto order = bitstring_layout.order_vec[i];
        const auto ones = bitstring_layout.get_ones_ptr(i);

        real sum = 0.0;

        const auto term_range_begin = cumulant_layout.cumulant_term_range_vec[order], term_range_end = cumulant_layout.cumulant_term_range_vec[order + 1];

        for (usize j = term_range_begin, k = cumulant_layout.cumulant_order_range_vec[order], l = cumulant_layout.cumulant_index_range_vec[order]; j < term_range_end; ++j) {
            const auto t = cumulant_layout.cumulant_term_vec[j], r = k + t;

            real term = 1.0;

            for (; k < r; ++k) {
                usize index = 0;
                for (usize m = 0, s = l + cumulant_layout.cumulant_order_vec[k]; l < s; ++l, ++m) {
                    index += gbs_index_arrays<N>[cutoff_order - m][ones[cumulant_layout.cumulant_index_vec[s - m - 1]]];
                }
                term *= data[index];
            }

            sum += coeff_vec[j] * term;
        }

        cumulants[i] = sum;
    }

    return Ok<std::vector<real>>{std::move(cumulants)};
}

}

}

#endif
