#ifndef FGBS_CUMULANTS_HPP
#define FGBS_CUMULANTS_HPP

#if FGBS_GPU_ENABLED
#include "fgbs/gpu/gpu_cumulants.cuh"
#else
#include "fgbs/cpu/cpu_cumulants.hpp"
#endif

#include "fgbs/core/validation.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/type_utils.hpp"

#include <vector>

namespace fgbs {

template<usize N>
Result<std::vector<real>> compute_cumulants(BitstringLayout<N> const& bitstring_layout, CumulantLayout<N> const& cumulant_layout, std::vector<real> const& data, usize sample_count, bool from_cumulants) {
    FGBS_VALIDATE_OR_RETURN(validate_bitstring_layout(bitstring_layout));
    FGBS_VALIDATE_OR_RETURN(validate_cumulant_layout<N>(bitstring_layout, cumulant_layout));
    if (data.size() != bitstring_layout.len) {
        return Err{"Input data has invalid length for cumulant computation."};
    }
    if (!from_cumulants && sample_count != 0 && sample_count < cumulant_layout.cutoff_order) {
        return Err{build_string("Finite-sample cumulant estimation requires at least ", cumulant_layout.cutoff_order, " samples for cutoff order ", cumulant_layout.cutoff_order, ".")};
    }

    #if FGBS_GPU_ENABLED
    auto cumulants_result = gpu::compute_cumulants<N>(bitstring_layout, cumulant_layout, data, sample_count, from_cumulants);
    #else
    auto cumulants_result = cpu::compute_cumulants<N>(bitstring_layout, cumulant_layout, data, sample_count, from_cumulants);
    #endif

    FGBS_VALIDATE_OR_RETURN_CTX(cumulants_result, "Failed to compute cumulants: ");
    return cumulants_result;
}

}

#endif
