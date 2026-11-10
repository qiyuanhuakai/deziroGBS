#ifndef FGBS_GPU_CUMULANTS_CUH
#define FGBS_GPU_CUMULANTS_CUH

#include "fgbs/core/structs.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/type_utils.hpp"

#include <vector>

namespace fgbs {

namespace gpu {

template<usize N>
Result<std::vector<real>> compute_cumulants(BitstringLayout<N> const& bitstring_layout, CumulantLayout<N> const& cumulant_layout, std::vector<real> const& data, usize sample_count, bool from_cumulants);

}

}

#endif
