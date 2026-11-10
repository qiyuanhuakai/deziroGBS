#ifndef FGBS_GPU_STATS_CUH
#define FGBS_GPU_STATS_CUH

#include "fgbs/core/structs.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/type_utils.hpp"

#include <vector>

namespace fgbs {

namespace gpu {

template<usize N>
Result<std::vector<real>> estimate_sample_correlator_sublists(SublistLayout<N> const& sublist_layout, std::vector<Bitstring<N>> const& samples);

}

}

#endif
