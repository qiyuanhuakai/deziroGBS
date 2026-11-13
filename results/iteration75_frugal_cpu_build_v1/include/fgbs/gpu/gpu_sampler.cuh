#ifndef FGBS_GPU_SAMPLER_CUH
#define FGBS_GPU_SAMPLER_CUH

#include "fgbs/core/structs.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/type_utils.hpp"

#include <vector>

namespace fgbs {

namespace gpu {

template<usize N>
Result<std::vector<Bitstring<N>>> generate_samples(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, usize sample_count, uint seed);

}

}

#endif
