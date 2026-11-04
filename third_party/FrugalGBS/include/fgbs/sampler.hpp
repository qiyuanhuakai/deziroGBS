#ifndef FGBS_SAMPLER_HPP
#define FGBS_SAMPLER_HPP

#if FGBS_GPU_ENABLED
#include "fgbs/gpu/gpu_sampler.cuh"
#else
#include "fgbs/cpu/cpu_sampler.hpp"
#endif

#include "fgbs/core/validation.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/type_utils.hpp"

#include "fgbs/layout.hpp"

#include <vector>

namespace fgbs {

template<usize N>
Result<std::vector<Bitstring<N>>> generate_samples(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, usize sample_count, uint seed) {
    FGBS_VALIDATE_OR_RETURN(validate_sampling_target<N>(bitstring_layout, sampling_target));

    #if FGBS_GPU_ENABLED
    auto samples_result = gpu::generate_samples<N>(bitstring_layout, sampling_target, sample_count, seed);
    #else
    auto samples_result = cpu::generate_samples<N>(bitstring_layout, sampling_target, sample_count, seed);
    #endif

    FGBS_VALIDATE_OR_RETURN_CTX(samples_result, "Failed to generate samples: ");

    auto samples = std::move(samples_result.value());
    if (samples.size() != sample_count) {
        return Err{build_string("Failed to generate requested sample count (requested ", sample_count, ", got ", samples.size(), ").")};
    }
    return Ok<std::vector<Bitstring<N>>>{std::move(samples)};
}

}

#endif
