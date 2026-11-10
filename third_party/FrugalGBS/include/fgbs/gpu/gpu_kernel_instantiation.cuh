#ifndef FGBS_GPU_KERNEL_INSTANTIATION_CUH
#define FGBS_GPU_KERNEL_INSTANTIATION_CUH

#include "fgbs/gpu/gpu_cumulants_impl.cuh"
#include "fgbs/gpu/gpu_sampler_impl.cuh"
#include "fgbs/gpu/gpu_stats_impl.cuh"

namespace fgbs {

namespace gpu {

#define FGBS_INSTANTIATE_GPU_COMPUTE_CUMULANTS(N) template Result<std::vector<real>> compute_cumulants<N>(BitstringLayout<N> const& bitstring_layout, CumulantLayout<N> const& cumulant_layout, std::vector<real> const& data, usize sample_count, bool from_cumulants)

#define FGBS_INSTANTIATE_GPU_GENERATE_SAMPLES(N) template Result<std::vector<Bitstring<N>>> generate_samples<N>(BitstringLayout<N> const& bitstring_layout, SamplingTarget<N> const& sampling_target, usize sample_count, uint seed)

#define FGBS_INSTANTIATE_GPU_ESTIMATE_SAMPLE_CORRELATOR_SUBLISTS(N) template Result<std::vector<real>> estimate_sample_correlator_sublists<N>(SublistLayout<N> const& sublist_layout, std::vector<Bitstring<N>> const& samples)

#define FGBS_INSTANTIATE_GPU_KERNELS(N)    \
FGBS_INSTANTIATE_GPU_COMPUTE_CUMULANTS(N); \
FGBS_INSTANTIATE_GPU_GENERATE_SAMPLES(N);  \
FGBS_INSTANTIATE_GPU_ESTIMATE_SAMPLE_CORRELATOR_SUBLISTS(N);

}

}

#endif
