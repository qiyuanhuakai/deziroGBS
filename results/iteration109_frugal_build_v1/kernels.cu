#include "fgbs/gpu/gpu_cumulants_impl.cuh"
#include "fgbs/gpu/gpu_sampler_impl.cuh"
namespace fgbs::gpu {
template Result<std::vector<Bitstring<16>>> generate_samples<16>(BitstringLayout<16> const&, SamplingTarget<16> const&, usize, uint);
template Result<std::vector<real>> compute_cumulants<16>(BitstringLayout<16> const&, CumulantLayout<16> const&, std::vector<real> const&, usize, bool);
}
