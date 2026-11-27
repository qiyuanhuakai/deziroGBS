#include "fgbs/gpu/gpu_cumulants_impl.cuh"
#include "fgbs/gpu/gpu_sampler_impl.cuh"
namespace fgbs::gpu {
template Result<std::vector<Bitstring<13>>> generate_samples<13>(BitstringLayout<13> const&, SamplingTarget<13> const&, usize, uint);
template Result<std::vector<real>> compute_cumulants<13>(BitstringLayout<13> const&, CumulantLayout<13> const&, std::vector<real> const&, usize, bool);
}
