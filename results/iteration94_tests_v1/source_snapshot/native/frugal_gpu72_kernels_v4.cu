#include "fgbs/gpu/gpu_cumulants_impl.cuh"
#include "fgbs/gpu/gpu_sampler_impl.cuh"
namespace fgbs::gpu {
template Result<std::vector<Bitstring<4>>> generate_samples<4>(BitstringLayout<4> const&, SamplingTarget<4> const&, usize, uint);
template Result<std::vector<real>> compute_cumulants<4>(BitstringLayout<4> const&, CumulantLayout<4> const&, std::vector<real> const&, usize, bool);
template Result<std::vector<Bitstring<8>>> generate_samples<8>(BitstringLayout<8> const&, SamplingTarget<8> const&, usize, uint);
template Result<std::vector<real>> compute_cumulants<8>(BitstringLayout<8> const&, CumulantLayout<8> const&, std::vector<real> const&, usize, bool);
template Result<std::vector<Bitstring<144>>> generate_samples<144>(BitstringLayout<144> const&, SamplingTarget<144> const&, usize, uint);
template Result<std::vector<real>> compute_cumulants<144>(BitstringLayout<144> const&, CumulantLayout<144> const&, std::vector<real> const&, usize, bool);
}
