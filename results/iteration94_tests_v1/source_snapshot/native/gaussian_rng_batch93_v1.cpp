// Batch the existing injective Philox addresses without changing a random word.
#include "gaussian_rng_domain84_v1.hpp"
#include <cstdio>
#include <random>
extern "C" __declspec(dllexport) int rng93_uniforms(std::uint64_t experiment,unsigned sector,std::uint64_t offset,int count,int modes,double* output,char* error,int capacity){try{
 if(sector>65535||offset>0x100000000ULL||count<0||std::uint64_t(count)>0x100000000ULL-offset||modes<1||modes>16384||!output)throw std::runtime_error("Batch RNG domain/dimension overflow");
 for(int row=0;row<count;++row){DomainRng84 rng(experiment,sector,Click84,offset+row);for(int j=0;j<modes;++j)output[std::size_t(row)*modes+j]=uniform84(rng);}return 0;
 }catch(const std::exception& e){if(error&&capacity>0)std::snprintf(error,capacity,"%s",e.what());return 1;}}
// Frugal75 uses mt19937_64 -> uniform size_t over its full range -> uint.
// Reproduce the actual first worker seed for its fixed one-worker quality ABI.
extern "C" __declspec(dllexport) int frugal93_worker_seeds(int count,const unsigned* outer,unsigned* inner){
 if(count<0||!outer||!inner)return 1;static_assert(sizeof(std::size_t)==8&&sizeof(unsigned)==4);
 for(int i=0;i<count;++i){std::mt19937_64 generator(outer[i]);std::uniform_int_distribution<std::size_t> distribution(0,~std::size_t(0));inner[i]=static_cast<unsigned>(distribution(generator));}return 0;
}
