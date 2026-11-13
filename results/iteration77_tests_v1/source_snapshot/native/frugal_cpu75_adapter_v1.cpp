// Thin C ABI over Frugal CPU; build uses the recorded sum-inlining ABI workaround.
#include "fgbs/data.hpp"
#include "fgbs/sampler.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <omp.h>
#define API extern "C" __declspec(dllexport)
using Clock72=std::chrono::steady_clock;
template<class T>T take72(fgbs::Result<T>&& result){if(!result.ok())throw std::runtime_error(result.error());return std::move(result.value());}
struct Threads72{int count,dynamic;Threads72(int n):count(omp_get_max_threads()),dynamic(omp_get_dynamic()){omp_set_dynamic(0);omp_set_num_threads(n);}~Threads72(){omp_set_num_threads(count);omp_set_dynamic(dynamic);}};
template<fgbs::usize N>static void run72(int cutoff,int count,unsigned seed,const double* covariance,unsigned char* output,double* timings){
    fgbs::Matrix<fgbs::f64> cov(2*N,2*N);for(fgbs::usize i=0;i<2*N;++i)for(fgbs::usize j=0;j<2*N;++j)cov(i,j)=covariance[i*2*N+j];
    auto start=Clock72::now();auto layout=take72(fgbs::generate_bitstring_layout<N>(cutoff));timings[0]=std::chrono::duration<double>(Clock72::now()-start).count();
    start=Clock72::now();auto cumulants=take72(fgbs::generate_cumulant_layout<N>(cutoff));timings[1]=std::chrono::duration<double>(Clock72::now()-start).count();
    start=Clock72::now();auto target=take72(fgbs::generate_sampling_target<N>(layout,cumulants,cov));timings[2]=std::chrono::duration<double>(Clock72::now()-start).count();
    start=Clock72::now();auto samples=take72(fgbs::generate_samples<N>(layout,target,count,seed));timings[3]=std::chrono::duration<double>(Clock72::now()-start).count();
    if(samples.size()!=std::size_t(count))throw std::runtime_error("Incomplete official Frugal CPU batch");start=Clock72::now();for(int i=0;i<count;++i)for(fgbs::usize j=0;j<N;++j)output[std::size_t(i)*N+j]=fgbs::get_bit<N>(samples[i],j);timings[4]=std::chrono::duration<double>(Clock72::now()-start).count();timings[5]=sizeof(fgbs::real)*8;timings[6]=fgbs::cpu::ProbSimd::size;
}
API int fc73_sample(int modes,int cutoff,int count,unsigned seed,int threads,const double* covariance,unsigned char* output,double* timings,char* error,int cap){try{
    if(cutoff<1||cutoff>std::min(5,modes)||count<0||threads<1||threads>256||!covariance||!output||!timings)throw std::runtime_error("Invalid Frugal CPU adapter arguments");Threads72 worker_budget(threads);std::fill(timings,timings+7,0);
    switch(modes){case 4:run72<4>(cutoff,count,seed,covariance,output,timings);break;case 8:run72<8>(cutoff,count,seed,covariance,output,timings);break;case 144:run72<144>(cutoff,count,seed,covariance,output,timings);break;case 192:run72<192>(cutoff,count,seed,covariance,output,timings);break;case 288:run72<288>(cutoff,count,seed,covariance,output,timings);break;case 384:run72<384>(cutoff,count,seed,covariance,output,timings);break;default:throw std::runtime_error("Frugal CPU adapter supports 4,8,144,192,288,384 modes");}return 0;
}catch(const std::exception& e){if(error&&cap>0){auto n=std::min(std::strlen(e.what()),std::size_t(cap-1));std::memcpy(error,e.what(),n);error[n]=0;}return 1;}}
