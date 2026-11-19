// Extend only the dimension dispatch. Official SIMD algorithm, prior ABI
// workaround and audited prefix implementation remain unchanged.
#include "frugal_prefix81_adapter_v1.cpp"
API int fc83_sample(int cutoff,int count,unsigned seed,int threads,const double* covariance,unsigned char* output,double* timings,char* error,int cap){try{
 if(cutoff<1||cutoff>2||count<0||threads<1||threads>64||!covariance||!output||!timings)throw std::runtime_error("Full Frugal adapter limits: 1152 modes, cutoff 1..2");Threads72 guard(threads);std::fill(timings,timings+7,0);run72<1152>(cutoff,count,seed,covariance,output,timings);return 0;
 }catch(const std::exception& e){if(error&&cap>0){auto n=std::min(std::strlen(e.what()),std::size_t(cap-1));std::memcpy(error,e.what(),n);error[n]=0;}return 1;}}
API int fc83_tree(int cutoff,int depth,int threads,const double* covariance,double* thresholds,double* previous,double* proposed,double* stats,char* error,int cap){try{
 if(cutoff<1||cutoff>2||depth<1||depth>8||threads<1||threads>64||!covariance||!thresholds||!previous||!proposed||!stats)throw std::runtime_error("Invalid full Frugal prefix arguments");Threads72 guard(threads);prefix81<1152>(cutoff,depth,covariance,thresholds,previous,proposed,stats);return 0;
 }catch(const std::exception& e){if(error&&cap>0){auto n=std::min(std::strlen(e.what()),std::size_t(cap-1));std::memcpy(error,e.what(),n);error[n]=0;}return 1;}}
