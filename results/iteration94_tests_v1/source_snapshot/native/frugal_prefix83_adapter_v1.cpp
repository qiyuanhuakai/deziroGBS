// Full 1152-mode official target, prefix-only tracing. The frozen trace routine
// accesses only prefix indices; allocate only the prefix's triangular buffers.
// No extra complete samples are generated during this probability audit.
#include "frugal_prefix81_adapter_v1.cpp"
API int fc83_prefix_only(int cutoff,int depth,int threads,const double* covariance,double* thresholds,double* stats,char* error,int cap){try{
 constexpr fgbs::usize N=1152;if(cutoff<1||cutoff>2||depth<3||depth>8||threads<1||threads>64||!covariance||!thresholds||!stats)throw std::runtime_error("Invalid full target prefix audit");Threads72 guard(threads);
 fgbs::Matrix<fgbs::f64> cov(2*N,2*N);for(fgbs::usize i=0;i<2*N;++i)for(fgbs::usize j=0;j<2*N;++j)cov(i,j)=covariance[i*2*N+j];auto layout=take72(fgbs::generate_bitstring_layout<N>(cutoff));auto cumulants=take72(fgbs::generate_cumulant_layout<N>(cutoff));auto target=take72(fgbs::generate_sampling_target<N>(layout,cumulants,cov));using namespace fgbs::cpu;
 auto two=std::size_t(depth)*(depth-1)/2,three=std::size_t(depth)*(depth-1)*(depth-2)/6;std::vector<ProbSimd> partial(two),elided(two),partial_elided(three),double_elided(three);
 for(int length=0;length<depth;++length)for(int bits=0;bits<(1<<length);++bits){std::string prefix;for(int i=0;i<length;++i)prefix.push_back((bits>>i)&1?'1':'0');auto r=trace_prefix<N>(layout,target,prefix,partial.data(),elided.data(),partial_elided.data(),double_elided.data());thresholds[(1<<length)-1+bits]=r.threshold;}
 stats[0]=(2*two+2*three)*sizeof(ProbSimd);stats[1]=N;stats[2]=depth;return 0;
 }catch(const std::exception& e){if(error&&cap>0){auto n=std::min(std::strlen(e.what()),std::size_t(cap-1));std::memcpy(error,e.what(),n);error[n]=0;}return 1;}}
