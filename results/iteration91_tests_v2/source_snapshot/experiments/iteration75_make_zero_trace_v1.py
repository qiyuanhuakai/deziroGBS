"""Record every threshold on an all-zero Frugal path without changing its arithmetic."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'results/iteration39_build/generated/frugal_extended_hooks.hpp').read_text(encoding='utf-8');start=source.index('template<usize N>\nPrefixRecord trace_prefix');end=source.index('\n}\n',start)+3;s=source[start:end]
s=s.replace('PrefixRecord trace_prefix','void trace_zero75').replace('std::string const& forced_prefix,','double* thresholds,double* previous,double* proposed,')
old='if (i == forced_prefix.size()) return {threshold.get(0), prev_prob.get(0), zero_next_prob.get(0)};';assert s.count(old)==1;s=s.replace(old,'thresholds[i]=threshold.get(0);previous[i]=prev_prob.get(0);proposed[i]=zero_next_prob.get(0);')
old="const auto mask = forced_prefix[i] == '1' ? get_one<T>() > get_zero<T>() : get_one<T>() < get_zero<T>();";assert s.count(old)==1;s=s.replace(old,'const auto mask = get_one<T>() < get_zero<T>();');s=s.replace('throw std::runtime_error("Prefix must be shorter than the mode count");','return;')
with (ROOT/'native/frugal_zero75_trace_v1.hpp').open('xb') as f:f.write(('#include "frugal_extended_hooks.hpp"\nnamespace fgbs::cpu {\n'+s+'\n}\n').encode())
adapter='''// Event instrumentation only; official sampling remains the included entry.
#include "frugal_cpu75_adapter_v1.cpp"
#include "frugal_zero75_trace_v1.hpp"
template<fgbs::usize N>static void zero75(int cutoff,const double* covariance,double* thresholds,double* previous,double* proposed){
    fgbs::Matrix<fgbs::f64> cov(2*N,2*N);for(fgbs::usize i=0;i<2*N;++i)for(fgbs::usize j=0;j<2*N;++j)cov(i,j)=covariance[i*2*N+j];auto layout=take72(fgbs::generate_bitstring_layout<N>(cutoff));auto cumulants=take72(fgbs::generate_cumulant_layout<N>(cutoff));auto target=take72(fgbs::generate_sampling_target<N>(layout,cumulants,cov));using namespace fgbs::cpu;std::vector<ProbSimd> partial(partial_marginal_arr_size<N>()),elided(elided_marginal_arr_size<N>()),partial_elided(partial_elided_marginal_arr_size<N>()),double_elided(double_elided_marginal_arr_size<N>());trace_zero75<N>(layout,target,thresholds,previous,proposed,partial.data(),elided.data(),partial_elided.data(),double_elided.data());
}
API int fc75_zero(int modes,int cutoff,int threads,const double* covariance,double* thresholds,double* previous,double* proposed,char* error,int cap){try{
    if(cutoff<1||cutoff>std::min(5,modes)||threads<1||threads>256||!covariance||!thresholds||!previous||!proposed)throw std::runtime_error("Invalid zero trace arguments");Threads72 worker_budget(threads);switch(modes){CASES default:throw std::runtime_error("Unsupported zero trace dimension");}return 0;
}catch(const std::exception& e){if(error&&cap>0){auto n=std::min(std::strlen(e.what()),std::size_t(cap-1));std::memcpy(error,e.what(),n);error[n]=0;}return 1;}}
'''
adapter=adapter.replace('CASES',''.join(f'case {n}:zero75<{n}>(cutoff,covariance,thresholds,previous,proposed);break;' for n in (4,8,144,192,288,384)))
with (ROOT/'native/frugal_zero75_adapter_v1.cpp').open('xb') as f:f.write(adapter.encode())
