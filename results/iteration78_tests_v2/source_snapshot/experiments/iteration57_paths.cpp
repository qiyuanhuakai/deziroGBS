// Iteration57: direct complete-path probes; pinned public generator and hooks unchanged.
#include "fgbs/data.hpp"
#include "fgbs/layout.hpp"
#include "fgbs/sampler.hpp"
#include "frugal_extended_hooks.hpp"
#include "frugal_extended_modes.hpp"
#include "iteration57_path_hooks.hpp"
#include <algorithm>
#include <chrono>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point s){return std::chrono::duration<double>(Clock::now()-s).count();}
std::string hex(double x){std::ostringstream s;s<<std::hexfloat<<x;return s.str();}
std::uint64_t number(std::string const& s){if(s.empty()||s.front()=='-')throw std::runtime_error("Nonnegative integer required");std::size_t n;auto v=std::stoull(s,&n);if(n!=s.size())throw std::runtime_error("Invalid integer");return v;}
struct Options {std::string input,probe_paths;std::size_t modes=0,cutoff=2,samples=0,max_layout=20000000,prefix_depth=0;unsigned int seed=0;int threads=1;bool audit=false,prefix_tree=false,reverse=false;};
Options options(int argc,char**argv){Options o;for(int i=1;i<argc;++i){std::string f=argv[i];if(f=="--audit"){o.audit=true;continue;}if(f=="--prefix-tree"){o.prefix_tree=true;continue;}if(f=="--reverse-prefixes"){o.reverse=true;continue;}if(f=="--help"){std::cout<<"frugal_extended --input covariance.csv --modes N --cutoff K --samples M --seed S --threads T [--audit --prefix-tree --prefix-depth D --reverse-prefixes --max-layout L]\nfrugal_extended --rng-probe words.txt\n";std::exit(0);}if(++i>=argc)throw std::runtime_error("Missing value");std::string v=argv[i];if(f=="--input")o.input=v;else if(f=="--probe-paths")o.probe_paths=v;else if(f=="--modes")o.modes=number(v);else if(f=="--cutoff")o.cutoff=number(v);else if(f=="--samples")o.samples=number(v);else if(f=="--max-layout")o.max_layout=number(v);else if(f=="--prefix-depth")o.prefix_depth=number(v);else if(f=="--seed"){auto n=number(v);if(n>UINT_MAX)throw std::runtime_error("Seed overflow");o.seed=n;}else if(f=="--threads"){auto n=number(v);if(!n||n>1024)throw std::runtime_error("Thread count out of range");o.threads=n;}else throw std::runtime_error("Unknown option: "+f);}if(o.input.empty())throw std::runtime_error("Input required");if(o.samples>1000000)throw std::runtime_error("Sample output budget exceeded");if(o.audit&&o.samples>2000000/std::max(std::size_t(1),o.modes))throw std::runtime_error("Audit record budget exceeded");if(o.prefix_depth&&!o.prefix_tree)throw std::runtime_error("Prefix depth requires prefix-tree");if(o.prefix_tree){if(!o.prefix_depth)o.prefix_depth=o.modes;if(o.prefix_depth>16||o.prefix_depth>o.modes)throw std::runtime_error("Prefix depth must be <=min(16,modes)");}return o;}
fgbs::Matrix<fgbs::f64> read_covariance(std::string const& path,std::size_t modes){std::ifstream f(path);if(!f)throw std::runtime_error("Cannot open covariance");std::size_t d=2*modes,row=0;fgbs::Matrix<fgbs::f64> v(d,d);std::string line;while(std::getline(f,line)){if(line.empty())continue;if(row==d)throw std::runtime_error("Too many rows");std::replace(line.begin(),line.end(),',',' ');std::istringstream s(line);for(std::size_t j=0;j<d;++j){double x;if(!(s>>x)||!std::isfinite(x))throw std::runtime_error("Finite square covariance required");v(row,j)=x;}std::string extra;if(s>>extra)throw std::runtime_error("Too many columns");++row;}if(row!=d)throw std::runtime_error("Wrong row count");return v;}
template<class T>T checked(fgbs::Result<T>&&r){if(!r.ok())throw std::runtime_error(r.error());return std::move(r.value());}
struct FixedWord {using result_type=std::uint64_t;std::uint64_t value,calls=0;static constexpr result_type min(){return 0;}static constexpr result_type max(){return UINT64_MAX;}result_type operator()(){++calls;return value;}};
int rng_probe(char const*path){std::ifstream f(path);if(!f)throw std::runtime_error("Cannot open words");std::cout<<"{\"mapping\":\"GCC uniform_real_distribution<double>, full-range uint64 engine\",\"rows\":[";std::string s;bool first=true;while(f>>s){auto w=number(s);FixedWord e{w};std::uniform_real_distribution<double>d(0,1);double u=d(e);if(!first)std::cout<<',';first=false;std::cout<<"[\""<<w<<"\",\""<<hex(u)<<"\","<<e.calls<<']';}std::cout<<"]}\n";return 0;}

template<fgbs::usize N>int run(Options const&o){
    if(o.cutoff<1||o.cutoff>5||o.cutoff>N)throw std::runtime_error("Cutoff must be 1..min(5,N)");
    fgbs::BitstringLayout<N> estimate(o.cutoff);if(estimate.len>o.max_layout)throw std::runtime_error("Declared layout-entry budget exceeded before allocation");
    auto start=Clock::now();auto covariance=read_covariance(o.input,N);double input_time=elapsed(start);start=Clock::now();auto bits=checked(fgbs::generate_bitstring_layout<N>(o.cutoff));double layout_time=elapsed(start);start=Clock::now();auto cumulants=checked(fgbs::generate_cumulant_layout<N>(o.cutoff));double cumulant_time=elapsed(start);start=Clock::now();auto target=checked(fgbs::generate_sampling_target<N>(bits,cumulants,covariance));double target_time=elapsed(start);
    start=Clock::now();auto samples=checked(fgbs::generate_samples<N>(bits,target,o.samples,o.seed));double sample_time=elapsed(start);
    std::vector<fgbs::cpu::AuditRecord> records;double audit_time=0;bool identical=true;
    if(o.audit){start=Clock::now();auto again=checked(fgbs::cpu::generate_samples_audited<N>(bits,target,o.samples,o.seed,records));identical=samples==again;audit_time=elapsed(start);if(!identical)throw std::runtime_error("Instrumented replay differs from the official samples");}
    std::vector<std::pair<std::string,fgbs::cpu::PrefixRecord>> prefixes;double prefix_time=0;bool prefix_reverse_verified=false;
    if(o.prefix_tree){start=Clock::now();using namespace fgbs::cpu;std::vector<ProbSimd> partial(partial_marginal_arr_size<N>()),elided(elided_marginal_arr_size<N>()),partial_elided(partial_elided_marginal_arr_size<N>()),double_elided(double_elided_marginal_arr_size<N>());for(std::size_t stage=0;stage<o.prefix_depth;++stage){std::size_t len=o.reverse?o.prefix_depth-1-stage:stage;for(std::size_t step=0;step<(std::size_t(1)<<len);++step){std::size_t mask=o.reverse?(std::size_t(1)<<len)-1-step:step;std::string p(len,'0');for(std::size_t j=0;j<len;++j)p[j]='0'+((mask>>j)&1);auto r=trace_prefix<N>(bits,target,p,partial.data(),elided.data(),partial_elided.data(),double_elided.data());prefixes.emplace_back(p,r);}}for(auto it=prefixes.rbegin();it!=prefixes.rend();++it){auto r=trace_prefix<N>(bits,target,it->first,partial.data(),elided.data(),partial_elided.data(),double_elided.data());auto const& old=it->second;if(hex(r.threshold)!=hex(old.threshold)||hex(r.previous)!=hex(old.previous)||hex(r.proposed)!=hex(old.proposed))throw std::runtime_error("Reverse prefix trace differs");}prefix_reverse_verified=true;prefix_time=elapsed(start);}
    std::vector<std::pair<std::string,std::vector<fgbs::cpu::PrefixRecord>>> probes;
    double probe_time=0;std::size_t probe_control_checks=0,public_path_checks=0;
    if(!o.probe_paths.empty()) {
        start=Clock::now();using namespace fgbs::cpu;
        std::ifstream file(o.probe_paths);if(!file)throw std::runtime_error("Cannot open probe paths");
        std::vector<ProbSimd> partial(partial_marginal_arr_size<N>()),elided(elided_marginal_arr_size<N>()),partial_elided(partial_elided_marginal_arr_size<N>()),double_elided(double_elided_marginal_arr_size<N>());
        auto same=[](PrefixRecord const&a,PrefixRecord const&b){return hex(a.threshold)==hex(b.threshold)&&hex(a.previous)==hex(b.previous)&&hex(a.proposed)==hex(b.proposed);};
        std::string path;
        while(file>>path){
            if(probes.size()>=2048)throw std::runtime_error("Probe output budget exceeded");
            auto trace=trace_path<N>(bits,target,path,partial.data(),elided.data(),partial_elided.data(),double_elided.data());
            for(std::size_t depth:{std::size_t(1),std::size_t(16),std::size_t(48),std::size_t(96),std::size_t(144)})if(depth<=N){auto r=trace_prefix<N>(bits,target,path.substr(0,depth-1),partial.data(),elided.data(),partial_elided.data(),double_elided.data());if(!same(r,trace[depth-1]))throw std::runtime_error("Original prefix probe mismatch");++probe_control_checks;}
            probes.emplace_back(path,std::move(trace));
        }
        for(auto it=probes.rbegin();it!=probes.rend();++it){auto trace=trace_path<N>(bits,target,it->first,partial.data(),elided.data(),partial_elided.data(),double_elided.data());for(std::size_t j=0;j<N;++j)if(!same(trace[j],it->second[j]))throw std::runtime_error("Reverse path probe mismatch");}
        if(o.audit)for(std::size_t i=0;i<samples.size();++i){std::string path(N,'0');for(std::size_t j=0;j<N;++j)path[j]='0'+fgbs::get_bit<N>(samples[i],j);auto trace=trace_path<N>(bits,target,path,partial.data(),elided.data(),partial_elided.data(),double_elided.data());for(std::size_t j=0;j<N;++j){auto const&r=records[i*N+j];if(!same(trace[j],PrefixRecord{r.threshold,r.previous,r.proposed}))throw std::runtime_error("Public audit path probe mismatch");++public_path_checks;}}
        probe_time=elapsed(start);
    }
    std::vector<std::uint64_t> counts(N+1);for(auto const&s:samples){std::size_t n=0;for(std::size_t j=0;j<N;++j)n+=fgbs::get_bit<N>(s,j);++counts[n];}
    std::uint64_t saturated=0,nonfinite=0;for(std::size_t s=0;s<samples.size()&&o.audit;++s)for(std::size_t j=0;j<N;++j){auto q=records[s*N+j].threshold;saturated+=q<0||q>1;nonfinite+=!std::isfinite(q);}
    std::cout<<std::setprecision(17)<<"{\"schema_version\":1,\"upstream_commit\":\"fa8b796f2afebd4845a2e335e457735952617ef8\",\"official_sampler_unchanged\":true,\"modes\":"<<N<<",\"cutoff\":"<<o.cutoff<<",\"samples\":"<<samples.size()<<",\"seed\":"<<o.seed<<",\"threads\":"<<omp_get_max_threads()<<",\"simd_lanes\":"<<fgbs::cpu::ProbSimd::size<<",\"prefix_depth\":"<<o.prefix_depth<<",\"prefix_reverse_verified\":"<<(prefix_reverse_verified?"true":"false")<<",\"layout_entries\":"<<bits.len<<",\"max_layout\":"<<o.max_layout<<",\"timings_seconds\":{\"input\":"<<input_time<<",\"layout\":"<<layout_time<<",\"cumulants\":"<<cumulant_time<<",\"target\":"<<target_time<<",\"official_sampling\":"<<sample_time<<",\"audit_replay\":"<<audit_time<<",\"prefix_tree\":"<<prefix_time<<"},\"outputs\":[";
    for(std::size_t i=0;i<samples.size();++i){if(i)std::cout<<',';std::cout<<'"';for(std::size_t j=0;j<N;++j)std::cout<<int(fgbs::get_bit<N>(samples[i],j));std::cout<<'"';}std::cout<<"],\"count_histogram\":[";for(std::size_t j=0;j<=N;++j){if(j)std::cout<<',';std::cout<<counts[j];}
    std::cout<<"],\"audit_enabled\":"<<(o.audit?"true":"false")<<",\"audit_identical_outputs\":"<<(o.audit&&identical?"true":"false")<<",\"audit_saturated_steps\":"<<saturated<<",\"audit_nonfinite_steps\":"<<nonfinite<<",\"audit_columns\":[\"uniform_hex\",\"zero_threshold_hex\",\"previous_hex\",\"proposed_zero_hex\"],\"audit_paths\":[";
    for(std::size_t i=0;i<samples.size()&&o.audit;++i){if(i)std::cout<<',';std::cout<<'[';for(std::size_t j=0;j<N;++j){if(j)std::cout<<',';auto const&r=records[i*N+j];std::cout<<"[\""<<hex(r.uniform)<<"\",\""<<hex(r.threshold)<<"\",\""<<hex(r.previous)<<"\",\""<<hex(r.proposed)<<"\"]";}std::cout<<']';}
    std::cout<<"],\"prefix_tree\":[";for(std::size_t i=0;i<prefixes.size();++i){if(i)std::cout<<',';auto const&[p,r]=prefixes[i];std::cout<<"{\"prefix\":\""<<p<<"\",\"threshold_hex\":\""<<hex(r.threshold)<<"\",\"previous_hex\":\""<<hex(r.previous)<<"\",\"proposed_hex\":\""<<hex(r.proposed)<<"\"}";}std::cout<<"],\"probe_seconds\":"<<probe_time<<",\"probe_control_checks\":"<<probe_control_checks<<",\"public_path_checks\":"<<public_path_checks<<",\"probe_reverse_verified\":"<<(!o.probe_paths.empty()?"true":"false")<<",\"path_probes\":[";
    for(std::size_t i=0;i<probes.size();++i){if(i)std::cout<<',';auto const&[p,trace]=probes[i];std::cout<<"{\"output\":\""<<p<<"\",\"steps\":[";for(std::size_t j=0;j<trace.size();++j){if(j)std::cout<<',';auto const&r=trace[j];std::cout<<"[\""<<hex(r.threshold)<<"\",\""<<hex(r.previous)<<"\",\""<<hex(r.proposed)<<"\"]";}std::cout<<"]}";}std::cout<<"]}\n";return 0;
}
}
int main(int argc,char**argv){try{static_assert(std::numeric_limits<double>::is_iec559&&std::numeric_limits<double>::digits==53);static_assert(fgbs::mt19937::min()==0&&fgbs::mt19937::max()==UINT64_MAX);if(std::fegetround()!=FE_TONEAREST)throw std::runtime_error("Nearest rounding required");if(argc==3&&std::string(argv[1])=="--rng-probe")return rng_probe(argv[2]);auto o=options(argc,argv);omp_set_dynamic(0);omp_set_num_threads(o.threads);switch(o.modes){
#define GBSLAB39_CASE(N) case N:return run<N>(o);
GBSLAB_FRUGAL_EXTENDED_MODES(GBSLAB39_CASE)
#undef GBSLAB39_CASE
default:throw std::runtime_error("Mode count not compiled");}}catch(std::exception const&e){std::cerr<<"Frugal extended adapter: "<<e.what()<<'\n';return 1;}}
