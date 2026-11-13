// Read-only export of the pinned official second-order sampling target.
#include "fgbs/data.hpp"
#include "fgbs/layout.hpp"
#include "fgbs/sampler.hpp"
#include "fgbs/core/consts.hpp"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
template<class T>T checked(fgbs::Result<T>&&r){if(!r.ok())throw std::runtime_error(r.error());return std::move(r.value());}
std::string hex(double x){std::ostringstream s;s<<std::hexfloat<<x;return s.str();}
template<fgbs::usize N>int run(std::string path){
    std::ifstream f(path);if(!f)throw std::runtime_error("Input missing");fgbs::Matrix<double> V(2*N,2*N);std::string line;std::size_t row=0;
    while(std::getline(f,line)){if(row==2*N)throw std::runtime_error("Extra row");std::replace(line.begin(),line.end(),',',' ');std::istringstream s(line);for(std::size_t j=0;j<2*N;++j){double x;if(!(s>>x)||!std::isfinite(x))throw std::runtime_error("Invalid covariance");V(row,j)=x;}std::string extra;if(s>>extra)throw std::runtime_error("Extra column");++row;}if(row!=2*N)throw std::runtime_error("Missing row");
    auto bits=checked(fgbs::generate_bitstring_layout<N>(2));auto cumulants=checked(fgbs::generate_cumulant_layout<N>(2));auto target=checked(fgbs::generate_sampling_target<N>(bits,cumulants,V));
    std::cout<<"{\"modes\":"<<N<<",\"cutoff\":2,\"first_hex\":[";
    for(std::size_t i=0;i<N;++i){if(i)std::cout<<',';std::cout<<'"'<<hex(target.first_correlator_cumulant_vec[i])<<'"';}
    std::cout<<"],\"pairs\":[";bool first=true;
    for(std::size_t i=0;i<N;++i)for(std::size_t j=0;j<i;++j){if(!first)std::cout<<',';first=false;auto offset=fgbs::gbs_index_arrays<N>[2][i]+fgbs::gbs_index_arrays<N>[1][j];std::cout<<'['<<j<<','<<i<<",\""<<hex(target.correlator_cumulant_vec[offset])<<"\"]";}
    std::cout<<"]}\n";return 0;
}
int main(int argc,char**argv){try{if(argc!=3)throw std::runtime_error("Usage: target_dump covariance.csv modes");omp_set_dynamic(0);omp_set_num_threads(1);int n=std::stoi(argv[2]);switch(n){case 2:return run<2>(argv[1]);case 10:return run<10>(argv[1]);case 144:return run<144>(argv[1]);default:throw std::runtime_error("Unsupported mode count");}}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}}
