// Sample number-correlated pairs followed by a passive interferometer on one arm.
#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace {
using Complex = std::complex<double>;
struct OpticalNetwork {
    int modes;
    std::vector<Complex> transfer;
    std::vector<std::vector<int>> support;
    std::vector<double> pair_probability;
};
struct Ladder {
    int photons=0;
    std::vector<int> stride,radix;
    std::vector<Complex> amplitude,lowered;
    explicit Ladder(const std::vector<int>& inputs,const std::vector<unsigned>& count) {
        int size=1;
        for(int i:inputs) {
            int n=count[i];photons+=n;stride.push_back(size);radix.push_back(n+1);
            if(n>=65536 || size>65536/(n+1))throw std::length_error("active Fock workspace");
            size*=n+1;
        }
        amplitude.resize(size);amplitude.back()=1.;lowered.resize(size);
    }
    double lower(const OpticalNetwork& net,const std::vector<int>& in,int output) {
        std::fill(lowered.begin(),lowered.end(),Complex(0.));
        for(int k=0;k<int(amplitude.size());++k)if(amplitude[k]!=Complex(0.)) {
            for(int j=0;j<int(in.size());++j) {
                int n=k/stride[j]%radix[j];
                if(n)lowered[k-stride[j]]+=net.transfer[output*net.modes+in[j]]*std::sqrt(double(n))*amplitude[k];
            }
        }
        double norm=0.;for(auto a:lowered)norm+=std::norm(a);return norm;
    }
    void advance(double norm) {
        if(!(norm>0))throw std::domain_error("empty detection");
        double scale=1./std::sqrt(norm);
        for(auto& a:lowered)a*=scale;
        amplitude.swap(lowered);--photons;
    }
};
int choose(const std::vector<double>& weights,double total,DomainRng84& rng) {
    if(!(total>0))throw std::domain_error("empty categorical");
    double u=uniform84(rng)*total,part=0.;int last=0;
    for(int j=0;j<int(weights.size());++j) {
        if(weights[j]>0)last=j;
        part+=weights[j];if(u<part)return j;
    }
    return last;
}
int ancestor(std::vector<int>& link,int a) {
    while(a!=link[a]){link[a]=link[link[a]];a=link[a];}return a;
}
void scatter(const OpticalNetwork& net,const std::vector<int>& inputs,const std::vector<unsigned>& n,
             DomainRng84& rng,unsigned* output) {
    int d=net.modes;
    if(inputs.size()==1) {
        int i=inputs.front();const auto& ports=net.support[i];
        if(ports.size()==1){output[ports[0]]+=n[i];return;}
        std::vector<double> weights;
        for(int b:ports)weights.push_back(std::norm(net.transfer[b*d+i]));
        double sum=std::accumulate(weights.begin(),weights.end(),0.);
        for(unsigned t=0;t<n[i];++t)++output[ports[choose(weights,sum,rng)]];
        return;
    }
    std::vector<int> ports;
    for(int i:inputs)ports.insert(ports.end(),net.support[i].begin(),net.support[i].end());
    std::sort(ports.begin(),ports.end());ports.erase(std::unique(ports.begin(),ports.end()),ports.end());
    Ladder state(inputs,n);std::vector<double> weights(ports.size());
    while(state.photons) {
        double sum=0.;
        for(int j=0;j<int(ports.size());++j)sum+=(weights[j]=state.lower(net,inputs,ports[j]));
        if(std::abs(sum-state.photons)>1e-9*state.photons)throw std::domain_error("detection normalization");
        int j=choose(weights,sum,rng);++output[ports[j]];
        if(state.photons==1)break;
        state.advance(state.lower(net,inputs,ports[j]));
    }
}
}
extern "C" __declspec(dllexport) void* herald96_prepare(int d,const double* matrix,const double* pair,unsigned* error) {
    if(!matrix||!pair||!error||d<1||d>4096)return nullptr;
    try {
        *error=1;auto net=std::make_unique<OpticalNetwork>();net->modes=d;
        net->support.resize(d);net->transfer.resize(std::size_t(d)*d);
        net->pair_probability.assign(pair,pair+d);
        for(int i=0;i<d;++i){
            if(!(pair[i]>=0 && pair[i]<1))return nullptr;
            for(int b=0;b<d;++b){
                auto index=std::size_t(b)*d+i;Complex z(matrix[2*index],matrix[2*index+1]);
                if(!std::isfinite(z.real())||!std::isfinite(z.imag()))return nullptr;
                net->transfer[index]=z;if(z!=Complex(0.))net->support[i].push_back(b);
            }
        }
        for(int i=0;i<d;++i)for(int j=0;j<=i;++j){
            Complex inner=0.;for(int b:net->support[i])inner+=std::conj(net->transfer[b*d+i])*net->transfer[b*d+j];
            if(std::abs(inner-double(i==j))>1e-11)return nullptr;
        }
        *error=0;return net.release();
    }catch(...){return nullptr;}
}
extern "C" __declspec(dllexport) void herald96_free(void* p){delete static_cast<OpticalNetwork*>(p);}
extern "C" __declspec(dllexport) int herald96_draw(const void* ptr,int records,std::uint64_t domain,
        std::uint64_t offset,int workers,unsigned* outputs,unsigned char* status) {
    if(!ptr||!outputs||!status||workers<1||workers>64||records<1||offset>0x100000000ULL||
       std::uint64_t(records)>0x100000000ULL-offset)return 1;
    auto& net=*static_cast<const OpticalNetwork*>(ptr);int d=net.modes;
    #pragma omp parallel for num_threads(workers) if(workers>1) schedule(static)
    for(int r=0;r<records;++r) {
        auto row=outputs+std::size_t(r)*2*d;std::fill(row,row+2*d,0u);status[r]=0;
        try {
            DomainRng84 pair(domain,0,145,offset+r),detect(domain,0,146,offset+r);
            std::vector<unsigned> n(d);std::vector<int> occupied,owner(d,-1),link(d);
            std::iota(link.begin(),link.end(),0);
            for(int i=0;i<d;++i){
                while(uniform84(pair)<net.pair_probability[i]){
                    if(n[i]==65535)throw std::length_error("source count");++n[i];
                }
                if(!n[i])continue;row[d+i]=n[i];occupied.push_back(i);
                for(int b:net.support[i]){
                    if(owner[b]<0)owner[b]=i;else link[ancestor(link,i)]=ancestor(link,owner[b]);
                }
            }
            std::vector<std::vector<int>> parts(d);
            for(int i:occupied)parts[ancestor(link,i)].push_back(i);
            for(int i:occupied)if(!parts[i].empty())scatter(net,parts[i],n,detect,row);
        }catch(const std::length_error&){status[r]=1;}catch(...){status[r]=2;}
    }
    return 0;
}
// Independent callers may inspect the probability of one ordered detection sequence.
extern "C" __declspec(dllexport) double herald96_ordered(const void* ptr,const unsigned* count,
        const int* sequence,int length) {
    if(!ptr||!count||!sequence||length<0)return -1.;
    auto& net=*static_cast<const OpticalNetwork*>(ptr);int d=net.modes;
    try{
        std::vector<unsigned> n(count,count+d);std::vector<int> in;
        for(int i=0;i<d;++i)if(n[i])in.push_back(i);
        Ladder state(in,n);if(state.photons!=length)return -1.;double prob=1.;
        for(int j=0;j<length;++j) {
            if(sequence[j]<0||sequence[j]>=d)return -1.;
            double norm=state.lower(net,in,sequence[j]);if(norm==0)return 0.;
            prob*=norm/state.photons;state.advance(norm);
        }
        return prob;
    }catch(...){return -1.;}
}
