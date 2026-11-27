#include <algorithm>
#include <complex>
#include <cstdint>
#include <vector>
#include "gaussian_rng_domain84_v1.hpp"
using Z=std::complex<double>;
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

// The initial categorical draw fixes a coarse joint word. Conditional CDF
// contractions then refine each unresolved cell without rejection.
API int coarse112(int n,int h,int pairs,int radix,int threads,uint64_t seed,uint32_t offset,
        const int* widths,const int* starts,const uint32_t* alias,const double* prob,
        const double* weights,const Z* coarse,const Z* fine,const Z* tail_prefix,
        const int* symbols,uint8_t* output){
    if(n<1||h<2||h>32||pairs<1||threads<1||uint64_t(offset)+n>(uint64_t(1)<<32))return 1;
    int ports=widths[0]+widths[1]+widths[2]+widths[3],bins=h*h*h*h,bad=0;
    #pragma omp parallel for num_threads(threads) reduction(|:bad)
    for(int row=0;row<n;++row){
        DomainRng84 rng(seed,0,30,offset+row),refine(seed,0,31,offset+row);
        uint64_t lim=(uint64_t(0)-uint64_t(bins))%uint64_t(bins),u;
        do{u=rng();}while(u<lim);
        int id=int(u%bins);
        if(uniform84(rng)>=prob[id])id=alias[id];
        int bucket[4],outcome[4];const Z* selected[4];
        for(int g=0;g<4;++g){
            bucket[g]=id%h;id/=h;
            outcome[g]=starts[g]+bucket[g];
            selected[g]=coarse+(size_t(g)*h+bucket[g])*pairs;
        }
        for(int g=0;g<4;++g)if(bucket[g]==h-1 && starts[g+1]-starts[g]>h){
            std::vector<Z> weight(pairs);
            for(int i=0;i<pairs;++i){
                Z v=weights[i];
                for(int j=0;j<4;++j)if(j!=g)v*=selected[j][i];
                weight[i]=v;
            }
            auto dot=[&](const Z* term){
                double z=0;
                for(int i=0;i<pairs;++i)z+=weight[i].real()*term[i].real()-weight[i].imag()*term[i].imag();
                return z;
            };
            double mass=dot(selected[g]);
            if(mass<=0){bad=2;break;}
            double cut=uniform84(refine)*mass;
            int lo=starts[g]+h-1,hi=starts[g+1]-1;
            while(lo<hi){
                int mid=lo+(hi-lo)/2;
                if(dot(tail_prefix+size_t(mid)*pairs)<=cut)lo=mid+1;else hi=mid;
            }
            outcome[g]=lo;
            selected[g]=fine+size_t(lo)*pairs;
        }
        uint8_t* dst=output+size_t(row)*ports;
        for(int g=0;g<4;++g){
            if(outcome[g]>=starts[g+1]){bad=3;break;}
            int code=symbols[outcome[g]];
            for(int i=0;i<widths[g];++i){*dst++=uint8_t(code%radix);code/=radix;}
        }
    }
    return bad;
}

// Matched fixed-register control for the exact, Frugal and projected click laws.
API int categorical112(int n,int bits,int threads,uint64_t seed,uint32_t offset,
        const uint32_t* alias,const double* prob,uint8_t* out){
    if(n<1||bits<1||bits>22||threads<1||uint64_t(offset)+n>(uint64_t(1)<<32))return 1;
    const uint32_t mask=(1u<<bits)-1;
    #pragma omp parallel for num_threads(threads)
    for(int row=0;row<n;++row){
        DomainRng84 rng(seed,0,32,offset+row);
        uint32_t word=uint32_t(rng())&mask;
        if(uniform84(rng)>=prob[word])word=alias[word];
        for(int j=0;j<bits;++j)out[size_t(row)*bits+j]=(word>>j)&1;
    }
    return 0;
}
