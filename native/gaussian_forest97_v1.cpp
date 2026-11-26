// Sample independent cell tables, then apply number-conserving terminal beamsplitters.
// Each random word used for an alias decision has independent column and coin bits.
#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <array>
#include <complex>
#include <unordered_map>
// Integer polynomial coefficients for four Fourier modes, with the common 1/2 removed.
extern "C" __declspec(dllexport) int forest97_cell(int cap,int capacity,std::uint32_t* keys,std::uint64_t* numerators){
    if(cap<0||cap>9||capacity<1||!keys||!numerators)return -1;
    try{
        std::array<std::uint64_t,10> fac{1};for(int i=1;i<=9;++i)fac[i]=fac[i-1]*i;
        using Z=std::complex<std::int64_t>;const Z units[]={Z(1,0),Z(0,1),Z(-1,0),Z(0,-1)};
        int pos=0;
        for(int a=0;a<=cap;++a)for(int b=0;b<=cap-a;++b)for(int c=0;c<=cap-a-b;++c)
        for(int d=0;d<=cap-a-b-c;++d){
            int n[]={a,b,c,d};unsigned packed=a|(b<<4)|(c<<8)|(d<<12);
            std::unordered_map<unsigned,Z> terms{{0,Z(1,0)}};
            for(int j=0;j<4;++j)for(int t=0;t<n[j];++t){
                std::unordered_map<unsigned,Z> next;
                for(auto [key,val]:terms)for(int i=0;i<4;++i)next[key+(1u<<(4*i))]+=val*units[(i*j)&3];
                terms.swap(next);
            }
            for(auto [key,val]:terms){
                std::uint64_t weight=std::uint64_t(val.real()*val.real()+val.imag()*val.imag());
                if(!weight)continue;
                for(int i=0;i<4;++i)weight*=fac[(key>>(4*i))&15];
                if(pos>=capacity)return -2;
                keys[pos]=key|(packed<<16);numerators[pos++]=weight;
            }
        }
        return pos;
    }catch(...){return -3;}
}
extern "C" __declspec(dllexport) int forest97_draw(int cells,int route,int size_bits,int cap,int pairs_cap,
        const std::uint64_t* cut,const std::uint32_t* alias,const std::uint32_t* payload,
        const std::uint64_t* bridge,int records,std::uint64_t experiment,std::uint64_t offset,int workers,unsigned char* out){
    if(cells<1||cells>16384||route<0||route>1||size_bits<1||size_bits>24||cap<1||cap>15||
       pairs_cap<1||pairs_cap>15||!cut||!alias||!payload||!bridge||!out||records<1||workers<1||workers>64||
       offset>0x100000000ULL||std::uint64_t(records)>0x100000000ULL-offset)return 1;
    int mask=(1<<size_bits)-1,d=4*cells,stride=route?pairs_cap+1:cap+1;
    #pragma omp parallel for num_threads(workers) if(workers>1) schedule(static)
    for(int r=0;r<records;++r){
        DomainRng84 cell_rng(experiment,0,155,offset+r),edge_rng(experiment,0,156,offset+r);
        auto row=out+std::size_t(r)*2*d;
        for(int c=0;c<cells;++c){
            std::uint64_t u=cell_rng();int i=int(u&mask);
            auto code=payload[(u>>size_bits)<cut[i]?i:alias[i]];int pos=4*c;
            if(route){
                for(int j=0;j<4;++j){row[pos+j]=(code>>(4*j))&15;row[d+pos+j]=(code>>(16+4*j))&15;}
            }else{
                row[pos]=code&15;row[pos+3]=(code>>4)&15;code>>=8;
                row[pos+1]=code&1;row[pos+2]=(code>>1)&1;
                for(int j=0;j<4;++j)row[d+pos+j]=(code>>(j+2))&1;
            }
        }
        for(int c=0;c<cells-1;++c){
            int j=4*c+3,n=row[j],m=row[j+1],sum=n+m;
            if(!sum)continue;
            if(route){
                auto cuts=bridge+(n*stride+m)*(2*pairs_cap+1);
                int k=int(std::upper_bound(cuts,cuts+sum+1,edge_rng()>>11)-cuts);
                row[j]=k;row[j+1]=sum-k;
            }else{
                auto cuts=bridge+2*(n*stride+m);auto u=edge_rng()>>11;
                row[j]=u>=cuts[0];row[j+1]=u<cuts[0]||u>=cuts[1];
            }
        }
        if(!route){row[0]=row[0]>0;row[d-1]=row[d-1]>0;}
    }
    return 0;
}
