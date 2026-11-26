#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <quadmath.h>
#include <vector>

using W99=std::uint64_t;
static unsigned choose99(DomainRng84& rng,unsigned n){
    auto threshold=(W99(0)-W99(n))%n;
    W99 u;do{u=rng();}while(u<threshold);
    return unsigned(u%n);
}
static unsigned alias99(DomainRng84& rng,int bits,const W99* cut,const unsigned* alias){
    auto u=rng();auto col=unsigned(u&((1u<<bits)-1));
    return (u>>bits)<cut[col]?col:alias[col];
}
extern "C" __declspec(dllexport) int mark99_law(int k,int sources,const double* marking,
        const double* pos,const double* neg,double* output){
    if(k<1||k>16||sources<1||!marking||!pos||!neg||!output)return 1;
    int size=1<<k;
    std::vector<__float128> dist(size,1),weights(size);
    for(int j=0;j<sources;++j){
        for(int i=1;i<size;++i){
            weights[i]=weights[i&(i-1)]+(__float128)marking[j*k+std::countr_zero(unsigned(i))];
            auto w=weights[i];
            dist[(size-1)^i]/=sqrtq((1+pos[j]*w)*(1+neg[j]*w));
        }
    }
    for(int step=1;step<size;step*=2)for(int b=0;b<size;b+=2*step)
        for(int i=0;i<step;++i)dist[b+step+i]-=dist[b+i];
    std::copy(dist.begin(),dist.end(),output);return 0;
}

extern "C" __declspec(dllexport) int mark99_draw(int modes,int sources,int limit,int bits,int route,
        const W99* counts,const W99* cuts,const unsigned* aliases,const int* group,int width,
        const W99* keep,const W99* single,int dbits,const W99* dcut,const unsigned* dalias,
        const unsigned* words,int shots,W99 seed,W99 offset,int workers,std::uint16_t* out,unsigned char* flags){
    if(modes<=width||sources<1||limit<1||std::uint64_t(sources)*limit>65535||
       bits<1||bits>20||modes>(1<<bits)||width<1||width>16||dbits<1||dbits>16||
       route<0||route>1||shots<1||workers<1||workers>64||offset>0xffffffffULL||
       offset+shots>0x100000000ULL||!counts||!cuts||!aliases||!group||!keep||!single||
       !dcut||!dalias||!words||!out||!flags)return 1;
    std::vector<int> membership(modes,-1);
    for(int i=0;i<width;++i){
        if(group[i]<0||group[i]>=modes||membership[group[i]]>=0)return 1;
        membership[group[i]]=i;
    }
    int size=1<<bits;
    #pragma omp parallel num_threads(workers)
    {
        std::vector<int> occupied,retained;
        occupied.reserve(256);retained.reserve(256);
        #pragma omp for schedule(static)
        for(int r=0;r<shots;++r){
            DomainRng84 totals(seed,0,161,offset+r),marks(seed,0,162,offset+r),
                adjust(seed,0,165,offset+r),thin(seed,0,166,offset+r),assign(seed,0,167,offset+r);
            auto row=out+std::size_t(r)*modes;std::fill(row,row+modes,0);
            occupied.clear();retained.clear();int total=0;unsigned observed=0;
            for(int j=0;j<sources;++j){
                auto c=counts+j*(limit+1);auto u=totals()>>11;if(u<c[0])continue;
                int n=int(std::upper_bound(c+1,c+limit+1,u)-c);total+=n;
                for(int h=0;h<n;++h){
                    auto port=alias99(marks,bits,cuts+j*size,aliases+j*size);
                    if(!row[port]){
                        occupied.push_back(port);
                        if(membership[port]>=0)observed|=1u<<membership[port];
                    }
                    if(route)++row[port];else row[port]=1;
                }
            }
            unsigned target=observed;unsigned flag=0;
            if((adjust()>>11)>=keep[observed]){
                target=words[alias99(adjust,dbits,dcut,dalias)];flag|=1;
                if(std::popcount(target)>total){target=observed;flag|=2;}
            }
            int released=0;
            for(auto port:occupied)if(membership[port]<0){
                if((thin()>>11)<single[port])retained.push_back(port);
                else{released+=row[port];row[port]=0;}
            }
            // Prioritize the designated marginal if the proposed mask would
            // require more occupied ports than the total number of photons.
            while(int(retained.size())+std::popcount(target)>total){
                auto index=choose99(adjust,unsigned(retained.size()));int port=retained[index];
                released+=row[port];row[port]=0;retained[index]=retained.back();retained.pop_back();flag|=4;
            }
            for(int i=0;i<width;++i){
                int port=group[i];
                if(target&(1u<<i)){
                    if(!row[port]){row[port]=1;--released;}
                    retained.push_back(port);
                }else{released+=row[port];row[port]=0;}
            }
            if(total&&retained.empty()){
                int port;do{port=int(choose99(adjust,modes));}while(membership[port]>=0);
                retained.push_back(port);row[port]=1;--released;flag|=8;
            }
            if(route){
                // Reassign excess photons on retained ports, preserving the mask.
                while(released>0){++row[retained[choose99(assign,unsigned(retained.size()))]];--released;}
                for(auto port:retained)if(released<0){
                    int take=std::min(-released,int(row[port])-1);row[port]-=take;released+=take;
                }
                if(released)flag|=16;
            }
            flags[r]=static_cast<unsigned char>(flag);
        }
    }
    return 0;
}
