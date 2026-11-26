#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <utility>

using Word=std::uint64_t;
static int choose(DomainRng84& rng,int n){
    Word bound=n,cut=(Word(0)-bound)%bound,u;
    do{u=rng();}while(u<cut);
    return int(u%bound);
}
// One designated branch in each of the first sixteen eight-way splitters.
// Swapping entire channel values preserves total photons and total clicks.
extern "C" __declspec(dllexport) int relabel100(int modes,int width,const Word* keep,int bits,const Word* cuts,
        const unsigned* alias,const unsigned* words,int shots,Word seed,Word offset,int workers,
        std::uint16_t* records,unsigned char* flags){
    if(width<2||width>8||modes<=16*width||!keep||bits<1||bits>16||!cuts||!alias||!words||shots<1||
       offset>0xffffffffULL||offset+shots>0x100000000ULL||workers<1||workers>64||!records||!flags)return 1;
    #pragma omp parallel for num_threads(workers)
    for(int r=0;r<shots;++r){
        DomainRng84 decide(seed,0,173,offset+r),donor(seed,0,174,offset+r);
        auto row=records+std::size_t(r)*modes;unsigned observed=0;
        for(int j=0;j<16;++j)observed|=unsigned(row[width*j]!=0)<<j;
        flags[r]=0;if((decide()>>11)<keep[observed])continue;
        auto u=decide();unsigned col=u&((1u<<bits)-1);
        unsigned target=words[(u>>bits)<cuts[col]?col:alias[col]];
        std::array<std::pair<int,int>,16> swaps;int n=0;bool ok=true;unsigned char flag=1;
        for(int pass=0;pass<2&&ok;++pass){
            unsigned change=pass?(target&~observed):(observed&~target);
            while(change){
                int j=std::countr_zero(change),port=j*width;change&=change-1;
                std::array<int,8> candidates{};int size=0,from=-1;
                for(int k=1;k<width;++k)if(int(row[port+k]>0)==pass)candidates[size++]=port+k;
                if(size)from=candidates[choose(donor,size)];
                else{
                    int region=16*width;int start=choose(donor,region);
                    for(int k=0;k<region&&from<0;++k){
                        int p=(start+k)%region;
                        if(p%width&&int(row[p]>0)==pass)from=p;
                    }
                    if(from<0){
                        int remain=modes-region;start=choose(donor,remain);
                        for(int k=0;k<remain&&from<0;++k){
                            int p=region+(start+k)%remain;
                            if(int(row[p]>0)==pass)from=p;
                        }
                        if(from>=0)flag|=4;
                    }
                }
                if(from<0){ok=false;break;}
                std::swap(row[port],row[from]);swaps[n++]={port,from};
            }
        }
        if(!ok){while(n){auto [a,b]=swaps[--n];std::swap(row[a],row[b]);}flag=3;}
        flags[r]=flag;
    }
    return 0;
}
