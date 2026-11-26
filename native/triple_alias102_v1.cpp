#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <cstdint>

extern "C" __declspec(dllexport) int triple102_draw(
        int ports,int shots,int workers,int route,std::uint64_t seed,std::uint64_t offset,
        int bits,const std::uint64_t* cuts,const std::uint32_t* alternate,
        const std::uint32_t* labels,std::uint16_t* output){
    if(ports<1 || shots<1 || workers<1 || workers>64 || route<0 || route>1 ||
       bits<1 || bits>24 || offset>0xffffffffULL || std::uint64_t(shots)>0x100000000ULL-offset) return 1;
    std::uint32_t mask=(std::uint32_t(1)<<bits)-1;
    #pragma omp parallel for schedule(static) num_threads(workers)
    for(int r=0;r<shots;++r){
        DomainRng84 rng(seed,0,route?186:185,offset+r);
        auto row=output+std::size_t(r)*ports;
        for(int base=0;base<ports;base+=3){
            auto word=rng();
            auto k=std::uint32_t(word)&mask;
            auto code=labels[(word>>bits)<cuts[k]?k:alternate[k]];
            int width=std::min(3,ports-base);
            for(int j=0;j<width;++j){
                row[base+j]=route?((code>>(8*j))&255):((code>>j)&1);
            }
        }
    }
    return 0;
}
