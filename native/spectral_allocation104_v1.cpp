#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

using Word=std::uint64_t;
static unsigned alias104(Word u,int bits,const Word* thresholds,const unsigned* fallback){
    unsigned i=unsigned(u&((Word(1)<<bits)-1));
    return (u>>bits)<thresholds[i]?i:fallback[i];
}
extern "C" __declspec(dllexport) int spectral104_draw(int m,int spectra,int rows,int pnr,int workers,
        Word seed,unsigned off,int port_bits,const int* nbits,const int* starts,const Word* totals,
        const unsigned* nalt,const Word* cuts,const unsigned* alt,unsigned* output){
    if(m<=0||spectra<=0||spectra>65536||rows<1||workers<1||port_bits<1||port_bits>30||
       Word(off)+rows>(Word(1)<<32))return 1;
    #pragma omp parallel for num_threads(workers) schedule(static)
    for(int i=0;i<rows;++i){
        auto* row=output+std::size_t(i)*m;
        std::fill_n(row,m,0U);
        for(int s=0;s<spectra;++s){
            DomainRng84 total(seed,s,199,Word(off)+i);
            DomainRng84 route(seed,s,200,Word(off)+i);
            unsigned n=alias104(total(),nbits[s],totals+starts[s],nalt+starts[s]);
            std::size_t base=std::size_t(s)<<port_bits;
            for(unsigned j=0;j<n;++j){
                unsigned port=alias104(route(),port_bits,cuts+base,alt+base);
                if(pnr)++row[port];else row[port]=1;
            }
        }
    }
    return 0;
}
