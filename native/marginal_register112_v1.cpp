#include <cstdint>
#include "gaussian_rng_domain84_v1.hpp"
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

API int cpu_register112(int rows,int cells,int size,unsigned domain,uint64_t seed,uint32_t offset,
        int width,int stride,int threads,const uint64_t* threshold,const uint32_t* alias,
        const uint32_t* words,uint8_t* output){
    if(rows<1 || cells<1 || size<1 || (size&(size-1)) || width<1 || width>4 || threads<1 ||
       uint64_t(cells)*width>uint64_t(stride) || uint64_t(offset)+rows>(uint64_t(1)<<32))return 1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t id=0;id<int64_t(rows)*cells;++id){
        uint32_t row=uint32_t(id/cells),cell=uint32_t(id%cells);
        auto v=philox84({0,11200+domain,offset+row,cell},{uint32_t(seed),uint32_t(seed>>32)});
        uint32_t k=v[0]&(size-1);
        uint64_t cut=((uint64_t(v[1])<<32)|v[2])>>11;
        uint32_t packed=words[cut<threshold[k]?k:alias[k]];
        uint8_t* dst=output+size_t(row)*stride+size_t(cell)*width;
        for(int j=0;j<width;++j){dst[j]=uint8_t(packed);packed>>=8;}
    }
    return 0;
}
