#include <array>
#include <complex>
#include <cstdint>
#include <map>

// Coefficients of the unnormalised four-port Fourier polynomial, including their phases.
extern "C" __declspec(dllexport) int cycle98_coefficients(int cap,int capacity,
        std::uint32_t* keys,std::int32_t* re,std::int32_t* im) {
    if(cap<1 || cap>9 || capacity<1 || !keys || !re || !im) return -1;
    using Z=std::complex<int>;
    const std::array<Z,4> units{Z{1,0},Z{0,1},Z{-1,0},Z{0,-1}};
    int size=0;
    for(int a=0;a<=cap;++a) for(int b=0;b<=cap-a;++b)
    for(int c=0;c<=cap-a-b;++c) for(int d=0;d<=cap-a-b-c;++d) {
        int m[]={a,b,c,d};
        std::map<unsigned,Z> poly{{0,Z{1,0}}};
        for(int j=0;j<4;++j) for(int k=0;k<m[j];++k) {
            std::map<unsigned,Z> next;
            for(auto [code,z]:poly) for(int i=0;i<4;++i)
                next[code+(1u<<(4*i))]+=z*units[(i*j)%4];
            poly.swap(next);
        }
        unsigned idler=a+(b<<4)+(c<<8)+(d<<12);
        for(auto [signal,z]:poly) if(z!=Z{}) {
            if(size>=capacity) return -2;
            keys[size]=signal+(idler<<16);re[size]=z.real();im[size]=z.imag();++size;
        }
    }
    return size;
}
