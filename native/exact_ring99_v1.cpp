#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>

namespace {
using Word=std::uint64_t;
using Wide=unsigned __int128;
using Z=std::complex<std::int64_t>;
constexpr int max_total=12;
struct Factors {
    std::array<std::array<int,13>,13> bin{};
    std::array<Word,13> fac{1};
    constexpr Factors(){
        for(int n=0;n<=max_total;++n){
            bin[n][0]=bin[n][n]=1;
            if(n)fac[n]=fac[n-1]*n;
            for(int i=1;i<n;++i)bin[n][i]=bin[n-1][i]+bin[n-1][i-1];
        }
    }
};
constexpr Factors f{};
int bs(int n,int m,int k){
    int sum=0;
    for(int j=std::max(0,k-m);j<=std::min(n,k);++j)
        sum+=((n-j)%2 ? -1 : 1)*f.bin[n][j]*f.bin[m][k-j];
    return sum;
}
int mark(int k,int n){return int(k>0)+2*int(k<n);}
Wide uniform_below(Wide bound,DomainRng84& rng){
    Wide reject=(Wide(0)-bound)%bound;
    for(;;){
        Wide high=rng();Wide low=rng();Wide v=(high<<64)|low;
        if(v>=reject)return v%bound;
    }
}
}
extern "C" __declspec(dllexport) int exact99_close2(int route,int bits,int cap,const Word* cut,
        const unsigned* alias,const unsigned* keys,const unsigned* groups,const std::int32_t* coeff,
        int shots,Word seed,Word offset,int workers,const unsigned char* flags,unsigned char* out) {
    if((route!=0&&route!=1)||bits<1||bits>24||cap<1||cap>9||!cut||!alias||!keys||!groups||!coeff||
       shots<1||offset>0xffffffffULL||offset+shots>0x100000000ULL||workers<1||workers>64||!flags||!out)return 1;
    int errors=0;
    #pragma omp parallel for num_threads(workers) reduction(+:errors)
    for(int r=0;r<shots;++r){
        if(!flags[r])continue;
        // Regenerate the two original alias decisions at exactly the same addresses.
        DomainRng84 local(seed,0,157,offset+r),decide(seed,0,164,offset+r);
        int left[2],right[2],total=0;unsigned group[2];
        for(int i=0;i<2;++i){
            auto u=local();unsigned col=u&((1u<<bits)-1);
            unsigned slot=(u>>bits)<cut[col]?col:alias[col];
            auto code=keys[slot];group[i]=groups[slot];
            left[i]=code&15;right[i]=(code>>12)&15;
            for(int j=0;j<4;++j)total+=(code>>(4*j))&15;
        }
        if(total>max_total)continue;
        auto row=out+16*std::size_t(r);
        int n0=right[0]+left[1],n1=right[1]+left[0];
        int observed=route ? row[3] : int(row[3])+2*int(row[4]);
        std::array<Wide,13> w{};
        for(int a=0;a<=n0;++a){
            if((route?a:mark(a,n0))!=observed)continue;
            for(int b=0;b<=n1;++b){
                Z amp{};
                for(int t=-std::min(right[0],right[1]);t<=std::min(left[0],left[1]);++t){
                    auto term=[&](int i){
                        auto k=2*(group[i]*(cap+1)+right[i]+t);
                        return Z(coeff[k],coeff[k+1]);
                    };
                    std::int64_t factors=bs(right[0]+t,left[1]-t,a)*bs(right[1]+t,left[0]-t,b);
                    amp+=term(0)*term(1)*factors;
                }
                Wide re=std::uint64_t(std::abs(amp.real())),im=std::uint64_t(std::abs(amp.imag()));
                w[route?b:mark(b,n1)]+=(re*re+im*im)*f.fac[a]*f.fac[n0-a]*f.fac[b]*f.fac[n1-b];
            }
        }
        Wide sum=0;for(auto z:w)sum+=z;
        if(!sum){++errors;continue;}
        auto u=uniform_below(sum,decide);
        int k=0;while(u>=w[k]){u-=w[k];++k;}
        row[7]=route?k:(k&1);row[0]=route?n1-k:((k>>1)&1);
    }
    return errors?2:0;
}
