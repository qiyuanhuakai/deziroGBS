#include "gaussian_cycle98_v1.cpp"
#include <bit>
#include <limits>

// The ring has emitted virtual source occupations in its idler slots. Condition
// the physical idler measurement on each occupation after commuting signal loss.
extern "C" __declspec(dllexport) int loss99_idlers(int pairs,int route,int cap,int limit,
        const Word98* cuts,int shots,Word98 seed,Word98 offset,int workers,unsigned char* out) {
    if(pairs<1 || pairs>65536 || (route!=0 && route!=1) || cap<1 || cap>9 ||
       limit<1 || limit>255 || !cuts || !out || shots<1 || workers<1 || workers>64 ||
       offset>0xffffffffULL || offset+shots>0x100000000ULL) return 1;
    int failures=0;
    #pragma omp parallel for num_threads(workers) reduction(+:failures)
    for(int r=0;r<shots;++r) {
        DomainRng84 rng(seed,0,160,offset+r);
        auto idler=out+std::size_t(r)*2*pairs+pairs;
        for(int j=0;j<pairs;++j) {
            int m=idler[j];
            if(m>cap){++failures;break;}
            auto c=cuts+((j%4)*(cap+1)+m)*(limit+1);
            auto u=rng()>>11;
            if(u<c[0]) idler[j]=0;
            else if(!route) idler[j]=1;
            else idler[j]=static_cast<unsigned char>(std::upper_bound(c+1,c+limit+1,u)-c);
        }
    }
    return failures ? 2 : 0;
}

// Each optical eigenvalue pair has an independent total count. Emission marks
// are physical ports. Click records can be written by setting bits directly.
extern "C" __declspec(dllexport) int loss99_sources(int modes,int sources,int limit,int bits,
        int route,const Word98* counts,const Word98* cut,const unsigned* alias,
        int shots,Word98 seed,Word98 offset,int workers,std::uint16_t* out) {
    if(modes<1 || sources<1 || limit<1 || bits<1 || bits>20 || modes>(1<<bits) ||
       std::uint64_t(sources)*limit>65535 || (route!=0 && route!=1) || shots<1 ||
       workers<1 || workers>64 || !counts || !cut || !alias || !out ||
       offset>0xffffffffULL || offset+shots>0x100000000ULL) return 1;
    int size=1<<bits;
    #pragma omp parallel for num_threads(workers)
    for(int r=0;r<shots;++r) {
        DomainRng84 totals(seed,0,161,offset+r),marks(seed,0,162,offset+r);
        auto row=out+std::size_t(r)*modes;
        std::fill(row,row+modes,0);
        for(int j=0;j<sources;++j) {
            auto c=counts+j*(limit+1);
            auto u=totals()>>11;
            if(u<c[0]) continue;
            int n=int(std::upper_bound(c+1,c+limit+1,u)-c);
            for(int i=0;i<n;++i) {
                auto v=marks();
                unsigned col=v&(size-1);
                auto slot=j*size+col;
                unsigned port=(v>>bits)<cut[slot] ? col : alias[slot];
                if(route) ++row[port]; else row[port]=1;
            }
        }
    }
    return 0;
}

// Void probabilities followed by inclusion-exclusion on the absent ports.
extern "C" __declspec(dllexport) int loss99_law(int modes,int sources,const double* marking,
        const double* pos,const double* neg,double* output) {
    if(modes<1 || modes>18 || sources<1 || !marking || !pos || !neg || !output) return 1;
    int size=1<<modes;
    std::vector<long double> dist(size),weights(size);
    for(int j=0;j<sources;++j) {
        for(int i=1;i<size;++i){
            auto port=std::countr_zero(unsigned(i));
            weights[i]=weights[i&(i-1)]+marking[j*modes+port];
            dist[(size-1)^i]-=0.5L*(std::log1p(pos[j]*weights[i])+std::log1p(neg[j]*weights[i]));
        }
    }
    for(auto& p:dist)p=std::exp(p);
    for(int step=1;step<size;step*=2)
        for(int block=0;block<size;block+=2*step)
            for(int i=0;i<step;++i)dist[block+step+i]-=dist[block+i];
    std::copy(dist.begin(),dist.end(),output);
    return 0;
}
