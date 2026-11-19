// Prepared threshold probabilities of independent connected optical components.
#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
namespace {
struct ComponentClick {
    std::vector<std::vector<std::uint64_t>> cuts;
    std::vector<std::vector<int>> ports;
    int modes;
};
}
extern "C" __declspec(dllexport) void* component96_prepare(int m,int groups,const int* sizes,
        const int* ports,const double* probabilities) {
    if(m<1||groups<1||groups>m||!sizes||!ports||!probabilities)return nullptr;
    try {
        auto net=std::make_unique<ComponentClick>();net->modes=m;std::vector<bool> seen(m);
        for(int i=0;i<groups;++i) {
            int d=sizes[i];if(d<1||d>18)return nullptr;
            for(int j=0;j<d;++j){int a=ports[j];if(a<0||a>=m||seen[a])return nullptr;seen[a]=true;}
            net->ports.emplace_back(ports,ports+d);ports+=d;int n=1<<d;
            auto& cuts=net->cuts.emplace_back(n);long double total=0,partial=0;
            for(int j=0;j<n;++j){
                double v=probabilities[j];if(!std::isfinite(v)||v<0)return nullptr;total+=v;
            }
            if(std::abs(total-1)>1e-8)return nullptr;
            for(int j=0;j<n;++j) {
                partial+=probabilities[j];cuts[j]=std::llround(std::ldexp(partial/total,53));
            }
            cuts.back()=1ULL<<53;probabilities+=n;
        }
        if(std::find(seen.begin(),seen.end(),false)!=seen.end())return nullptr;
        return net.release();
    }catch(...){return nullptr;}
}
extern "C" __declspec(dllexport) void component96_free(void* p){delete static_cast<ComponentClick*>(p);}
extern "C" __declspec(dllexport) int component96_draw(const void* ptr,int n,std::uint64_t experiment,
        std::uint64_t offset,int workers,unsigned char* output) {
    if(!ptr||workers<1||workers>64||n<1||!output||offset>0x100000000ULL||
        std::uint64_t(n)>0x100000000ULL-offset)return 1;
    auto& net=*static_cast<const ComponentClick*>(ptr);
    #pragma omp parallel for num_threads(workers) if(workers>1) schedule(static)
    for(int r=0;r<n;++r) {
        DomainRng84 rng(experiment,0,147,offset+r);
        auto row=output+std::size_t(r)*net.modes;
        for(std::size_t j=0;j<net.ports.size();++j) {
            auto& cuts=net.cuts[j];int x=int(std::upper_bound(cuts.begin(),cuts.end(),rng()>>11)-cuts.begin());
            for(int a:net.ports[j]){row[a]=x&1;x>>=1;}
        }
    }
    return 0;
}
extern "C" __declspec(dllexport) int component96_cuts(const void* ptr,int group,std::uint64_t* out){
    if(!ptr||!out)return 1;
    auto& net=*static_cast<const ComponentClick*>(ptr);if(group<0||group>=int(net.cuts.size()))return 1;
    auto& cuts=net.cuts[group];std::copy(cuts.begin(),cuts.end(),out);return 0;
}
