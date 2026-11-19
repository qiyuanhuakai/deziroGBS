// Compile the reference window planner and add immutable integer sampling tables.
#include "gaussian_click_window95_v1.cpp"
#include <cstring>
#include <thread>
namespace {
struct Choice95 { std::uint64_t threshold; unsigned char alternate; };
struct FastTile95 {
    int first,width,history,next_history;
    std::vector<Choice95> choices;
};
struct Ensemble95 {
    int m;
    std::vector<std::vector<FastTile95>> components;
    std::vector<double> weights;
    std::vector<int> output_ports;
    std::uint64_t unpack[256]{};
};
void alias95(const double* cdf,int size,Choice95* output) {
    std::vector<int> small,large;
    std::vector<double> scaled(size);
    for(int i=0;i<size;++i) {
        scaled[i]=size*(cdf[i]-(i?cdf[i-1]:0.));
        (scaled[i]<1?small:large).push_back(i);
    }
    if(cdf[size-1]!=1) throw std::domain_error("empty or unnormalized context");
    while(!small.empty()&&!large.empty()) {
        int a=small.back(),b=large.back();small.pop_back();large.pop_back();
        output[a]={static_cast<std::uint64_t>(std::ceil(std::ldexp(scaled[a],53))),static_cast<unsigned char>(b)};
        scaled[b]+=scaled[a]-1;
        (scaled[b]<1?small:large).push_back(b);
    }
    for(auto* rest:{&small,&large}) for(int i:*rest) output[i]={1ULL<<53,static_cast<unsigned char>(i)};
}
}
extern "C" __declspec(dllexport) void* ensemble95_prepare(int n,void** plans,const double* weights) {
    if(n<1||n>64||!plans||!weights) return nullptr;
    try {
        auto e=std::make_unique<Ensemble95>();
        double sum=0;
        for(int i=0;i<n;++i) {
            if(!plans[i]||!std::isfinite(weights[i])||weights[i]<=0) return nullptr;
            sum+=weights[i];e->weights.push_back(sum);
            const auto& p=*static_cast<const Plan*>(plans[i]);
            if(i && e->m!=p.modes) return nullptr;
            e->m=p.modes;
            auto& comp=e->components.emplace_back();
            for(std::size_t j=0;j<p.tiles.size();++j) {
                const auto& t=p.tiles[j];
                if(t.width>8) return nullptr;
                auto& f=comp.emplace_back(FastTile95{t.first,t.width,t.history,
                    j+1<p.tiles.size()?p.tiles[j+1].history:0,{}});
                int size=1<<t.width;
                f.choices.resize(t.cdf.size());
                for(int h=0;h<(1<<t.history);++h)
                    alias95(t.cdf.data()+h*size,size,f.choices.data()+h*size);
            }
        }
        if(std::abs(sum-1)>1e-12) return nullptr;
        for(double& w:e->weights) w/=sum;
        for(int word=0;word<256;++word) {
            unsigned char bytes[8];
            for(int j=0;j<8;++j) bytes[j]=(word>>j)&1;
            std::memcpy(&e->unpack[word],bytes,8);
        }
        return e.release();
    } catch(const std::exception&) {return nullptr;}
}
extern "C" __declspec(dllexport) void ensemble95_destroy(void* p) {delete static_cast<Ensemble95*>(p);}
extern "C" __declspec(dllexport) int ensemble95_output_order(void* ptr,const int* ports) {
    if(!ptr||!ports) return 1;
    auto& e=*static_cast<Ensemble95*>(ptr);
    std::vector<int> v(ports,ports+e.m),check=v;
    std::sort(check.begin(),check.end());
    for(int i=0;i<e.m;++i) if(check[i]!=i) return 1;
    if(check!=v) e.output_ports=std::move(v);else e.output_ports.clear();
    return 0;
}
extern "C" __declspec(dllexport) int ensemble95_order(int m,int sectors,const double* q,int* order) {
    if(m<1||m>4096||sectors<1||sectors>8||!q||!order) return 1;
    std::vector<bool> used(m,false);
    order[0]=0;used[0]=true;
    for(int pos=1;pos<m;++pos) {
        int selected=-1;double best=-1;
        for(int j=0;j<m;++j) if(!used[j]) {
            double score=0;
            for(int back=1;back<=std::min(4,pos);++back) {
                int i=order[pos-back];
                for(int s=0;s<sectors;++s) {
                    Eigen::Map<const Dense> a(q+std::size_t(s)*4*m*m,2*m,2*m);
                    for(int k=0;k<2;++k) for(int l=0;l<2;++l) {
                        double cross=a(i+k*m,j+l*m);
                        score+=cross*cross/(back*a(i+k*m,i+k*m)*a(j+l*m,j+l*m));
                    }
                }
            }
            if(score>best) {best=score;selected=j;}
        }
        if(selected<0) return 2;
        order[pos]=selected;used[selected]=true;
    }
    return 0;
}
// Audit the output probabilities implied by the integer alias thresholds,
// reporting the sum of worst-context TVs across tiles (coupling bound).
extern "C" __declspec(dllexport) double ensemble95_alias_error(void* ptr,int component,void* original) {
    if(!ptr||!original) return -1;
    auto& e=*static_cast<Ensemble95*>(ptr);auto& p=*static_cast<Plan*>(original);
    if(component<0||component>=int(e.components.size())) return -1;
    long double total=0;
    for(std::size_t j=0;j<p.tiles.size();++j) {
        auto& tile=e.components[component][j];
        int size=1<<tile.width;long double worst=0;
        for(int ctx=0;ctx<(1<<tile.history);++ctx) {
            std::vector<long double> prob(size);
            for(int i=0;i<size;++i) {
                const auto& c=tile.choices[ctx*size+i];
                long double f=std::ldexp(static_cast<long double>(c.threshold),-53);
                prob[i]+=f/size;prob[c.alternate]+=(1-f)/size;
            }
            long double error=0;
            auto* row=p.tiles[j].cdf.data()+ctx*size;
            for(int i=0;i<size;++i) error+=std::abs(prob[i]-(static_cast<long double>(row[i])-(i?row[i-1]:0.)));
            worst=std::max(worst,error/2);
        }
        total+=worst;
    }
    return double(total);
}
// Exact eight-bit marginal of a window plan by forward summation over separators.
extern "C" __declspec(dllexport) int ensemble95_marginal(void* ptr,int history,const int* ports,double* out) {
    if(!ptr||history<0||history>11||!ports||!out) return 1;
    auto& p=*static_cast<Plan*>(ptr);
    std::vector<int> mark(p.modes,0);
    for(int j=0;j<8;++j) {
        if(ports[j]<0||ports[j]>=p.modes||mark[ports[j]]) return 1;
        mark[ports[j]]=1<<j;
    }
    std::vector<double> current((1<<history)*256),next(current.size());
    current[0]=1;
    for(const auto& t:p.tiles) {
        if(t.history!=std::min(history,t.first)) return 1;
        std::fill(next.begin(),next.end(),0.);
        int size=1<<t.width;
        for(int w=0;w<size;++w) {
            int markbits=0;
            for(int b=0;b<t.width;++b) if(w&(1<<b)) markbits|=mark[t.first+b];
            for(int ctx=0;ctx<(1<<t.history);++ctx) {
                const auto* row=t.cdf.data()+ctx*size;
                double chance=row[w]-(w?row[w-1]:0.);
                if(!chance) continue;
                int after=(ctx|(w<<t.history))>>std::max(0,t.width+t.history-history);
                auto* dest=next.data()+after*256;
                const auto* src=current.data()+ctx*256;
                for(int mask=0;mask<256;++mask) dest[mask|markbits]+=chance*src[mask];
            }
        }
        current.swap(next);
    }
    std::fill(out,out+256,0.);
    for(int ctx=0;ctx<(1<<history);++ctx)
        for(int mask=0;mask<256;++mask) out[mask]+=current[ctx*256+mask];
    return 0;
}
extern "C" __declspec(dllexport) int ensemble95_sample(void* ptr,int n,std::uint64_t exp,
        std::uint64_t offset,int threads,unsigned char* output) {
    if(!ptr||!output||n<0||threads<1||threads>256||offset>0x100000000ULL||
       std::uint64_t(n)>0x100000000ULL-offset) return 1;
    const auto& e=*static_cast<const Ensemble95*>(ptr);
    auto worker=[&](int begin,int end) {
        for(int r=begin;r<end;++r) {
            DomainRng84 rng(exp,0,Click84,offset+r);
            std::size_t component=0;
            if(e.components.size()>1) {
                DomainRng84 pick(exp,0,Resplitting84,offset+r);
                component=std::upper_bound(e.weights.begin(),e.weights.end(),uniform84(pick))-e.weights.begin();
            }
            auto* row=output+std::size_t(r)*e.m;
            int history=0;
            for(const auto& t:e.components[component]) {
                auto u=rng();
                int size=1<<t.width,column=int(u&(size-1));
                const auto& choice=t.choices[(history<<t.width)+column];
                int word=(u>>11)<choice.threshold?column:choice.alternate;
                if(e.output_ports.empty()) std::memcpy(row+t.first,&e.unpack[word],t.width);
                else for(int j=0;j<t.width;++j) row[e.output_ports[t.first+j]]=(word>>j)&1;
                history=(history|(word<<t.history))>>(t.width+t.history-t.next_history);
            }
        }
    };
    if(threads==1||n<2) {worker(0,n);return 0;}
    int workers=std::min(n,threads);
    std::vector<std::thread> pool;
    try {
        for(int i=1;i<workers;++i) pool.emplace_back(worker,int(std::int64_t(n)*i/workers),
            int(std::int64_t(n)*(i+1)/workers));
        worker(0,n/workers);
    } catch(const std::exception&) {for(auto& t:pool)t.join();return 2;}
    for(auto& t:pool)t.join();
    return 0;
}
