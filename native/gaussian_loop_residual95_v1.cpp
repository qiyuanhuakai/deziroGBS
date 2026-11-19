// Split a count correction into a bounded rejection component and its residual.
// For a rare residual total, sample a center and recursively split the subtotal.
#include "gaussian_loop_alloc95_v1.cpp"
namespace {
struct TotalPlan95 {
    const Table95* base;
    int limit,leaves;
    double bulk,acceptance;
    std::vector<std::vector<PolyFilter>> tree;
    std::vector<double> q,target,residual,center_cdf;
    std::vector<std::uint64_t> accept;
};
PolyFilter convolution95(const PolyFilter& a,const PolyFilter& b,int bins) {
    PolyFilter p(std::min<int>(bins,int(a.size()+b.size()-1)),0.);
    for(int j=0;j<std::min<int>(a.size(),p.size());++j)
        for(int k=0;k<std::min<int>(b.size(),p.size()-j);++k) p[j+k]+=a[j]*b[k];
    while(p.size()>1&&p.back()<1e-85) p.pop_back();
    return p;
}
int choice95(const double* cdf,int n,DomainRng84& rng) {
    return std::min<int>(n-1,int(std::upper_bound(cdf,cdf+n,uniform84(rng))-cdf));
}
void conditioned95(const TotalPlan95& plan,int c,int node,int total,DomainRng84& rng,unsigned char* out) {
    if(node>=plan.leaves) {
        if(node-plan.leaves<plan.base->m) out[node-plan.leaves]=static_cast<unsigned char>(total);
        return;
    }
    if(!total) return; // the caller zeroes the residual record
    const auto& tree=plan.tree[c];
    auto& a=tree[2*node];auto& b=tree[2*node+1];
    double u=uniform84(rng)*tree[node][total],running=0;
    int lo=std::max<int>(0,total+1-int(b.size())),hi=std::min<int>(a.size()-1,total),left=lo;
    for(int j=lo;j<=hi;++j) {
        double mass=a[j]*b[total-j];
        if(mass>0) left=j;
        running+=mass;
        if(u<running) break;
    }
    conditioned95(plan,c,2*node,left,rng,out);
    conditioned95(plan,c,2*node+1,total-left,rng,out);
}
}
extern "C" __declspec(dllexport) void* residual95_prepare(void* ptr,int limit,const double* target,double bound,
        int workers,double* law,double* diag) {
    if(!ptr||!target||!law||!diag||limit<1||limit>1024||bound<1||workers<1||workers>64) return nullptr;
    try {
        auto plan=std::make_unique<TotalPlan95>();auto& p=*static_cast<Table95*>(ptr);
        plan->base=&p;plan->limit=limit;int bins=limit+1;
        int leaves=1;while(leaves<p.m) leaves*=2;plan->leaves=leaves;
        plan->tree.resize(p.centers);plan->center_cdf.resize(std::size_t(bins)*p.centers,0.);
        #pragma omp parallel for num_threads(workers) schedule(static)
        for(int c=0;c<p.centers;++c) {
            auto& tree=plan->tree[c];tree.resize(2*leaves);
            for(int j=0;j<leaves;++j) tree[leaves+j]=j<p.m?port95(p,c,j,true):PolyFilter{1.};
            for(int j=leaves-1;j>0;--j) tree[j]=convolution95(tree[2*j],tree[2*j+1],bins);
        }
        plan->q.resize(bins);plan->target.assign(target,target+bins);
        plan->accept.resize(bins);plan->residual.resize(bins);
        double acc=0,remaining=0,tsum=0,worst=0;
        for(int t=0;t<bins;++t) {
            double sum=0;auto* cdf=plan->center_cdf.data()+std::size_t(t)*p.centers;
            for(int c=0;c<p.centers;++c) {
                auto& v=plan->tree[c][1];if(t<int(v.size())) sum+=v[t];cdf[c]=sum;
            }
            double q=sum/p.centers;law[t]=q;plan->q[t]=q;
            if(target[t]<0||!std::isfinite(target[t])||(target[t]>0&&q<=0)) return nullptr;
            if(q>0) for(int c=0;c<p.centers;++c) cdf[c]/=sum;
            double ratio=q>0?target[t]/q:0.;worst=std::max(worst,ratio);
            auto cut=std::uint64_t(std::floor(word95*std::min(1.,ratio/bound)));
            plan->accept[t]=cut;double accepted=q*(cut/word95);acc+=accepted;
            remaining+=std::max(0.,target[t]-bound*accepted);plan->residual[t]=remaining;tsum+=target[t];
        }
        if(std::abs(tsum-1)>1e-11||acc<=0||std::abs(bound*acc+remaining-1)>1e-11) return nullptr;
        if(remaining>0) for(auto& v:plan->residual) v/=remaining;
        plan->bulk=1.-remaining;plan->acceptance=acc;
        diag[0]=remaining;diag[1]=acc;diag[2]=worst;diag[3]=0;
        for(const auto& tree:plan->tree) for(const auto& v:tree) diag[3]+=sizeof(double)*v.size();
        return plan.release();
    } catch(const std::exception&) {return nullptr;}
}
extern "C" __declspec(dllexport) void residual95_free(void* ptr) {delete static_cast<TotalPlan95*>(ptr);}
extern "C" __declspec(dllexport) int residual95_marginals(void* ptr,int workers,double* out) {
    if(!ptr||!out||workers<1||workers>64) return 1;
    const auto& plan=*static_cast<const TotalPlan95*>(ptr);const auto& p=*plan.base;
    int levels=p.levels+1,size=p.m*levels;
    std::fill(out,out+size,0.);
    #pragma omp parallel num_threads(workers)
    {
        std::vector<double> partial(size);
        std::vector<PolyFilter> w(2*plan.leaves);
        #pragma omp for schedule(static)
        for(int c=0;c<p.centers;++c) {
            const auto& tree=plan.tree[c];
            w[1].resize(plan.limit+1);
            for(int t=0;t<=plan.limit;++t) w[1][t]=plan.q[t]>0?plan.target[t]/plan.q[t]:0.;
            for(int i=1;i<plan.leaves;++i) for(int side=0;side<2;++side) {
                int child=2*i+side;const auto& sib=tree[2*i+1-side];auto& v=w[child];
                v.assign(tree[child].size(),0.);
                for(int k=0;k<int(v.size());++k)
                    for(int t=0;t<std::min<int>(sib.size(),int(w[i].size())-k);++t)
                        v[k]+=sib[t]*w[i][k+t];
            }
            for(int g=0;g<p.m;++g) for(int k=0;k<int(tree[plan.leaves+g].size());++k)
                partial[g*levels+k]+=tree[plan.leaves+g][k]*w[plan.leaves+g][k];
        }
        #pragma omp critical
        for(int i=0;i<size;++i) out[i]+=partial[i]/p.centers;
    }
    return 0;
}
extern "C" __declspec(dllexport) int residual95_draw(void* ptr,int n,std::uint64_t key,std::uint64_t offset,
        int workers,unsigned char* output,std::uint32_t* attempts,unsigned char* branch) {
    if(!ptr||!output||!attempts||!branch||n<1||workers<1||workers>64||offset>0x100000000ULL||
       std::uint64_t(n)>0x100000000ULL-offset) return 1;
    const auto& plan=*static_cast<const TotalPlan95*>(ptr);const auto& p=*plan.base;
    std::atomic<bool> failed{false};
    #pragma omp parallel for num_threads(workers) schedule(static) if(n>1 && workers>1)
    for(int i=0;i<n;++i) {
        DomainRng84 mix(key,0,132,offset+i),component(key,0,123,offset+i),ports(key,0,124,offset+i),
                    selection(key,0,126,offset+i),split(key,0,133,offset+i);
        auto* row=output+std::size_t(i)*p.m;
        if(uniform84(mix)>=plan.bulk) {
            int t=choice95(plan.residual.data(),plan.limit+1,split);
            int c=choice95(plan.center_cdf.data()+std::size_t(t)*p.centers,p.centers,split);
            std::fill(row,row+p.m,0);conditioned95(plan,c,1,t,split,row);
            attempts[i]=0;branch[i]=1;continue;
        }
        branch[i]=0;
        for(std::uint32_t trial=1;trial<=2048;++trial) {
            auto start=std::size_t(component()&(p.centers-1))*p.m;int total=0;
            for(int j=0;j<p.m;j+=2) {
                auto pair=ports();
                for(int h=0;h<std::min(2,p.m-j);++h) {
                    auto u=std::uint32_t(pair>>(32*h));auto pos=start+j+h;
                    int k=0;while(k<p.levels&&u<p.above[k][pos]) ++k;
                    row[j+h]=static_cast<unsigned char>(k);total+=k;
                }
            }
            auto u=std::uint32_t(selection());
            if(total<=plan.limit&&u<plan.accept[total]) {attempts[i]=trial;break;}
            if(trial==2048) {attempts[i]=0;failed=true;}
        }
    }
    return failed?2:0;
}
extern "C" __declspec(dllexport) int residual95_expand(int ng,int n,int width,int last,int kind,
        std::uint64_t key,std::uint64_t offset,int workers,int stride,const std::uint64_t* splits,
        const std::uint64_t* masks,int degree,const int* offsets,const std::uint64_t* low,
        const std::uint32_t* patterns,const unsigned char* totals,unsigned char* out) {
    if(ng<1||n<1||width<1||width>8||last<1||last>width||kind<0||kind>1||degree<0||degree>15||
       stride<3||workers<1||workers>64||!splits||!masks||!offsets||!low||!patterns||!totals||!out||
       offset>0x100000000ULL||std::uint64_t(n)>0x100000000ULL-offset) return 1;
    int m=(ng-1)*width+last;
    #pragma omp parallel for num_threads(workers) schedule(static) if(n>1 && workers>1)
    for(int i=0;i<n;++i) {
        DomainRng84 rng(key,0,kind?131:130,offset+i);
        for(int g=0;g<ng;++g) {
            auto* dest=out+std::size_t(i)*m+width*g;int w=g==ng-1?last:width;
            int t=totals[std::size_t(i)*ng+g];
            if(!kind) {
                int mask=0;
                if(t>=w) mask=(1<<w)-1;
                else if(t) {
                    auto* r=masks+(std::size_t(g)*9+t)*256;
                    mask=int(std::upper_bound(r,r+(1<<w),std::uint32_t(rng()))-r);
                }
                for(int j=0;j<w;++j) dest[j]=(mask>>j)&1;
            } else if(t>0&&t<=degree) {
                int pos=g*(degree+2)+t;
                int k=int(std::upper_bound(low+offsets[pos],low+offsets[pos+1],std::uint32_t(rng()))-low);
                auto word=patterns[k];
                for(int j=0;j<w;++j) dest[j]=(word>>(4*j))&15;
            } else allocate95(w,0,1,t,stride,splits+std::size_t(g)*16*stride*stride,rng,dest);
        }
    }
    return 0;
}
