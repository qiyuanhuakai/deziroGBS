// Optical block laws and a tree sampler for threshold and photon-count output.
#include "gaussian_loop_alloc95_v1.cpp"
#include <numeric>
namespace {
using Mat96=Eigen::Matrix<long double,-1,-1>;
using Vec96=Eigen::Matrix<long double,-1,1>;
long double vacuum96(const double* q,const double* mean,int m,int mask) {
    if(!mask) return 1;
    std::vector<int> ids;
    for(int i=0;i<m;++i) if(mask&(1<<i)) {ids.push_back(i);ids.push_back(m+i);}
    int n=ids.size();Mat96 sub(n,n);Vec96 v(n);
    for(int i=0;i<n;++i) {
        v[i]=mean[ids[i]];
        for(int j=0;j<n;++j) sub(i,j)=q[ids[i]*2*m+ids[j]];
    }
    Eigen::LLT<Mat96> f(sub);
    if(f.info()!=Eigen::Success) throw std::domain_error("covariance");
    Vec96 y=f.matrixL().solve(v);
    return std::exp(-.5L*y.squaredNorm()-f.matrixLLT().diagonal().array().log().sum());
}
}
extern "C" __declspec(dllexport) int group96_click(int m,int sectors,const double* q,const double* mean,
        int workers,double* law,double* audit) {
    if(m<1||m>18||sectors<1||sectors>8||!q||!mean||workers<1||workers>64||!law||!audit) return 1;
    int n=1<<m;std::vector<long double> v(n,1.L);std::atomic<bool> valid{true};
    #pragma omp parallel for num_threads(workers) if(n>256) schedule(static)
    for(int mask=1;mask<n;++mask) {
        try {for(int s=0;s<sectors;++s) v[mask]*=vacuum96(q+s*4*m*m,mean+s*2*m,m,mask);}
        catch(const std::exception&) {valid=false;}
    }
    if(!valid) return 2;
    for(int bit=1;bit<n;bit*=2) for(int mask=0;mask<n;++mask) if(!(mask&bit)) v[mask]-=v[mask|bit];
    long double neg=0.,sum=0.;for(int i=0;i<n;++i) {neg+=std::max(0.L,-v[i]);sum+=std::max(0.L,v[i]);}
    audit[0]=double(neg);audit[1]=double(sum);
    if(!(sum>0)||neg>1e-9L) return 3;
    for(int i=0;i<n;++i) law[i]=double(std::max(0.L,v[n-1-i])/sum);
    return 0;
}

// Transform unconstrained/zero/one probabilities into 0,1,2-or-more states.
extern "C" __declspec(dllexport) int group96_ternary(int m,int sectors,const double* q,const double* mu,
        double* law,double* audit) {
    if(m<1||m>8||sectors<1||sectors>8||!q||!mu||!law||!audit) return 1;
    int n=1;for(int i=0;i<m;++i)n*=3;
    std::vector<long double> table(n);table[0]=1;
    for(int mask=1;mask<(1<<m);++mask) {
        std::vector<int> ports;for(int j=0;j<m;++j)if(mask&(1<<j)) ports.push_back(j);
        int d=ports.size(),h=1<<d;
        std::vector<double> sub(sectors*4*d*d),mean(sectors*2*d),dirs(h*d),terms(h*(d+1));
        for(int s=0;s<sectors;++s) for(int i=0;i<2*d;++i) {
            int a=ports[i%d]+(i/d)*m;mean[s*2*d+i]=mu[s*2*m+a];
            for(int j=0;j<2*d;++j) sub[(s*2*d+i)*2*d+j]=q[(s*2*m+a)*2*m+ports[j%d]+(j/d)*m];
        }
        for(int b=0;b<h;++b)for(int i=0;i<d;++i) dirs[b*d+i]=(b>>i)&1;
        if(alloc95_low(d,sectors,d,h,sub.data(),mean.data(),dirs.data(),terms.data()))return 2;
        for(int b=0;b<h;++b) {
            int t=__builtin_popcount(unsigned(b));long double prob=0;
            for(int k=b;;k=(k-1)&b) {
                int parity=(t-__builtin_popcount(unsigned(k)))%2;
                prob+=(parity?-1.L:1.L)*terms[k*(d+1)+t];if(k==0) break;
            }
            int pos=0,scale=1,next=0;
            for(int i=0;i<m;++i,scale*=3) if(mask&(1<<i)) pos+=scale*(1+((b>>next++)&1));
            table[pos]=prob;
        }
    }
    for(int b=1;b<n;b*=3)for(int i=0;i<n;i+=3*b)for(int j=i;j<i+b;++j) {
        auto a=table[j],z=table[j+b],o=table[j+2*b];table[j]=z;table[j+b]=o;table[j+2*b]=a-z-o;
    }
    long double neg=0,sum=0;
    for(auto x:table) {neg+=std::max(0.L,-x);sum+=std::max(0.L,x);}
    if(!(sum>0)||neg>1e-8L) return 3;
    for(int i=0;i<n;++i) law[i]=double(std::max(0.L,table[i])/sum);
    audit[0]=double(neg);audit[1]=double(sum);return 0;
}
namespace {
using Poly96=std::vector<double>;
constexpr std::uint64_t scale96=1ULL<<53;
struct Node96 {
    int parent,width;
    std::vector<int> ports,children;
    std::vector<Poly96> message;
    // Per parent state, suffix[0] is the entire subtree and suffix[1]
    // sums all children. The last suffix contains only zero photons.
    std::vector<std::vector<Poly96>> suffix;
    std::vector<Poly96> child_total;
};
struct Tree96 {
    int m,g,states,cap,limit,pnr;
    std::vector<Node96> nodes;
    std::vector<double> transition,emission,proposal,target,residual;
    std::vector<std::uint64_t> state_cdf,total_cdf,split_cdf,accept;
    double bulk=1,acceptance=1;
};
Poly96 convolution96(const Poly96& a,const Poly96& b,int bins) {
    Poly96 v(std::min<int>(bins,int(a.size()+b.size()-1)),0.);
    for(int i=0;i<std::min<int>(a.size(),v.size());++i) if(a[i])
        for(int j=0;j<std::min<int>(b.size(),v.size()-i);++j) v[i+j]+=a[i]*b[j];
    while(v.size()>1&&v.back()<1e-115) v.pop_back();
    return v;
}
bool cdf96(double* prob,int size,std::uint64_t* out) {
    double sum=0,part=0;for(int i=0;i<size;++i) {if(prob[i]<0||!std::isfinite(prob[i]))return false;sum+=prob[i];}
    if(!(sum>0)) return false;
    std::uint64_t old=0;
    for(int i=0;i<size;++i) {
        part+=prob[i];auto cut=std::min(scale96,std::uint64_t(std::llround(scale96*part/sum)));
        if(i==size-1)cut=scale96;
        out[i]=cut;prob[i]=double(cut-old)/scale96;old=cut;
    }
    return true;
}
int index96(const std::uint64_t* cdf,int n,DomainRng84& rng) {
    return int(std::upper_bound(cdf,cdf+n,rng()>>11)-cdf);
}
double at96(const Poly96& v,int t) {return t>=0&&t<int(v.size())?v[t]:0.;}
int proportion96(const Poly96& a,const Poly96& b,int total,DomainRng84& rng) {
    int lo=std::max(0,total+1-int(b.size())),hi=std::min<int>(total,int(a.size())-1);
    double sum=0;for(int j=lo;j<=hi;++j)sum+=a[j]*b[total-j];
    if(!(sum>0))throw std::domain_error("empty subtotal");
    double u=uniform84(rng)*sum;sum=0;int last=lo;
    for(int j=lo;j<=hi;++j) {
        double value=a[j]*b[total-j];if(value>0) last=j;
        sum+=value;if(u<sum)return j;
    }
    return last;
}
int conditional_state96(const Tree96& p,int g,int par,int count,DomainRng84& rng) {
    const auto& node=p.nodes[g];const auto* tr=p.transition.data()+(std::size_t(g)*p.states+par)*p.states;
    double sum=0;for(int s=0;s<p.states;++s)sum+=tr[s]*at96(node.message[s],count);
    if(!(sum>0)) throw std::domain_error("empty conditional state");
    double u=uniform84(rng)*sum;sum=0;int last=0;
    for(int s=0;s<p.states;++s) {
        double v=tr[s]*at96(node.message[s],count);if(v>0)last=s;sum+=v;if(u<sum)return s;
    }
    return last;
}
}
extern "C" __declspec(dllexport) void* group96_prepare(int m,int g,int states,int cap,int pnr,int limit,
        const int* parents,const int* ports,const double* transition,const double* emission,
        const std::uint64_t* splits,double* law) {
    if(m<1||m>8192||g<1||g>m||states<2||states>81||cap<1||cap>126||limit<1||limit>1024||
       !parents||!ports||!transition||!emission||!splits||!law||(pnr!=0&&pnr!=1))return nullptr;
    try {
        auto h=std::make_unique<Tree96>();auto& p=*h;p.m=m;p.g=g;p.states=states;p.cap=cap;p.limit=limit;p.pnr=pnr;
        int n=cap+2;size_t rows=std::size_t(g)*states;
        p.transition.assign(transition,transition+rows*states);p.state_cdf.resize(rows*states);
        p.emission.assign(emission,emission+rows*n);p.total_cdf.resize(rows*n);
        if(pnr) p.split_cdf.assign(splits,splits+rows*n*n);
        p.nodes.resize(g);std::vector<bool> seen(m,false);
        for(int i=0;i<g;++i) {
            auto& node=p.nodes[i];node.parent=parents[i];
            if((i==0&&parents[i]!=-1)||(i>0&&(parents[i]<0||parents[i]>=i)))return nullptr;
            if(i)p.nodes[parents[i]].children.push_back(i);
            for(int j=0;j<8;++j)if(ports[i*8+j]>=0) {
                int k=ports[i*8+j];if(k>=m||seen[k])return nullptr;seen[k]=true;node.ports.push_back(k);
            }
            node.width=node.ports.size();if(node.width<1||(pnr&&node.width>2))return nullptr;
        }
        if(std::find(seen.begin(),seen.end(),false)!=seen.end())return nullptr;
        for(std::size_t i=0;i<rows;++i)
            if(!cdf96(p.transition.data()+i*states,states,p.state_cdf.data()+i*states)||
               !cdf96(p.emission.data()+i*n,n,p.total_cdf.data()+i*n))return nullptr;
        for(int i=g-1;i>=0;--i) {
            auto& node=p.nodes[i];node.message.resize(states);node.child_total.resize(states);
            node.suffix.resize(states);
            for(int s=0;s<states;++s) {
                auto& suf=node.suffix[s];int kids=node.children.size();
                suf.resize(kids+2);suf[kids+1]={1.};
                for(int j=kids-1;j>=0;--j) {
                    auto& child=p.nodes[node.children[j]];
                    suf[j+1]=convolution96(child.child_total[s],suf[j+2],limit+1);
                }
                const auto* e=p.emission.data()+(std::size_t(i)*states+s)*n;
                Poly96 local(e,e+n);while(local.size()>1&&local.back()==0) local.pop_back();
                suf[0]=convolution96(local,suf[1],limit+1);node.message[s]=suf[0];
            }
            for(int parent=0;parent<states;++parent) {
                auto& mix=node.child_total[parent];mix.assign(1,0.);
                for(int s=0;s<states;++s) {
                    double prob=p.transition[(std::size_t(i)*states+parent)*states+s];if(!prob)continue;
                    auto& v=node.message[s];if(mix.size()<v.size())mix.resize(v.size());
                    for(std::size_t j=0;j<v.size();++j)mix[j]+=prob*v[j];
                }
            }
        }
        p.proposal.resize(limit+1);for(int t=0;t<=limit;++t)law[t]=p.proposal[t]=at96(p.nodes[0].child_total[0],t);
        return h.release();
    } catch(const std::exception&) {return nullptr;}
}
extern "C" __declspec(dllexport) int group96_target(void* ptr,const double* target,double bound,double* diag) {
    if(!ptr||!target||!diag||!(bound>=1))return 1;
    auto& p=*static_cast<Tree96*>(ptr);int n=p.limit+1;p.target.assign(target,target+n);
    p.accept.resize(n);p.residual.resize(n);double excess=0,acc=0,total=0;
    for(int t=0;t<n;++t) {
        if(!std::isfinite(target[t])||target[t]<0||(target[t]>0&&p.proposal[t]<=0))return 2;
        double ratio=p.proposal[t]>0?target[t]/(p.proposal[t]*bound):0;
        auto cut=std::uint64_t(std::floor(std::min(1.,ratio)*scale96));
        p.accept[t]=cut;double mass=p.proposal[t]*(double(cut)/scale96);
        acc+=mass;excess+=std::max(0.,target[t]-bound*mass);p.residual[t]=excess;total+=target[t];
    }
    if(std::abs(total-1)>1e-9||acc<=0)return 3;
    if(excess>0)for(auto& x:p.residual)x/=excess;
    p.bulk=1.-excess;p.acceptance=acc;diag[0]=excess;diag[1]=acc;return 0;
}
extern "C" __declspec(dllexport) void group96_destroy(void* ptr){delete static_cast<Tree96*>(ptr);}
extern "C" __declspec(dllexport) int group96_draw(void* ptr,int n,std::uint64_t exp,std::uint64_t offset,
        int calibrated,int workers,unsigned char* out,std::uint32_t* tries,unsigned char* tail,unsigned char* excess) {
    if(!ptr||n<1||workers<1||workers>64||!out||!tries||!tail||!excess||
       offset>0x100000000ULL||std::uint64_t(n)>0x100000000ULL-offset)return 1;
    const auto& p=*static_cast<const Tree96*>(ptr);int k=p.states,support=p.cap+2;
    if(calibrated&&p.target.empty())return 1;
    std::atomic<bool> ok{true};
    #pragma omp parallel num_threads(workers) if(n>1&&workers>1)
    {
        std::vector<int> mark(p.g),count(p.g),sub(p.g);
        #pragma omp for schedule(static)
        for(int r=0;r<n;++r) {
            try {
                DomainRng84 coarse(exp,0,140,offset+r),fine(exp,0,141,offset+r),mix(exp,0,142,offset+r),
                            conditional(exp,0,143,offset+r),allocation(exp,0,144,offset+r);
                bool rare=calibrated&&uniform84(mix)>=p.bulk;excess[r]=rare;tries[r]=0;tail[r]=0;
                if(rare) {
                    sub[0]=std::min(p.limit,int(std::upper_bound(p.residual.begin(),p.residual.end(),uniform84(conditional))-p.residual.begin()));
                    for(int i=0;i<p.g;++i) {
                        const auto& node=p.nodes[i];int par=i?mark[node.parent]:0;
                        int s=mark[i]=conditional_state96(p,i,par,sub[i],conditional);
                        const auto& suf=node.suffix[s];
                        const auto* e=p.emission.data()+(std::size_t(i)*k+s)*support;
                        int left=sub[i];count[i]=p.pnr?proportion96(Poly96(e,e+support),suf[1],left,conditional):__builtin_popcount(unsigned(s));
                        left-=count[i];
                        for(int j=0;j<int(node.children.size());++j) {
                            int child=node.children[j];
                            int t=(j+1==int(node.children.size()))?left:proportion96(p.nodes[child].child_total[s],suf[j+2],left,conditional);
                            sub[child]=t;left-=t;
                        }
                    }
                } else for(unsigned trial=1;trial<=2048;++trial) {
                    int total=0;
                    for(int i=0;i<p.g;++i) {
                        int par=i?mark[p.nodes[i].parent]:0;
                        int s=mark[i]=index96(p.state_cdf.data()+(std::size_t(i)*k+par)*k,k,coarse);
                        count[i]=p.pnr?index96(p.total_cdf.data()+(std::size_t(i)*k+s)*support,support,fine):__builtin_popcount(unsigned(s));
                        total+=count[i];
                    }
                    if(!calibrated||(total<=p.limit&&(mix()>>11)<p.accept[total])) {tries[r]=trial;break;}
                    if(trial==2048)ok=false;
                }
                auto* row=out+std::size_t(r)*p.m;
                for(int i=0;i<p.g;++i) {
                    const auto& ids=p.nodes[i].ports;int s=mark[i],t=count[i];
                    if(!p.pnr) {for(int j=0;j<int(ids.size());++j)row[ids[j]]=(s>>j)&1;continue;}
                    if(t==p.cap+1)tail[r]=1;
                    if(ids.size()==1)row[ids[0]]=static_cast<unsigned char>(t);
                    else {
                        int a;
                        if(s%3<2)a=s%3;
                        else if(s/3<2)a=t-s/3;
                        else a=index96(p.split_cdf.data()+((std::size_t(i)*k+s)*support+t)*support,t+1,allocation);
                        row[ids[0]]=static_cast<unsigned char>(a);row[ids[1]]=static_cast<unsigned char>(t-a);
                    }
                }
            } catch(const std::exception&) {ok=false;}
        }
    }
    return ok?0:2;
}

#include <unordered_map>
#include <functional>
// Enumerate all even occupations up to a cutoff for a pure undisplaced state.
extern "C" __declspec(dllexport) int group96_fock(int m,int max_total,int capacity,const double* barg,
        double vacuum,std::uint64_t* keys,double* probabilities) {
    if(m<1||m>16||max_total<0||max_total>15||capacity<1||!barg||!keys||!probabilities||!(vacuum>0))return -1;
    try {
        using C=std::complex<double>;std::unordered_map<std::uint64_t,C> amp;amp.reserve(capacity);
        std::vector<int> n(m);int pos=0;
        std::function<void(int,int,std::uint64_t)> visit=[&](int i,int remaining,std::uint64_t key){
            if(i<m) {
                for(int j=(i+1==m?remaining:0);j<=remaining;++j) {
                    n[i]=j;visit(i+1,remaining-j,key|(std::uint64_t(j)<<(4*i)));
                }
                return;
            }
            if(pos>=capacity)throw std::length_error("capacity");
            int a=0;while(a<m&&!n[a])++a;C v=std::sqrt(vacuum);
            if(a<m) {
                v=0.;
                for(int j=0;j<m;++j) if(n[j]>(a==j)) {
                    auto prev=key-(1ULL<<(4*j))-(1ULL<<(4*a));int k=2*(a*m+j);
                    v+=C(barg[k],barg[k+1])*std::sqrt(double(n[j]-(a==j))/n[a])*amp.at(prev);
                }
            }
            amp.emplace(key,v);keys[pos]=key;probabilities[pos++]=std::norm(v);
        };
        for(int t=0;t<=max_total;t+=2)visit(0,t,0);
        return pos;
    } catch(const std::exception&) {return -2;}
}
