// Iteration 71: bounded signed Gaussian branches for threshold sampling.
// Exact Gaussian-difference identity: Quesada et al., arXiv:1807.01639.
// Compression to signed first/second moments is approximate. No clipping.
#include "gaussian_direct_click.cpp"

struct Branch71 {Real x;Vector w;long double weight;};
struct Trace71 {double evaluations=0,merges=0,peak=1;long double cancellation=1;};
static long double sum71(const std::vector<long double>& values){
    long double s=0,c=0;for(auto x:values){auto y=x-c,t=s+y;c=(t-s)-y;s=t;}return s;
}
static void audit71(const std::vector<Branch71>& branches,Trace71& trace){
    std::vector<long double> w,a;for(auto& b:branches){w.push_back(b.weight);a.push_back(std::abs(b.weight));}
    long double mass=sum71(w),condition=sum71(a);
    need(std::isfinite(mass)&&std::isfinite(condition)&&std::abs(mass-1)<1e-10L,"Signed branch normalization lost");
    trace.peak=std::max(trace.peak,double(branches.size()));trace.cancellation=std::max(trace.cancellation,condition);
}
static Branch71 merge71(const std::vector<Branch71>& branches){
    int d=branches.front().w.size();Branch71 out{Real::Zero(d,d),Vector::Zero(d),1};
    // Accumulate both moments in extended precision; negative weights are kept.
    std::vector<long double> weights;for(auto& b:branches)weights.push_back(b.weight);auto mass=sum71(weights);
    need(std::isfinite(mass)&&mass>0,"Invalid signed mass during merge");
    std::vector<long double> terms(branches.size()),mu(d);
    for(int i=0;i<d;++i){for(std::size_t k=0;k<branches.size();++k)terms[k]=branches[k].weight*branches[k].w[i];mu[i]=sum71(terms)/mass;out.w[i]=double(mu[i]);}
    for(int i=0;i<d;++i)for(int j=0;j<=i;++j){
        for(std::size_t k=0;k<branches.size();++k){auto& b=branches[k];terms[k]=b.weight*((long double)b.x(i,j)+((long double)b.w[i]-mu[i])*((long double)b.w[j]-mu[j]));}
        out.x(i,j)=out.x(j,i)=double(sum71(terms)/mass);
    }
    need(out.x.allFinite()&&out.w.allFinite(),"Signed branch moment overflow");return out;
}
struct Step71 {std::vector<ClickStep70> parts;long double p0,p1;};
static Step71 parameters71(const ClickModel70& model,const std::vector<Branch71>& branches,int mode,Trace71& trace){
    Step71 result;result.parts.reserve(branches.size());std::vector<long double> zero,one;
    for(auto& b:branches){auto p=click_parameters70(model,b.x,b.w,mode);zero.push_back(b.weight*p.p0);one.push_back(b.weight*p.p1);result.parts.push_back(std::move(p));}
    auto z=sum71(zero),o=sum71(one),mass=z+o;
    need(std::isfinite(z)&&std::isfinite(o)&&z>=0&&o>=0&&mass>0,"Signed conditional probability invalid; no clipping");
    result.p0=z/mass;result.p1=o/mass;trace.evaluations+=branches.size();return result;
}
static void update71(std::vector<Branch71>& branches,const Step71& step,bool clicked,int budget,Trace71& trace){
    auto probability=clicked?step.p1:step.p0;need(probability>0,"Forced zero-probability branch");
    // Preserve the exact floating-point path of iteration 70 for budget one.
    if(budget==1){click_update70(branches[0].x,branches[0].w,step.parts[0],clicked);if(clicked)++trace.merges;return;}
    std::vector<Branch71> next;next.reserve(branches.size()*(clicked?2:1));
    for(std::size_t k=0;k<branches.size();++k){auto& b=branches[k];auto& p=step.parts[k];
        if(clicked)next.push_back({b.x,b.w,b.weight/probability});
        if(p.p0>0&&b.weight!=0){Branch71 vacuum=b;click_update70(vacuum.x,vacuum.w,p,false);vacuum.weight=b.weight*p.p0/probability*(clicked?-1:1);next.push_back(std::move(vacuum));}
    }
    need(!next.empty(),"Empty signed state");
    std::vector<long double> weights;for(auto& b:next)weights.push_back(b.weight);auto mass=sum71(weights);need(std::isfinite(mass)&&mass>0,"Conditional signed normalization failed");
    for(auto& b:next)b.weight/=mass;audit71(next,trace);
    if(next.size()>std::size_t(budget)){auto merged=merge71(next);next.clear();next.push_back(std::move(merged));++trace.merges;}
    branches=std::move(next);
}
// Fixed-mean forced prefix probability. No random displacement integration.
API int tc71_prefix(void* handle,const double* mean,int length,const std::uint8_t* bits,int budget,double* probability,double* stats,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&mean&&bits&&probability&&stats&&budget>=1&&budget<=65536,"Invalid branch probe");
    auto& model=*static_cast<ClickModel70*>(handle);int d=2*model.model->n;need(length>=0&&length<=model.model->m,"Invalid prefix length");
    Vector w(d);for(int i=0;i<d;++i){w[i]=mean[i];need(std::isfinite(w[i]),"Nonfinite prefix mean");}
    std::vector<Branch71> branches{{model.model->initial,w,1}};Trace71 trace;long double joint=1;
    for(int mode=0;mode<length;++mode){need(bits[mode]<=1,"Prefix must contain bits");auto step=parameters71(model,branches,mode,trace);joint*=bits[mode]?step.p1:step.p0;if(joint==0)break;if(mode+1<length)update71(branches,step,bits[mode],budget,trace);}
    *probability=double(joint);stats[0]=trace.evaluations;stats[1]=trace.merges;stats[2]=trace.peak;stats[3]=double(trace.cancellation);return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
// Stats: clicks, detector steps, zero-click samples, workers, completed,
// Gaussian component evaluations, compressions, transient peak components,
// maximum signed absolute weight sum. The transient peak may be 2*budget.
API int tc71_sample(void* handle,int count,std::uint64_t seed,int threads,int budget,const double* mean,const double* variance,std::uint8_t* output,double* stats,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&count>=0&&threads>=1&&threads<=256&&budget>=1&&budget<=65536&&mean&&variance&&output&&stats,"Invalid branch batch");
    auto& model=*static_cast<ClickModel70*>(handle);int d=2*model.model->n,m=model.model->m;Vector center(d),deviation(d);
    for(int j=0;j<d;++j){need(std::isfinite(mean[j])&&std::isfinite(variance[j])&&variance[j]>=0,"Invalid branch displacement");center[j]=mean[j];deviation[j]=std::sqrt(variance[j]);}
    std::fill(stats,stats+9,0);std::fill(output,output+std::size_t(count)*m,0);if(!count)return 0;
    int workers=std::min(count,threads);std::atomic<std::int64_t> next{0},completed{0};std::atomic<bool> failed{false};std::vector<std::exception_ptr> failures(workers);std::vector<std::array<double,7>> local(workers);std::vector<std::thread> pool;pool.reserve(workers);
    auto worker=[&](int id){try{PrecisionGuard precision;for(;;){if(failed.load())break;auto row=next.fetch_add(1);if(row>=count)break;
        std::mt19937_64 detector(mix(seed+2ULL*row+1)),noise(mix(seed+0x81c6f2d54a970e31ULL+row));std::normal_distribution<double> normal;Vector w=center;for(int j=0;j<d;++j)if(deviation[j]>0)w[j]+=deviation[j]*normal(noise);
        std::vector<Branch71> branches{{model.model->initial,w,1}};Trace71 trace;int clicks=0;
        for(int mode=0;mode<m;++mode){auto step=parameters71(model,branches,mode,trace);bool clicked=(long double)uniform(detector)<step.p1;output[std::size_t(row)*m+mode]=clicked;clicks+=clicked;if(mode+1<m)update71(branches,step,clicked,budget,trace);}
        local[id][0]+=clicks;local[id][1]+=m;local[id][2]+=(clicks==0);local[id][3]+=trace.evaluations;local[id][4]+=trace.merges;local[id][5]=std::max(local[id][5],trace.peak);local[id][6]=std::max(local[id][6],double(trace.cancellation));++completed;
    }}catch(...){failures[id]=std::current_exception();failed.store(true);}};
    try{for(int id=0;id<workers;++id)pool.emplace_back(worker,id);}catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}
    for(auto& thread:pool)thread.join();for(auto& row:local){for(int j=0;j<3;++j)stats[j]+=row[j];stats[5]+=row[3];stats[6]+=row[4];stats[7]=std::max(stats[7],row[5]);stats[8]=std::max(stats[8],row[6]);}
    stats[3]=workers;stats[4]=completed.load();for(auto& failure:failures)if(failure)std::rethrow_exception(failure);need(completed.load()==count,"Incomplete signed branch batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
