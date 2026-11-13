// Keep a moving window of click inclusion-exclusion bits.
// On overflow merge only pairs differing in the oldest retained bit.
// Signed cluster masses may be negative; normalized cluster moments are kept.
#include "gaussian_click_branches_v1.cpp"
static void rolling_update80(std::vector<Branch71>& branches,const Step71& step,bool clicked,int budget,Trace71& trace){
 if(budget==1){update71(branches,step,clicked,1,trace);return;}
 update71(branches,step,clicked,65536,trace);
 if(branches.size()<=std::size_t(budget))return;
 need(branches.size()==std::size_t(2*budget),"Rolling branch topology lost");
 std::vector<Branch71> merged;merged.reserve(budget);
 for(int i=0;i<budget;++i){
   std::vector<Branch71> pair{branches[i],branches[i+budget]};
   long double mass=sum71({pair[0].weight,pair[1].weight});
   need(std::isfinite(mass)&&mass!=0,"Zero or nonfinite rolling cluster mass");
   for(auto& item:pair)item.weight/=mass;
   auto value=merge71(pair);value.weight=mass;merged.push_back(std::move(value));
 }
 branches=std::move(merged);++trace.merges;audit71(branches,trace);
}
// Fixed-mean forced prefix probability. No random displacement integration.
API int rc80_prefix(void* handle,const double* mean,int length,const std::uint8_t* bits,int budget,double* probability,double* stats,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&mean&&bits&&probability&&stats&&budget>=1&&budget<=128&&(budget&(budget-1))==0,"Invalid branch probe");
    auto& model=*static_cast<ClickModel70*>(handle);int d=2*model.model->n;need(length>=0&&length<=model.model->m,"Invalid prefix length");
    Vector w(d);for(int i=0;i<d;++i){w[i]=mean[i];need(std::isfinite(w[i]),"Nonfinite prefix mean");}
    std::vector<Branch71> branches{{model.model->initial,w,1}};Trace71 trace;long double joint=1;
    for(int mode=0;mode<length;++mode){need(bits[mode]<=1,"Prefix must contain bits");auto step=parameters71(model,branches,mode,trace);joint*=bits[mode]?step.p1:step.p0;if(joint==0)break;if(mode+1<length)rolling_update80(branches,step,bits[mode],budget,trace);}
    *probability=double(joint);stats[0]=trace.evaluations;stats[1]=trace.merges;stats[2]=trace.peak;stats[3]=double(trace.cancellation);return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
// Stats: clicks, detector steps, zero-click samples, workers, completed,
// Gaussian component evaluations, compressions, transient peak components,
// maximum signed absolute weight sum. The transient peak may be 2*budget.
API int rc80_sample(void* handle,int count,std::uint64_t seed,int threads,int budget,const double* mean,const double* variance,std::uint8_t* output,double* stats,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&count>=0&&threads>=1&&threads<=256&&budget>=1&&budget<=128&&(budget&(budget-1))==0&&mean&&variance&&output&&stats,"Invalid branch batch");
    auto& model=*static_cast<ClickModel70*>(handle);int d=2*model.model->n,m=model.model->m;Vector center(d),deviation(d);
    for(int j=0;j<d;++j){need(std::isfinite(mean[j])&&std::isfinite(variance[j])&&variance[j]>=0,"Invalid branch displacement");center[j]=mean[j];deviation[j]=std::sqrt(variance[j]);}
    std::fill(stats,stats+9,0);std::fill(output,output+std::size_t(count)*m,0);if(!count)return 0;
    int workers=std::min(count,threads);std::atomic<std::int64_t> next{0},completed{0};std::atomic<bool> failed{false};std::vector<std::exception_ptr> failures(workers);std::vector<std::array<double,7>> local(workers);std::vector<std::thread> pool;pool.reserve(workers);
    auto worker=[&](int id){try{PrecisionGuard precision;for(;;){if(failed.load())break;auto row=next.fetch_add(1);if(row>=count)break;
        std::mt19937_64 detector(mix(seed+2ULL*row+1)),noise(mix(seed+0x81c6f2d54a970e31ULL+row));std::normal_distribution<double> normal;Vector w=center;for(int j=0;j<d;++j)if(deviation[j]>0)w[j]+=deviation[j]*normal(noise);
        std::vector<Branch71> branches{{model.model->initial,w,1}};Trace71 trace;int clicks=0;
        for(int mode=0;mode<m;++mode){auto step=parameters71(model,branches,mode,trace);bool clicked=(long double)uniform(detector)<step.p1;output[std::size_t(row)*m+mode]=clicked;clicks+=clicked;if(mode+1<m)rolling_update80(branches,step,clicked,budget,trace);}
        local[id][0]+=clicks;local[id][1]+=m;local[id][2]+=(clicks==0);local[id][3]+=trace.evaluations;local[id][4]+=trace.merges;local[id][5]=std::max(local[id][5],trace.peak);local[id][6]=std::max(local[id][6],double(trace.cancellation));++completed;
    }}catch(...){failures[id]=std::current_exception();failed.store(true);}};
    try{for(int id=0;id<workers;++id)pool.emplace_back(worker,id);}catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}
    for(auto& thread:pool)thread.join();for(auto& row:local){for(int j=0;j<3;++j)stats[j]+=row[j];stats[5]+=row[3];stats[6]+=row[4];stats[7]=std::max(stats[7],row[5]);stats[8]=std::max(stats[8],row[6]);}
    stats[3]=workers;stats[4]=completed.load();for(auto& failure:failures)if(failure)std::rethrow_exception(failure);need(completed.load()==count,"Incomplete signed branch batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
