// Deterministic admissibility guard for rolling moment compression.
// Before any draw, invalid branch probabilities trigger all-to-one compression.
// No random retry, clipping or sample rejection; the changed law is approximate.
#include "gaussian_click_rolling80_v1.cpp"
struct Trace80:Trace71 {double parameter_recoveries=0,update_recoveries=0;};
static Step71 guarded_parameters80(const ClickModel70& model,std::vector<Branch71>& branches,int mode,Trace80& trace){
 try{return parameters71(model,branches,mode,trace);}
 catch(const std::runtime_error&){
  if(branches.size()==1)throw;
  trace.evaluations+=branches.size();auto merged=merge71(branches);branches.clear();branches.push_back(std::move(merged));++trace.merges;++trace.parameter_recoveries;
  return parameters71(model,branches,mode,trace);
 }
}
static void guarded_update80(std::vector<Branch71>& branches,const Step71& step,bool clicked,int budget,Trace80& trace){
 try{rolling_update80(branches,step,clicked,budget,trace);}
 catch(const std::runtime_error&){
  if(branches.size()!=std::size_t(2*budget))throw;
  auto merged=merge71(branches);branches.clear();branches.push_back(std::move(merged));++trace.merges;++trace.update_recoveries;
 }
}
// Fixed-mean forced prefix probability. No random displacement integration.
API int rg80_prefix(void* handle,const double* mean,int length,const std::uint8_t* bits,int budget,double* probability,double* stats,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&mean&&bits&&probability&&stats&&budget>=1&&budget<=128&&(budget&(budget-1))==0,"Invalid branch probe");
    auto& model=*static_cast<ClickModel70*>(handle);int d=2*model.model->n;need(length>=0&&length<=model.model->m,"Invalid prefix length");
    Vector w(d);for(int i=0;i<d;++i){w[i]=mean[i];need(std::isfinite(w[i]),"Nonfinite prefix mean");}
    std::vector<Branch71> branches{{model.model->initial,w,1}};Trace80 trace;long double joint=1;
    for(int mode=0;mode<length;++mode){need(bits[mode]<=1,"Prefix must contain bits");auto step=guarded_parameters80(model,branches,mode,trace);joint*=bits[mode]?step.p1:step.p0;if(joint==0)break;if(mode+1<length)guarded_update80(branches,step,bits[mode],budget,trace);}
    *probability=double(joint);stats[0]=trace.evaluations;stats[1]=trace.merges;stats[2]=trace.peak;stats[3]=double(trace.cancellation);stats[4]=trace.parameter_recoveries;stats[5]=trace.update_recoveries;return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
// Stats: clicks, detector steps, zero-click samples, workers, completed,
// Gaussian component evaluations, compressions, transient peak components,
// maximum signed absolute weight sum. The transient peak may be 2*budget.
API int rg80_sample(void* handle,int count,std::uint64_t seed,int threads,int budget,const double* mean,const double* variance,std::uint8_t* output,double* stats,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&count>=0&&threads>=1&&threads<=256&&budget>=1&&budget<=128&&(budget&(budget-1))==0&&mean&&variance&&output&&stats,"Invalid branch batch");
    auto& model=*static_cast<ClickModel70*>(handle);int d=2*model.model->n,m=model.model->m;Vector center(d),deviation(d);
    for(int j=0;j<d;++j){need(std::isfinite(mean[j])&&std::isfinite(variance[j])&&variance[j]>=0,"Invalid branch displacement");center[j]=mean[j];deviation[j]=std::sqrt(variance[j]);}
    std::fill(stats,stats+11,0);std::fill(output,output+std::size_t(count)*m,0);if(!count)return 0;
    int workers=std::min(count,threads);std::atomic<std::int64_t> next{0},completed{0};std::atomic<bool> failed{false};std::vector<std::exception_ptr> failures(workers);std::vector<std::array<double,9>> local(workers);std::vector<std::thread> pool;pool.reserve(workers);
    auto worker=[&](int id){try{PrecisionGuard precision;for(;;){if(failed.load())break;auto row=next.fetch_add(1);if(row>=count)break;
        std::mt19937_64 detector(mix(seed+2ULL*row+1)),noise(mix(seed+0x81c6f2d54a970e31ULL+row));std::normal_distribution<double> normal;Vector w=center;for(int j=0;j<d;++j)if(deviation[j]>0)w[j]+=deviation[j]*normal(noise);
        std::vector<Branch71> branches{{model.model->initial,w,1}};Trace80 trace;int clicks=0;
        for(int mode=0;mode<m;++mode){auto step=guarded_parameters80(model,branches,mode,trace);bool clicked=(long double)uniform(detector)<step.p1;output[std::size_t(row)*m+mode]=clicked;clicks+=clicked;if(mode+1<m)guarded_update80(branches,step,clicked,budget,trace);}
        local[id][0]+=clicks;local[id][1]+=m;local[id][2]+=(clicks==0);local[id][3]+=trace.evaluations;local[id][4]+=trace.merges;local[id][5]=std::max(local[id][5],trace.peak);local[id][6]=std::max(local[id][6],double(trace.cancellation));local[id][7]+=trace.parameter_recoveries;local[id][8]+=trace.update_recoveries;++completed;
    }}catch(...){failures[id]=std::current_exception();failed.store(true);}};
    try{for(int id=0;id<workers;++id)pool.emplace_back(worker,id);}catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}
    for(auto& thread:pool)thread.join();for(auto& row:local){for(int j=0;j<3;++j)stats[j]+=row[j];stats[5]+=row[3];stats[6]+=row[4];stats[7]=std::max(stats[7],row[5]);stats[8]=std::max(stats[8],row[6]);stats[9]+=row[7];stats[10]+=row[8];}
    stats[3]=workers;stats[4]=completed.load();for(auto& failure:failures)if(failure)std::rethrow_exception(failure);need(completed.load()==count,"Incomplete signed branch batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
