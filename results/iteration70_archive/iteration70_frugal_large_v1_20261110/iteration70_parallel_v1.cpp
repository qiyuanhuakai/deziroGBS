#include "iteration70_flow_v2.cpp"

// Independent trajectories, deterministic row seeds, shared read-only model.
// Stats 0..17 are summed worker CPU component times / counters (not wall time),
// except maxima 6,12,13,16. 18=workers launched, 19=completed rows.
#include <thread>
#include <atomic>
#include <array>
API int g70_sample_parallel(void* handle,int count,std::uint64_t seed,int selector,int algorithm,void* eigen_function,int threads,const double* mean,const double* variance,std::uint32_t* output,double* aggregate,char* message,int cap){try{
    PrecisionGuard rounding;
    need(handle&&count>=0&&selector>=0&&selector<=5&&algorithm>=0&&algorithm<=2&&threads>=1&&threads<=256&&mean&&variance&&output&&aggregate,"Invalid parallel sampler call");
    const auto& model=*static_cast<Model*>(handle);need(algorithm!=2||eigen_function,"Missing LAPACKE function");
    Vector center(2*model.n),deviation(2*model.n);
    for(int j=0;j<2*model.n;++j){need(std::isfinite(mean[j])&&std::isfinite(variance[j])&&variance[j]>=0.,"Invalid parallel displacement");center[j]=mean[j];deviation[j]=std::sqrt(variance[j]);}
    std::fill(aggregate,aggregate+20,0.);std::fill(output,output+std::size_t(count)*model.m,0);
    if(count==0)return 0;
    int workers=std::min(threads,count);
    std::vector<std::array<double,18>> local(workers);std::vector<std::string> failures(workers);
    std::atomic<int> next{0},completed{0};std::atomic<bool> failed{false};std::vector<std::thread> pool;pool.reserve(workers);
    auto worker=[&](int id){try{
        PrecisionGuard worker_rounding;
        EVDWorkspace70 workspace(2*model.n,algorithm==2?reinterpret_cast<EVD70>(eigen_function):nullptr);
        double* stats=local[id].data();
        for(;;){if(failed.load())break;int sample=next.fetch_add(1);if(sample>=count)break;
        std::mt19937_64 waits(mix(seed+2ULL*std::uint64_t(sample))),detectors(mix(seed+2ULL*std::uint64_t(sample)+1)),noise(mix(seed+0x81c6f2d54a970e31ULL+sample));std::normal_distribution<double> normal;Real x=model.initial;Vector w=center;for(int j=0;j<w.size();++j)if(deviation[j]>0.)w[j]+=deviation[j]*normal(noise);Anchor anchor;
        if(selector==5){auto start=Clock::now();anchor=make_anchor(model,w);stats[7]+=std::chrono::duration<double>(Clock::now()-start).count();}std::uint64_t photons=0,trials=0;
        for(;;){auto start=Clock::now();auto flow=flow70(x,w,uniform(waits),algorithm,stats,workspace);stats[2]+=std::chrono::duration<double>(Clock::now()-start).count();if(flow.stopped)break;x=std::move(flow.x);w=std::move(flow.w);start=Clock::now();long double lambda=0.;int mode;if(selector==5){auto h=occupation(x);mode=choose_anchor(model,h,w,anchor,detectors,trials,lambda);}else{auto h=total_occupation(x,w);mode=choose(model,h,selector,detectors,trials,lambda);}stats[3]+=std::chrono::duration<double>(Clock::now()-start).count();start=Clock::now();displaced_jump(model,x,w,mode,lambda);stats[4]+=std::chrono::duration<double>(Clock::now()-start).count();auto& value=output[std::size_t(sample)*model.m+mode];need(value<std::numeric_limits<std::uint32_t>::max(),"Profiled photon storage overflow");++value;++photons;}
        stats[0]+=double(photons);stats[1]+=double(trials);stats[5]+=(photons==0);stats[6]=std::max(stats[6],double(photons));
        ++completed;
        }
    }catch(const std::exception& e){failures[id]=e.what();failed.store(true);}catch(...){failures[id]="Nonstandard worker exception";failed.store(true);}};
    try{for(int id=0;id<workers;++id)pool.emplace_back(worker,id);}catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}
    for(auto& thread:pool)thread.join();
    for(int id=0;id<workers;++id)for(int j=0;j<18;++j){if(j==6||j==12||j==13||j==16)aggregate[j]=std::max(aggregate[j],local[id][j]);else aggregate[j]+=local[id][j];}
    aggregate[18]=workers;aggregate[19]=completed.load();
    if(failed.load()){for(auto& failure:failures)if(!failure.empty())throw std::runtime_error(failure);throw std::runtime_error("Parallel worker creation failed");}
    need(completed.load()==count,"Incomplete parallel batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
