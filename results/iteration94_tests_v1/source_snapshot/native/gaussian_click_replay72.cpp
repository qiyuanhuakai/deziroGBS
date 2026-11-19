// Common explicit random inputs and CPU reference for the CUDA branch kernel.
#include "gaussian_click_branches_v1.cpp"
API int tc72_draws(int m,int d,int count,std::uint64_t seed,const double* mean,const double* variance,double* means,double* draws,char* message,int cap){try{
    PrecisionGuard precision;need(m>0&&d>0&&count>=0&&mean&&variance&&means&&draws,"Invalid replay random buffers");Vector deviation(d);for(int i=0;i<d;++i){need(std::isfinite(mean[i])&&std::isfinite(variance[i])&&variance[i]>=0,"Invalid replay displacement");deviation[i]=std::sqrt(variance[i]);}
    for(int row=0;row<count;++row){std::mt19937_64 detector(mix(seed+2ULL*row+1)),noise(mix(seed+0x81c6f2d54a970e31ULL+row));std::normal_distribution<double> normal;
        for(int i=0;i<d;++i)means[std::size_t(row)*d+i]=mean[i]+(deviation[i]>0?deviation[i]*normal(noise):0);
        for(int mode=0;mode<m;++mode)draws[std::size_t(row)*m+mode]=uniform(detector);
    }return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int tc72_cpu(void* handle,int count,int threads,int budget,const double* means,const double* draws,std::uint8_t* output,double* probabilities,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&count>=0&&threads>=1&&threads<=256&&budget>=1&&budget<=65536&&means&&draws&&output&&probabilities,"Invalid replay batch");auto& c=*static_cast<ClickModel70*>(handle);int d=2*c.model->n,m=c.model->m;int workers=std::min(count,threads);std::atomic<std::int64_t> next{0},completed{0};std::atomic<bool> failed{false};std::vector<std::exception_ptr> failures(workers);std::vector<std::thread> pool;
    auto work=[&](int id){try{PrecisionGuard precision;for(;;){if(failed.load())break;auto row=next.fetch_add(1);if(row>=count)break;Vector w(d);for(int j=0;j<d;++j){w[j]=means[std::size_t(row)*d+j];need(std::isfinite(w[j]),"Nonfinite replay mean");}std::vector<Branch71> states{{c.model->initial,w,1}};Trace71 trace;
        for(int mode=0;mode<m;++mode){double u=draws[std::size_t(row)*m+mode];need(std::isfinite(u)&&u>=0&&u<1,"Invalid replay uniform");auto step=parameters71(c,states,mode,trace);bool clicked=(long double)u<step.p1;output[std::size_t(row)*m+mode]=clicked;probabilities[std::size_t(row)*m+mode]=double(step.p1);if(mode+1<m)update71(states,step,clicked,budget,trace);}++completed;
    }}catch(...){failures[id]=std::current_exception();failed.store(true);}};
    try{for(int i=0;i<workers;++i)pool.emplace_back(work,i);}catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}
    for(auto& thread:pool)thread.join();for(auto& failure:failures)if(failure)std::rethrow_exception(failure);need(completed.load()==count,"Incomplete replay batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
