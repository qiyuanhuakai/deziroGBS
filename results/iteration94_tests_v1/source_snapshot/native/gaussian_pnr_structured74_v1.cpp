// Explicit product-pure squeezing parameters preserve endpoint structure.
// After the first moment compression, use the frozen general Gaussian core.
#include "gaussian_pnr_polynomial73_v1.cpp"
#include "gaussian_positive_norm74_v1.hpp"
static LD73 pure_norm74(const ClickModel70& c,const Vector& squeezing,const LVec73& tilted_mean,const std::vector<int>& modes,LD73 remaining){
    int k=modes.size(),n=squeezing.size();if(!k)return 1;CVector74 alpha(k);CMatrix74 anomalous=CMatrix74::Zero(k,k),factor(k,n);
    for(int j=0;j<k;++j){alpha[j]=Complex74(c.a.col(modes[j]).cast<LD73>().dot(tilted_mean),c.b.col(modes[j]).cast<LD73>().dot(tilted_mean))/std::sqrt(2.L);
        for(int q=0;q<n;++q){LD73 t=squeezing[q],den=1-remaining*remaining*t*t;need(den>0,"Structured squeeze domain");auto v=c.model->v(modes[j],q);factor(j,q)=Complex74(v.real(),v.imag())*std::sqrt(remaining*t*t/den);}}
    for(int j=0;j<k;++j)for(int l=0;l<k;++l)for(int q=0;q<n;++q){LD73 t=squeezing[q],den=1-remaining*remaining*t*t;auto v=c.model->v(modes[j],q),w=c.model->v(modes[l],q);anomalous(j,l)+=Complex74(v.real(),v.imag())*Complex74(w.real(),w.imag())*(-t/den);}
    return positive_norm74(alpha,anomalous,factor);
}
static bool pure_flow74(const ClickModel70& c,State73& state,Vector& squeezing,double uniform,Stats73& stats){
    ++stats.flow_calls;int n=squeezing.size(),d=2*n;Real forms=forms73(c,state.factors);LD73 normalization=1;bool normalized=false;
    auto evaluate=[&](LD73 s,bool derivative){LD73 logp=0,slope=0;LVec73 tilted(d),mu=LVec73::Zero(forms.cols()),mud=mu;LMat73 cov=LMat73::Zero(forms.cols(),forms.cols()),covd=cov;
        for(int i=0;i<d;++i){LD73 t=squeezing[i%n],lambda=i<n?-t/(1+t):t/(1-t),a=state.mean[i],den=1+s*lambda,z=a/den,x=lambda/den;tilted[i]=z;logp-=.5L*(std::log1p(s*lambda)+s*a*a/den);slope-=.5L*(x+z*z);
            if(derivative)for(int j=0;j<forms.cols();++j){LD73 u=forms(i,j);mu[j]+=u*z;mud[j]-=u*x*z;for(int k=0;k<forms.cols();++k){LD73 uv=u*forms(i,k);cov(j,k)+=uv*x;covd(j,k)-=uv*x*x;}}}
        LD73 mass=pure_norm74(c,squeezing,tilted,state.factors,1-s),gaussian=std::exp(logp),p=gaussian*mass/normalization,dp=0;
        if(derivative){auto moments=polynomial73(mu,cov,true);LD73 prime=moments.gradient.dot(mud)+.5L*(moments.hessian.cwiseProduct(covd)).sum();dp=gaussian*(slope*mass+prime)/normalization;}
        need(std::isfinite(p)&&p>=0&&(!normalized||p<=1),"Structured survival outside [0,1]; no clipping");return std::pair<LD73,LD73>(p,dp);
    };
    auto initial=evaluate(0,true);normalization=initial.first;need(normalization>0,"Structured normalization zero");normalized=true;
    if(uniform<=evaluate(1,false).first)return false;LD73 lo=0,hi=1,s=.5L;auto guess=std::log((LD73)uniform)/(initial.second/normalization);if(guess>0&&guess<1&&std::isfinite(guess))s=guess;bool converged=false;
    for(int i=0;i<32;++i){auto [p,dp]=evaluate(s,true);++stats.root_iterations;LD73 residual=p-uniform;if(std::abs(residual)<=64*std::numeric_limits<LD73>::epsilon()*(1+std::abs(p))){converged=true;break;}if(residual>0)lo=s;else hi=s;auto proposed=s-residual/dp;if(!(dp<0&&std::isfinite(proposed)&&proposed>lo&&proposed<hi))proposed=(lo+hi)/2;s=proposed;}
    if(!converged){++stats.root_fallbacks;for(int i=0;i<64;++i){s=(lo+hi)/2;++stats.root_iterations;if(evaluate(s,false).first>uniform)lo=s;else hi=s;}s=(lo+hi)/2;}
    double rounded=double(s);need(rounded>=0&&rounded<1,"Structured root rounded to endpoint");stats.residual=std::max(stats.residual,double(std::abs(evaluate(rounded,false).first-uniform)));state.x.setZero();
    for(int i=0;i<n;++i){LD73 t=squeezing[i],remain=1-(LD73)rounded;state.mean[i]*=double(std::sqrt(remain)*(1+t)/(1+remain*t));state.mean[i+n]*=double(std::sqrt(remain)*(1-t)/(1-remain*t));squeezing[i]=double(remain*t);t=squeezing[i];state.x(i,i)=double(-t/(1+t));state.x(i+n,i+n)=double(t/(1-t));}return true;
}
static int pure_detector74(const ClickModel70& c,const State73& state,const Vector& squeezing,double uniform){std::vector<int> modes=state.factors;modes.push_back(0);std::vector<LD73> cumulative;LD73 total=0;
    for(int i=0;i<c.model->m;++i){modes.back()=i;LD73 rate=pure_norm74(c,squeezing,state.mean.cast<LD73>(),modes,1);LD73 next=total+rate;need(!(rate>0&&next<=total),"Structured detector interval lost");total=next;cumulative.push_back(total);}return categorical(cumulative,total,uniform);
}
API int pnrs74_norm(int modes,int rank,const double* mean,const double* anomalous,const double* factor,double* result,char* message,int cap){try{
    PrecisionGuard guard;need(modes>=0&&modes<=8&&rank>=0&&rank<=256&&mean&&anomalous&&factor&&result,"Positive norm probe arguments");CVector74 a(modes);CMatrix74 m(modes,modes),l(modes,rank);
    for(int i=0;i<modes;++i){a[i]=Complex74(mean[2*i],mean[2*i+1]);for(int j=0;j<modes;++j){int z=2*(i*modes+j);m(i,j)=Complex74(anomalous[z],anomalous[z+1]);}for(int j=0;j<rank;++j){int z=2*(i*rank+j);l(i,j)=Complex74(factor[z],factor[z+1]);}}*result=double(positive_norm74(a,m,l));return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int pnrs74_endpoint(void* handle,const double* squeezing,const double* mean,int length,const int* modes,double remaining,double* result,char* message,int cap){try{
    PrecisionGuard guard;need(handle&&squeezing&&mean&&modes&&result&&length>=0&&length<=8&&remaining>=0&&remaining<=1,"Endpoint arguments");auto& c=*static_cast<ClickModel70*>(handle);Vector t(c.model->n),w(2*c.model->n);std::vector<int> ids;
    for(int i=0;i<t.size();++i){t[i]=squeezing[i];need(std::isfinite(t[i])&&std::abs(t[i])<1,"Squeezing outside (-1,1)");}for(int i=0;i<w.size();++i)w[i]=mean[i];for(int j=0;j<length;++j){need(modes[j]>=0&&modes[j]<c.model->m,"Mode index");ids.push_back(modes[j]);}*result=double(pure_norm74(c,t,w.cast<LD73>(),ids,remaining));return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int pnrs74_sample(void* handle,const double* squeezing,int count,std::uint64_t seed,int threads,int retained,const double* mean,const double* variance,std::uint32_t* output,double* stats,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&squeezing&&count>=0&&threads>=1&&threads<=256&&retained>=0&&retained<=3&&mean&&variance&&output&&stats,"Invalid polynomial PNR batch");auto& c=*static_cast<ClickModel70*>(handle);int d=2*c.model->n,m=c.model->m;Vector center(d),deviation(d),initial_t(d/2);for(int j=0;j<d/2;++j){initial_t[j]=squeezing[j];need(std::isfinite(initial_t[j])&&std::abs(initial_t[j])<1,"Invalid structured squeezing");}for(int j=0;j<d;++j){need(std::isfinite(mean[j])&&std::isfinite(variance[j])&&variance[j]>=0,"Invalid polynomial displacement");center[j]=mean[j];deviation[j]=std::sqrt(variance[j]);}
    std::fill(stats,stats+9,0);std::fill(output,output+std::size_t(count)*m,0);if(count==0)return 0;int workers=std::min(count,threads);std::atomic<std::int64_t> next{0},completed{0};std::atomic<bool> failed{false};std::vector<Stats73> local(workers);std::vector<std::exception_ptr> failures(workers);std::vector<std::thread> pool;
    auto worker=[&](int id){try{PrecisionGuard guard;auto& trace=local[id];for(;;){if(failed.load())break;auto row=next.fetch_add(1);if(row>=count)break;std::mt19937_64 waits(mix(seed+2ULL*row)),detectors(mix(seed+2ULL*row+1)),noise(mix(seed+0x81c6f2d54a970e31ULL+row));std::normal_distribution<double> normal;State73 state{Real::Zero(d,d),center,{}};Vector current_t=initial_t;bool pure=true;for(int j=0;j<d/2;++j){LD73 t=current_t[j];state.x(j,j)=double(-t/(1+t));state.x(j+d/2,j+d/2)=double(t/(1-t));}for(int j=0;j<d;++j)if(deviation[j]>0)state.mean[j]+=deviation[j]*normal(noise);std::uint64_t photons=0;
        while(pure?pure_flow74(c,state,current_t,uniform(waits),trace):flow73(c,state,uniform(waits),trace)){int mode=pure?pure_detector74(c,state,current_t,uniform(detectors)):detector73(c,state,uniform(detectors));state.factors.push_back(mode);trace.maximum_factors=std::max(trace.maximum_factors,double(state.factors.size()));if(state.factors.size()>std::size_t(retained)){compress73(c,state,trace);pure=false;}auto& value=output[std::size_t(row)*m+mode];need(value<std::numeric_limits<std::uint32_t>::max(),"PNR storage overflow");++value;++photons;}
        trace.photons+=double(photons);++trace.completed;trace.maximum_photons=std::max(trace.maximum_photons,double(photons));++completed;
    }}catch(...){failures[id]=std::current_exception();failed.store(true);}};
    try{for(int i=0;i<workers;++i)pool.emplace_back(worker,i);}catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}for(auto& thread:pool)thread.join();for(auto& t:local){stats[0]+=t.photons;stats[1]+=t.completed;stats[2]=std::max(stats[2],t.maximum_photons);stats[3]+=t.compressions;stats[4]=std::max(stats[4],t.maximum_factors);stats[5]+=t.flow_calls;stats[6]+=t.root_iterations;stats[7]+=t.root_fallbacks;stats[8]=std::max(stats[8],t.residual);}for(auto& e:failures)if(e)std::rethrow_exception(e);need(completed.load()==count,"Incomplete polynomial PNR batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
