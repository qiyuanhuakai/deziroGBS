// Gaussian P moment functional times homogeneous photon-subtraction factors.
// X may be indefinite: no real P-distribution sampling is performed. Wick
// identities are analytic Gaussian moments. Retain at most K photon factors;
// after K+1 jumps match exact first/second moments back to a Gaussian.
// No finite tap, photon cutoff, probability clipping or failed-row rejection.
#include "gaussian_direct_click.cpp"
#include <sstream>
#include <iomanip>
using LD73=long double;
using LVec73=Eigen::Matrix<LD73,Eigen::Dynamic,1>;
using LMat73=Eigen::Matrix<LD73,Eigen::Dynamic,Eigen::Dynamic>;
struct Moments73{LD73 value;LVec73 gradient;LMat73 hessian;};
struct Wick73 {
    int length;const LVec73& mean;const LMat73& cov;std::vector<int> indices;std::vector<LD73> cache;std::vector<unsigned char> known;
    Wick73(const LVec73& m,const LMat73& c,std::vector<int> ids):length(int(ids.size())),mean(m),cov(c),indices(std::move(ids)),cache(std::size_t(1)<<length),known(cache.size(),0){cache[0]=1;known[0]=1;}
    LD73 get(unsigned mask){if(known[mask])return cache[mask];unsigned first=__builtin_ctz(mask),rest=mask&~(1U<<first);LD73 value=mean[indices[first]]*get(rest);for(unsigned scan=rest;scan;scan&=scan-1){auto j=__builtin_ctz(scan);value+=cov(indices[first],indices[j])*get(rest&~(1U<<j));}known[mask]=1;return cache[mask]=value;}
};
static Moments73 polynomial73(const LVec73& mean,const LMat73& cov,bool derivatives){
    int channels=int(mean.size()),factors=channels/2;need(channels%2==0&&factors<=4,"Polynomial degree limit");Moments73 result{0,LVec73::Zero(channels),LMat73::Zero(channels,channels)};
    for(int choice=0;choice<(1<<factors);++choice){std::vector<int> indices;for(int j=0;j<factors;++j){int index=2*j+((choice>>j)&1);indices.push_back(index);indices.push_back(index);}Wick73 w(mean,cov,std::move(indices));unsigned full=(1U<<(2*factors))-1;result.value+=w.get(full);
        if(derivatives)for(int j=0;j<factors;++j){int a=w.indices[2*j];unsigned without=full&~(1U<<(2*j));result.gradient[a]+=2*w.get(without);result.hessian(a,a)+=2*w.get(without&~(1U<<(2*j+1)));
            for(int k=0;k<j;++k){int b=w.indices[2*k];auto value=4*w.get(without&~(1U<<(2*k)));result.hessian(a,b)+=value;result.hessian(b,a)+=value;}}
    }LD73 scale=std::ldexp(1.L,-factors);result.value*=scale;result.gradient*=scale;result.hessian*=scale;return result;
}
static Real forms73(const ClickModel70& c,const std::vector<int>& modes){Real forms(2*c.model->n,2*modes.size());for(std::size_t i=0;i<modes.size();++i){forms.col(2*i)=c.a.col(modes[i]);forms.col(2*i+1)=c.b.col(modes[i]);}return forms;}
static Moments73 moments73(const Real& x,const Vector& mean,const Real& forms,bool derivatives){
    Real products=x*forms;LVec73 mu(forms.cols());LMat73 cov(forms.cols(),forms.cols());for(int i=0;i<forms.cols();++i){mu[i]=dot70(forms.col(i),mean);for(int j=0;j<forms.cols();++j)cov(i,j)=dot70(forms.col(i),products.col(j));}return polynomial73(mu,(cov+cov.transpose()).eval()/2,derivatives);
}
struct State73{Real x;Vector mean;std::vector<int> factors;};
struct Stats73{double photons=0,completed=0,maximum_photons=0,compressions=0,maximum_factors=0,flow_calls=0,root_iterations=0,root_fallbacks=0,residual=0;};
static bool flow73(const ClickModel70& c,State73& state,double uniform,Stats73& stats){
    ++stats.flow_calls;Eigen::SelfAdjointEigenSolver<Real> eig(state.x);need(eig.info()==Eigen::Success,"Polynomial flow eigensolver failure");Vector values=eig.eigenvalues(),rotated=eig.eigenvectors().transpose()*state.mean;need(values.minCoeff()>-1,"Polynomial determinant domain");
    Real forms=forms73(c,state.factors),rforms=eig.eigenvectors().transpose()*forms;int channels=forms.cols(),d=values.size();LD73 normalization=1;bool normalized=false;
    auto evaluate=[&](LD73 s,bool derivatives){
        LVec73 mu=LVec73::Zero(channels),mud=LVec73::Zero(channels);LMat73 cov=LMat73::Zero(channels,channels),covd=LMat73::Zero(channels,channels);LD73 logp=0,slope=0;
        for(int i=0;i<d;++i){LD73 lambda=values[i],a=rotated[i],den=1+s*lambda,t=lambda/den,z=a/den;logp-=.5L*(std::log1p(s*lambda)+s*a*a/den);slope-=.5L*(t+z*z);
            for(int j=0;j<channels;++j){LD73 u=rforms(i,j);mu[j]+=u*z;if(derivatives)mud[j]-=u*t*z;for(int k=0;k<channels;++k){LD73 uv=u*rforms(i,k);cov(j,k)+=uv*t;if(derivatives)covd(j,k)-=uv*t*t;}}
        }
        auto moments=polynomial73(mu,cov,derivatives);LD73 gaussian=std::exp(logp),p=gaussian*moments.value/normalization,derivative=0;
        if(derivatives){LD73 qprime=moments.gradient.dot(mud)+.5L*(moments.hessian.cwiseProduct(covd)).sum();derivative=gaussian*(slope*moments.value+qprime)/normalization;}
        if(!(std::isfinite(p)&&p>=0&&(!normalized||p<=1))){std::ostringstream text;text<<std::setprecision(24)<<"Polynomial survival failure s="<<s<<" p="<<p<<" moment="<<moments.value<<" normalization="<<normalization<<" factors="<<state.factors.size()<<" eig="<<values.transpose()<<" mean="<<state.mean.transpose();throw std::runtime_error(text.str());}return std::pair<LD73,LD73>(p,derivative);
    };
    auto initial=evaluate(0,true);normalization=initial.first;need(std::isfinite(normalization)&&normalization>0,"Polynomial state normalization invalid");normalized=true;
    if(uniform<=evaluate(1,false).first)return false;LD73 lo=0,hi=1,s=.5L;auto guess=std::log((LD73)uniform)/(initial.second/normalization);if(guess>0&&guess<1&&std::isfinite(guess))s=guess;bool converged=false;
    for(int i=0;i<32;++i){auto [p,derivative]=evaluate(s,true);++stats.root_iterations;LD73 residual=p-uniform;if(std::abs(residual)<=64*std::numeric_limits<LD73>::epsilon()*(1+std::abs(p))){converged=true;break;}if(residual>0)lo=s;else hi=s;auto proposed=s-residual/derivative;if(!(derivative<0&&std::isfinite(proposed)&&proposed>lo&&proposed<hi))proposed=(lo+hi)/2;s=proposed;}
    if(!converged){++stats.root_fallbacks;for(int i=0;i<64;++i){s=(lo+hi)/2;++stats.root_iterations;if(evaluate(s,false).first>uniform)lo=s;else hi=s;}s=(lo+hi)/2;}
    double rounded=double(s);need(rounded>=0&&rounded<1,"Polynomial root rounded to endpoint");stats.residual=std::max(stats.residual,double(std::abs(evaluate(rounded,false).first-uniform)));
    for(int i=0;i<d;++i){double den=1+rounded*values[i];rotated[i]*=std::sqrt(1-rounded)/den;values[i]*=(1-rounded)/den;}state.x=eig.eigenvectors()*values.asDiagonal()*eig.eigenvectors().transpose();state.x=(state.x+state.x.transpose()).eval()/2;state.mean=eig.eigenvectors()*rotated;return true;
}
static int detector73(const ClickModel70& c,const State73& state,double uniform){
    std::vector<LD73> cumulative;LD73 total=0;std::vector<int> factors=state.factors;factors.push_back(0);
    for(int mode=0;mode<c.model->m;++mode){factors.back()=mode;auto rate=moments73(state.x,state.mean,forms73(c,factors),false).value;need(std::isfinite(rate)&&rate>=0,"Polynomial detector rate invalid; no clipping");auto next=total+rate;need(!(rate>0&&next<=total),"Polynomial detector interval lost");total=next;cumulative.push_back(total);}return categorical(cumulative,total,uniform);
}
static void compress73(const ClickModel70& c,State73& state,Stats73& stats){
    Real forms=forms73(c,state.factors),products=state.x*forms;auto moment=moments73(state.x,state.mean,forms,true);need(std::isfinite(moment.value)&&moment.value>0,"Polynomial compression normalization invalid");
    Vector delta=(products.cast<LD73>()*(moment.gradient/moment.value)).cast<double>();Real correction=(products.cast<LD73>()*(moment.hessian/moment.value)*products.transpose().cast<LD73>()).cast<double>();state.mean+=delta;state.x+=correction-delta*delta.transpose();state.x=(state.x+state.x.transpose()).eval()/2;need(state.x.allFinite()&&state.mean.allFinite(),"Polynomial compression overflow");state.factors.clear();++stats.compressions;
}
API int pnr74_sample(void* handle,int count,std::uint64_t seed,int threads,int retained,const double* mean,const double* variance,std::uint32_t* output,double* stats,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&count>=0&&threads>=1&&threads<=256&&retained>=0&&retained<=3&&mean&&variance&&output&&stats,"Invalid polynomial PNR batch");auto& c=*static_cast<ClickModel70*>(handle);int d=2*c.model->n,m=c.model->m;Vector center(d),deviation(d);for(int j=0;j<d;++j){need(std::isfinite(mean[j])&&std::isfinite(variance[j])&&variance[j]>=0,"Invalid polynomial displacement");center[j]=mean[j];deviation[j]=std::sqrt(variance[j]);}
    std::fill(stats,stats+9,0);std::fill(output,output+std::size_t(count)*m,0);if(count==0)return 0;int workers=std::min(count,threads);std::atomic<std::int64_t> next{0},completed{0};std::atomic<bool> failed{false};std::vector<Stats73> local(workers);std::vector<std::exception_ptr> failures(workers);std::vector<std::thread> pool;
    auto worker=[&](int id){try{PrecisionGuard guard;auto& trace=local[id];for(;;){if(failed.load())break;auto row=next.fetch_add(1);if(row>=count)break;std::mt19937_64 waits(mix(seed+2ULL*row)),detectors(mix(seed+2ULL*row+1)),noise(mix(seed+0x81c6f2d54a970e31ULL+row));std::normal_distribution<double> normal;State73 state{c.model->initial,center,{}};for(int j=0;j<d;++j)if(deviation[j]>0)state.mean[j]+=deviation[j]*normal(noise);std::uint64_t photons=0;
        while(flow73(c,state,uniform(waits),trace)){int mode=detector73(c,state,uniform(detectors));state.factors.push_back(mode);trace.maximum_factors=std::max(trace.maximum_factors,double(state.factors.size()));if(state.factors.size()>std::size_t(retained))compress73(c,state,trace);auto& value=output[std::size_t(row)*m+mode];need(value<std::numeric_limits<std::uint32_t>::max(),"PNR storage overflow");++value;++photons;}
        trace.photons+=double(photons);++trace.completed;trace.maximum_photons=std::max(trace.maximum_photons,double(photons));++completed;
    }}catch(...){failures[id]=std::current_exception();failed.store(true);}};
    try{for(int i=0;i<workers;++i)pool.emplace_back(worker,i);}catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}for(auto& thread:pool)thread.join();for(auto& t:local){stats[0]+=t.photons;stats[1]+=t.completed;stats[2]=std::max(stats[2],t.maximum_photons);stats[3]+=t.compressions;stats[4]=std::max(stats[4],t.maximum_factors);stats[5]+=t.flow_calls;stats[6]+=t.root_iterations;stats[7]+=t.root_fallbacks;stats[8]=std::max(stats[8],t.residual);}for(auto& e:failures)if(e)std::rethrow_exception(e);need(completed.load()==count,"Incomplete polynomial PNR batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
// Probe arbitrary Gaussian polynomial moments for independent Wick/Fock checks.
API int pnr74_probe(void* handle,const double* x,const double* mean,int length,const int* modes,double* value,double* output,double* outmean,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&x&&mean&&modes&&value&&output&&outmean&&length>=1&&length<=4,"Invalid polynomial probe");auto& c=*static_cast<ClickModel70*>(handle);int d=2*c.model->n;State73 state{Real(d,d),Vector(d),{}};for(int j=0;j<d;++j){state.mean[j]=mean[j];for(int k=0;k<d;++k)state.x(j,k)=x[j*d+k];}for(int j=0;j<length;++j){need(modes[j]>=0&&modes[j]<c.model->m,"Invalid probe mode");state.factors.push_back(modes[j]);}*value=double(moments73(state.x,state.mean,forms73(c,state.factors),false).value);Stats73 stats;compress73(c,state,stats);for(int j=0;j<d;++j){outmean[j]=state.mean[j];for(int k=0;k<d;++k)output[j*d+k]=state.x(j,k);}return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
