// Experimental displaced reset, Lim arXiv:2610.06526v1 S62--S65.
// Detector rejection applies to the TOTAL occupation including the mean.
// This is a different approximate law from centered reset; no exact GBS claim.
#include "iteration69_reset_blas_v5.cpp"
using Vector=Eigen::VectorXd;
struct DisplacedFlow {Real x;Vector w;double s=0.,p0=1.;bool stopped=true;};
static Complex total_occupation(const Real& x,const Vector& w){
    int n=int(w.size()/2);Eigen::VectorXcd z(n);
    for(int j=0;j<n;++j)z[j]=std::complex<double>(w[j],w[n+j])/std::sqrt(2.);
    return occupation(x)+z*z.adjoint();
}
static DisplacedFlow displaced_evolve(const Real& x,const Vector& w,double u){
    need(std::isfinite(u)&&u>=0.&&u<1.,"Invalid displaced waiting uniform");
    Eigen::SelfAdjointEigenSolver<Real> solver(x);need(solver.info()==Eigen::Success,"Displaced eigensolver failure");
    const auto& values=solver.eigenvalues();need(values.minCoeff()>-1.,"Invalid displaced determinant domain");
    Vector rotated=solver.eigenvectors().transpose()*w;
    auto log_survival=[&](long double s){long double logp=0.;for(int j=0;j<values.size();++j){long double d=1+s*values[j],a=rotated[j];logp-=.5L*(std::log1p(s*values[j])+s*a*a/d);}return logp;};
    long double logp=log_survival(1.);need(logp<=0.&&std::isfinite(logp),"Invalid displaced vacuum probability");
    DisplacedFlow result;result.p0=double(std::exp(logp));if((long double)u<=std::exp(logp))return result;
    long double target=std::log((long double)u),lo=0.,hi=1.;for(int step=0;step<80;++step){long double middle=(lo+hi)/2;if(log_survival(middle)>target)lo=middle;else hi=middle;}
    result.s=double((lo+hi)/2);need(result.s>=0.&&result.s<1.,"Displaced inversion rounded to terminal endpoint");
    Vector damped=values;double attenuation=std::sqrt(1-result.s);
    for(int j=0;j<values.size();++j){double denominator=1+result.s*values[j];damped[j]=(1-result.s)*values[j]/denominator;rotated[j]*=attenuation/denominator;}
    result.x=solver.eigenvectors()*damped.asDiagonal()*solver.eigenvectors().transpose();result.x=(result.x+result.x.transpose()).eval()/2.;result.w=solver.eigenvectors()*rotated;result.stopped=false;return result;
}
static void displaced_jump(const Model& model,Real& x,Vector& w,int mode,long double lambda){
    need(lambda>0.,"Zero displaced rate selected");Vector a(2*model.n),b(2*model.n);
    for(int k=0;k<model.n;++k){a[k]=model.v(mode,k).real();a[k+model.n]=-model.v(mode,k).imag();b[k]=model.v(mode,k).imag();b[k+model.n]=model.v(mode,k).real();}
    Vector xa=x*a,xb=x*b,increment=(xa*a.dot(w)+xb*b.dot(w))/double(lambda);
    x+=(xa*xa.transpose()+xb*xb.transpose())/double(lambda)-increment*increment.transpose();w+=increment;x=(x+x.transpose()).eval()/2.;need(x.allFinite()&&w.allFinite(),"Displaced reset overflow");
}
API int ds_sample(void* handle,int count,std::uint64_t seed,int method,const double* mean,const double* variance,std::uint32_t* output,double* stats,char* message,int cap){try{PrecisionGuard rounding;
    need(handle&&count>=0&&method>=0&&method<=4&&mean&&variance&&output&&stats,"Invalid displaced sampler call");auto& model=*static_cast<Model*>(handle);
    Vector center(2*model.n),deviation(2*model.n);for(int j=0;j<2*model.n;++j){need(std::isfinite(mean[j])&&std::isfinite(variance[j])&&variance[j]>=0.,"Invalid displacement distribution");center[j]=mean[j];deviation[j]=std::sqrt(variance[j]);}
    std::fill(stats,stats+7,0.);std::fill(output,output+std::size_t(count)*model.m,0);
    for(int sample=0;sample<count;++sample){std::mt19937_64 waits(mix(seed+2*sample)),detectors(mix(seed+2*sample+1)),noise(mix(seed+0x81c6f2d54a970e31ULL+sample));std::normal_distribution<double> normal;Real x=model.initial;Vector w=center;
        for(int j=0;j<w.size();++j)if(deviation[j]>0.)w[j]+=deviation[j]*normal(noise);
        std::uint64_t photons=0,trials=0;for(;;){auto start=Clock::now();auto flow=displaced_evolve(x,w,uniform(waits));stats[2]+=std::chrono::duration<double>(Clock::now()-start).count();if(flow.stopped)break;x=std::move(flow.x);w=std::move(flow.w);start=Clock::now();auto h=total_occupation(x,w);long double lambda=0.;int mode=choose(model,h,method,detectors,trials,lambda);stats[3]+=std::chrono::duration<double>(Clock::now()-start).count();start=Clock::now();displaced_jump(model,x,w,mode,lambda);stats[4]+=std::chrono::duration<double>(Clock::now()-start).count();auto& value=output[std::size_t(sample)*model.m+mode];need(value<std::numeric_limits<std::uint32_t>::max(),"Displaced photon storage overflow");++value;++photons;}
        stats[0]+=double(photons);stats[1]+=double(trials);stats[5]+=(photons==0);stats[6]=std::max(stats[6],double(photons));}
    return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int ds_probe(void* handle,const double* input,const double* mean,double waiting,int forced,double* output,double* next_mean,double* rates,double* info,char* message,int cap){try{PrecisionGuard rounding;
    need(handle&&input&&mean&&output&&next_mean&&rates&&info,"Invalid displaced probe buffers");auto& model=*static_cast<Model*>(handle);int d=2*model.n;Real x(d,d);Vector w(d);
    for(int i=0;i<d;++i){w[i]=mean[i];for(int j=0;j<d;++j)x(i,j)=input[std::size_t(i)*d+j];}need(x.allFinite()&&w.allFinite(),"Nonfinite displaced probe state");
    auto flow=displaced_evolve(x,w,waiting);info[0]=flow.stopped;info[1]=flow.p0;info[2]=flow.s;if(flow.stopped)return 0;
    auto h=total_occupation(flow.x,flow.w);for(int i=0;i<model.m;++i)rates[i]=double(rate(model,h,i));if(forced>=0){need(forced<model.m,"Invalid displaced forced detector");displaced_jump(model,flow.x,flow.w,forced,rate(model,h,forced));}
    for(int i=0;i<d;++i){next_mean[i]=flow.w[i];for(int j=0;j<d;++j)output[std::size_t(i)*d+j]=flow.x(i,j);}return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
