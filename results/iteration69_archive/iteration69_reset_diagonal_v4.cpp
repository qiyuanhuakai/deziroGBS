// Exploratory centered Gaussian photon-subtraction reset sampler.
// Reset law: Lim arXiv:2610.06526v1, S6-S7. It is an approximate PNR law.
// New implementation option: static occupied-subspace rejection proposals.
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#define API extern "C" __declspec(dllexport)
using Real=Eigen::MatrixXd;
using Complex=Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using Clock=std::chrono::steady_clock;
// NumPy on Windows can leave x87 precision at 53 bits although long double
// storage has 64 significand bits. Enclose native arithmetic in 64-bit x87
// precision and restore the caller's complete control word on every exit.
struct PrecisionGuard {
    unsigned short saved;
    PrecisionGuard(){asm volatile("fnstcw %0" : "=m"(saved));unsigned short current=(saved & ~0x0f00u)|0x0300u;asm volatile("fldcw %0" : : "m"(current));}
    ~PrecisionGuard(){asm volatile("fldcw %0" : : "m"(saved));}
};
API void gr_precision(double* values){unsigned short word;asm volatile("fnstcw %0" : "=m"(word));values[0]=word;volatile long double one=1.L,small=0x1.0p-60L;values[1]=double((one+small)-one);{PrecisionGuard guard;asm volatile("fnstcw %0" : "=m"(word));values[2]=word;values[3]=double((one+small)-one);}asm volatile("fnstcw %0" : "=m"(word));values[4]=word;}

static void need(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
static void error(char* out,int cap,const std::string& s){if(out&&cap>0){auto n=std::min(s.size(),std::size_t(cap-1));std::memcpy(out,s.data(),n);out[n]=0;}}
static long double up(long double x){return std::nextafter(x,std::numeric_limits<long double>::infinity());}
static double uniform(std::mt19937_64& rng){return double(rng()>>11)*0x1.0p-53;}
static std::uint64_t mix(std::uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
struct Model {int m,n;Complex v;Real initial;std::vector<long double> ell,cdf,column_cdf,column_total,entry_weights;long double total=0.;};
static Complex occupation(const Real& x){int n=int(x.rows()/2);Complex h(n,n);for(int i=0;i<n;++i)for(int j=0;j<n;++j)h(i,j)={.5*(x(i,j)+x(n+i,n+j)),.5*(x(n+i,j)-x(i,n+j))};return h;}
static long double rate(const Model& model,const Complex& h,int row){
    long double sum=0.;for(int i=0;i<model.n;++i){long double a=model.v(row,i).real(),b=model.v(row,i).imag();sum+=h(i,i).real()*(a*a+b*b);
        for(int j=i+1;j<model.n;++j){long double c=h(i,j).real(),d=h(i,j).imag(),e=model.v(row,j).real(),f=model.v(row,j).imag();sum+=2*((a*c-b*d)*e+(a*d+b*c)*f);}}
    need(std::isfinite(sum)&&sum>=0.,"Negative/nonfinite detector rate; no clipping");return sum;
}
static long double bound(const Complex& h){long double maximum=0.;for(int i=0;i<h.rows();++i){long double row=0.;for(int j=0;j<h.cols();++j){row=up(row+std::abs((long double)h(i,j).real()));row=up(row+std::abs((long double)h(i,j).imag()));}maximum=std::max(maximum,row);}long double guard=64.L*h.rows()*h.rows()*std::numeric_limits<long double>::epsilon();return up(maximum*(1+guard));}
static int categorical(const std::vector<long double>& cumulative,long double total,double u){need(total>0.&&std::isfinite(total),"Empty categorical law");auto it=std::upper_bound(cumulative.begin(),cumulative.end(),(long double)u*total);need(it!=cumulative.end(),"Categorical endpoint rounded outside support");return int(it-cumulative.begin());}
static int choose(const Model& model,const Complex& h,int method,std::mt19937_64& rng,std::uint64_t& trials,long double& chosen_rate){
    if(method==3){std::vector<long double> diagonal(model.n),mix_cdf(model.n);long double mix_total=0.;
        for(int j=0;j<model.n;++j){long double d=0.;for(int k=0;k<model.n;++k){d=up(d+std::abs((long double)h(j,k).real()));d=up(d+std::abs((long double)h(j,k).imag()));}diagonal[j]=up(d*(1+64.L*model.n*model.n*std::numeric_limits<long double>::epsilon()));long double weight=diagonal[j]*model.column_total[j],next=mix_total+weight;need(!(weight>0.&&next<=mix_total),"Positive column-mixture interval lost");mix_total=next;mix_cdf[j]=mix_total;}
        for(;;){need(trials<std::numeric_limits<std::uint64_t>::max(),"Proposal counter overflow");++trials;int column=categorical(mix_cdf,mix_total,uniform(rng));const auto* first=model.column_cdf.data()+std::size_t(column)*model.m;long double coordinate=(long double)uniform(rng)*model.column_total[column];const auto* found=std::upper_bound(first,first+model.m,coordinate);need(found!=first+model.m,"Column proposal endpoint lost");int i=int(found-first);long double ceiling=0.;for(int j=0;j<model.n;++j)ceiling+=diagonal[j]*model.entry_weights[std::size_t(i)*model.n+j];long double r=rate(model,h,i);need(ceiling>0.&&r<=ceiling,"Adaptive diagonal envelope violated");if((long double)uniform(rng)*ceiling<r){chosen_rate=r;return i;}}
    }

    if(method==2){long double c=bound(h);need(c>0.,"Empty occupation bound");for(;;){need(trials<std::numeric_limits<std::uint64_t>::max(),"Proposal counter overflow");++trials;int i=categorical(model.cdf,model.total,uniform(rng));long double r=rate(model,h,i),ceiling=c*model.ell[i];if(!(ceiling>0.&&r<=ceiling)){std::ostringstream dump;dump<<std::setprecision(25)<<"Rejection envelope violated rate="<<r<<" ceiling="<<ceiling<<" c="<<c<<" ell="<<model.ell[i]<<" detector="<<i<<" H="<<h<<" Vrow="<<model.v.row(i);throw std::runtime_error(dump.str());}if((long double)uniform(rng)*ceiling<r){chosen_rate=r;return i;}}}
    std::vector<long double> cumulative(model.m);long double total=0.;
    Complex multiplied;if(method==1)multiplied.noalias()=model.v*h;
    for(int i=0;i<model.m;++i){long double r=0.;if(method==0)r=rate(model,h,i);else for(int j=0;j<model.n;++j)r+=(long double)multiplied(i,j).real()*model.v(i,j).real()+(long double)multiplied(i,j).imag()*model.v(i,j).imag();need(r>=0.&&std::isfinite(r),"GEMM rate negative/nonfinite");long double next=total+r;need(!(r>0.&&next<=total),"Positive categorical interval lost");total=next;cumulative[i]=total;}
    ++trials;int i=categorical(cumulative,total,uniform(rng));chosen_rate=rate(model,h,i);return i;
}
struct Flow {Real x;double s=0.,p0=1.;bool stopped=true;};
static Flow evolve(const Real& x,double u){
    need(std::isfinite(u)&&u>=0.&&u<1.,"Invalid waiting uniform");Eigen::SelfAdjointEigenSolver<Real> solver(x);need(solver.info()==Eigen::Success,"Eigensolver failure");auto values=solver.eigenvalues();
    need(values.minCoeff()>-1.,"Invalid no-click determinant domain");long double logp=0.;for(int i=0;i<values.size();++i)logp-=.5L*std::log1p((long double)values[i]);need(logp<=0.&&std::isfinite(logp),"Invalid vacuum log probability");Flow result;result.p0=double(std::exp(logp));
    if((long double)u<=std::exp(logp))return result;
    long double target=std::log((long double)u),lo=0.,hi=1.;for(int step=0;step<80;++step){long double middle=(lo+hi)/2,logsurvival=0.;for(int i=0;i<values.size();++i)logsurvival-=.5L*std::log1p(middle*values[i]);if(logsurvival>target)lo=middle;else hi=middle;}
    result.s=double((lo+hi)/2);need(result.s>=0.&&result.s<1.,"No-click inversion rounded to terminal endpoint");Eigen::VectorXd damped=values;for(int i=0;i<values.size();++i)damped[i]=(1-result.s)*values[i]/(1+result.s*values[i]);result.x=solver.eigenvectors()*damped.asDiagonal()*solver.eigenvectors().transpose();result.x=(result.x+result.x.transpose()).eval()/2.;result.stopped=false;return result;
}
static void jump(const Model& model,Real& x,int mode,long double lambda){need(lambda>0.,"Zero rate selected");Eigen::VectorXd a(2*model.n),b(2*model.n);for(int k=0;k<model.n;++k){a[k]=model.v(mode,k).real();a[k+model.n]=-model.v(mode,k).imag();b[k]=model.v(mode,k).imag();b[k+model.n]=model.v(mode,k).real();}Eigen::VectorXd xa=x*a,xb=x*b;x+=(xa*xa.transpose()+xb*xb.transpose())/double(lambda);x=(x+x.transpose()).eval()/2.;need(x.allFinite(),"Reset overflow");}
API void* gr_create(int m,int n,const double* v,const double* x0,char* message,int cap){Model* model=nullptr;try{PrecisionGuard rounding;
    need(m>=n&&n>0&&v&&x0,"Invalid model dimensions");model=new Model;model->m=m;model->n=n;model->v.resize(m,n);model->initial.resize(2*n,2*n);
    for(int i=0;i<m;++i)for(int j=0;j<n;++j){auto k=2*(std::size_t(i)*n+j);need(std::isfinite(v[k])&&std::isfinite(v[k+1]),"Nonfinite isometry");model->v(i,j)={v[k],v[k+1]};}
    for(int i=0;i<2*n;++i)for(int j=0;j<2*n;++j){double value=x0[std::size_t(i)*2*n+j];need(std::isfinite(value)&&value==x0[std::size_t(j)*2*n+i],"Initial covariance is not finite symmetric");model->initial(i,j)=value;}
    need((model->v.adjoint()*model->v-Complex::Identity(n,n)).norm()<1e-10,"Input columns not orthonormal");model->ell.resize(m);model->cdf.resize(m);
    for(int i=0;i<m;++i){long double e=0.;for(int j=0;j<n;++j){long double a=model->v(i,j).real(),b=model->v(i,j).imag();if(a)e=up(e+up(a*a));if(b)e=up(e+up(b*b));}model->ell[i]=e;long double next=model->total+e;need(!(e>0.&&next<=model->total),"Positive proposal interval lost");model->total=next;model->cdf[i]=next;}
    model->column_cdf.resize(std::size_t(m)*n);model->entry_weights.resize(std::size_t(m)*n);model->column_total.resize(n);
    for(int j=0;j<n;++j){long double total=0.;for(int i=0;i<m;++i){long double a=model->v(i,j).real(),b=model->v(i,j).imag(),w=0.;if(a)w=up(a*a);if(b)w=up(w+up(b*b));model->entry_weights[std::size_t(i)*n+j]=w;long double next=total+w;need(!(w>0.&&next<=total),"Positive fixed-column proposal interval lost");total=next;model->column_cdf[std::size_t(j)*m+i]=total;}model->column_total[j]=total;}
    return model;
}catch(const std::exception& e){delete model;error(message,cap,e.what());return nullptr;}}
API void gr_destroy(void* handle){delete static_cast<Model*>(handle);}
// stats: photons, proposal trials, elapsed flow / selection / reset seconds,
// zero-count samples, maximum photons in one sample; times include native work.
API int gr_sample(void* handle,int count,std::uint64_t seed,int method,std::uint32_t* output,double* stats,char* message,int cap){try{PrecisionGuard rounding;
    need(handle&&count>=0&&method>=0&&method<=3&&output&&stats,"Invalid sampler call");auto& model=*static_cast<Model*>(handle);std::fill(stats,stats+7,0.);std::fill(output,output+std::size_t(count)*model.m,0);
    for(int sample=0;sample<count;++sample){std::mt19937_64 waits(mix(seed+2*sample)),detectors(mix(seed+2*sample+1));Real x=model.initial;std::uint64_t photons=0,trials=0;
        for(;;){auto start=Clock::now();auto flow=evolve(x,uniform(waits));stats[2]+=std::chrono::duration<double>(Clock::now()-start).count();if(flow.stopped)break;x=std::move(flow.x);start=Clock::now();auto h=occupation(x);long double lambda=0.;int mode=choose(model,h,method,detectors,trials,lambda);stats[3]+=std::chrono::duration<double>(Clock::now()-start).count();start=Clock::now();jump(model,x,mode,lambda);stats[4]+=std::chrono::duration<double>(Clock::now()-start).count();auto& value=output[std::size_t(sample)*model.m+mode];need(value<std::numeric_limits<std::uint32_t>::max(),"Photon storage overflow");++value;++photons;}
        stats[0]+=double(photons);stats[1]+=double(trials);stats[5]+=(photons==0);stats[6]=std::max(stats[6],double(photons));}
    return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int gr_probe(void* handle,const double* input,double waiting,int forced,double* output,double* rates,double* info,char* message,int cap){try{PrecisionGuard rounding;
    need(handle&&input&&output&&rates&&info,"Invalid probe buffers");auto& model=*static_cast<Model*>(handle);Real x(2*model.n,2*model.n);for(int i=0;i<x.rows();++i)for(int j=0;j<x.cols();++j)x(i,j)=input[std::size_t(i)*x.cols()+j];auto flow=evolve(x,waiting);info[0]=flow.stopped;info[1]=flow.p0;info[2]=flow.s;if(flow.stopped)return 0;
    auto h=occupation(flow.x);info[3]=double(bound(h));info[4]=double(model.total);for(int i=0;i<model.m;++i)rates[i]=double(rate(model,h,i));if(forced>=0){need(forced<model.m,"Invalid forced detector");jump(model,flow.x,forced,rate(model,h,forced));}
    for(int i=0;i<x.rows();++i)for(int j=0;j<x.cols();++j)output[std::size_t(i)*x.cols()+j]=flow.x(i,j);return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
