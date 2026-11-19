// Direct threshold Gaussian moment closure, with O(M N^2) work per trajectory.
// Exact no-click Gaussian conditioning; on click retain conditional first and
// second moments only. The click-conditioned state is NOT itself Gaussian.
// Gaussian difference identity: Quesada/Arrazola/Killoran, arXiv:1807.01639.
#include "gaussian_reset_cpu.cpp"
#include <memory>
#include <thread>
#include <atomic>
#include <array>
#include <exception>
struct ClickModel70 {std::unique_ptr<Model> model;Real a,b;};
API void* tc70_create(int m,int n,const double* v,const double* x,char* message,int cap){try{
    PrecisionGuard rounding;std::unique_ptr<ClickModel70> c(new ClickModel70);
    c->model.reset(static_cast<Model*>(gr_create(m,n,v,x,message,cap)));if(!c->model)return nullptr;
    c->a.resize(2*n,m);c->b.resize(2*n,m);
    for(int i=0;i<m;++i)for(int j=0;j<n;++j){auto z=c->model->v(i,j);c->a(j,i)=z.real();c->a(j+n,i)=-z.imag();c->b(j,i)=z.imag();c->b(j+n,i)=z.real();}
    return c.release();
}catch(const std::exception& e){error(message,cap,e.what());return nullptr;}}
API void tc70_destroy(void* handle){delete static_cast<ClickModel70*>(handle);}
static long double dot70(const Vector& a,const Vector& b){long double s=0.;for(int j=0;j<a.size();++j)s+=(long double)a[j]*b[j];return s;}
struct ClickStep70 {Vector xa,xb,delta;long double p0,p1,i00,i01,i11;};
static ClickStep70 click_parameters70(const ClickModel70& c,const Real& x,const Vector& w,int mode){
    Vector a=c.a.col(mode),b=c.b.col(mode);ClickStep70 p;p.xa=x*a;p.xb=x*b;
    long double b00=1+dot70(a,p.xa),b11=1+dot70(b,p.xb),b01=(dot70(a,p.xb)+dot70(b,p.xa))/2,det=b00*b11-b01*b01;
    need(b00>0.&&b11>0.&&det>0.&&std::isfinite(det),"Invalid direct-click Husimi covariance");
    p.i00=b11/det;p.i01=-b01/det;p.i11=b00/det;long double mq=dot70(a,w),mp=dot70(b,w);
    long double sq=p.i00*mq+p.i01*mp,sp=p.i01*mq+p.i11*mp;
    long double logp=-.5L*(std::log(det)+mq*sq+mp*sp);
    need(std::isfinite(logp)&&logp<=0.,"Invalid direct-click vacuum probability; no clipping");
    p.p0=std::exp(logp);p.p1=-std::expm1(logp);p.delta=double(sq)*p.xa+double(sp)*p.xb;return p;
}
static void click_update70(Real& x,Vector& w,const ClickStep70& p,bool click){
    Real u=p.xa*(double(p.i00)*p.xa+double(p.i01)*p.xb).transpose()+p.xb*(double(p.i01)*p.xa+double(p.i11)*p.xb).transpose();
    if(click){need(p.p1>0.,"Forced impossible click");long double q=p.p0/p.p1;x+=double(q)*u-double(q/p.p1)*(p.delta*p.delta.transpose());w+=double(q)*p.delta;}
    else{need(p.p0>0.,"Forced impossible no-click");x-=u;w-=p.delta;}
    x=(x+x.transpose()).eval()/2.;need(x.allFinite()&&w.allFinite(),"Direct-click moment overflow");
}
API int tc70_probe(void* handle,const double* input,const double* mean,int mode,int clicked,double* output,double* next_mean,double* probability,char* message,int cap){try{
    PrecisionGuard rounding;need(handle&&input&&mean&&output&&next_mean&&probability&&(clicked==0||clicked==1),"Invalid direct-click probe");auto& c=*static_cast<ClickModel70*>(handle);int d=2*c.model->n;need(mode>=0&&mode<c.model->m,"Invalid direct-click mode");Real x(d,d);Vector w(d);
    for(int i=0;i<d;++i){w[i]=mean[i];for(int j=0;j<d;++j)x(i,j)=input[std::size_t(i)*d+j];}need(x.allFinite()&&w.allFinite(),"Nonfinite direct-click probe");
    auto p=click_parameters70(c,x,w,mode);probability[0]=double(p.p0);probability[1]=double(p.p1);click_update70(x,w,p,clicked);
    for(int i=0;i<d;++i){next_mean[i]=w[i];for(int j=0;j<d;++j)output[std::size_t(i)*d+j]=x(i,j);}return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
// Output is M bits stored as uint8_t per row. No photon counts are generated.
// Stats: clicks, detector steps, zero-click records, launched workers, completed.
API int tc70_sample(void* handle,int count,std::uint64_t seed,int threads,const double* mean,const double* variance,std::uint8_t* output,double* stats,char* message,int cap){try{
    PrecisionGuard rounding;need(handle&&count>=0&&threads>=1&&threads<=256&&mean&&variance&&output&&stats,"Invalid direct-click batch");auto& c=*static_cast<ClickModel70*>(handle);int d=2*c.model->n,m=c.model->m;Vector center(d),deviation(d);
    for(int j=0;j<d;++j){need(std::isfinite(mean[j])&&std::isfinite(variance[j])&&variance[j]>=0.,"Invalid direct-click displacement");center[j]=mean[j];deviation[j]=std::sqrt(variance[j]);}
    std::fill(stats,stats+5,0.);std::fill(output,output+std::size_t(count)*m,0);if(!count)return 0;int workers=std::min(count,threads);std::atomic<std::int64_t> next{0},completed{0};std::atomic<bool> failed{false};std::vector<std::exception_ptr> failures(workers);std::vector<std::array<double,3>> local(workers);std::vector<std::thread> pool;pool.reserve(workers);
    auto worker=[&](int id){try{PrecisionGuard precision;for(;;){if(failed.load())break;auto row=next.fetch_add(1);if(row>=count)break;std::mt19937_64 detector(mix(seed+2ULL*row+1)),noise(mix(seed+0x81c6f2d54a970e31ULL+row));std::normal_distribution<double> normal;Real x=c.model->initial;Vector w=center;for(int j=0;j<d;++j)if(deviation[j]>0.)w[j]+=deviation[j]*normal(noise);int clicks=0;
        for(int mode=0;mode<m;++mode){auto p=click_parameters70(c,x,w,mode);bool clicked=(long double)uniform(detector)<p.p1;output[std::size_t(row)*m+mode]=clicked;clicks+=clicked;if(mode+1<m)click_update70(x,w,p,clicked);}
        local[id][0]+=clicks;local[id][1]+=m;local[id][2]+=(clicks==0);++completed;
    }}catch(...){failures[id]=std::current_exception();failed.store(true);}};
    try{for(int id=0;id<workers;++id)pool.emplace_back(worker,id);}catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}
    for(auto& thread:pool)thread.join();for(auto& a:local)for(int j=0;j<3;++j)stats[j]+=a[j];stats[3]=workers;stats[4]=completed.load();for(auto& failure:failures)if(failure)std::rethrow_exception(failure);need(completed.load()==count,"Incomplete direct-click batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
