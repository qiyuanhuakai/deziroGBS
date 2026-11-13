// Zero-mean signed branches: extended-precision Gaussian state, binary128
// mixture weights, and exactly complementary binary128 per-component masses.
// Same finite-budget moment compression; not a new exact all-size algorithm.
#include "gaussian_click_branches_v1.cpp"
#include <quadmath.h>
using Q82=__float128;
using Mat82=Eigen::Matrix<long double,Eigen::Dynamic,Eigen::Dynamic>;
using Vec82=Eigen::Matrix<long double,Eigen::Dynamic,1>;
struct StableBranch82{Mat82 x;Q82 weight;};
struct StablePart82{Vec82 xa,xb;long double i00,i01,i11;Q82 p0,p1;};
struct StableStep82{std::vector<StablePart82> parts;Q82 p0,p1;};
struct StableTrace82{double evaluations=0,merges=0,peak=1;long double condition=1,residual=0;};
static bool finite82(Q82 x){return !isnanq(x)&&!isinfq(x);}
static StableStep82 step82(const ClickModel70& c,const std::vector<StableBranch82>& branches,int mode,StableTrace82& trace){
 StableStep82 out;Q82 z=0,o=0;Vec82 a=c.a.col(mode).cast<long double>(),b=c.b.col(mode).cast<long double>();
 for(auto& v:branches){StablePart82 p;p.xa=v.x*a;p.xb=v.x*b;long double b00=1+a.dot(p.xa),b11=1+b.dot(p.xb),b01=(a.dot(p.xb)+b.dot(p.xa))/2;Q82 det=Q82(b00)*b11-Q82(b01)*b01;
  need(b00>0&&b11>0&&det>0&&finite82(det),"Stable Husimi covariance invalid");p.i00=static_cast<long double>(Q82(b11)/det);p.i01=static_cast<long double>(-Q82(b01)/det);p.i11=static_cast<long double>(Q82(b00)/det);Q82 logp=-logq(det)/2;need(logp<=0&&finite82(logp),"Stable component vacuum probability invalid");p.p1=-expm1q(logp);p.p0=1-p.p1;z+=v.weight*p.p0;o+=v.weight*p.p1;out.parts.push_back(std::move(p));
 }
 Q82 mass=z+o;need(z>=0&&o>=0&&mass>0&&finite82(mass),"Stable signed conditional probability invalid; no clipping");out.p0=z/mass;out.p1=o/mass;trace.evaluations+=branches.size();return out;
}
static void update82(std::vector<StableBranch82>& branches,const StableStep82& step,bool click,int budget,StableTrace82& trace){
 Q82 probability=click?step.p1:step.p0;need(probability>0,"Stable forced zero-probability branch");std::vector<StableBranch82> next;next.reserve(branches.size()*(click?2:1));
 for(std::size_t k=0;k<branches.size();++k){auto& v=branches[k];auto& p=step.parts[k];if(click)next.push_back({v.x,v.weight/probability});if(p.p0>0&&v.weight!=0){StableBranch82 b=v;Mat82 u=p.xa*(p.i00*p.xa+p.i01*p.xb).transpose()+p.xb*(p.i01*p.xa+p.i11*p.xb).transpose();b.x-=u;b.x=((b.x+b.x.transpose())/2).eval();b.weight=v.weight*p.p0/probability*(click?-1:1);need(b.x.allFinite(),"Stable covariance overflow");next.push_back(std::move(b));}}
 need(!next.empty(),"Stable empty branch list");Q82 mass=0;for(auto& b:next)mass+=b.weight;need(mass>0&&finite82(mass),"Stable normalization failed");for(auto& b:next)b.weight/=mass;Q82 after=0,condition=0;for(auto& b:next){after+=b.weight;condition+=fabsq(b.weight);}Q82 residual=fabsq(after-1);need(residual<Q82(1e-18L),"Stable binary128 normalization lost");trace.residual=std::max(trace.residual,static_cast<long double>(residual));trace.condition=std::max(trace.condition,static_cast<long double>(condition));trace.peak=std::max(trace.peak,double(next.size()));
 if(next.size()>std::size_t(budget)){int d=next[0].x.rows();StableBranch82 merged{Mat82::Zero(d,d),1};for(int i=0;i<d;++i)for(int j=0;j<=i;++j){Q82 value=0;for(auto& b:next)value+=b.weight*Q82(b.x(i,j));merged.x(i,j)=merged.x(j,i)=static_cast<long double>(value/after);}need(merged.x.allFinite(),"Stable moment compression overflow");next.clear();next.push_back(std::move(merged));++trace.merges;}branches=std::move(next);
}
API int tc82_prefix(void* ptr,int length,const std::uint8_t* bits,int budget,double* probability,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(ptr&&bits&&probability&&stats&&budget>=1&&budget<=128,"Invalid stable prefix arguments");auto& c=*static_cast<ClickModel70*>(ptr);need(length>=0&&length<=std::min(16,c.model->m),"Stable prefix audit limit");std::fill(stats,stats+6,0);std::vector<StableBranch82> branches{{c.model->initial.cast<long double>(),1}};StableTrace82 trace;Q82 joint=1;
 for(int i=0;i<length;++i){need(bits[i]<=1,"Invalid bit");auto p=step82(c,branches,i,trace);joint*=bits[i]?p.p1:p.p0;if(joint==0)break;if(i+1<length)update82(branches,p,bits[i],budget,trace);stats[5]=i+1;}
 *probability=double(joint);stats[0]=trace.evaluations;stats[1]=trace.merges;stats[2]=trace.peak;stats[3]=double(trace.condition);stats[4]=double(trace.residual);return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int tc82_sample(void* ptr,int count,std::uint64_t seed,int threads,int budget,std::uint8_t* output,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(ptr&&output&&stats&&count>=0&&count<=1048576&&threads>=1&&threads<=64&&budget>=1&&budget<=128,"Invalid stable sample arguments");auto& c=*static_cast<ClickModel70*>(ptr);int m=c.model->m;std::fill(stats,stats+7,0);std::fill(output,output+std::size_t(count)*m,0);if(!count)return 0;int workers=std::min(threads,count);std::atomic<int> next{0},completed{0};std::atomic<bool> failed{false};std::vector<std::exception_ptr> failures(workers);std::vector<std::array<double,7>> traces(workers);std::vector<std::thread> pool;
 auto worker=[&](int id){try{PrecisionGuard guard;for(;;){if(failed.load())break;int row=next.fetch_add(1);if(row>=count)break;std::mt19937_64 rng(mix(seed+2ULL*row+1));std::vector<StableBranch82> branches{{c.model->initial.cast<long double>(),1}};StableTrace82 trace;int clicks=0;for(int mode=0;mode<m;++mode){auto p=step82(c,branches,mode,trace);bool bit=Q82(uniform(rng))<p.p1;output[std::size_t(row)*m+mode]=bit;clicks+=bit;if(mode+1<m)update82(branches,p,bit,budget,trace);}auto& t=traces[id];t[0]+=clicks;t[1]++;t[2]+=trace.evaluations;t[3]+=trace.merges;t[4]=std::max(t[4],trace.peak);t[5]=std::max(t[5],double(trace.condition));t[6]=std::max(t[6],double(trace.residual));++completed;}}catch(...){failures[id]=std::current_exception();failed.store(true);}};
 try{for(int i=0;i<workers;++i)pool.emplace_back(worker,i);}catch(...){failed.store(true);for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(auto& t:traces)for(int j=0;j<7;++j){if(j<4)stats[j]+=t[j];else stats[j]=std::max(stats[j],t[j]);}for(auto& f:failures)if(f)std::rethrow_exception(f);need(completed.load()==count,"Incomplete stable batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
