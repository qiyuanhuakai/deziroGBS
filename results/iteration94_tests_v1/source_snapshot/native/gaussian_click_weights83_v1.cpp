// Targeted signed-weight precision experiment. Two binary128 components retain
// about 226 significant binary bits away from overflow/underflow. Covariances
// remain long double as in stable82, so this does not certify state arithmetic.
// Same finite-budget Gaussian moment compression; no threshold relaxation.
#include "gaussian_click_stable82_v1.cpp"

struct Weight83 {
 Q82 hi=0,lo=0;
 Weight83()=default;
 Weight83(Q82 v):hi(v),lo(0){}
 Weight83(Q82 h,Q82 l):hi(h),lo(l){}
};
static Weight83 sum83(Q82 a,Q82 b){Q82 s=a+b,v=s-a;return {s,(a-(s-v))+(b-v)};}
static Weight83 operator+(Weight83 a,Weight83 b){auto s=sum83(a.hi,b.hi),t=sum83(a.lo,b.lo);s.lo+=t.hi;s=sum83(s.hi,s.lo);s.lo+=t.lo;return sum83(s.hi,s.lo);}
static Weight83 operator-(Weight83 a){return {-a.hi,-a.lo};}
static Weight83 operator-(Weight83 a,Weight83 b){return a+(-b);}
static Weight83 operator*(Weight83 a,Weight83 b){Q82 p=a.hi*b.hi;Weight83 r{p,fmaq(a.hi,b.hi,-p)};r=r+Weight83(a.hi*b.lo);r=r+Weight83(a.lo*b.hi);return r+Weight83(a.lo*b.lo);}
static Weight83 operator/(Weight83 a,Weight83 b){Q82 q1=a.hi/b.hi;Weight83 r=a-b*Weight83(q1);Q82 q2=r.hi/b.hi;r=r-b*Weight83(q2);Q82 q3=r.hi/b.hi;return Weight83(q1)+Weight83(q2)+Weight83(q3);}
static bool operator<(Weight83 a,Weight83 b){return a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo);}
static bool operator>(Weight83 a,Weight83 b){return b<a;}
static bool operator>=(Weight83 a,Weight83 b){return !(a<b);}
static bool finite83(Weight83 a){return finite82(a.hi)&&finite82(a.lo);}
static Weight83 abs83(Weight83 a){return a<Weight83(0)?-a:a;}
static long double ld83(Weight83 a){return static_cast<long double>(a.hi)+static_cast<long double>(a.lo);}
static double double83(Weight83 a){return static_cast<double>(a.hi)+static_cast<double>(a.lo);}
struct WeightBranch83{Mat82 x;Weight83 weight;};
struct WeightPart83{Vec82 xa,xb;long double i00,i01,i11;Weight83 p0,p1;};
struct WeightStep83{std::vector<WeightPart83> parts;Weight83 p0,p1;};
static WeightStep83 weight_step83(const ClickModel70& c,const std::vector<WeightBranch83>& branches,int mode,StableTrace82& trace){
 WeightStep83 out;Weight83 z,o;Vec82 a=c.a.col(mode).cast<long double>(),b=c.b.col(mode).cast<long double>();
 for(auto& v:branches){WeightPart83 p;p.xa=v.x*a;p.xb=v.x*b;long double b00=1+a.dot(p.xa),b11=1+b.dot(p.xb),b01=(a.dot(p.xb)+b.dot(p.xa))/2;Q82 det=Q82(b00)*b11-Q82(b01)*b01;
  need(b00>0&&b11>0&&det>0&&finite82(det),"Weight83 Husimi covariance invalid");p.i00=static_cast<long double>(Q82(b11)/det);p.i01=static_cast<long double>(-Q82(b01)/det);p.i11=static_cast<long double>(Q82(b00)/det);Q82 logp=-logq(det)/2;need(logp<=0&&finite82(logp),"Weight83 component vacuum probability invalid");p.p1=Weight83(-expm1q(logp));p.p0=Weight83(1)-p.p1;z=z+v.weight*p.p0;o=o+v.weight*p.p1;out.parts.push_back(std::move(p));
 }
 Weight83 mass=z+o;need(z>=Weight83(0)&&o>=Weight83(0)&&mass>Weight83(0)&&finite83(mass),"Weight83 signed conditional probability invalid; no clipping");out.p0=z/mass;out.p1=o/mass;trace.evaluations+=branches.size();return out;
}
static void weight_update83(std::vector<WeightBranch83>& branches,const WeightStep83& step,bool click,int budget,StableTrace82& trace){
 Weight83 probability=click?step.p1:step.p0;need(probability>Weight83(0),"Weight83 forced zero-probability branch");std::vector<WeightBranch83> next;next.reserve(branches.size()*(click?2:1));
 for(std::size_t k=0;k<branches.size();++k){auto& v=branches[k];auto& p=step.parts[k];if(click)next.push_back({v.x,v.weight/probability});if(p.p0>Weight83(0)&&(v.weight.hi!=0||v.weight.lo!=0)){WeightBranch83 b=v;Mat82 u=p.xa*(p.i00*p.xa+p.i01*p.xb).transpose()+p.xb*(p.i01*p.xa+p.i11*p.xb).transpose();b.x-=u;b.x=((b.x+b.x.transpose())/2).eval();b.weight=v.weight*p.p0/probability*Weight83(click?-1:1);need(b.x.allFinite(),"Weight83 covariance overflow");next.push_back(std::move(b));}}
 need(!next.empty(),"Weight83 empty branch list");Weight83 mass;for(auto& b:next)mass=mass+b.weight;need(mass>Weight83(0)&&finite83(mass),"Weight83 normalization failed");for(auto& b:next)b.weight=b.weight/mass;Weight83 after,condition;for(auto& b:next){after=after+b.weight;condition=condition+abs83(b.weight);}Weight83 residual=abs83(after-Weight83(1));need(finite83(residual)&&residual<Weight83(Q82(1e-18L)),"Weight83 normalization lost");trace.residual=std::max(trace.residual,ld83(residual));trace.condition=std::max(trace.condition,ld83(condition));trace.peak=std::max(trace.peak,double(next.size()));
 if(next.size()>std::size_t(budget)){int d=next[0].x.rows();WeightBranch83 merged{Mat82::Zero(d,d),Weight83(1)};for(int i=0;i<d;++i)for(int j=0;j<=i;++j){Weight83 value;for(auto& b:next)value=value+b.weight*Weight83(Q82(b.x(i,j)));merged.x(i,j)=merged.x(j,i)=ld83(value/after);}need(merged.x.allFinite(),"Weight83 moment compression overflow");next.clear();next.push_back(std::move(merged));++trace.merges;}branches=std::move(next);
}
// Expose scalar arithmetic as decimal high/low components for independent mpmath
// tests. Inputs parse to binary128; tests use exactly representable dyadics.
API int weight83_probe(int op,const char* ah,const char* al,const char* bh,const char* bl,char* rh,char* rl,int size,char* message,int cap){try{
 need(op>=0&&op<=3&&ah&&al&&bh&&bl&&rh&&rl&&size>=128,"Invalid Weight83 probe arguments");auto parse=[](const char* s){char* end=nullptr;Q82 q=strtoflt128(s,&end);need(end&&*end==0&&finite82(q),"Invalid Weight83 scalar");return q;};Weight83 a=sum83(parse(ah),parse(al)),b=sum83(parse(bh),parse(bl)),r;
 if(op==0)r=a+b;else if(op==1)r=a-b;else if(op==2)r=a*b;else{need(b.hi!=0,"Weight83 division by zero");r=a/b;}need(finite83(r),"Weight83 scalar overflow");int h=quadmath_snprintf(rh,size,"%.85Qg",r.hi),l=quadmath_snprintf(rl,size,"%.85Qg",r.lo);need(h>=0&&h<size&&l>=0&&l<size,"Weight83 output buffer too short");return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int tc83w_prefix(void* ptr,int length,const std::uint8_t* bits,int budget,double* probability,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(ptr&&bits&&probability&&stats&&budget>=1&&budget<=128,"Invalid Weight83 prefix arguments");auto& c=*static_cast<ClickModel70*>(ptr);need(length>=0&&length<=std::min(16,c.model->m),"Weight83 prefix audit limit");std::fill(stats,stats+6,0);std::vector<WeightBranch83> branches{{c.model->initial.cast<long double>(),Weight83(1)}};StableTrace82 trace;Weight83 joint(1);
 for(int i=0;i<length;++i){need(bits[i]<=1,"Invalid Weight83 bit");auto p=weight_step83(c,branches,i,trace);joint=joint*(bits[i]?p.p1:p.p0);if(!(joint>Weight83(0)))break;if(i+1<length)weight_update83(branches,p,bits[i],budget,trace);stats[5]=i+1;}
 *probability=double83(joint);stats[0]=trace.evaluations;stats[1]=trace.merges;stats[2]=trace.peak;stats[3]=double(trace.condition);stats[4]=double(trace.residual);return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int tc83w_sample(void* ptr,int count,std::uint64_t seed,int threads,int budget,const double* draws,std::uint8_t* bits,double* probabilities,int* errors,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(ptr&&count>=0&&count<=4096&&threads>=1&&threads<=64&&budget>=1&&budget<=128&&bits&&probabilities&&errors&&stats,"Invalid Weight83 sample arguments");auto& c=*static_cast<ClickModel70*>(ptr);int m=c.model->m;std::fill(bits,bits+std::size_t(count)*m,0);std::fill(probabilities,probabilities+std::size_t(count)*m,std::numeric_limits<double>::quiet_NaN());std::fill(errors,errors+2*count,0);std::fill(stats,stats+6*count,0);std::atomic<int> next{0};std::vector<std::string> messages(count);std::vector<std::thread> pool;
 auto worker=[&](){PrecisionGuard precision;for(;;){int row=next.fetch_add(1);if(row>=count)break;int mode=0,stage=1;StableTrace82 trace;try{std::mt19937_64 rng(mix(seed+2ULL*row+1));std::vector<WeightBranch83> branches{{c.model->initial.cast<long double>(),Weight83(1)}};for(;mode<m;++mode){stage=1;double u=draws?draws[std::size_t(row)*m+mode]:uniform(rng);need(std::isfinite(u)&&u>=0&&u<1,"Invalid Weight83 draw");auto p=weight_step83(c,branches,mode,trace);bool bit=Weight83(Q82(u))<p.p1;probabilities[std::size_t(row)*m+mode]=double83(p.p1);bits[std::size_t(row)*m+mode]=bit;if(mode+1<m){stage=2;weight_update83(branches,p,bit,budget,trace);}}}catch(const std::exception& e){errors[2*row]=stage;errors[2*row+1]=mode;messages[row]=e.what();}stats[6*row]=trace.evaluations;stats[6*row+1]=trace.merges;stats[6*row+2]=trace.peak;stats[6*row+3]=double(trace.condition);stats[6*row+4]=double(trace.residual);stats[6*row+5]=mode;}};
 try{for(int i=0;i<std::min(count,threads);++i)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[2*row]){error(message,cap,"row="+std::to_string(row)+" mode="+std::to_string(errors[2*row+1])+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
