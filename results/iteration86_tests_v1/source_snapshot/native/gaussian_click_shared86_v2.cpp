// Shared Gaussian base plus a set of retained click constraints.
// Small vacuum-subset Schur complements replace independently rounded dense
// signed branch covariances. Gaussian moment closure occurs at the same budget.
// The bundled MinGW libquadmath fmaq leaves SSE rounding at toward-zero.
// Preserve its caller's MXCSR control bits around every included Weight83 FMA.
#include <quadmath.h>
#include <xmmintrin.h>
static __attribute__((noinline)) __float128 preserving_fmaq86(__float128 a,__float128 b,__float128 c){
 unsigned saved=_mm_getcsr();volatile __float128 result=fmaq(a,b,c);unsigned after=_mm_getcsr();_mm_setcsr((saved&~0x3fu)|((saved|after)&0x3fu));return result;
}
#define fmaq preserving_fmaq86
#include "gaussian_click_exactsum84_v1.cpp"
#undef fmaq
struct SharedRoundGuard86 {
 PrecisionGuard x87;unsigned saved;
 SharedRoundGuard86():saved(_mm_getcsr()){_mm_setcsr(saved&~0x6000u);}
 ~SharedRoundGuard86(){unsigned after=_mm_getcsr();_mm_setcsr((saved&~0x3fu)|((saved|after)&0x3fu));}
};
static Q82 sqrt86(Q82 a){need(a>0&&finite82(a),"Shared86 positive square root");return sqrtq(a);}
static Weight83 sqrt86(Weight83 a){need(a>Weight83(0)&&finite83(a),"Shared86 wide square root");Weight83 r(sqrtq(a.hi));r=(r+a/r)/Weight83(2);r=(r+a/r)/Weight83(2);return r;}
static long double ld86(Q82 a){return static_cast<long double>(a);}
static long double ld86(Weight83 a){return ld83(a);}
static bool finite86(Q82 a){return finite82(a);}
static bool finite86(Weight83 a){return finite83(a);}
template<class T> static T abs86(T a){return a<T(0)?-a:a;}
template<class T> struct Subset86 {std::vector<int> ids;std::vector<T> inverse;T vacuum=T(1);};
template<class T> struct VacuumSum86 {std::vector<Subset86<T>> subsets;T mass=T(0),zero=T(0),absolute=T(0);int clicks=0;};
template<class T> static VacuumSum86<T> vacuum_sum86(const Mat82& gram,int clicks){
 VacuumSum86<T> result;result.clicks=clicks;int masks=1<<(clicks+1);result.subsets.resize(masks);
 for(int mask=1;mask<masks;++mask){int bit=31-__builtin_clz(unsigned(mask)),parent=mask^(1<<bit);auto& before=result.subsets[parent];auto& current=result.subsets[mask];int n=int(before.ids.size()),d=n+2;current.ids=before.ids;current.ids.push_back(2*bit);current.ids.push_back(2*bit+1);std::vector<T> u0(n,T(0)),u1(n,T(0));
  for(int i=0;i<n;++i)for(int j=0;j<n;++j){u0[i]=u0[i]+before.inverse[i*n+j]*T(Q82(gram(before.ids[j],2*bit)));u1[i]=u1[i]+before.inverse[i*n+j]*T(Q82(gram(before.ids[j],2*bit+1)));}
  T s00(Q82(gram(2*bit,2*bit))),s01(Q82(gram(2*bit,2*bit+1))),s11(Q82(gram(2*bit+1,2*bit+1)));
  for(int i=0;i<n;++i){s00=s00-T(Q82(gram(before.ids[i],2*bit)))*u0[i];s01=s01-T(Q82(gram(before.ids[i],2*bit)))*u1[i];s11=s11-T(Q82(gram(before.ids[i],2*bit+1)))*u1[i];}
  T det=s00*s11-s01*s01;need(s00>T(0)&&s11>T(0)&&det>T(0)&&finite86(det),"Shared86 subset Husimi determinant invalid");T i00=s11/det,i01=-s01/det,i11=s00/det;current.vacuum=before.vacuum/sqrt86(det);need(current.vacuum>T(0)&&finite86(current.vacuum),"Shared86 subset vacuum invalid");current.inverse.assign(d*d,T(0));
  for(int i=0;i<n;++i){T v0=u0[i]*i00+u1[i]*i01,v1=u0[i]*i01+u1[i]*i11;for(int j=0;j<n;++j)current.inverse[i*d+j]=before.inverse[i*n+j]+v0*u0[j]+v1*u1[j];current.inverse[i*d+n]=current.inverse[n*d+i]=-v0;current.inverse[i*d+n+1]=current.inverse[(n+1)*d+i]=-v1;}
  current.inverse[n*d+n]=i00;current.inverse[n*d+n+1]=current.inverse[(n+1)*d+n]=i01;current.inverse[(n+1)*d+n+1]=i11;
 }
 int last=1<<clicks;for(int mask=0;mask<last;++mask){T sign(__builtin_popcount(unsigned(mask))%2?-1:1);result.mass=result.mass+sign*result.subsets[mask].vacuum;result.zero=result.zero+sign*result.subsets[mask|last].vacuum;result.absolute=result.absolute+abs86(result.subsets[mask].vacuum)+abs86(result.subsets[mask|last].vacuum);}return result;
}
struct SharedState86 {Mat82 x;std::vector<int> clicks;};
struct SharedTrace86 {double evaluations=0,merges=0,peak=0,condition=1,wide=0;};
struct SharedStep86 {Mat82 forms,products;Vec82 xa,xb;long double i00=0,i01=0,i11=0;Weight83 p0,p1;std::unique_ptr<VacuumSum86<Q82>> narrow;std::unique_ptr<VacuumSum86<Weight83>> wide;};
static SharedStep86 shared_step86(const ClickModel70& c,const SharedState86& state,int mode,SharedTrace86& trace){
 SharedStep86 step;int k=int(state.clicks.size()),d=state.x.rows();need(k<=7,"Shared86 retained click limit");step.forms.resize(d,2*(k+1));for(int j=0;j<k+1;++j){int port=j<k?state.clicks[j]:mode;step.forms.col(2*j)=c.a.col(port).cast<long double>();step.forms.col(2*j+1)=c.b.col(port).cast<long double>();}step.products=state.x*step.forms;Mat82 gram=step.forms.transpose()*step.products;gram=((gram+gram.transpose())/2).eval();gram.diagonal().array()+=1;step.xa=step.products.col(2*k);step.xb=step.products.col(2*k+1);Q82 b00=gram(2*k,2*k),b11=gram(2*k+1,2*k+1),b01=gram(2*k,2*k+1),det=b00*b11-b01*b01;need(b00>0&&b11>0&&det>0,"Shared86 new-port determinant invalid");step.i00=static_cast<long double>(b11/det);step.i01=static_cast<long double>(-b01/det);step.i11=static_cast<long double>(b00/det);
 bool wider=false;try{step.narrow=std::make_unique<VacuumSum86<Q82>>(vacuum_sum86<Q82>(gram,k));auto& s=*step.narrow;Q82 click=s.mass-s.zero;Q82 smallest=fminq(fabsq(s.mass),fminq(fabsq(s.zero),fabsq(click)));wider=!(s.mass>0&&s.zero>=0&&click>=0&&finite82(s.mass))||smallest==0||s.absolute/smallest>Q82(1e16L);if(!wider){step.p0=Weight83(s.zero/s.mass);step.p1=Weight83(click/s.mass);trace.condition=std::max(trace.condition,double(s.absolute/smallest));}}catch(const std::exception&){wider=true;}
 if(wider){step.wide=std::make_unique<VacuumSum86<Weight83>>(vacuum_sum86<Weight83>(gram,k));auto& s=*step.wide;Weight83 click=s.mass-s.zero;need(s.mass>Weight83(0)&&s.zero>=Weight83(0)&&click>=Weight83(0)&&finite83(s.mass),"Shared86 wide signed probability invalid; no clipping");step.p0=s.zero/s.mass;step.p1=click/s.mass;long double small=std::min({std::abs(ld83(s.mass)),std::abs(ld83(s.zero)),std::abs(ld83(click))});if(small>0)trace.condition=std::max(trace.condition,double(ld83(s.absolute)/small));++trace.wide;}
 trace.evaluations+=1<<(k+1);return step;
}
template<class T> static Mat82 correction86(const VacuumSum86<T>& sums){
 int k=sums.clicks+1,d=2*k;T mass=sums.mass-sums.zero;need(mass>T(0),"Shared86 zero click compression mass");std::vector<T> out(d*d,T(0));
 for(int mask=1;mask<(1<<k);++mask){auto& sub=sums.subsets[mask];int n=int(sub.ids.size());T weight=sub.vacuum*T(__builtin_popcount(unsigned(mask))%2?-1:1)/mass;for(int i=0;i<n;++i)for(int j=0;j<n;++j){auto at=sub.ids[i]*d+sub.ids[j];out[at]=out[at]+weight*sub.inverse[i*n+j];}}
 Mat82 result(d,d);for(int i=0;i<d;++i)for(int j=0;j<d;++j)result(i,j)=ld86(out[i*d+j]);return result;
}
static void shared_update86(SharedState86& state,SharedStep86& step,int mode,bool click,int budget,SharedTrace86& trace){
 if(!click){need(step.p0>Weight83(0),"Shared86 impossible zero");state.x-=step.xa*(step.i00*step.xa+step.i01*step.xb).transpose()+step.xb*(step.i01*step.xa+step.i11*step.xb).transpose();}
 else{need(step.p1>Weight83(0),"Shared86 impossible click");state.clicks.push_back(mode);trace.peak=std::max(trace.peak,double(state.clicks.size()));if((1U<<state.clicks.size())>unsigned(budget)){Mat82 R=step.wide?correction86(*step.wide):correction86(*step.narrow);state.x-=step.products*R*step.products.transpose();state.clicks.clear();++trace.merges;}}
 state.x=((state.x+state.x.transpose())/2).eval();need(state.x.allFinite(),"Shared86 covariance overflow");
}
API int tc86_shared_prefix(void* handle,int length,const std::uint8_t* bits,int budget,double* probability,double* stats,char* message,int cap){try{
 SharedRoundGuard86 guard;need(handle&&bits&&probability&&stats&&budget>=1&&budget<=128,"Shared86 prefix arguments");auto& c=*static_cast<ClickModel70*>(handle);need(length>=0&&length<=std::min(16,c.model->m),"Shared86 prefix limit");SharedState86 state{c.model->initial.cast<long double>(),{}};SharedTrace86 trace;Weight83 joint(1);std::fill(stats,stats+6,0);
 for(int mode=0;mode<length;++mode){need(bits[mode]<=1,"Shared86 bit");auto step=shared_step86(c,state,mode,trace);joint=joint*(bits[mode]?step.p1:step.p0);if(!(joint>Weight83(0)))break;if(mode+1<length)shared_update86(state,step,mode,bits[mode],budget,trace);stats[5]=mode+1;}
 *probability=double83(joint);stats[0]=trace.evaluations;stats[1]=trace.merges;stats[2]=trace.peak;stats[3]=trace.condition;stats[4]=trace.wide;return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
// Replay a frozen prefix, including the next conditional at length. No joint
// probability product: long prefixes must not disappear through underflow.
API int tc86_shared_conditionals(void* handle,int length,const std::uint8_t* bits,int budget,double* probabilities,double* stats,char* message,int cap){try{
 SharedRoundGuard86 guard;need(handle&&bits&&probabilities&&stats&&budget>=1&&budget<=128,"Shared86 conditional replay arguments");auto& c=*static_cast<ClickModel70*>(handle);need(length>=0&&length<c.model->m,"Shared86 conditional replay length");SharedState86 state{c.model->initial.cast<long double>(),{}};SharedTrace86 trace;std::fill(stats,stats+6,0);std::fill(probabilities,probabilities+length+1,std::numeric_limits<double>::quiet_NaN());
 for(int mode=0;mode<=length;++mode){auto step=shared_step86(c,state,mode,trace);probabilities[mode]=double83(step.p1);stats[5]=mode+1;if(mode<length){need(bits[mode]<=1,"Shared86 replay bit");shared_update86(state,step,mode,bits[mode],budget,trace);}}
 stats[0]=trace.evaluations;stats[1]=trace.merges;stats[2]=trace.peak;stats[3]=trace.condition;stats[4]=trace.wide;return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int tc86_shared_sample(void* handle,int count,int threads,int budget,const double* draws,std::uint8_t* bits,double* probabilities,int* errors,double* stats,char* message,int cap){try{
 SharedRoundGuard86 guard;need(handle&&count>=0&&count<=4096&&threads>=1&&threads<=64&&budget>=1&&budget<=128&&draws&&bits&&probabilities&&errors&&stats,"Shared86 sample arguments; explicit draws required");auto& c=*static_cast<ClickModel70*>(handle);int m=c.model->m;std::fill(bits,bits+std::size_t(count)*m,0);std::fill(probabilities,probabilities+std::size_t(count)*m,std::numeric_limits<double>::quiet_NaN());std::fill(errors,errors+2*count,0);std::fill(stats,stats+6*count,0);std::atomic<int> next{0};std::vector<std::string> messages(count);std::vector<std::thread> pool;
 auto worker=[&](){SharedRoundGuard86 precision;for(;;){int row=next.fetch_add(1);if(row>=count)break;int mode=0,stage=1;SharedTrace86 trace;try{SharedState86 state{c.model->initial.cast<long double>(),{}};for(;mode<m;++mode){stage=1;double u=draws[std::size_t(row)*m+mode];need(std::isfinite(u)&&u>=0&&u<1,"Shared86 explicit draw");auto step=shared_step86(c,state,mode,trace);bool bit=Weight83(Q82(u))<step.p1;bits[std::size_t(row)*m+mode]=bit;probabilities[std::size_t(row)*m+mode]=double83(step.p1);if(mode+1<m){stage=2;shared_update86(state,step,mode,bit,budget,trace);}}}catch(const std::exception& e){errors[2*row]=stage;errors[2*row+1]=mode;messages[row]=e.what();}stats[6*row]=trace.evaluations;stats[6*row+1]=trace.merges;stats[6*row+2]=trace.peak;stats[6*row+3]=trace.condition;stats[6*row+4]=trace.wide;stats[6*row+5]=mode;}};
 try{for(int i=0;i<std::min(count,threads);++i)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[2*row]){error(message,cap,"row="+std::to_string(row)+" mode="+std::to_string(errors[2*row+1])+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}

API int tc86_rounding_test(void* handle,int rounding,const std::uint8_t* bits,double* probability,unsigned* state,char* message,int cap){
 if(rounding<0||rounding>3||!state)return 1;unsigned saved=_mm_getcsr();_mm_setcsr((saved&~0x6000u)|(unsigned(rounding)<<13));state[0]=_mm_getcsr();double stats[6];int rc=tc86_shared_prefix(handle,2,bits,64,probability,stats,message,cap);state[1]=_mm_getcsr();_mm_setcsr(saved);return rc;
}
