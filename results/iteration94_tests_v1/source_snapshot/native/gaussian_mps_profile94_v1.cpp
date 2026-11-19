// Gaussian background remains analytic; an MPS stores its coherent Fock core.
// All physical projections and Schmidt truncations charge trace-distance budget.
#include "gaussian_rng_domain84_v1.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <xmmintrin.h>
#define API93 extern "C" __declspec(dllexport)
using Z93=std::complex<double>;
using Mat93=Eigen::MatrixXcd;
using Vec93=Eigen::VectorXd;
using Site93=std::vector<Mat93>;
using Mps93=std::vector<Site93>;
struct Guard93{unsigned short saved;unsigned sse;Guard93():sse(_mm_getcsr()){asm volatile("fnstcw %0":"=m"(saved));unsigned short w=(saved&~0x0f00u)|0x0300u;asm volatile("fldcw %0"::"m"(w));_mm_setcsr(sse&~0x6000u);}~Guard93(){asm volatile("fldcw %0"::"m"(saved));unsigned a=_mm_getcsr();_mm_setcsr((sse&~0x3fu)|((sse|a)&0x3fu));}};
// Diagnostic build only: inclusive timers, calls, and attempted projection weights.
static thread_local double* profile94=nullptr;
struct Timer94{int slot;std::chrono::steady_clock::time_point start;
 explicit Timer94(int s):slot(s),start(std::chrono::steady_clock::now()){}
 ~Timer94(){if(profile94){profile94[18+slot]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();profile94[27+slot]+=1;}}};
static void weights94(long double local,long double schmidt){if(profile94){profile94[36]+=double(local);profile94[37]+=double(schmidt);profile94[38]=std::max(profile94[38],double(local));profile94[39]=std::max(profile94[39],double(schmidt));profile94[40]+=std::sqrt(local);profile94[41]+=std::sqrt(schmidt);}}
struct Failure93:std::runtime_error{int code;Failure93(int c,const char* s):std::runtime_error(s),code(c){}};
static void need93(bool b,const char* s){if(!b)throw Failure93(4,s);}
static void message93(char* out,int cap,const std::string& value){if(out&&cap>0){std::strncpy(out,value.c_str(),cap-1);out[cap-1]=0;}}
static long double norm93(const Mat93& m){long double out=0;for(Eigen::Index i=0;i<m.size();++i){auto z=m.data()[i];out+=(long double)z.real()*z.real()+(long double)z.imag()*z.imag();}return out;}
// Optional pinned LAPACKE backend, configured before any worker starts.
using Gesdd93=int(*)(int,char,int,int,Z93*,int,double*,Z93*,int,Z93*,int);
using Geqrf93=int(*)(int,int,int,Z93*,int,Z93*);
using Ungqr93=int(*)(int,int,int,int,Z93*,int,const Z93*);
static Gesdd93 gesdd93=nullptr;static Geqrf93 geqrf93=nullptr;static Ungqr93 ungqr93=nullptr;
API93 int mps93_set_lapack(void* svd,void* qr,void* q){if(!svd||!qr||!q)return 1;gesdd93=reinterpret_cast<Gesdd93>(svd);geqrf93=reinterpret_cast<Geqrf93>(qr);ungqr93=reinterpret_cast<Ungqr93>(q);return 0;}
static void thin_qr93(const Mat93& a,Mat93& Q,Mat93& R,bool want_q){Timer94 timer94(7);int m=a.rows(),n=a.cols(),k=std::min(m,n);if(geqrf93){Mat93 packed=a;std::vector<Z93> tau(k);need93(geqrf93(102,m,n,packed.data(),m,tau.data())==0,"LAPACK QR failed");R=packed.topRows(k).template triangularView<Eigen::Upper>();if(want_q){Q=packed.leftCols(k);need93(ungqr93(102,m,k,k,Q.data(),m,tau.data())==0,"LAPACK Q failed");}}else{Eigen::HouseholderQR<Mat93> qr(a);R=qr.matrixQR().topRows(k).template triangularView<Eigen::Upper>();if(want_q)Q=qr.householderQ()*Mat93::Identity(m,k);}}
static Mat93 qr_factor93(const Mat93& a){Mat93 q,r;thin_qr93(a,q,r,false);return r;}
static void thin_svd93(const Mat93& a,Mat93& U,Vec93& values,Mat93& Vt){Timer94 timer94(8);if(gesdd93){int m=a.rows(),n=a.cols(),k=std::min(m,n);Mat93 packed=a;U.resize(m,k);Vt.resize(k,n);values.resize(k);need93(gesdd93(102,'S',m,n,packed.data(),m,values.data(),U.data(),m,Vt.data(),k)==0,"LAPACK SVD failed");}else{Eigen::JacobiSVD<Mat93> svd(a,Eigen::ComputeThinU|Eigen::ComputeThinV);need93(svd.info()==Eigen::Success,"MPS SVD failed");U=svd.matrixU();values=svd.singularValues();Vt=svd.matrixV().adjoint();}}
// Right-canonicalize without truncation. Return incoming squared norm, normalize.
static long double canonical93(Mps93& c,bool allow_zero=false){Timer94 timer94(6);
 for(int i=int(c.size())-1;i>0;--i){int dl=c[i][0].rows(),dr=c[i][0].cols(),d=c[i].size();Mat93 flat(dl,d*dr);for(int s=0;s<d;++s)flat.middleCols(s*dr,dr)=c[i][s];Mat93 adj=flat.adjoint();int rank=std::min(adj.rows(),adj.cols());Mat93 Q,R;thin_qr93(adj,Q,R,true);Mat93 right=Q.adjoint();for(int s=0;s<d;++s)c[i][s]=right.middleCols(s*dr,dr);Mat93 left=R.adjoint();for(auto& a:c[i-1])a=(a*left).eval();}
 long double n=0;for(auto& a:c[0])n+=norm93(a);if(n==0&&allow_zero)return 0;need93(n>0&&std::isfinite(n),"Zero/nonfinite MPS norm");double factor=1/std::sqrt(n);for(auto& a:c[0])a*=factor;return n;
}
static double canonical_residual93(const Mps93& c){double error=0;for(int i=1;i<int(c.size());++i){Mat93 g=Mat93::Zero(c[i][0].rows(),c[i][0].rows());for(const auto& a:c[i])g.noalias()+=a*a.adjoint();error=std::max(error,(g-Mat93::Identity(g.rows(),g.cols())).cwiseAbs().maxCoeff());}return error;}
struct Gaussian93{std::vector<double> t;std::vector<Z93> alpha;};
struct Local93{Mat93 map;double t;Z93 alpha;long double log_weight;};
static Local93 local93(int d,double t,Z93 alpha,double eta){
 double tp=eta*t,den=1-tp*tp;Z93 ap=std::sqrt(eta)*((1-eta*t*t)*alpha-t*(1-eta)*std::conj(alpha))/den;double scale=std::sqrt(eta*(1-t*t)/den),beta=-t*(1-eta*eta)/den;Z93 gamma=std::sqrt(1-t*t)*(std::sqrt(eta)*std::conj(ap)-std::conj(alpha));std::vector<Z93> e(d);e[0]=1;
 for(int j=1;j<d;++j)e[j]=(gamma*e[j-1]+(j>1?beta*e[j-2]:Z93(0)))/double(j);
 Mat93 M=Mat93::Zero(d,d);double power=1;for(int a=0;a<d;++a){double factor=1;for(int b=a;b<d;++b){if(b>a)factor*=std::sqrt(double(b));M(a,b)=power*factor*e[b-a];}power*=scale;}
 long double logw=.5L*std::log((1-t*t)/den)-std::norm(alpha)+t*std::real(std::conj(alpha)*std::conj(alpha))+std::norm(ap)-tp*std::real(std::conj(ap)*std::conj(ap));return {std::move(M),tp,ap,logw};
}
static Site93 apply93(const Site93& site,const Mat93& op){Site93 out(op.rows(),Mat93::Zero(site[0].rows(),site[0].cols()));for(int s=0;s<op.rows();++s)for(int t=0;t<op.cols();++t)if(op(s,t)!=Z93(0))out[s].noalias()+=op(s,t)*site[t];return out;}
static Mat93 physical93(int d,double t,Z93 alpha){Mat93 op=alpha*Mat93::Identity(d,d);double a=1/std::sqrt(1-t*t);for(int j=1;j<d;++j){op(j-1,j)=a*std::sqrt(double(j));op(j,j-1)=t*a*std::sqrt(double(j));}return op;}
static Mat93 transfer93(const Mat93& x,const Site93& bra,const Site93& ket){Mat93 out=Mat93::Zero(bra[0].cols(),ket[0].cols());for(int s=0;s<int(bra.size());++s)out.noalias()+=bra[s].adjoint()*x*ket[s];return out;}
struct Rates93{Mat93 gram;std::vector<long double> outside;long double total;};
static Rates93 rates93(const Mps93& c,const Gaussian93& g,bool full){Timer94 timer94(1);
 int n=c.size();std::vector<Site93> acted(n);std::vector<Mat93> left(n);std::vector<long double> outside(n);Mat93 F=Mat93::Ones(1,1),G=Mat93::Zero(n,n);long double total=0;
 for(int i=0;i<n;++i){int dl=c[i][0].rows(),dr=c[i][0].cols(),d=c[i].size();left[i]=F.adjoint()*F;acted[i]=apply93(c[i],physical93(d,g.t[i],g.alpha[i]));long double inside=0;for(auto& a:acted[i])inside+=norm93((F*a).eval());outside[i]=double(d)*g.t[i]*g.t[i]/(1-g.t[i]*g.t[i])*norm93((F*c[i].back()).eval());G(i,i)=double(inside+outside[i]);total+=inside+outside[i];Mat93 stack(d*F.rows(),dr);for(int s=0;s<d;++s)stack.middleRows(s*F.rows(),F.rows())=F*c[i][s];F=qr_factor93(stack);}
 if(full)for(int i=0;i<n;++i){Mat93 x=transfer93(left[i],acted[i],c[i]);for(int j=i+1;j<n;++j){Z93 z=transfer93(x,c[j],acted[j]).trace();G(i,j)=z;G(j,i)=std::conj(z);x=transfer93(x,c[j],c[j]);}}
 need93(std::isfinite(total)&&total>=0,"Invalid total photon hazard");return {std::move(G),std::move(outside),total};
}
static long double attenuate93(const Mps93& c,const Gaussian93& g,double eta,Mps93* transformed,Gaussian93* next){Timer94 timer94(2);
 int n=c.size();long double logw=0;Mps93 out;Gaussian93 gp;gp.t.resize(n);gp.alpha.resize(n);if(eta==0){Mat93 amplitude=Mat93::Ones(1,1);for(int i=0;i<n;++i){auto map=local93(c[i].size(),g.t[i],g.alpha[i],eta);Mat93 a=Mat93::Zero(c[i][0].rows(),c[i][0].cols());for(int s=0;s<int(c[i].size());++s)a.noalias()+=map.map(0,s)*c[i][s];amplitude=(amplitude*a).eval();logw+=map.log_weight;}return std::exp(logw)*norm93(amplitude);}
 out.reserve(n);for(int i=0;i<n;++i){auto map=local93(c[i].size(),g.t[i],g.alpha[i],eta);out.push_back(apply93(c[i],map.map));gp.t[i]=map.t;gp.alpha[i]=map.alpha;logw+=map.log_weight;}auto norm=canonical93(out);long double p=std::exp(logw)*norm;if(transformed)*transformed=std::move(out);if(next)*next=std::move(gp);return p;
}
static bool wait93(Mps93& c,Gaussian93& g,double u,double* stats){Timer94 timer94(0);
 long double vacuum=attenuate93(c,g,0,nullptr,nullptr);++stats[3];need93(vacuum>=0&&vacuum<=1+1e-10L,"Invalid MPS vacuum probability");if(u<=vacuum)return false;
 // P(eta)=E[eta^N] >= eta^E[N] by Jensen. Start near the high-rate
 // root without changing the bracket, stopping rule or consumed random words.
 // The bound motivates a guess only: all bracket updates use evaluated P.
 auto initial_rate=rates93(c,g,false).total;
 need93(initial_rate>0&&std::isfinite(initial_rate),"Invalid initial waiting hazard");
 double guess=double(std::exp(std::log((long double)u)/initial_rate));
 double lo=0,hi=1,eta=std::isfinite(guess)&&guess>0&&guess<1?guess:.5;
 for(int iteration=0;iteration<80;++iteration){Mps93 cp;Gaussian93 gp;auto p=attenuate93(c,g,eta,&cp,&gp);++stats[2];++stats[3];need93(p>0&&p<=1+1e-9L&&std::isfinite(p),"Invalid MPS survival");long double residual=p-u;if(std::abs(residual)<2e-12L||hi-lo<4e-15){stats[4]=std::max(stats[4],double(std::abs(residual)));c=std::move(cp);g=std::move(gp);return true;}if(residual>0)hi=eta;else lo=eta;auto rate=rates93(cp,gp,false);long double derivative=p*rate.total/eta;double proposed=eta-double(residual/derivative);eta=std::isfinite(proposed)&&proposed>lo&&proposed<hi?proposed:(lo+hi)/2;}
 throw Failure93(4,"MPS waiting root failed");
}
// MPO of sum_i coefficient_i O_i. The chosen local basis projection is explicit.
static Mps93 jump_mpo93(const Mps93& c,const Gaussian93& g,const std::vector<Z93>& coefficients){
 int n=c.size();Mps93 out(n);for(int i=0;i<n;++i){auto acted=apply93(c[i],physical93(c[i].size(),g.t[i],g.alpha[i]));int dl=c[i][0].rows(),dr=c[i][0].cols(),d=c[i].size();out[i].resize(d);for(int s=0;s<d;++s){Mat93 b=coefficients[i]*acted[s];if(n==1)out[i][s]=b;else if(i==0){out[i][s]=Mat93::Zero(1,2*dr);out[i][s].leftCols(dr)=c[i][s];out[i][s].rightCols(dr)=b;}else if(i==n-1){out[i][s]=Mat93::Zero(2*dl,1);out[i][s].topRows(dl)=b;out[i][s].bottomRows(dl)=c[i][s];}else{out[i][s]=Mat93::Zero(2*dl,2*dr);out[i][s].topLeftCorner(dl,dr)=c[i][s];out[i][s].topRightCorner(dl,dr)=b;out[i][s].bottomRightCorner(dl,dr)=c[i][s];}}}return out;
}
static void charge93(double delta,double limit,double* stats){need93(delta>=0&&std::isfinite(delta),"Invalid discarded weight");if(stats[5]+delta>limit)throw Failure93(2,"Trace-distance budget exhausted");stats[5]+=delta;stats[6]=std::max(stats[6],delta);}
static long double compress93(Mps93& c,int maxbond,double* stats){Timer94 timer94(5);
 long double log_retention=0;
 for(int i=0;i+1<int(c.size());++i){int dl=c[i][0].rows(),dr=c[i][0].cols(),d=c[i].size();Mat93 flat(d*dl,dr);for(int s=0;s<d;++s)flat.middleRows(s*dl,dl)=c[i][s];Mat93 Ufull,Vt;Vec93 values;thin_svd93(flat,Ufull,values,Vt);int keep=std::min<int>(maxbond,values.size());while(keep>1&&values[keep-1]<=1e-13*values[0])--keep;long double mass=0,discard=0;for(int j=0;j<values.size();++j){long double w=(long double)values[j]*values[j];mass+=w;if(j>=keep)discard+=w;}need93(mass>0,"MPS SVD zero norm");log_retention+=std::log1p(-discard/mass);if(discard>0)++stats[7];Mat93 U=Ufull.leftCols(keep),T=values.head(keep).asDiagonal()*Vt.topRows(keep);T/=std::sqrt(mass-discard);for(int s=0;s<d;++s)c[i][s]=U.middleRows(s*dl,dl);for(auto& a:c[i+1])a=(T*a).eval();stats[8]=std::max(stats[8],double(keep));}
 auto norm=canonical93(c);stats[9]=std::max(stats[9],double(std::abs(norm-1)));auto residual=canonical_residual93(c);stats[10]=std::max(stats[10],residual);need93(residual<1e-9&&std::abs(norm-1)<1e-8L,"MPS canonical residual failed");return log_retention;
}
static int select_jump_full93(Mps93& c,const Gaussian93& g,const Mat93& optical,double uniform,int maxbond,double budget,double* stats){Timer94 timer94(3);
 auto rates=rates93(c,g,true);int m=optical.rows(),n=optical.cols();std::vector<long double> port(m);long double total=0;std::vector<Z93> weights(n);
 for(int j=0;j<m;++j){long double value=0,scale=0;for(int a=0;a<n;++a)for(int b=0;b<n;++b){auto z=std::conj(optical(j,a))*optical(j,b)*rates.gram(a,b);value+=z.real();scale+=std::abs(z);}if(value<=1e-12L*scale){for(int a=0;a<n;++a)weights[a]=optical(j,a);auto state=jump_mpo93(c,g,weights);long double outside=0;for(int a=0;a<n;++a)outside+=std::norm(weights[a])*rates.outside[a];value=canonical93(state,true)+outside;++stats[11];}need93(value>=0&&std::isfinite(value),"Negative/nonfinite MPS port hazard");port[j]=value;total+=value;}
 need93(total>0&&std::isfinite(total),"Zero MPS output hazard");stats[12]=std::max(stats[12],double(std::abs(total-rates.total)/total));long double point=uniform*total;int chosen=m-1;for(int j=0;j<m;++j){if(point<port[j]){chosen=j;break;}point-=port[j];}for(int a=0;a<n;++a)weights[a]=optical(chosen,a);auto state=jump_mpo93(c,g,weights);long double outside=0;for(int a=0;a<n;++a)outside+=std::norm(weights[a])*rates.outside[a];auto inside=canonical93(state);long double full=inside+outside;stats[13]=std::max(stats[13],double(std::abs(full-port[chosen])/full));need93(std::abs(full-port[chosen])<1e-8L*full,"Selected MPS hazard mismatch");long double schmidt94=compress93(state,maxbond,stats);weights94(outside/full,-std::expm1(schmidt94));long double log_retention=std::log1p(-outside/full)+schmidt94;charge93(std::sqrt(-std::expm1(log_retention)),budget,stats);c=std::move(state);return chosen;
}

struct Centered93{std::vector<Z93> mean;std::vector<long double> variance,outside;long double coherent=0,fluctuation=0;};
static Centered93 centered93(const Mps93& c,const Gaussian93& g){Timer94 timer94(4);
 int n=c.size();Centered93 result;result.mean.resize(n);result.variance.resize(n);result.outside.resize(n);Mat93 F=Mat93::Ones(1,1);
 for(int i=0;i<n;++i){int d=c[i].size(),dr=c[i][0].cols();auto acted=apply93(c[i],physical93(d,g.t[i],g.alpha[i]));std::vector<Mat93> x(d),y(d);std::complex<long double> mean=0;
  for(int a=0;a<d;++a){x[a]=F*c[i][a];y[a]=F*acted[a];for(Eigen::Index j=0;j<x[a].size();++j){auto p=x[a].data()[j],q=y[a].data()[j];mean+=std::conj(std::complex<long double>(p.real(),p.imag()))*std::complex<long double>(q.real(),q.imag());}}
  result.mean[i]=Z93(double(mean.real()),double(mean.imag()));long double variance=0;for(int a=0;a<d;++a)variance+=norm93((y[a]-result.mean[i]*x[a]).eval());auto outside=double(d)*g.t[i]*g.t[i]/(1-g.t[i]*g.t[i])*norm93(x.back());variance+=outside;result.outside[i]=outside;result.variance[i]=variance;result.coherent+=std::norm(result.mean[i]);result.fluctuation+=variance;Mat93 stack(d*F.rows(),dr);for(int a=0;a<d;++a)stack.middleRows(a*F.rows(),F.rows())=x[a];F=qr_factor93(stack);
 }
 return result;
}
// Positive decomposition of rates, NOT a decomposition of the post-jump state.
// The selected port always applies the original coherent sum of all operators.
static int select_jump93(Mps93& c,const Gaussian93& g,const Mat93& optical,DomainRng84& rng,int maxbond,double budget,double* stats){
 auto centered=centered93(c,g);auto expected=centered.coherent+centered.fluctuation;need93(expected>0&&std::isfinite(expected),"Invalid centered hazard");double first=uniform84(rng);
 if(centered.coherent<.8L*expected)return select_jump_full93(c,g,optical,first,maxbond,budget,stats);
 int m=optical.rows(),n=optical.cols();std::vector<Z93> mean(m),weights(n);std::vector<long double> coherent(m),proposal(m);long double cmass=0,vmass=0;
 for(int j=0;j<m;++j){for(int i=0;i<n;++i){mean[j]+=optical(j,i)*centered.mean[i];proposal[j]+=std::norm(optical(j,i))*centered.variance[i];}coherent[j]=std::norm(mean[j]);cmass+=coherent[j];vmass+=proposal[j];}
 long double total=cmass+vmass;need93(total>0&&std::isfinite(total),"Invalid centered output mass");stats[12]=std::max(stats[12],double(std::abs(total-expected)/total));long double point=first*total;int chosen=m-1;bool residual_branch=point>=cmass;long double chosen_variance=0;
 auto categorical=[&](const std::vector<long double>& rates,long double p){for(int j=0;j<m;++j){if(p<rates[j])return j;p-=rates[j];}throw Failure93(4,"Centered categorical endpoint");};
 if(!residual_branch)chosen=categorical(coherent,point);else{
  need93(vmass>0,"Zero centered proposal mass");Gaussian93 fluctuation=g;for(int i=0;i<n;++i)fluctuation.alpha[i]-=centered.mean[i];long double draw=point-cmass;bool accepted=false;
  for(int attempt=0;attempt<100000;++attempt){int port=categorical(proposal,draw);for(int i=0;i<n;++i)weights[i]=optical(port,i);auto temp=jump_mpo93(c,fluctuation,weights);long double rate=canonical93(temp,true);for(int i=0;i<n;++i)rate+=std::norm(weights[i])*centered.outside[i];long double envelope=n*proposal[port];++stats[11];need93(rate>=0&&std::isfinite(rate)&&rate<=envelope*(1+1e-10L),"Centered rejection bound violated");if(uniform84(rng)*envelope<rate){chosen=port;chosen_variance=rate;accepted=true;break;}draw=uniform84(rng)*vmass;}
  if(!accepted)throw Failure93(4,"Centered rejection limit exceeded");
 }
 for(int i=0;i<n;++i)weights[i]=optical(chosen,i);auto state=jump_mpo93(c,g,weights);long double inside=canonical93(state),outside=0;for(int i=0;i<n;++i)outside+=std::norm(weights[i])*centered.outside[i];long double full=inside+outside;
 if(residual_branch){long double reconstructed=coherent[chosen]+chosen_variance;stats[13]=std::max(stats[13],double(std::abs(full-reconstructed)/full));need93(std::abs(full-reconstructed)<1e-8L*full,"Centered selected hazard mismatch");}
 long double schmidt94=compress93(state,maxbond,stats);weights94(outside/full,-std::expm1(schmidt94));long double retention=std::log1p(-outside/full)+schmidt94;charge93(std::sqrt(-std::expm1(retention)),budget,stats);c=std::move(state);return chosen;
}
API93 int mps93_local(int d,double t,double ar,double ai,double eta,double* output,double* stats,char* error,int cap){try{Guard93 guard;need93(d>=1&&d<=64&&std::abs(t)<1&&eta>=0&&eta<=1&&output&&stats,"Local map arguments");auto map=local93(d,t,{ar,ai},eta);for(int a=0;a<d;++a)for(int b=0;b<d;++b){auto z=map.map(a,b);output[2*(a*d+b)]=z.real();output[2*(a*d+b)+1]=z.imag();}stats[0]=map.t;stats[1]=map.alpha.real();stats[2]=map.alpha.imag();stats[3]=double(map.log_weight);return 0;}catch(const std::exception& e){message93(error,cap,e.what());return 1;}}
API93 int mps93_sample(int m,int n,const double* optical,const double* squeezing,const double* displacement,const double* noise,int d,int bond,double budget,int photon_cap,int count,std::uint64_t experiment,unsigned sector,std::uint64_t offset,int threads,std::uint32_t* output,double* statistics,int* errors,char* message,int cap){try{
 Guard93 guard;need93(n>=1&&n<=128&&m>=n&&m<=16384&&optical&&squeezing&&displacement&&noise&&d>=2&&d<=32&&bond>=1&&bond<=256&&budget>=0&&budget<=1&&photon_cap>=0&&photon_cap<=256&&count>=0&&sector<=65535&&offset<=0x100000000ULL&&std::uint64_t(count)<=0x100000000ULL-offset&&threads>=1&&threads<=64&&output&&statistics&&errors,"MPS sampling arguments");Mat93 U(m,n);Gaussian93 initial;initial.t.resize(n);initial.alpha.resize(n);for(int i=0;i<m;++i)for(int j=0;j<n;++j){int at=2*(i*n+j);U(i,j)={optical[at],optical[at+1]};}need93(U.allFinite()&&(U.adjoint()*U-Mat93::Identity(n,n)).cwiseAbs().maxCoeff()<1e-11,"MPS optical isometry");for(int i=0;i<n;++i){initial.t[i]=squeezing[i];initial.alpha[i]={displacement[2*i],displacement[2*i+1]};need93(std::isfinite(squeezing[i])&&std::abs(squeezing[i])<1&&std::isfinite(std::abs(initial.alpha[i])),"Gaussian input parameters");}for(int i=0;i<4*n*n;++i)need93(std::isfinite(noise[i]),"Gaussian noise factor");std::fill(output,output+std::size_t(count)*m,0);std::fill(statistics,statistics+std::size_t(count)*48,0);std::fill(errors,errors+count,0);std::atomic<int> next{0};std::vector<std::string> messages(count);std::vector<std::thread> pool;
 auto worker=[&](){Guard93 guard;for(;;){int row=next.fetch_add(1);if(row>=count)break;double* stats=statistics+48*row;profile94=stats;DomainRng84 waits(experiment,sector,Waiting84,offset+row),detectors(experiment,sector,Detector84,offset+row),normalrng(experiment,sector,Displacement84,offset+row);int photons=0;try{auto g=initial;std::normal_distribution<double> normal;std::vector<double> z(2*n);for(auto& v:z)v=normal(normalrng);for(int i=0;i<n;++i){double re=0,im=0;for(int j=0;j<2*n;++j){re+=noise[i*2*n+j]*z[j];im+=noise[(i+n)*2*n+j]*z[j];}g.alpha[i]+=Z93(re,im);}Mps93 c(n,Site93(d,Mat93::Zero(1,1)));for(auto& site:c)site[0](0,0)=1;stats[8]=1;for(;;){if(!wait93(c,g,uniform84(waits),stats))break;if(photons==photon_cap)throw Failure93(1,"Explicit photon-cap outcome");int port=select_jump93(c,g,U,detectors,bond,budget,stats);++output[std::size_t(row)*m+port];++photons;}stats[1]=1;}catch(const Failure93& e){errors[row]=e.code;messages[row]=e.what();}catch(const std::exception& e){errors[row]=4;messages[row]=e.what();}stats[0]=photons;stats[14]=double(waits.used);stats[15]=double(detectors.used);stats[16]=double(normalrng.used);stats[17]=budget;}};
 try{for(int i=0;i<std::min(count,threads);++i)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[row]){message93(message,cap,"row="+std::to_string(row)+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){message93(message,cap,e.what());return 2;}}
