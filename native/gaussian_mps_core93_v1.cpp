// Gaussian background remains analytic; an MPS stores its coherent Fock core.
// All physical projections and Schmidt truncations charge trace-distance budget.
#include "gaussian_rng_domain84_v1.hpp"
#include <Eigen/Dense>
#include <algorithm>
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
struct Failure93:std::runtime_error{int code;Failure93(int c,const char* s):std::runtime_error(s),code(c){}};
static void need93(bool b,const char* s){if(!b)throw Failure93(4,s);}
static void message93(char* out,int cap,const std::string& value){if(out&&cap>0){std::strncpy(out,value.c_str(),cap-1);out[cap-1]=0;}}
static long double norm93(const Mat93& m){long double out=0;for(Eigen::Index i=0;i<m.size();++i){auto z=m.data()[i];out+=(long double)z.real()*z.real()+(long double)z.imag()*z.imag();}return out;}
static Mat93 qr_factor93(const Mat93& a){int k=std::min(a.rows(),a.cols());Eigen::HouseholderQR<Mat93> qr(a);return qr.matrixQR().topRows(k).template triangularView<Eigen::Upper>();}
// Right-canonicalize without truncation. Return incoming squared norm, normalize.
static long double canonical93(Mps93& c,bool allow_zero=false){
 for(int i=int(c.size())-1;i>0;--i){int dl=c[i][0].rows(),dr=c[i][0].cols(),d=c[i].size();Mat93 flat(dl,d*dr);for(int s=0;s<d;++s)flat.middleCols(s*dr,dr)=c[i][s];Mat93 adj=flat.adjoint();int rank=std::min(adj.rows(),adj.cols());Eigen::HouseholderQR<Mat93> qr(adj);Mat93 Q=qr.householderQ()*Mat93::Identity(adj.rows(),rank);Mat93 R=qr.matrixQR().topRows(rank).template triangularView<Eigen::Upper>();Mat93 right=Q.adjoint();for(int s=0;s<d;++s)c[i][s]=right.middleCols(s*dr,dr);Mat93 left=R.adjoint();for(auto& a:c[i-1])a=(a*left).eval();}
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
static Rates93 rates93(const Mps93& c,const Gaussian93& g,bool full){
 int n=c.size();std::vector<Site93> acted(n);std::vector<Mat93> left(n);std::vector<long double> outside(n);Mat93 F=Mat93::Ones(1,1),G=Mat93::Zero(n,n);long double total=0;
 for(int i=0;i<n;++i){int dl=c[i][0].rows(),dr=c[i][0].cols(),d=c[i].size();left[i]=F.adjoint()*F;acted[i]=apply93(c[i],physical93(d,g.t[i],g.alpha[i]));long double inside=0;for(auto& a:acted[i])inside+=norm93((F*a).eval());outside[i]=double(d)*g.t[i]*g.t[i]/(1-g.t[i]*g.t[i])*norm93((F*c[i].back()).eval());G(i,i)=double(inside+outside[i]);total+=inside+outside[i];Mat93 stack(d*F.rows(),dr);for(int s=0;s<d;++s)stack.middleRows(s*F.rows(),F.rows())=F*c[i][s];F=qr_factor93(stack);}
 if(full)for(int i=0;i<n;++i){Mat93 x=transfer93(left[i],acted[i],c[i]);for(int j=i+1;j<n;++j){Z93 z=transfer93(x,c[j],acted[j]).trace();G(i,j)=z;G(j,i)=std::conj(z);x=transfer93(x,c[j],c[j]);}}
 need93(std::isfinite(total)&&total>=0,"Invalid total photon hazard");return {std::move(G),std::move(outside),total};
}
static long double attenuate93(const Mps93& c,const Gaussian93& g,double eta,Mps93* transformed,Gaussian93* next){
 int n=c.size();long double logw=0;Mps93 out;Gaussian93 gp;gp.t.resize(n);gp.alpha.resize(n);if(eta==0){Mat93 amplitude=Mat93::Ones(1,1);for(int i=0;i<n;++i){auto map=local93(c[i].size(),g.t[i],g.alpha[i],eta);Mat93 a=Mat93::Zero(c[i][0].rows(),c[i][0].cols());for(int s=0;s<int(c[i].size());++s)a.noalias()+=map.map(0,s)*c[i][s];amplitude=(amplitude*a).eval();logw+=map.log_weight;}return std::exp(logw)*norm93(amplitude);}
 out.reserve(n);for(int i=0;i<n;++i){auto map=local93(c[i].size(),g.t[i],g.alpha[i],eta);out.push_back(apply93(c[i],map.map));gp.t[i]=map.t;gp.alpha[i]=map.alpha;logw+=map.log_weight;}auto norm=canonical93(out);long double p=std::exp(logw)*norm;if(transformed)*transformed=std::move(out);if(next)*next=std::move(gp);return p;
}
static bool wait93(Mps93& c,Gaussian93& g,double u,double* stats){
 long double vacuum=attenuate93(c,g,0,nullptr,nullptr);++stats[3];need93(vacuum>=0&&vacuum<=1+1e-10L,"Invalid MPS vacuum probability");if(u<=vacuum)return false;double lo=0,hi=1,eta=.5;
 for(int iteration=0;iteration<80;++iteration){Mps93 cp;Gaussian93 gp;auto p=attenuate93(c,g,eta,&cp,&gp);++stats[2];++stats[3];need93(p>0&&p<=1+1e-9L&&std::isfinite(p),"Invalid MPS survival");long double residual=p-u;if(std::abs(residual)<2e-12L||hi-lo<4e-15){stats[4]=std::max(stats[4],double(std::abs(residual)));c=std::move(cp);g=std::move(gp);return true;}if(residual>0)hi=eta;else lo=eta;auto rate=rates93(cp,gp,false);long double derivative=p*rate.total/eta;double proposed=eta-double(residual/derivative);eta=std::isfinite(proposed)&&proposed>lo&&proposed<hi?proposed:(lo+hi)/2;}
 throw Failure93(4,"MPS waiting root failed");
}
// MPO of sum_i coefficient_i O_i. The chosen local basis projection is explicit.
static Mps93 jump_mpo93(const Mps93& c,const Gaussian93& g,const std::vector<Z93>& coefficients){
 int n=c.size();Mps93 out(n);for(int i=0;i<n;++i){auto acted=apply93(c[i],physical93(c[i].size(),g.t[i],g.alpha[i]));int dl=c[i][0].rows(),dr=c[i][0].cols(),d=c[i].size();out[i].resize(d);for(int s=0;s<d;++s){Mat93 b=coefficients[i]*acted[s];if(n==1)out[i][s]=b;else if(i==0){out[i][s]=Mat93::Zero(1,2*dr);out[i][s].leftCols(dr)=c[i][s];out[i][s].rightCols(dr)=b;}else if(i==n-1){out[i][s]=Mat93::Zero(2*dl,1);out[i][s].topRows(dl)=b;out[i][s].bottomRows(dl)=c[i][s];}else{out[i][s]=Mat93::Zero(2*dl,2*dr);out[i][s].topLeftCorner(dl,dr)=c[i][s];out[i][s].topRightCorner(dl,dr)=b;out[i][s].bottomRightCorner(dl,dr)=c[i][s];}}}return out;
}
static void charge93(double delta,double limit,double* stats){need93(delta>=0&&std::isfinite(delta),"Invalid discarded weight");if(stats[5]+delta>limit)throw Failure93(2,"Trace-distance budget exhausted");stats[5]+=delta;stats[6]=std::max(stats[6],delta);}
static void compress93(Mps93& c,int maxbond,double budget,double* stats){
 for(int i=0;i+1<int(c.size());++i){int dl=c[i][0].rows(),dr=c[i][0].cols(),d=c[i].size();Mat93 flat(d*dl,dr);for(int s=0;s<d;++s)flat.middleRows(s*dl,dl)=c[i][s];Eigen::JacobiSVD<Mat93> svd(flat,Eigen::ComputeThinU|Eigen::ComputeThinV);need93(svd.info()==Eigen::Success,"MPS SVD failed");auto values=svd.singularValues();int keep=std::min<int>(maxbond,values.size());long double mass=0,discard=0;for(int j=0;j<values.size();++j){long double w=(long double)values[j]*values[j];mass+=w;if(j>=keep)discard+=w;}need93(mass>0,"MPS SVD zero norm");charge93(std::sqrt(discard/mass),budget,stats);if(discard>0)++stats[7];Mat93 U=svd.matrixU().leftCols(keep),T=values.head(keep).asDiagonal()*svd.matrixV().leftCols(keep).adjoint();T/=std::sqrt(mass-discard);for(int s=0;s<d;++s)c[i][s]=U.middleRows(s*dl,dl);for(auto& a:c[i+1])a=(T*a).eval();stats[8]=std::max(stats[8],double(keep));}
 auto norm=canonical93(c);stats[9]=std::max(stats[9],double(std::abs(norm-1)));auto residual=canonical_residual93(c);stats[10]=std::max(stats[10],residual);need93(residual<1e-9&&std::abs(norm-1)<1e-8L,"MPS canonical residual failed");
}
static int select_jump93(Mps93& c,const Gaussian93& g,const Mat93& optical,double uniform,int maxbond,double budget,double* stats){
 auto rates=rates93(c,g,true);int m=optical.rows(),n=optical.cols();std::vector<long double> port(m);long double total=0;std::vector<Z93> weights(n);
 for(int j=0;j<m;++j){long double value=0,scale=0;for(int a=0;a<n;++a)for(int b=0;b<n;++b){auto z=std::conj(optical(j,a))*optical(j,b)*rates.gram(a,b);value+=z.real();scale+=std::abs(z);}if(value<=1e-12L*scale){for(int a=0;a<n;++a)weights[a]=optical(j,a);auto state=jump_mpo93(c,g,weights);long double outside=0;for(int a=0;a<n;++a)outside+=std::norm(weights[a])*rates.outside[a];value=canonical93(state,true)+outside;++stats[11];}need93(value>=0&&std::isfinite(value),"Negative/nonfinite MPS port hazard");port[j]=value;total+=value;}
 need93(total>0&&std::isfinite(total),"Zero MPS output hazard");stats[12]=std::max(stats[12],double(std::abs(total-rates.total)/total));long double point=uniform*total;int chosen=m-1;for(int j=0;j<m;++j){if(point<port[j]){chosen=j;break;}point-=port[j];}for(int a=0;a<n;++a)weights[a]=optical(chosen,a);auto state=jump_mpo93(c,g,weights);long double outside=0;for(int a=0;a<n;++a)outside+=std::norm(weights[a])*rates.outside[a];auto inside=canonical93(state);long double full=inside+outside;stats[13]=std::max(stats[13],double(std::abs(full-port[chosen])/full));need93(std::abs(full-port[chosen])<1e-8L*full,"Selected MPS hazard mismatch");charge93(std::sqrt(outside/full),budget,stats);compress93(state,maxbond,budget,stats);c=std::move(state);return chosen;
}
API93 int mps93_local(int d,double t,double ar,double ai,double eta,double* output,double* stats,char* error,int cap){try{Guard93 guard;need93(d>=1&&d<=64&&std::abs(t)<1&&eta>=0&&eta<=1&&output&&stats,"Local map arguments");auto map=local93(d,t,{ar,ai},eta);for(int a=0;a<d;++a)for(int b=0;b<d;++b){auto z=map.map(a,b);output[2*(a*d+b)]=z.real();output[2*(a*d+b)+1]=z.imag();}stats[0]=map.t;stats[1]=map.alpha.real();stats[2]=map.alpha.imag();stats[3]=double(map.log_weight);return 0;}catch(const std::exception& e){message93(error,cap,e.what());return 1;}}
API93 int mps93_sample(int m,int n,const double* optical,const double* squeezing,const double* displacement,const double* noise,int d,int bond,double budget,int photon_cap,int count,std::uint64_t experiment,unsigned sector,std::uint64_t offset,int threads,std::uint32_t* output,double* statistics,int* errors,char* message,int cap){try{
 Guard93 guard;need93(n>=1&&n<=128&&m>=n&&m<=16384&&optical&&squeezing&&displacement&&noise&&d>=2&&d<=32&&bond>=1&&bond<=256&&budget>=0&&budget<=1&&photon_cap>=0&&photon_cap<=256&&count>=0&&sector<=65535&&offset<=0x100000000ULL&&std::uint64_t(count)<=0x100000000ULL-offset&&threads>=1&&threads<=64&&output&&statistics&&errors,"MPS sampling arguments");Mat93 U(m,n);Gaussian93 initial;initial.t.resize(n);initial.alpha.resize(n);for(int i=0;i<m;++i)for(int j=0;j<n;++j){int at=2*(i*n+j);U(i,j)={optical[at],optical[at+1]};}need93(U.allFinite()&&(U.adjoint()*U-Mat93::Identity(n,n)).cwiseAbs().maxCoeff()<1e-11,"MPS optical isometry");for(int i=0;i<n;++i){initial.t[i]=squeezing[i];initial.alpha[i]={displacement[2*i],displacement[2*i+1]};need93(std::isfinite(squeezing[i])&&std::abs(squeezing[i])<1&&std::isfinite(std::abs(initial.alpha[i])),"Gaussian input parameters");}for(int i=0;i<4*n*n;++i)need93(std::isfinite(noise[i]),"Gaussian noise factor");std::fill(output,output+std::size_t(count)*m,0);std::fill(statistics,statistics+std::size_t(count)*18,0);std::fill(errors,errors+count,0);std::atomic<int> next{0};std::vector<std::string> messages(count);std::vector<std::thread> pool;
 auto worker=[&](){Guard93 guard;for(;;){int row=next.fetch_add(1);if(row>=count)break;double* stats=statistics+18*row;DomainRng84 waits(experiment,sector,Waiting84,offset+row),detectors(experiment,sector,Detector84,offset+row),normalrng(experiment,sector,Displacement84,offset+row);int photons=0;try{auto g=initial;std::normal_distribution<double> normal;std::vector<double> z(2*n);for(auto& v:z)v=normal(normalrng);for(int i=0;i<n;++i){double re=0,im=0;for(int j=0;j<2*n;++j){re+=noise[i*2*n+j]*z[j];im+=noise[(i+n)*2*n+j]*z[j];}g.alpha[i]+=Z93(re,im);}Mps93 c(n,Site93(d,Mat93::Zero(1,1)));for(auto& site:c)site[0](0,0)=1;stats[8]=1;for(;;){if(!wait93(c,g,uniform84(waits),stats))break;if(photons==photon_cap)throw Failure93(1,"Explicit photon-cap outcome");int port=select_jump93(c,g,U,uniform84(detectors),bond,budget,stats);++output[std::size_t(row)*m+port];++photons;}stats[1]=1;}catch(const Failure93& e){errors[row]=e.code;messages[row]=e.what();}catch(const std::exception& e){errors[row]=4;messages[row]=e.what();}stats[0]=photons;stats[14]=double(waits.used);stats[15]=double(detectors.used);stats[16]=double(normalrng.used);stats[17]=budget;}};
 try{for(int i=0;i<std::min(count,threads);++i)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[row]){message93(message,cap,"row="+std::to_string(row)+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){message93(message,cap,e.what());return 2;}}
