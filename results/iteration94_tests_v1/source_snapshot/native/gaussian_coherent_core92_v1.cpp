// Exact coherent finite core over a rank-two two-mode squeezed background.
// No occupation-table enumeration, Gaussian reset, tail refill or rate clipping.
#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
using Real92=long double;
using Complex92=std::complex<Real92>;
#define API92 extern "C" __declspec(dllexport)
static void require92(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void message92(char* out,int cap,const std::string& value){if(out&&cap>0){std::strncpy(out,value.c_str(),cap-1);out[cap-1]=0;}}
struct Core92 {
 int k;std::vector<Complex92> data;
 explicit Core92(int degree):k(degree),data(std::size_t(k+1)*(k+1)){}
 Complex92& at(int a,int b){return data[std::size_t(a)*(k+1)+b];}
 const Complex92& at(int a,int b)const{return data[std::size_t(a)*(k+1)+b];}
 Real92 norm()const{Real92 n=0;for(auto z:data)n+=std::norm(z);return n;}
 void normalize(){auto n=norm();require92(n>0&&std::isfinite(n),"Invalid coherent norm");auto scale=1/std::sqrt(n);for(auto& z:data)z*=scale;}
};
static Core92 read92(int k,const double* input){
 require92(k>=0&&k<=32&&input,"Invalid core input");Core92 c(k);
 for(int a=0;a<=k;++a)for(int b=0;b<=k;++b){auto i=std::size_t(a)*(k+1)+b;Complex92 z(input[2*i],input[2*i+1]);require92(std::isfinite(z.real())&&std::isfinite(z.imag()),"Nonfinite core");require92(a+b<=k||z==Complex92(0),"Core outside total-degree bound");c.data[i]=z;}
 c.normalize();return c;
}
// S(r')^dag eta^(N/2) S(r) = c alpha^N exp(beta a b), r'=eta r.
static Core92 attenuate92(const Core92& c,Real92 r,Real92 eta){
 Core92 d(c.k);Real92 rp=r*eta,base=std::sqrt((1-r*r)/(1-rp*rp));Real92 alpha=std::sqrt(eta)*base,beta=-r*(1-eta*eta)/(1-rp*rp);
 std::vector<Real92> powers(c.k+1,1);for(int j=1;j<=c.k;++j)powers[j]=powers[j-1]*alpha;
 for(int a=0;a<=c.k;++a)for(int b=0;b<=c.k-a;++b){Complex92 z=0;Real92 f=1;for(int l=0;2*l<=c.k-a-b;++l){z+=f*c.at(a+l,b+l);f*=beta*std::sqrt(Real92(a+l+1)*(b+l+1))/(l+1);}d.at(a,b)=base*powers[a+b]*z;}
 return d;
}
static std::pair<Core92,Core92> annihilate92(const Core92& c,Real92 r){
 Core92 a(c.k+1),b(c.k+1);Real92 scale=1/std::sqrt(1-r*r);
 for(int i=0;i<=c.k;++i)for(int j=0;j<=c.k-i;++j){auto z=scale*c.at(i,j);if(i)a.at(i-1,j)+=std::sqrt(Real92(i))*z;if(j)b.at(i,j-1)+=std::sqrt(Real92(j))*z;a.at(i,j+1)+=r*std::sqrt(Real92(j+1))*z;b.at(i+1,j)+=r*std::sqrt(Real92(i+1))*z;}
 return {std::move(a),std::move(b)};
}
static bool wait92(Core92& c,Real92& r,Real92 u,double* stats){
 auto vacuum=attenuate92(c,r,0).norm();require92(vacuum<=1+1e-13L,"Vacuum probability exceeds one");++stats[3];
 if(u<=vacuum)return false;
 Real92 lo=0,hi=1,eta=.5L;
 for(int iteration=0;iteration<100;++iteration){auto d=attenuate92(c,r,eta);auto p=d.norm();require92(std::isfinite(p)&&p>0&&p<=1+1e-12L,"Invalid survival norm");++stats[1];++stats[3];Real92 residual=p-u;
  if(std::abs(residual)<=8e-18L||hi-lo<=8*std::numeric_limits<Real92>::epsilon()){stats[4]=std::max(stats[4],double(std::abs(residual)));d.normalize();c=std::move(d);r*=eta;return true;}
  if(residual>0)hi=eta;else lo=eta;
  auto ab=annihilate92(d,r*eta);Real92 derivative=(ab.first.norm()+ab.second.norm())/eta;Real92 next=eta-residual/derivative;
  eta=std::isfinite(next)&&next>lo&&next<hi?next:(lo+hi)/2;
 }
 throw std::runtime_error("Waiting root did not converge");
}
static int jump92(Core92& c,Real92 r,const std::vector<Complex92>& optics,int m,Real92 uniform,double* stats){
 auto ab=annihilate92(c,r);auto& a=ab.first;auto& b=ab.second;Real92 na=a.norm(),nb=b.norm();bool swapped=nb>na;const auto& first=swapped?b:a;const auto& second=swapped?a:b;Real92 n=swapped?nb:na;require92(n>0&&std::isfinite(n),"Zero jump hazard");Complex92 overlap=0;
 for(std::size_t i=0;i<first.data.size();++i)overlap+=std::conj(first.data[i])*second.data[i];Complex92 projection=overlap/n;Real92 residual=0;for(std::size_t i=0;i<first.data.size();++i)residual+=std::norm(second.data[i]-projection*first.data[i]);
 std::vector<Real92> rates(m);Real92 total=0;for(int j=0;j<m;++j){auto u=optics[2*j+(swapped?1:0)],v=optics[2*j+(swapped?0:1)];rates[j]=n*std::norm(u+v*projection)+residual*std::norm(v);total+=rates[j];}
 require92(total>0&&std::isfinite(total),"Invalid output hazard");stats[7]=std::max(stats[7],double(std::abs(total-na-nb)/total));Real92 point=uniform*total;int port=m-1;for(int j=0;j<m;++j){if(point<rates[j]){port=j;break;}point-=rates[j];}
 for(std::size_t i=0;i<a.data.size();++i)a.data[i]=optics[2*port]*a.data[i]+optics[2*port+1]*b.data[i];a.normalize();c=std::move(a);return port;
}
API92 int coherent92_flow(int degree,const double* input,double r,double eta,double* output,double* survival,char* message,int cap){try{
 require92(r>=0&&r<1&&eta>=0&&eta<=1&&output&&survival,"Invalid flow arguments");auto core=read92(degree,input);auto out=attenuate92(core,r,eta);*survival=double(out.norm());for(std::size_t i=0;i<out.data.size();++i){output[2*i]=double(out.data[i].real());output[2*i+1]=double(out.data[i].imag());}return 0;
 }catch(const std::exception& e){message92(message,cap,e.what());return 1;}}
API92 int coherent92_sample(int m,const double* optical,int degree,const double* input,double r,int count,std::uint64_t experiment,unsigned sector,std::uint64_t offset,int threads,int photon_cap,std::uint32_t* output,double* statistics,int* errors,char* message,int cap){try{
 require92(m>=2&&m<=16384&&optical&&r>=0&&r<1&&count>=0&&sector<=65535&&offset<=0x100000000ULL&&std::uint64_t(count)<=0x100000000ULL-offset&&threads>=1&&threads<=64&&photon_cap>=0&&photon_cap<=256&&output&&statistics&&errors,"Invalid sampling arguments");auto initial=read92(degree,input);std::vector<Complex92> optics(2*m);Complex92 gram[4]={};for(int j=0;j<m;++j)for(int a=0;a<2;++a){auto i=2*j+a;optics[i]={optical[2*i],optical[2*i+1]};require92(std::isfinite(optics[i].real())&&std::isfinite(optics[i].imag()),"Nonfinite optics");}
 for(int a=0;a<2;++a)for(int b=0;b<2;++b){for(int j=0;j<m;++j)gram[2*a+b]+=std::conj(optics[2*j+a])*optics[2*j+b];require92(std::abs(gram[2*a+b]-Complex92(a==b))<1e-12L,"Optical columns must be isometric");}
 std::fill(output,output+std::size_t(count)*m,0);std::fill(statistics,statistics+std::size_t(count)*8,0);std::fill(errors,errors+count,0);std::vector<std::string> messages(count);std::atomic<int> next{0};std::vector<std::thread> pool;
 auto worker=[&](){for(;;){int row=next.fetch_add(1);if(row>=count)break;auto core=initial;Real92 squeeze=r;auto stats=statistics+8*row;DomainRng84 waits(experiment,sector,Waiting84,offset+row),detectors(experiment,sector,Detector84,offset+row);int photons=0;try{for(;;){stats[2]=std::max(stats[2],double(core.k));if(!wait92(core,squeeze,uniform84(waits),stats))break;if(photons==photon_cap){errors[row]=1;messages[row]="Explicit photon-cap outcome";break;}int port=jump92(core,squeeze,optics,m,uniform84(detectors),stats);++output[std::size_t(row)*m+port];++photons;}}catch(const std::exception& e){errors[row]=2;messages[row]=e.what();}stats[0]=photons;stats[5]=double(waits.used);stats[6]=double(detectors.used);}};
 try{for(int j=0;j<std::min(threads,count);++j)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[row]){message92(message,cap,"row="+std::to_string(row)+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){message92(message,cap,e.what());return 2;}}
