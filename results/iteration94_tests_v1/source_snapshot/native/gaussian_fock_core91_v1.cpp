// Positive finite-occupation core U S_2(r)|k,0>, through arbitrary m x 2
// complex isometries. Enumerates complete fixed-total Fock layers. This is not
// a polynomial-cost general GBS sampler. All omitted mass is reported as tail.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <vector>
#define API extern "C" __declspec(dllexport)
using Z91=std::complex<long double>;
using Layer91=std::map<std::uint64_t,Z91>;
static void require91(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
static Layer91 create91(const Layer91& a,const std::vector<Z91>& u,long double divisor){
 Layer91 b;
 for(const auto& [key,value]:a)for(int j=0;j<int(u.size());++j){unsigned n=(key>>(8*j))&255;auto next=key+(std::uint64_t(1)<<(8*j));b[next]+=value*u[j]*std::sqrt((n+1)/divisor);}
 return b;
}
API int fock91_law(int modes,int photons,double r,int cutoff,const double* optical,std::uint64_t capacity,std::uint64_t* keys,double* probabilities,std::uint64_t* written,double* stats,char* msg,int cap){try{
 require91(modes>=2&&modes<=6&&photons>=0&&photons<=64&&cutoff>=photons&&cutoff<=96&&std::isfinite(r)&&r>=0&&r<1&&optical&&keys&&probabilities&&written&&stats,"Fock91 arguments");
 std::vector<Z91> a(modes),b(modes);for(int j=0;j<modes;++j){a[j]={optical[4*j],optical[4*j+1]};b[j]={optical[4*j+2],optical[4*j+3]};require91(std::isfinite(std::abs(a[j]))&&std::isfinite(std::abs(b[j])),"Fock91 finite isometry");}
 long double aa=0,bb=0;Z91 ab=0;for(int j=0;j<modes;++j){aa+=std::norm(a[j]);bb+=std::norm(b[j]);ab+=std::conj(a[j])*b[j];}long double iso=std::max({std::abs(aa-1),std::abs(bb-1),std::abs(ab)});require91(iso<1e-12L,"Fock91 columns must be an isometry");
 Layer91 layer{{0,Z91(1)}};for(int j=0;j<photons;++j)layer=create91(layer,a,j+1);
 long double s=static_cast<long double>(r)*r,weight=std::pow(1-s,photons+1),mass=0,maxnorm=0;std::uint64_t at=0;
 for(int pairs=0;photons+2*pairs<=cutoff;++pairs){long double norm=0;for(const auto& [key,z]:layer){require91(at<capacity,"Fock91 output capacity");long double p=weight*std::norm(z);require91(std::isfinite(p)&&p>=0,"Fock91 probability");keys[at]=key;probabilities[at++]=double(p);norm+=std::norm(z);}maxnorm=std::max(maxnorm,std::abs(norm-1));mass+=weight;
  if(photons+2*(pairs+1)>cutoff)break;
  layer=create91(layer,a,photons+pairs+1);layer=create91(layer,b,pairs+1);weight*=s*(photons+pairs+1)/(pairs+1);
 }
 *written=at;stats[0]=double(mass);stats[1]=double(1-mass);stats[2]=double(maxnorm);stats[3]=double(iso);return 0;
 }catch(const std::exception& e){if(msg&&cap>0)std::snprintf(msg,cap,"%s",e.what());return 1;}}
// The caller keeps the unresolved probability as a separate failure outcome.
// It is never renormalized or filled by drawing a replacement record.
API int fock91_draw(std::uint64_t size,const std::uint64_t* keys,const double* probabilities,int count,const double* uniforms,std::uint64_t* output,int* failures,char* msg,int cap){try{
 require91(size>0&&keys&&probabilities&&count>=0&&uniforms&&output&&failures,"Fock91 draw arguments");std::vector<long double> cdf(size);long double sum=0;for(std::uint64_t j=0;j<size;++j){require91(std::isfinite(probabilities[j])&&probabilities[j]>=0,"Fock91 finite nonnegative law");sum+=probabilities[j];cdf[j]=sum;}require91(sum<=1+1e-12L,"Fock91 mass above one");
 for(int i=0;i<count;++i){require91(std::isfinite(uniforms[i])&&uniforms[i]>=0&&uniforms[i]<1,"Fock91 explicit uniform");auto it=std::upper_bound(cdf.begin(),cdf.end(),static_cast<long double>(uniforms[i]));failures[i]=it==cdf.end();output[i]=it==cdf.end()?0:keys[it-cdf.begin()];}return 0;
 }catch(const std::exception& e){if(msg&&cap>0)std::snprintf(msg,cap,"%s",e.what());return 1;}}
