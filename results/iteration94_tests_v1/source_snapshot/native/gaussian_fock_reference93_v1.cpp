// Independent pure-Gaussian Bargmann coefficient recurrence, reference only.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <xmmintrin.h>
#define API93 extern "C" __declspec(dllexport)
using Z=std::complex<long double>;
using Layer=std::unordered_map<std::uint64_t,Z>;
struct RefGuard93{unsigned short saved;unsigned sse;RefGuard93():sse(_mm_getcsr()){asm volatile("fnstcw %0":"=m"(saved));unsigned short w=(saved&~0x0f00u)|0x0300u;asm volatile("fldcw %0"::"m"(w));_mm_setcsr(sse&~0x6000u);}~RefGuard93(){asm volatile("fldcw %0"::"m"(saved));unsigned a=_mm_getcsr();_mm_setcsr((sse&~0x3fu)|((sse|a)&0x3fu));}};
static void require93(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
API93 int gaussian93_law(int m,int cutoff,const double* matrix,const double* linear,double log_norm_real,double log_norm_imag,std::uint64_t capacity,std::uint64_t* keys,double* probabilities,std::uint64_t* written,double* stats,char* message,int cap){try{
 RefGuard93 guard;require93(m>=1&&m<=6&&cutoff>=0&&cutoff<=64&&matrix&&linear&&keys&&probabilities&&written&&stats,"Reference arguments");std::vector<Z>B(m*m),z(m);for(int i=0;i<m;++i){z[i]={linear[2*i],linear[2*i+1]};require93(std::isfinite(std::abs(z[i])),"Reference finite displacement");for(int j=0;j<m;++j){int at=i*m+j;B[at]={matrix[2*at],matrix[2*at+1]};require93(std::isfinite(std::abs(B[at])),"Reference finite matrix");}}for(int i=0;i<m;++i)for(int j=0;j<m;++j)require93(std::abs(B[i*m+j]-B[j*m+i])<1e-12L,"Reference symmetric B");Z vacuum=std::exp(Z(log_norm_real,log_norm_imag));Layer previous{{0,vacuum}},previous2;std::uint64_t at=0;long double mass=0;
 auto emit=[&](std::uint64_t key,Z value){require93(at<capacity,"Reference capacity");long double p=std::norm(value);require93(std::isfinite(p)&&p>=0,"Reference finite probability");keys[at]=key;probabilities[at++]=double(p);mass+=p;};emit(0,vacuum);
 for(int total=1;total<=cutoff;++total){Layer layer;std::vector<int> occupation(m);std::function<void(int,int,std::uint64_t)> visit=[&](int axis,int remaining,std::uint64_t key){if(axis+1<m){for(int k=0;k<=remaining;++k){occupation[axis]=k;visit(axis+1,remaining-k,key|(std::uint64_t(k)<<(8*axis)));}return;}occupation[axis]=remaining;key|=std::uint64_t(remaining)<<(8*axis);int i=0;while(!occupation[i])++i;std::uint64_t first=key-(std::uint64_t(1)<<(8*i));auto one=previous.find(first);Z value=one==previous.end()?Z(0):z[i]*one->second;for(int j=0;j<m;++j){int copies=occupation[j]-(i==j);if(copies){auto two=previous2.find(first-(std::uint64_t(1)<<(8*j)));if(two!=previous2.end())value+=B[i*m+j]*std::sqrt((long double)copies)*two->second;}}value/=std::sqrt((long double)occupation[i]);layer.emplace(key,value);emit(key,value);};visit(0,total,0);previous2=std::move(previous);previous=std::move(layer);}
 *written=at;stats[0]=double(mass);stats[1]=double(std::norm(vacuum));stats[2]=double(at);return 0;
 }catch(const std::exception& e){if(message&&cap>0)std::snprintf(message,cap,"%s",e.what());return 1;}}
