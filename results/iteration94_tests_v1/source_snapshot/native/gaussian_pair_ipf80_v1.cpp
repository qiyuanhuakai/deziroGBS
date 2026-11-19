// Entropic pair-table calibration with explicit convergence and work bounds.
// Alternating exact marginal projections; no negative-probability clipping.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#define API extern "C" __declspec(dllexport)
API int pair80_ipf(int m,int latent,const double* weights,const double* marginals,const double* target,int maximum,double tolerance,double* table,double* stats,char* message,int cap){try{
 auto need=[](bool ok,const char* why){if(!ok)throw std::runtime_error(why);};
 need(m>=1&&m<=128&&latent>=1&&latent<=4096&&weights&&marginals&&target&&table&&stats&&maximum>=1&&std::isfinite(tolerance)&&tolerance>0&&tolerance<=1e-4,"Invalid pair IPF arguments");
 std::size_t mm=std::size_t(m)*m,cells=mm*latent;need(cells<=8000000&&cells*std::uint64_t(maximum)<=500000000ULL,"Pair IPF work limit");std::fill(stats,stats+4,0);
 long double sw=0;for(int l=0;l<latent;++l){need(std::isfinite(weights[l])&&weights[l]>0,"Latent weights must be positive");sw+=weights[l];long double sum=0;for(int i=0;i<m;++i){double p=marginals[l*m+i];need(std::isfinite(p)&&p>=0,"Invalid latent marginal");sum+=p;}need(std::abs(sum-1)<1e-10L,"Latent marginal normalization");}need(std::abs(sw-1)<1e-10L,"Latent weight normalization");
 for(std::size_t k=0;k<mm;++k){need(std::isfinite(target[k])&&target[k]>=0,"Invalid pair target");for(int l=0;l<latent;++l)table[std::size_t(l)*mm+k]=target[k];}
 auto audit=[&](){double residual=0;
  for(std::size_t k=0;k<mm;++k){long double mass=0;for(int l=0;l<latent;++l)mass+=weights[l]*table[std::size_t(l)*mm+k];residual=std::max(residual,double(std::abs(mass-target[k])));}
  for(int l=0;l<latent;++l)for(int i=0;i<m;++i){long double row=0,col=0;for(int j=0;j<m;++j){row+=table[std::size_t(l)*mm+i*m+j];col+=table[std::size_t(l)*mm+j*m+i];}residual=std::max(residual,double(std::abs(row-marginals[l*m+i])));residual=std::max(residual,double(std::abs(col-marginals[l*m+i])));}
  return residual;
 };
 for(int iteration=1;iteration<=maximum;++iteration){
  for(std::size_t k=0;k<mm;++k){long double mass=0;for(int l=0;l<latent;++l)mass+=weights[l]*table[std::size_t(l)*mm+k];need(mass>0||target[k]==0,"Incompatible pair support");double factor=mass>0?double(target[k]/mass):0;for(int l=0;l<latent;++l)table[std::size_t(l)*mm+k]*=factor;}
  for(int l=0;l<latent;++l)for(int i=0;i<m;++i){long double mass=0;for(int j=0;j<m;++j)mass+=table[std::size_t(l)*mm+i*m+j];double want=marginals[l*m+i];need(mass>0||want==0,"Incompatible row support");double factor=mass>0?double(want/mass):0;for(int j=0;j<m;++j)table[std::size_t(l)*mm+i*m+j]*=factor;}
  for(int l=0;l<latent;++l)for(int j=0;j<m;++j){long double mass=0;for(int i=0;i<m;++i)mass+=table[std::size_t(l)*mm+i*m+j];double want=marginals[l*m+j];need(mass>0||want==0,"Incompatible column support");double factor=mass>0?double(want/mass):0;for(int i=0;i<m;++i)table[std::size_t(l)*mm+i*m+j]*=factor;}
  stats[0]=iteration;if(iteration%16==0||iteration==maximum){stats[1]=audit();if(stats[1]<=tolerance){
   for(int l=0;l<latent;++l)for(int i=0;i<m;++i)for(int j=0;j<i;++j){double value=.5*(table[std::size_t(l)*mm+i*m+j]+table[std::size_t(l)*mm+j*m+i]);table[std::size_t(l)*mm+i*m+j]=table[std::size_t(l)*mm+j*m+i]=value;}
   stats[2]=audit();need(stats[2]<=tolerance,"Symmetric IPF residual exceeded tolerance");stats[3]=double(cells)*iteration;return 0;
  }}
 }
 stats[3]=double(cells)*maximum;throw std::runtime_error("Pair IPF failed to converge within explicit work budget");
}catch(const std::exception& e){if(message&&cap>0){auto n=std::min(std::strlen(e.what()),std::size_t(cap-1));std::memcpy(message,e.what(),n);message[n]=0;}return 1;}}
