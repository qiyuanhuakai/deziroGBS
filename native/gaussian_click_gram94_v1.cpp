// Gaussian-to-purification bridge, not yet a complete compressed sampler.
// For |g_S> = (P_vac(S) tensor I)|Psi_rho>, <g_S|g_T> = Pr(vac(S union T)).
// No Fock-space table or hidden photon count is constructed.
#include "gaussian_mps_core94_v2.cpp"
#include <map>
using Real94=Eigen::Matrix<long double,Eigen::Dynamic,Eigen::Dynamic>;
using RealVec94=Eigen::Matrix<long double,Eigen::Dynamic,1>;
API93 int gaussian94_gram(int m,const double* q,const double* mean,int count,
 const std::uint64_t* masks,int next_mode,double* gram,double* zero_gram,
 double* stats,char* error,int cap){try{
 Guard93 guard;need93(m>=1&&m<=2048&&q&&mean&&count>=1&&count<=128&&masks
  &&next_mode>=0&&next_mode<m&&gram&&zero_gram&&stats,"Gaussian Gram arguments");
 int dim=2*m,words=(m+63)/64;
 for(int i=0;i<dim;++i){need93(std::isfinite(mean[i]),"Nonfinite Gaussian mean");for(int j=0;j<dim;++j)
  need93(std::isfinite(q[std::size_t(i)*dim+j])&&std::abs(q[std::size_t(i)*dim+j]-q[std::size_t(j)*dim+i])<1e-11,"Nonfinite/asymmetric Gaussian Q");}
 if(m%64)for(int a=0;a<count;++a)need93((masks[std::size_t(a)*words+words-1]>>(m%64))==0,"Mask exceeds physical modes");
 std::map<std::vector<std::uint64_t>,long double> cache;int largest=0;
 auto vacuum=[&](const std::vector<std::uint64_t>& mask){
  auto found=cache.find(mask);if(found!=cache.end())return found->second;
  std::vector<int> selected;for(int i=0;i<m;++i)if((mask[i/64]>>(i%64))&1)selected.push_back(i);
  int k=selected.size();largest=std::max(largest,k);long double probability=1;
  if(k){std::vector<int> indices=selected;for(int i:selected)indices.push_back(i+m);Real94 sub(2*k,2*k);RealVec94 mu(2*k);
   for(int i=0;i<2*k;++i){mu[i]=mean[indices[i]];for(int j=0;j<2*k;++j)sub(i,j)=q[std::size_t(indices[i])*dim+indices[j]];}
   Eigen::LLT<Real94> chol(sub);need93(chol.info()==Eigen::Success,"Vacuum covariance is not positive definite");Real94 L=chol.matrixL();long double logdet_half=0;for(int i=0;i<2*k;++i){need93(L(i,i)>0,"Nonpositive Cholesky diagonal");logdet_half+=std::log(L(i,i));}
   RealVec94 whitened=L.template triangularView<Eigen::Lower>().solve(mu);long double quadratic=whitened.squaredNorm();probability=std::exp(-.5L*quadratic-logdet_half);
   need93(probability>=0&&probability<=1+1e-12L&&std::isfinite(probability),"Invalid Gaussian vacuum probability");
  }
  cache.emplace(mask,probability);return probability;
 };
 for(int a=0;a<count;++a)for(int b=0;b<=a;++b){std::vector<std::uint64_t> mask(words);for(int w=0;w<words;++w)mask[w]=masks[std::size_t(a)*words+w]|masks[std::size_t(b)*words+w];
  double g=double(vacuum(mask));mask[next_mode/64]|=std::uint64_t(1)<<(next_mode%64);double z=double(vacuum(mask));gram[a*count+b]=gram[b*count+a]=g;zero_gram[a*count+b]=zero_gram[b*count+a]=z;
 }
 stats[0]=cache.size();stats[1]=largest;return 0;
 }catch(const std::exception& e){message93(error,cap,e.what());return 1;}}
