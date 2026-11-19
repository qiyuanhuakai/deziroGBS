// Factorized IPF: P_l(i,j)=K(i,j)*A(l,i)*B(l,j).
// No L*m*m allocation. Final physical pair table is (P_l+P_l^T)/2.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <omp.h>
#define API extern "C" __declspec(dllexport)
namespace factor81 {
using Matrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void error(char* out,int cap,const char* why){if(out&&cap>0){auto n=std::min(std::strlen(why),std::size_t(cap-1));std::memcpy(out,why,n);out[n]=0;}}
struct Threads {int previous;Threads(int n):previous(omp_get_max_threads()){omp_set_num_threads(n);}~Threads(){omp_set_num_threads(previous);}};
}
API int factor81_ipf(int m,int latent,const double* weights,const double* marginals,const double* target,int maximum,double tolerance,int threads,double* kernel,double* left,double* right,double* stats,char* message,int cap){try{
 using namespace factor81;need(m>=1&&m<=1024&&latent>=1&&latent<=131072&&weights&&marginals&&target&&kernel&&left&&right&&stats&&maximum>=1&&maximum<=10000&&threads>=1&&threads<=64&&std::isfinite(tolerance)&&tolerance>0&&tolerance<=1e-4,"Invalid factor IPF arguments");
 std::uint64_t lm=std::uint64_t(latent)*m;need(lm<=64000000&&lm*m*maximum<=500000000000ULL,"Factor IPF explicit work limit");Threads guard(threads);std::fill(stats,stats+6,0);
 Eigen::Map<const Matrix> p(marginals,latent,m),Q(target,m,m);Eigen::Map<Matrix> K(kernel,m,m),A(left,latent,m),B(right,latent,m);
 need(p.allFinite()&&Q.allFinite()&&p.minCoeff()>=0&&Q.minCoeff()>=0&&(Q-Q.transpose()).cwiseAbs().maxCoeff()<1e-10,"Invalid factor IPF probabilities");
 long double sw=0;for(int l=0;l<latent;++l){need(std::isfinite(weights[l])&&weights[l]>0,"Latent weights must be positive");sw+=weights[l];need(std::abs(p.row(l).sum()-1)<1e-10,"Latent marginal normalization");}need(std::abs(sw-1)<1e-10L&&std::abs(Q.sum()-1)<1e-10,"Pair normalization");
 K=Q;A.setOnes();B.setOnes();Matrix weighted(latent,m),den(latent,m),G(m,m),rows(latent,m);stats[3]=8.*(m*std::uint64_t(m)+2*lm);stats[4]=8.*(2*m*std::uint64_t(m)+5*lm);
 auto gram=[&](){weighted=B;for(int l=0;l<latent;++l)weighted.row(l)*=weights[l];G.noalias()=A.transpose()*weighted;};
 auto audit=[&](){gram();Matrix average=K.cwiseProduct(G);double residual=(average-Q).cwiseAbs().maxCoeff();den.noalias()=B*K.transpose();rows=A.cwiseProduct(den);residual=std::max(residual,(rows-p).cwiseAbs().maxCoeff());den.noalias()=A*K;den.array()*=B.array();residual=std::max(residual,(den-p).cwiseAbs().maxCoeff());stats[2]=std::max((.5*(average+average.transpose())-Q).cwiseAbs().maxCoeff(),(.5*(rows+den)-p).cwiseAbs().maxCoeff());return residual;};
 for(int iteration=1;iteration<=maximum;++iteration){
  gram();for(int i=0;i<m;++i)for(int j=0;j<m;++j){need(G(i,j)>0||Q(i,j)==0,"Incompatible factor pair support");K(i,j)=G(i,j)>0?Q(i,j)/G(i,j):0;}
  den.noalias()=B*K.transpose();for(int l=0;l<latent;++l)for(int i=0;i<m;++i){need(den(l,i)>0||p(l,i)==0,"Incompatible factor row support");A(l,i)=den(l,i)>0?p(l,i)/den(l,i):0;}
  den.noalias()=A*K;for(int l=0;l<latent;++l)for(int j=0;j<m;++j){need(den(l,j)>0||p(l,j)==0,"Incompatible factor column support");B(l,j)=den(l,j)>0?p(l,j)/den(l,j):0;}
  // Label-wise reciprocal scaling preserves every P_l entry.
  for(int l=0;l<latent;++l){double am=A.row(l).maxCoeff(),bm=B.row(l).maxCoeff();need(am>0&&bm>0&&std::isfinite(am)&&std::isfinite(bm),"Factor scale overflow");double scale=std::sqrt(am)/std::sqrt(bm);A.row(l)/=scale;B.row(l)*=scale;}
  need(K.allFinite()&&A.allFinite()&&B.allFinite(),"Nonfinite IPF factors");stats[0]=iteration;
  if(iteration%16==0||iteration==maximum){stats[1]=audit();stats[5]=double(lm)*m*iteration;if(stats[1]<=tolerance){need(stats[2]<=tolerance,"Symmetric factor residual exceeded tolerance");return 0;}}
 }
 throw std::runtime_error("Factor IPF failed to converge within explicit work budget");
}catch(const std::exception& e){factor81::error(message,cap,e.what());return 1;}}

// Finite-field 4-wise signs plus coordinate axes, generated in native code.
API int factor81_design(int n,int m,int latent,const double* real,const double* imag,int threads,double* directions,double* weights,double* marginals,char* message,int cap){try{
 using namespace factor81;need((n==16||n==32||n==128)&&m>=n&&m<=1024&&real&&imag&&directions&&weights&&marginals&&threads>=1&&threads<=64,"Invalid factor design arguments");int bits=0;for(int v=n;v;v>>=1)++bits;int q=1<<bits,poly=bits==5?0x25:bits==6?0x43:0x11b;need(latent==n+q*q,"Wrong factor design size");Threads guard(threads);
 auto multiply=[&](int a,int b){int value=0;while(b){if(b&1)value^=a;b>>=1;a<<=1;if(a&q)a^=poly;}return value;};std::vector<int> labels(n),cubes(n);
 for(int i=0;i<n;++i){int x=i+1;cubes[i]=multiply(multiply(x,x),x);labels[i]=x|(cubes[i]<<bits);need(labels[i]!=0,"Zero design label");}
 for(int i=0;i<n;++i)for(int j=0;j<i;++j){need((labels[i]^labels[j])!=0,"Dependent design pair");for(int k=0;k<j;++k){need((labels[i]^labels[j]^labels[k])!=0,"Dependent design triple");for(int h=0;h<k;++h)need((labels[i]^labels[j]^labels[k]^labels[h])!=0,"Dependent design quadruple");}}
 Eigen::Map<Matrix> U(directions,latent,n),p(marginals,latent,m);U.setZero();for(int i=0;i<n;++i){U(i,i)=1;weights[i]=2./(n*(n+2.));}
 for(int a=0;a<q;++a)for(int b=0;b<q;++b){int l=n+a*q+b;weights[l]=double(n)/((n+2.)*q*q);for(int i=0;i<n;++i)U(l,i)=((__builtin_popcount((a&(i+1))^(b&cubes[i]))&1)?-1.:1.)/std::sqrt(double(n));}
 Eigen::Map<const Matrix> vr(real,m,n),vi(imag,m,n);need(vr.allFinite()&&vi.allFinite(),"Nonfinite design optics");Matrix projection=U*vr.transpose();p=projection.array().square().matrix();projection.noalias()=U*vi.transpose();p.array()+=projection.array().square();return 0;
}catch(const std::exception& e){factor81::error(message,cap,e.what());return 1;}}
