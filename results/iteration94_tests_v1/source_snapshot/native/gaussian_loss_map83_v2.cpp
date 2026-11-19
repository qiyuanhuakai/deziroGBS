// v1 transformation unchanged; native full-covariance export for equal-input
// Frugal comparisons. Export is preparation work, never a quality certificate.
#include "gaussian_loss_map83_v1.cpp"
#include <omp.h>
API int loss83_covariance(int m,int n,const double* vv,const double* xx,int threads,double* output,char* message,int cap){try{
 PrecisionGuard guard;need(m>=n&&n>0&&n<=128&&m<=4096&&vv&&xx&&output&&threads>=1&&threads<=64,"Invalid full covariance export");Eigen::setNbThreads(threads);Real a(2*m,2*n),x(2*n,2*n);
 for(int i=0;i<m;++i)for(int j=0;j<n;++j){auto k=2*(std::size_t(i)*n+j);double re=vv[k],im=vv[k+1];need(std::isfinite(re)&&std::isfinite(im),"Nonfinite map");a(i,j)=re;a(i,n+j)=-im;a(m+i,j)=im;a(m+i,n+j)=re;}
 for(int i=0;i<2*n;++i)for(int j=0;j<2*n;++j){x(i,j)=xx[std::size_t(i)*2*n+j];need(std::isfinite(x(i,j)),"Nonfinite covariance");}need((x-x.transpose()).norm()<1e-12,"Nonsymmetric covariance");
 Real c=2*(a*x)*a.transpose();c.diagonal().array()+=1;c=((c+c.transpose())/2).eval();for(int i=0;i<2*m;++i)for(int j=0;j<2*m;++j)output[std::size_t(i)*2*m+j]=c(i,j);return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
