// Exact-in-real-arithmetic first two total-click moments of independent
// zero-mean Gaussian sectors combined by OR. Numerical Cholesky, no clipping.
#include "gaussian_reset_cpu.cpp"
#include <omp.h>
API int click83_moments(int m,int sectors,const double* covariances,int threads,double* singles,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(m>=1&&m<=2048&&sectors>=1&&sectors<=8&&covariances&&singles&&stats&&threads>=1&&threads<=64,"Invalid click moment inputs");std::vector<long double> vacuum(m,1);long double mean=0,variance=0;int d=2*m;
 auto vacuum_subset=[&](int s,int i,int j){int size=j<0?2:4;int ids[4]={i,m+i,j,m+j};Eigen::Matrix<long double,Eigen::Dynamic,Eigen::Dynamic> q(size,size);for(int a=0;a<size;++a)for(int b=0;b<size;++b){double v=covariances[std::size_t(s)*d*d+std::size_t(ids[a])*d+ids[b]];need(std::isfinite(v),"Nonfinite Gaussian covariance");q(a,b)=(static_cast<long double>(v)+(a==b))/2;}Eigen::LLT<decltype(q)> chol(q);need(chol.info()==Eigen::Success,"No-click Cholesky failed");long double detroot=1;for(int a=0;a<size;++a)detroot*=chol.matrixL()(a,a);long double p=1/detroot;need(p>0&&p<=1+1e-14L,"Invalid vacuum moment");return p;};
 for(int i=0;i<m;++i){for(int s=0;s<sectors;++s)vacuum[i]*=vacuum_subset(s,i,-1);singles[i]=double(1-vacuum[i]);mean+=1-vacuum[i];variance+=(1-vacuum[i])*vacuum[i];}
 // Each thread reports an exception after joining; do not throw through OpenMP.
 std::vector<std::exception_ptr> errors(threads);long double cross=0;
 #pragma omp parallel num_threads(threads) reduction(+:cross)
 {PrecisionGuard precision;
  #pragma omp for schedule(dynamic,8)
  for(int i=0;i<m;++i){try{for(int j=0;j<i;++j){long double p=1;for(int s=0;s<sectors;++s)p*=vacuum_subset(s,i,j);cross+=2*(p-vacuum[i]*vacuum[j]);}}catch(...){errors[omp_get_thread_num()]=std::current_exception();}}
 }
 for(auto& e:errors)if(e)std::rethrow_exception(e);variance+=cross;need(variance>=0&&std::isfinite(variance),"Invalid total click variance");stats[0]=double(mean);stats[1]=double(variance);stats[2]=double(cross);stats[3]=std::uint64_t(m)*(m-1)/2;return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
