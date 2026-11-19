// Lossy passive map T=V R, with V an isometry. Vacuum loss is absorbed in
// the active excess covariance: X'=realify(R) X realify(R)^T.
// This transformation is exact in real arithmetic; it does not make a
// finite-budget click sampler exact. No covariance eigenvalue clipping.
#include "gaussian_reset_cpu.cpp"
#include <Eigen/QR>
API int loss83_prepare(int m,int n,int sectors,const double* transmission,
 const double* inputs,double* isometry,double* outputs,double* stats,
 char* message,int cap){try{
 PrecisionGuard guard;need(m>=n&&n>=1&&n<=128&&m<=65536&&sectors>=1&&sectors<=8&&transmission&&inputs&&isometry&&outputs&&stats,"Invalid lossy map dimensions or buffers");
 Complex t(m,n);for(int i=0;i<m;++i)for(int j=0;j<n;++j){auto k=2*(std::size_t(i)*n+j);need(std::isfinite(transmission[k])&&std::isfinite(transmission[k+1]),"Nonfinite transmission");t(i,j)={transmission[k],transmission[k+1]};}
 Eigen::MatrixXcd gram=t.adjoint()*t;Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eig(gram,Eigen::EigenvaluesOnly);need(eig.info()==Eigen::Success&&eig.eigenvalues().maxCoeff()<=1+1e-10,"Transmission is not passive contraction");
 Eigen::HouseholderQR<Eigen::MatrixXcd> qr(t);Eigen::MatrixXcd v=qr.householderQ()*Eigen::MatrixXcd::Identity(m,n),r=qr.matrixQR().topRows(n).template triangularView<Eigen::Upper>();
 Real a(2*n,2*n);a.topLeftCorner(n,n)=r.real();a.topRightCorner(n,n)=-r.imag();a.bottomLeftCorner(n,n)=r.imag();a.bottomRightCorner(n,n)=r.real();
 stats[0]=(v.adjoint()*v-Eigen::MatrixXcd::Identity(n,n)).norm();stats[1]=(v*r-t).norm()/std::max(1.,t.norm());stats[2]=eig.eigenvalues().maxCoeff();stats[3]=std::numeric_limits<double>::infinity();stats[4]=0;
 need(stats[0]<1e-10&&stats[1]<1e-12,"Lossy QR residual too large");
 for(int s=0;s<sectors;++s){Real x(2*n,2*n);for(int i=0;i<2*n;++i)for(int j=0;j<2*n;++j){auto k=std::size_t(s)*4*n*n+std::size_t(i)*2*n+j;x(i,j)=inputs[k];need(std::isfinite(x(i,j)),"Nonfinite input covariance");}need((x-x.transpose()).norm()<1e-12,"Input covariance not symmetric");x=((x+x.transpose())/2).eval();
  auto physical=[&](const Real& xx){Eigen::MatrixXcd c=xx.cast<std::complex<double>>();c.diagonal().array()+=.5;for(int i=0;i<n;++i){c(i,n+i)+=std::complex<double>(0,.5);c(n+i,i)-=std::complex<double>(0,.5);}Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> e(c,Eigen::EigenvaluesOnly);need(e.info()==Eigen::Success&&e.eigenvalues().minCoeff()>=-1e-11,"Covariance fails numerical physicality check");return e.eigenvalues().minCoeff();};physical(x);
  Real y=a*x*a.transpose();stats[4]=std::max(stats[4],(y-y.transpose()).norm());y=((y+y.transpose())/2).eval();stats[3]=std::min(stats[3],physical(y));for(int i=0;i<2*n;++i)for(int j=0;j<2*n;++j)outputs[std::size_t(s)*4*n*n+std::size_t(i)*2*n+j]=y(i,j);
 }
 for(int i=0;i<m;++i)for(int j=0;j<n;++j){auto k=2*(std::size_t(i)*n+j);isometry[k]=v(i,j).real();isometry[k+1]=v(i,j).imag();}return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
