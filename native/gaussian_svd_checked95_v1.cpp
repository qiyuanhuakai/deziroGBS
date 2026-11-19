// LAPACKE_zgesdd adapter for the thin, column-major decomposition in the PNR sweep.
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <mutex>
using Matrix95=Eigen::MatrixXcd;
using Complex95=std::complex<double>;
using Lapack95=int(*)(int,char,int,int,Complex95*,int,double*,Complex95*,int,Complex95*,int);
static Lapack95 primary95=nullptr;
static std::mutex audit95;
static int attempts95=0,recovered95=0,last95=0;
static double worst95=0;
extern "C" __declspec(dllexport) int svd95_setup(void* fn) {
    if(!fn) return 1;
    primary95=reinterpret_cast<Lapack95>(fn);return 0;
}
extern "C" __declspec(dllexport) void svd95_report(int* counts,double* residual) {
    std::lock_guard<std::mutex> lock(audit95);
    counts[0]=attempts95;counts[1]=recovered95;counts[2]=last95;*residual=worst95;
}
extern "C" __declspec(dllexport) int svd95_checked(int layout,char job,int m,int n,Complex95* a,int lda,
        double* values,Complex95* u,int ldu,Complex95* vt,int ldvt) {
    int k=std::min(m,n);
    if(!primary95||layout!=102||job!='S'||m<1||n<1||lda!=m||ldu!=m||ldvt!=k||
       !a||!values||!u||!vt) return -1;
    try {
        Matrix95 original=Eigen::Map<const Matrix95>(a,m,n);
        int info=primary95(layout,job,m,n,a,lda,values,u,ldu,vt,ldvt);
        if(info==0) return 0;
        std::lock_guard<std::mutex> lock(audit95);
        ++attempts95;last95=info;
        if(info<0||!original.allFinite()) return info;
        Eigen::JacobiSVD<Matrix95> alternative(original,Eigen::ComputeThinU|Eigen::ComputeThinV);
        if(alternative.info()!=Eigen::Success) return info;
        const auto& left=alternative.matrixU();const auto& right=alternative.matrixV();
        const auto& spectrum=alternative.singularValues();
        if(!left.allFinite()||!right.allFinite()||!spectrum.allFinite()||spectrum.minCoeff()<0) return info;
        Matrix95 identity=Matrix95::Identity(k,k);
        double scale=original.norm();
        if(!std::isfinite(scale)||scale<=0) return info;
        double residual=(original-left*spectrum.asDiagonal()*right.adjoint()).norm()/scale;
        double error=std::max({residual,(left.adjoint()*left-identity).norm(),
                               (right.adjoint()*right-identity).norm()});
        if(!std::isfinite(error)||error>1e-10) return info;
        Eigen::Map<Matrix95>(u,m,k)=left;
        Eigen::Map<Matrix95>(vt,k,n)=right.adjoint();
        Eigen::Map<Eigen::VectorXd>(values,k)=spectrum;
        ++recovered95;worst95=std::max(worst95,error);
        return 0;
    } catch(const std::exception&) {return -1;}
}
