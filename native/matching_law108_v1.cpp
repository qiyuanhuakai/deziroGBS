#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <cmath>
#include <vector>
using R=long double;
using Z=std::complex<R>;
using M=Eigen::Matrix<Z,Eigen::Dynamic,Eigen::Dynamic>;

extern "C" __declspec(dllexport) int matching_law108(int h,int k,int spectra,int cap,int workers,const double* raw,
    const double* lambda,const double* keep,double* out){
    if(h<1||h>18||k<1||k>128||(k>12&&h>8)||spectra<1||cap<1||cap>6||workers<1)return 1;
    int d=2*k,states=1<<h,laws=cap+2;
    const auto* tr=reinterpret_cast<const std::complex<double>*>(raw);
    std::vector<M> sub(h,M(d,d));
    for(int i=0;i<h;++i)for(int j=0;j<d;++j)for(int l=0;l<d;++l)
        sub[i](j,l)=std::conj(Z(tr[i*d+j]))*Z(tr[i*d+l]);
    std::vector<R> values(std::size_t(laws)*states,1);
    std::vector<M> pair(spectra,M::Zero(d,d));std::vector<R> vacuum(spectra,1);
    for(int s=0;s<spectra;++s)for(int j=0;j<k;++j){
        R l=lambda[s*k+j];pair[s](2*j,2*j+1)=pair[s](2*j+1,2*j)=l;vacuum[s]*=1-l*l;
    }
    int error=0;
    #pragma omp parallel for num_threads(workers) schedule(static) reduction(max:error)
    for(int bits=0;bits<states;++bits){
        M gram=M::Identity(d,d);
        for(unsigned w=bits;w;w&=w-1)gram-=sub[std::countr_zero(w)];
        int dest=states-1-bits;
        for(int s=0;s<spectra;++s){
            M x=pair[s]*gram.transpose()*pair[s].adjoint()*gram;
            auto det=(M::Identity(d,d)-x).determinant();
            if(!(det.real()>0.)||std::abs(det.imag())>1e-12L){error=2;continue;}
            R exact=vacuum[s]/std::sqrt(det.real());values[dest]*=exact;
            std::array<R,7> quantum{},ind{},trace{};quantum[0]=ind[0]=1;
            M power=x;
            for(int n=1;n<=cap;++n){
                trace[n]=power.trace().real();
                for(int j=1;j<=n;++j)quantum[n]+=trace[j]*quantum[n-j];
                quantum[n]/=2*n;
                if(n<cap)power=(power*x).eval();
            }
            R independent=vacuum[s];
            for(int j=0;j<k;++j){
                R l=lambda[s*k+j],f=l*l*gram(2*j,2*j).real()*gram(2*j+1,2*j+1).real();
                independent/=1-f;
                for(int n=1;n<=cap;++n)ind[n]+=f*ind[n-1];
            }
            values[states+dest]*=independent;
            for(int n=1;n<=cap;++n){
                independent+=keep[s*(cap+1)+n]*vacuum[s]*(quantum[n]-ind[n]);
                values[(n+1)*states+dest]*=independent;
            }
        }
    }
    if(error)return error;
    for(int l=0;l<laws;++l){
        R* p=values.data()+l*states;
        for(int bit=1;bit<states;bit*=2)for(int j=0;j<states;++j)if(j&bit)p[j]-=p[j^bit];
        for(int j=0;j<states;++j)out[l*states+j]=double(p[j]);
    }
    return 0;
}
