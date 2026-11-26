#include <Eigen/Dense>
#include <complex>
#include <array>
#include <bit>
#include <cmath>
#include <omp.h>
using Z107=std::complex<double>;

// Wick expectation of the normally ordered generating function after its
// vacuum projection. The small repeated matrix contains at most ten rows.
static Z107 wick107(unsigned mask,const Z107* h,const int* ids,int stride){
    if(!mask)return Z107(1);
    int first=std::countr_zero(mask);unsigned rest=mask&(mask-1);
    if(!(rest&(rest-1)))return h[ids[first]*stride+ids[std::countr_zero(rest)]];
    Z107 sum{};
    for(unsigned choices=rest;choices;choices&=choices-1){
        int j=std::countr_zero(choices);
        sum+=h[ids[first]*stride+ids[j]]*wick107(rest^(1u<<j),h,ids,stride);
    }
    return sum;
}
extern "C" __declspec(dllexport) int patterns107(int m,int states,int workers,const double* raw,
    const unsigned long long* codes,double* output){
    if(m<1||m>16||states<1||workers<1)return 1;
    using Matrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
    int dim=2*m;Eigen::Map<const Matrix> q(raw,dim,dim);
    Matrix k=Matrix::Identity(dim,dim)-q.inverse();
    double p0=1/std::sqrt(q.determinant());if(!(p0>0))return 2;
    Eigen::Matrix<Z107,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> h(dim,dim);
    for(int i=0;i<m;++i)for(int j=0;j<m;++j){
        auto an=Z107(k(i,j)-k(i+m,j+m),k(i,j+m)+k(i+m,j))*.5;
        auto n=Z107(k(i,j)+k(i+m,j+m),k(i+m,j)-k(i,j+m))*.5;
        h(i,j)=an;h(i+m,j+m)=std::conj(an);h(i,m+j)=h(m+j,i)=n;
    }
    int error=0;
    #pragma omp parallel for num_threads(workers) reduction(max:error) schedule(static)
    for(int x=0;x<states;++x){
        std::array<int,10> ids{};int ct=0;double denominator=1;
        for(int j=0;j<m;++j){
            unsigned a=(codes[x]>>(4*j))&15u;
            for(unsigned b=1;b<=a;++b){
                if(ct>=5){error=3;break;}
                ids[ct++]=j;denominator*=b;
            }
        }
        if(error>0)continue;
        for(int j=0;j<ct;++j)ids[ct+j]=m+ids[j];
        auto value=wick107((1u<<(2*ct))-1,h.data(),ids.data(),dim)*p0/denominator;
        if(std::abs(value.imag())>1e-10||value.real() < -1e-12)error=4;
        output[x]=std::max(0.,value.real());
    }
    return error;
}
