#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>
#include "gaussian_rng_domain84_v1.hpp"
using Z=std::complex<double>;

struct FockRow { Z f[3]; };
struct Conditional {
    Z vacuum;
    double linear,total;
};
struct Site {
    Z c,a,h,b,overlap;
    double linear,one,two;
    Conditional apply(const FockRow& row)const {
        const Z f=row.f[0],v=row.f[1]*b;
        const Z z=f*c+row.f[1]*a+row.f[2]*h;
        double lin=std::norm(f)*linear+std::norm(v)*one+2*(std::conj(f)*v*overlap).real();
        lin=std::max(0.,lin);
        return {z,lin,std::norm(z)+lin+std::norm(f*b)*two};
    }
};
extern "C" int displaced115(int rows,int m,int r,int clicks,int threads,uint64_t seed,
        uint32_t offset,int spectrum,const Z* q,const Z* v,const Z* gv,const Z* diagonal,
        const double* one,const double* two,const Z* alpha,unsigned char* records,double* maxerr) {
    if(rows<1||m<1||r<1||threads<1||spectrum<0||spectrum>65535||
       uint64_t(offset)+rows>(uint64_t(1)<<32))return 1;
    constexpr double invsqrt2=.7071067811865475244;
    int failed=0;double error=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4) reduction(|:failed) reduction(max:error)
    for(int row=0;row<rows;++row) {
        DomainRng84 rng(seed,spectrum,clicks?307:308,offset+row);
        auto* dest=records+size_t(row)*m;std::vector<Z> u(r);
        Z c=1/std::sqrt(1+two[0]),b=c;
        double linear=0;
        for(int i=0;i<m;++i) {
            const Z* t=q+size_t(i)*r;const Z* g=gv+size_t(i)*r;
            Z a{},overlap{};
            for(int j=0;j<r;++j){a+=t[j]*u[j];overlap+=std::conj(u[j])*g[j];}
            Site s{c,a,b*diagonal[i]*invsqrt2,b,overlap,
                   std::max(0.,linear-std::norm(a)),one[i],two[i+1]};
            double total=s.linear+std::norm(c)+std::norm(a)+std::norm(s.h)+std::norm(b)*(s.one+s.two);
            error=std::max(error,std::abs(total-1));
            double target=uniform84(rng)*total;
            const Z field=alpha[size_t(row)*m+i],bar=std::conj(field);
            const double intensity=std::norm(field);
            Z coh=std::exp(-intensity/2);
            FockRow vacuum{{coh,-bar*coh,bar*bar*coh*invsqrt2}},chosen=vacuum;
            Conditional next=s.apply(chosen);
            int count=0;
            if(target>=next.total) {
                target-=next.total;
                if(clicks) {
                    const double last=1+std::sqrt(std::max(0.,1-std::exp(-intensity)*(1+intensity+.5*intensity*intensity)));
                    for(int k=0;k<3;++k) {
                        for(int j=0;j<3;++j)chosen.f[j]=(k==j?1.:0.)-std::conj(vacuum.f[k])*vacuum.f[j]/last;
                        next=s.apply(chosen);
                        if(target<next.total||k==2)break;
                        target-=next.total;
                    }
                    count=1;
                } else {
                    Z prev=0;
                    for(count=1;count<=255;++count) {
                        const double root=std::sqrt(double(count));
                        const Z current=coh*field/root;
                        chosen={{current,root*coh-bar*current,
                                 (std::sqrt(double(count*(count-1)))*prev-2.*bar*root*coh+bar*bar*current)*invsqrt2}};
                        next=s.apply(chosen);
                        if(target<next.total)break;
                        target-=next.total;prev=coh;coh=current;
                    }
                }
            }
            if(!(next.total>0)||count>255||!std::isfinite(total)){failed|=2;break;}
            dest[i]=static_cast<unsigned char>(count);
            const double scale=1/std::sqrt(next.total);
            Z left=chosen.f[0]*scale,right=chosen.f[1]*b*scale;
            for(int j=0;j<r;++j)u[j]=left*u[j]+right*v[size_t(i)*r+j];
            c=next.vacuum*scale;b*=left;linear=next.linear/next.total;
        }
    }
    *maxerr=error;return failed;
}
