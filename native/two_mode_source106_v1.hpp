#pragma once
#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

using C=std::complex<double>;
using U=std::uint64_t;

static unsigned alias105(U roll,int bits,const U* cut,const unsigned* alt){
    unsigned cell=unsigned(roll&((U(1)<<bits)-1));
    return roll>>bits<cut[cell]?cell:alt[cell];
}

// Direction stores normalized transmission amplitudes. The proposal correction
// accounts for rounding the port distribution to its integer alias table.
struct Direction105 {C u,v;double u2,uv_real,uv_imag,correction;};
static int route_source105(int m,int pnr,int n,int bits,const int* info,const double* initial,
    const U* cuts,const unsigned* alt,std::size_t first,const Direction105* dir,
    const double* roots,std::vector<C>& psi,DomainRng84& port,DomainRng84& accept,unsigned* record){
    if(info[2]==2){
        // Source emits the same number into each arm, with path
        // labels retained so that signal and idler cannot exchange.
        for(int k=0;k<n;++k){
            unsigned j;
            do{
                j=alias105(port(),bits,cuts+first,alt+first);
            }while(uniform84(accept)>=(k<n/2?dir[j].u2:std::norm(dir[j].v))*dir[j].correction);
            if(j<unsigned(m)){if(pnr)++record[j];else record[j]=1;}
        }
        return 0;
    }
    if(info[2]){
        std::fill_n(psi.begin(),n+1,C{});
        psi[n/2]=1.;
    }else{
        int k=n/2;
        const double* wave=initial+info[3]+std::size_t(k)*k;
        std::copy_n(wave,n+1,psi.begin());
    }
    while(n){
        double a=0,b=0;C cross=0;
        for(int k=0;k<=n;++k){
            double q=std::norm(psi[k]);
            a+=k*q;b+=(n-k)*q;
            if(k<n)cross+=roots[k+1]*roots[n-k]*psi[k+1]*std::conj(psi[k]);
        }
        double scale=a+b;
        if(!(scale>0.)||!std::isfinite(scale))return 2;
        unsigned j;
        double d;
        do{
            j=alias105(port(),bits,cuts+first,alt+first);
            auto &u=dir[j];
            d=b+(a-b)*u.u2+2*(u.uv_real*cross.real()-u.uv_imag*cross.imag());
        }while(uniform84(accept)*scale>=d*dir[j].correction);
        if(j<unsigned(m)){if(pnr)++record[j];else record[j]=1;}
        if(n>1){
            auto &u=dir[j];
            // Rescale by the measured norm after cancellation.
            double norm=0.;
            for(int k=0;k<n;++k){
                psi[k]=u.u*(roots[k+1]*psi[k+1])+u.v*(roots[n-k]*psi[k]);
                norm+=std::norm(psi[k]);
            }
            if(!(norm>0.)||!std::isfinite(norm))return 3;
            double inv=1./std::sqrt(norm);
            for(int k=0;k<n;++k)psi[k]*=inv;
        }
        --n;
    }
    return 0;
}
