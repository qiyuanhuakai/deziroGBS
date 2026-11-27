#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>
#include <random>
#include "gaussian_rng_domain84_v1.hpp"

using Z=std::complex<double>;
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

// Contract a photon in the symmetric N-particle Fock basis, recording one local POVM word.
API int sparse_emit112(int rows,int cells,int r,int cap,int alphabet,int threads,uint64_t seed,uint32_t offset,
        const double* number_cdf,const int* dims,const int* starts,const int* parent,const double* factor,
        const double* initial,const Z* prefix,const Z* eigfactor,const int* words,int* locations,int* packets,int* totals){
    if(rows<1 || cells<1 || cap>cells || cap<1 || r<1 || alphabet<2 || threads<1 ||
       uint64_t(offset)+rows>(uint64_t(1)<<32))return 1;
    int bad=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1) reduction(|:bad)
    for(int row=0;row<rows;++row){
        DomainRng84 number(seed,0,20,offset+row), site(seed,0,21,offset+row), word(seed,0,22,offset+row), mark(seed,0,23,offset+row);
        int total=std::upper_bound(number_cdf,number_cdf+cap+1,uniform84(number))-number_cdf;
        totals[row]=total;if(!total)continue;
        std::vector<Z> psi(initial+starts[total],initial+starts[total]+dims[total]);
        int* dest=locations+size_t(row)*cap;
        for(int step=0;step<total;++step){
            const int n=total-step,dim=dims[n-1];
            int loc;
            do {loc=std::uniform_int_distribution<int>(0,cells-1)(site);}
            while(std::find(dest,dest+step,loc)!=dest+step);
            dest[step]=loc;
            std::vector<Z> a(size_t(dim)*r), gram(size_t(r)*r);
            for(int j=0;j<dim;++j)for(int i=0;i<r;++i){
                size_t at=size_t(starts[n-1]+j)*r+i;
                a[size_t(j)*r+i]=factor[at]*psi[parent[at]];
            }
            for(int j=0;j<dim;++j){
                const Z* v=&a[size_t(j)*r];
                for(int i=0;i<r;++i)for(int k=0;k<=i;++k)gram[i*r+k]+=std::conj(v[i])*v[k];
            }
            double norm=0;
            for(int i=0;i<r;++i)norm+=gram[i*r+i].real();
            if(!std::isfinite(norm)||norm<=0){bad=2;break;}
            auto contraction=[&](const Z* e){
                double value=0;
                for(int i=0;i<r;++i){
                    value+=(e[i*r+i]*gram[i*r+i]).real();
                    for(int k=0;k<i;++k)value+=2*(e[k*r+i]*gram[i*r+k]).real();
                }
                return value;
            };
            double threshold=norm*uniform84(word);
            int lo=0,hi=alphabet-1;
            while(lo<hi){
                int mid=(lo+hi)/2;
                if(contraction(prefix+size_t(mid)*r*r)<=threshold)lo=mid+1;else hi=mid;
            }
            packets[size_t(row)*cap+step]=words[lo];
            if(n==1)break;
            const Z* basis=eigfactor+size_t(lo)*r*r;
            std::vector<double> cdf(r);
            for(int k=0;k<r;++k){
                double mass=0;
                for(int i=0;i<r;++i){
                    Z bi=basis[i*r+k];
                    mass+=std::norm(bi)*gram[i*r+i].real();
                    for(int h=0;h<i;++h)mass+=2*(std::conj(bi)*gram[i*r+h]*basis[h*r+k]).real();
                }
                cdf[k]=std::max(0.,mass)+(k?cdf[k-1]:0.);
            }
            if(cdf.back()<=0){bad=3;break;}
            int k=std::upper_bound(cdf.begin(),cdf.end(),uniform84(mark)*cdf.back())-cdf.begin();
            if(k==r){bad=4;break;}
            psi.assign(dim,Z{});
            double inv=1/std::sqrt(cdf[k]-(k?cdf[k-1]:0));
            for(int j=0;j<dim;++j)for(int i=0;i<r;++i)psi[j]+=a[size_t(j)*r+i]*(basis[i*r+k]*inv);
        }
    }
    return bad;
}
