// Batched correlated classical displacement followed by direct click or photon-count detection.
#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>
using Gemm97=void(*)(const char*,const char*,const int*,const int*,const int*,const double*,const double*,
                      const int*,const double*,const int*,const double*,double*,const int*);
extern "C" __declspec(dllexport) int field97_draw(int m,int sectors,int rank,int n,int route,int workers,
        std::uint64_t seed,std::uint64_t offset,const double* factor,const double* mean,Gemm97 gemm,
        std::uint16_t* output,unsigned char* errors){
    if(m<1||sectors<1||rank<1||n<1||workers<1||workers>64||(route!=0&&route!=1)||!factor||!mean||
       !gemm||!output||!errors||offset>0x100000000ULL||std::uint64_t(n)>0x100000000ULL-offset)return 1;
    try{
        int dim=2*m*sectors;const char no='N';double one=1,zero=0;
        // Column-major BLAS maps a contiguous [record,rank] buffer to [record,all quadratures].
        for(int start=0;start<n;start+=256){
            int batch=std::min(n-start,256);std::vector<double> z(std::size_t(batch)*rank),field(std::size_t(batch)*dim);
            #pragma omp parallel for num_threads(workers) if(workers>1) schedule(static)
            for(int r=0;r<batch;++r){
                DomainRng84 rng(seed,0,153,offset+start+r);double* dst=z.data()+std::size_t(r)*rank;
                for(int j=0;j<rank;j+=2){
                    double radius=std::sqrt(-2.*std::log(1.-uniform84(rng)));
                    double angle=6.2831853071795864769*uniform84(rng);
                    dst[j]=radius*std::cos(angle);if(j+1<rank)dst[j+1]=radius*std::sin(angle);
                }
            }
            gemm(&no,&no,&dim,&batch,&rank,&one,factor,&dim,z.data(),&rank,&zero,field.data(),&dim);
            #pragma omp parallel for num_threads(workers) if(workers>1) schedule(static)
            for(int r=0;r<batch;++r){
                int record=start+r;DomainRng84 detect(seed,0,154,offset+record);errors[record]=0;
                auto row=field.data()+std::size_t(r)*dim;auto out=output+std::size_t(record)*m;
                for(int i=0;i<m;++i){
                    double lambda=0.;
                    for(int s=0;s<sectors;++s){
                        int j=2*s*m+i;double x=row[j]+mean[j],y=row[j+m]+mean[j+m];
                        lambda+=.5*(x*x+y*y);
                    }
                    if(!std::isfinite(lambda)){errors[record]=1;out[i]=0;continue;}
                    if(route==0){out[i]=uniform84(detect)<-std::expm1(-lambda);continue;}
                    // A sum of independent Poisson variables has the desired mean, including bright ports.
                    unsigned count=0;
                    while(lambda>0&&count<65536){
                        double part=std::min(32.,lambda),prob=std::exp(-part),u=uniform84(detect);unsigned k=0;
                        while(u>=prob&&k<256){u-=prob;prob*=part/(++k);}
                        if(k==256){errors[record]=2;break;}
                        count+=k;lambda-=part;
                    }
                    if(count>65535)errors[record]=3;
                    out[i]=std::uint16_t(std::min(65535u,count));
                }
            }
        }
        return 0;
    }catch(...){return 2;}
}
