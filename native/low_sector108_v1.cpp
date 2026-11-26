#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>

struct Low108 {
    int m,h,z,d,s;
    const int *order,*ranges;
    const double *row,*v,*stats,*prefix,*first,*mix;
    using RNG=DomainRng84;
    int select(const double* p,int lo,int hi,RNG& rng) const {
        if(hi<=lo+1)return lo;
        double a=p[lo],b=p[hi];
        if(!(b>a))return lo+int(uniform84(rng)*(hi-lo));
        double cut=std::min(std::nextafter(b,a),a+(b-a)*uniform84(rng));
        return int(std::upper_bound(p+lo+1,p+hi+1,cut)-p)-1;
    }
    int one(int group,int law,RNG& rng) const {
        return select(prefix+std::size_t(law)*(m+1),ranges[2*group],ranges[2*group+1],rng);
    }
    double pair_mass(int i,int j) const {
        const double* a=v+std::size_t(i)*s*2*d;
        const double* b=row+std::size_t(j)*2*d;
        double corr=0.;
        for(int u=0;u<2*s;++u){
            double x=0.,y=0.;
            for(int k=0;k<d;++k){x+=a[u*d+k]*b[k];y+=a[u*d+k]*b[d+k];}
            corr+=x*x+y*y;
        }
        return stats[i]*stats[j]+.5*corr;
    }
    void pair(int a,int b,bool distinct,RNG& rng,int& i,int& j) const {
        int lo=ranges[2*b],hi=ranges[2*b+1];
        const double* masses=first+(std::size_t(distinct)*(z+1)+b)*(m+1);
        i=select(masses,ranges[2*a],ranges[2*a+1],rng);
        if(hi==lo+1){j=lo;return;}
        const double* leverage=stats+m;
        double bound=stats[2*m+i];
        const double* lev_cdf=prefix+m+1;
        double p=stats[i]*(prefix[hi]-prefix[lo]),q=bound*(lev_cdf[hi]-lev_cdf[lo]);
        for(int k=0;k<24;++k){
            j=select(uniform84(rng)*(p+q)<p?prefix:lev_cdf,lo,hi,rng);
            if(distinct&&i==j)continue;
            double envelope=stats[i]*stats[j]+bound*leverage[j];
            if(envelope>0. && uniform84(rng)*envelope<pair_mass(i,j))return;
        }
        // Exact inversion after the bounded rejection path, also for a
        // nearly rank-one support concentrated on the excluded diagonal.
        std::vector<double> cum(hi-lo+1);
        for(int k=lo;k<hi;++k)cum[k-lo+1]=cum[k-lo]+(distinct&&k==i?0.:pair_mass(i,k));
        j=lo+select(cum.data(),0,hi-lo,rng);
    }
    template<class T>
    int run(T* x,bool pnr,bool sparse,int label,const int* shape,RNG& rng) const {
        auto emit=[&](int j,unsigned count){
            if(j<0||j>=m)return 2;
            T& dest=x[order[j]];
            if(pnr){
                if(count>std::numeric_limits<T>::max()-dest)return 1;
                dest+=T(count);
            }else dest=1;
            return 0;
        };
        if(pnr){
            unsigned a=0,b=0,c=0;
            if(sparse){c=label%shape[2];label/=shape[2];b=label%shape[1];a=label/shape[1];}
            else {
                a=x[order[0]];b=x[order[1]];if(a+b>2)return 0;
                for(int k=2;k<m;++k){c+=x[order[k]];if(a+b+c>2)return 0;}
            }
            if(!sparse&&c==0)return 0;
            if(!sparse)std::fill_n(x,m,T(0));
            if(a+b+c<=2){
                int sites[2],n=0;
                for(unsigned k=0;k<a;++k)sites[n++]=0;
                for(unsigned k=0;k<b;++k)sites[n++]=1;
                for(unsigned k=0;k<c;++k)sites[n++]=z;
                if(n==0)return 0;
                if(n==1)return emit(one(sites[0],0,rng),1);
                int i,j;pair(sites[0],sites[1],false,rng,i,j);
                return std::max(emit(i,1),emit(j,1));
            }
            int status=std::max(emit(0,a),emit(1,b));
            for(unsigned k=0;k<c&&!status;++k)status=emit(one(z,1,rng),1);
            return status;
        }
        int occupied=0,only=-1;
        if(!sparse){
            label=0;
            for(int g=0;g<z;++g){
                for(int k=ranges[2*g];k<ranges[2*g+1];++k)if(x[order[k]]){
                    label|=1<<g;only=k;++occupied;
                }
            }
        }
        if(!label)return 0;
        double u=uniform84(rng);const double *frac=mix+2*label;
        int a=std::countr_zero(unsigned(label)),b=std::bit_width(unsigned(label))-1;
        if(u<frac[0]){
            if(!sparse)std::fill_n(x,m,T(0));
            return emit(one(a,2,rng),1);
        }
        if(u<frac[0]+frac[1]){
            if(!sparse)std::fill_n(x,m,T(0));
            int i,j;pair(a,b,true,rng,i,j);
            if(i==j)return 2;
            emit(i,1);return emit(j,1);
        }
        if(sparse){
            for(int g=0;g<z;++g)if(label&(1<<g)){only=one(g,1,rng);emit(only,1);++occupied;}
        }
        if(occupied==1&&ranges[2*a+1]-ranges[2*a]>1){
            // The remaining optical component contains at least two clicks.
            int n=ranges[2*a+1]-ranges[2*a]-1;
            int j=ranges[2*a]+int(uniform84(rng)*n);
            return emit(j+(j>=only),1);
        }
        return 0;
    }
};
extern "C" __declspec(dllexport) int low108_draw(int m,int n,int h,int z,int d,int spectra,int pnr,int sparse,int bytes,int workers,
    int count_states,int click_states,std::uint64_t seed,std::uint32_t offset,const int* order,const int* ranges,
    const double* row,const double* v,const double* stats,const double* prefix,const double* first,const double* mix,
    const double* cdf,const int* shape,void* buffer){
    if(m<2||n<1||workers<1||z<h||z>20||(bytes!=1&&bytes!=4))return 2;
    Low108 sampler{m,h,z,d,spectra,order,ranges,row,v,stats,prefix,first,mix};
    int error=0;
    #pragma omp parallel for num_threads(workers) reduction(max:error) schedule(static)
    for(int k=0;k<n;++k){
        DomainRng84 rng(seed,0,310+pnr+2*sparse,offset+k);
        int state=0;
        if(sparse){
            int len=pnr?count_states:click_states;
            state=std::upper_bound(cdf,cdf+len,uniform84(rng))-cdf;
            if(state==len){error=2;continue;}
        }
        auto start=std::size_t(k)*m;
        int rc=bytes==1?sampler.run(static_cast<unsigned char*>(buffer)+start,pnr,sparse,state,shape,rng)
                       :sampler.run(static_cast<std::uint32_t*>(buffer)+start,pnr,sparse,state,shape,rng);
        error=std::max(error,rc);
    }
    return error;
}
