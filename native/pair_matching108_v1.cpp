#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>
#include <omp.h>

using C108=std::complex<double>;
using R108=DomainRng84;
static double open108(R108& r){return (double(r()>>12)+.5)*0x1p-52;}
static int sample108(const double* cdf,int size,R108& rng){
    return int(std::upper_bound(cdf,cdf+size,open108(rng))-cdf);
}
struct Wick108{C108 amplitude;double proposal;};
static Wick108 matching108(unsigned mask,const C108* matrix,int dim){
    if(!mask)return {1.,1.};
    int i=std::countr_zero(mask);unsigned rest=mask&(mask-1);
    Wick108 sum{0.,0.};
    for(auto c=rest;c;c&=c-1){
        int j=std::countr_zero(c);auto x=matrix[i*dim+j];auto sub=matching108(rest^(1u<<j),matrix,dim);
        sum.amplitude+=x*sub.amplitude;sum.proposal+=std::norm(x)*sub.proposal;
    }
    return sum;
}
struct Circuit108{
    int m,k,rows;const C108* u;const double *lev,*leverage_cdf,*columns;
    C108 amplitude(int i,int j,const double* lambda)const{
        const C108* a=u+std::size_t(i)*2*k;const C108* b=u+std::size_t(j)*2*k;
        C108 result=0.;
        for(int l=0;l<k;++l)result+=lambda[l]*(a[2*l]*b[2*l+1]+a[2*l+1]*b[2*l]);
        return result;
    }
    bool paired(const double* lambda,const double* first,R108& rng,int& i,int& j,std::vector<C108>& work)const{
        i=sample108(first,rows,rng);if(i>=rows)return false;
        const C108* a=u+std::size_t(i)*2*k;double weight=0.;
        for(int l=0;l<k;++l){
            work[2*l]=lambda[l]*a[2*l+1];work[2*l+1]=lambda[l]*a[2*l];
            weight+=std::norm(work[2*l])+std::norm(work[2*l+1]);
        }
        if(!(weight>0))return false;
        for(unsigned rep=0;rep<unsigned(80*k+256);++rep){
            j=sample108(leverage_cdf,rows,rng);if(j>=rows)return false;
            const C108* b=u+std::size_t(j)*2*k;C108 amp=0.;
            for(int l=0;l<2*k;++l)amp+=b[l]*work[l];
            if(open108(rng)*weight*lev[j]<std::norm(amp))return true;
        }
        std::vector<double> cumulative(rows);
        double mass=0.;
        for(int p=0;p<rows;++p){mass+=std::norm(amplitude(i,p,lambda));cumulative[p]=mass;}
        if(!(mass>0))return false;
        double z=std::min(std::nextafter(mass,0.),open108(rng)*mass);
        j=std::upper_bound(cumulative.begin(),cumulative.end(),z)-cumulative.begin();return j<rows;
    }
};
template<class T>
static int sample_row108(const Circuit108& c,int spectra,int cap,int max_tries,bool pnr,
    std::uint64_t seed,std::uint32_t row,const double* lambdas,const double* hazard,const double* first,T* out,
    std::uint64_t* tally,std::vector<int>* sparse=nullptr,const double* single_pair=nullptr,int sector_offset=0,bool known_active=false,bool initialize=true,const double* cached=nullptr,const int* cache_index=nullptr){
    if(!sparse&&initialize)std::fill_n(out,c.m,T(0));
    auto emit=[&](int j){
        if(j>=c.m)return 0;
        if(sparse){sparse->push_back(j);return 0;}
        if(!pnr){out[j]=1;return 0;}
        if(out[j]==std::numeric_limits<T>::max())return 1;
        ++out[j];return 0;
    };
    std::vector<C108> work;std::vector<std::pair<int,int>> sources;
    for(int s=0;s<spectra;++s){
        const auto* h=hazard+s*(c.k+1);const auto* lambda=lambdas+s*c.k;
        int sector=s+sector_offset;
        R108 source(seed,sector,320,row);int active=0,total=0;sources.clear();
        while(active<c.k){
            double z=(active==0 && known_active)?-std::log1p(-open108(source)*(-std::expm1(-h[c.k]))):h[active]-std::log(open108(source));
            int j=std::upper_bound(h+active+1,h+c.k+1,z)-h-1;if(j>=c.k)break;
            int n=1+int(std::floor(std::log(open108(source))/(2*std::log(std::abs(lambda[j])))));
            if(n<1||total>10000000-n)return 2;
            total+=n;sources.emplace_back(j,n);active=j+1;
        }
        auto* st=tally+std::size_t(s)*(cap+2)*3+3*std::min(total,cap+1);
        ++st[0];
        if(!total)continue;
        if(total==1 && single_pair){
            const double* table=single_pair+std::size_t(s)*(2*c.m+2);
            R108 choose(seed,sector,325,row);
            double pick=open108(choose);
            ++st[1];
            if(pick<table[0])continue;
            if(pick<table[1]){
                int j=sample108(table+2,c.m,choose);
                if(j==c.m)return 2;
                if(int rc=emit(j))return rc;
                continue;
            }
            int i=sample108(table+2+c.m,c.m,choose);
            if(i==c.m)return 2;
            if(int rc=emit(i))return rc;
            if(cache_index && cache_index[i]>=0){
                int j=sample108(cached+std::size_t(cache_index[i])*c.m,c.m,choose);
                if(j>=c.m)return 2;
                if(int rc=emit(j))return rc;
                continue;
            }
            const C108* firstrow=c.u+std::size_t(i)*2*c.k;
            work.resize(2*c.k);double norm=0.;
            for(int l=0;l<c.k;++l){
                work[2*l]=lambda[l]*firstrow[2*l+1];work[2*l+1]=lambda[l]*firstrow[2*l];
                norm+=std::norm(work[2*l])+std::norm(work[2*l+1]);
            }
            if(!(norm>0.))return 2;
            int dest=-1;
            for(unsigned rep=0;rep<unsigned(80*c.k+256);++rep){
                double w=open108(choose)*c.leverage_cdf[c.m-1];
                int j=std::upper_bound(c.leverage_cdf,c.leverage_cdf+c.m,w)-c.leverage_cdf;
                if(j>=c.m)continue;
                C108 amp=0.;auto* b=c.u+std::size_t(j)*2*c.k;
                for(int t=0;t<2*c.k;++t)amp+=work[t]*b[t];
                if(open108(choose)*norm*c.lev[j]<std::norm(amp)){dest=j;break;}
            }
            if(dest<0){
                std::vector<double> cumulative(c.m);double sum=0.;
                for(int j=0;j<c.m;++j){sum+=std::norm(c.amplitude(i,j,lambda));cumulative[j]=sum;}
                if(!(sum>0.))return 2;
                dest=std::upper_bound(cumulative.begin(),cumulative.end(),std::min(std::nextafter(sum,0.),open108(choose)*sum))-cumulative.begin();
            }
            if(int rc=emit(dest))return rc;
            continue;
        }
        const int dim=2*total;
        if(total<=cap){
            R108 proposal(seed,sector,321,row);R108 test(seed,sector,322,row);
            work.resize(2*c.k);
            std::array<int,12> sites{};std::array<C108,144> mat{};
            double matches=1.;for(int l=1;l<total;++l)matches*=2*l+1;
            bool accepted=false;
            for(int rep=0;rep<max_tries&&!accepted;++rep){
                ++st[1];
                for(int l=0;l<total;++l)
                    if(!c.paired(lambda,first+std::size_t(s)*c.rows,proposal,sites[2*l],sites[2*l+1],work))return 2;
                if(total==1)accepted=true;
                else{
                    double scale=0.;
                    for(int i=0;i<dim;++i)for(int j=i+1;j<dim;++j){
                        auto a=c.amplitude(sites[i],sites[j],lambda);mat[i*dim+j]=a;scale=std::max(scale,std::abs(a));
                    }
                    if(!(scale>0.))return 2;
                    for(int i=0;i<dim;++i)for(int j=i+1;j<dim;++j)mat[i*dim+j]/=scale;
                    auto p=matching108((1u<<dim)-1,mat.data(),dim);
                    if(!(p.proposal>0.))return 2;
                    double a=std::norm(p.amplitude)/(matches*p.proposal);
                    if(a>1.+1e-9||!std::isfinite(a))return 2;
                    accepted=open108(test)<a;
                }
            }
            if(accepted){
                for(int i=0;i<dim;++i)if(int rc=emit(sites[i]))return rc;
                continue;
            }
        }
        ++st[2];R108 fallback(seed,sector,323,row);
        for(auto [j,n]:sources)for(int l=0;l<n;++l)for(int arm=0;arm<2;++arm){
            int dest=sample108(c.columns+std::size_t(j*2+arm)*c.rows,c.rows,fallback);
            if(dest>=c.rows)return 2;
            if(int rc=emit(dest))return rc;
        }
    }
    return 0;
}
extern "C" __declspec(dllexport) int matching108_draw(int m,int k,int spectra,int rows,int n,int cap,int tries,int pnr,int wide,int workers,
    std::uint64_t seed,std::uint32_t offset,const C108* u,const double* lambda,const double* hazard,const double* first,
    const double* lev,const double* leverage_cdf,const double* column,void* x,std::uint64_t* tally){
    if(k<1||spectra<1||spectra>65536||cap<1||cap>6||tries<1||workers<1||rows<m)return 2;
    Circuit108 c{m,k,rows,u,lev,leverage_cdf,column};
    int error=0;int size=spectra*(cap+2)*3;
    #pragma omp parallel num_threads(workers) reduction(max:error)
    {
        std::vector<std::uint64_t> counts(size);
        #pragma omp for schedule(static)
        for(int j=0;j<n;++j){
            int rc;
            if(wide)rc=sample_row108(c,spectra,cap,tries,pnr,seed,offset+j,lambda,hazard,first,
                        static_cast<std::uint32_t*>(x)+std::size_t(m)*j,counts.data());
            else rc=sample_row108(c,spectra,cap,tries,pnr,seed,offset+j,lambda,hazard,first,
                        static_cast<unsigned char*>(x)+std::size_t(m)*j,counts.data());
            error=std::max(error,rc);
        }
        #pragma omp critical
        for(int j=0;j<size;++j)tally[j]+=counts[j];
    }
    return error;
}
