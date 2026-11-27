#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <quadmath.h>
#include "gaussian_rng_domain84_v1.hpp"

using Q=__float128;
constexpr int limit=12;

// K restricted to a detector prefix equals L R L^T. Update the source-space
// matrix R when one two-quadrature detector is appended to the vacuum prefix.
extern "C" int prefix115(int m,int d,const double* basis,const double* lambda,
                         double* history,double* logvac) {
    if(m<1||d<1)return 1;
    size_t stride=size_t(d)*d;
    std::fill_n(history,stride,0.);
    for(int j=0;j<d;++j)history[j*d+j]=lambda[j];
    logvac[0]=0;
    std::vector<double> v(2*d),w(2*d);
    for(int site=0;site<m;++site) {
        const double* u=basis+size_t(site)*2*d;
        const double* prev=history+size_t(site)*stride;
        double* next=history+size_t(site+1)*stride;
        for(int a=0;a<2;++a)for(int i=0;i<d;++i) {
            double z=0;for(int j=0;j<d;++j)z+=prev[i*d+j]*u[a*d+j];v[a*d+i]=z;
        }
        double xx=0,xp=0,pp=0;
        for(int i=0;i<d;++i){xx+=u[i]*v[i];xp+=u[i]*v[d+i];pp+=u[d+i]*v[d+i];}
        double delta=xx+pp+xx*pp-xp*xp,det=1+delta;
        if(!(det>0))return 2;
        logvac[site+1]=logvac[site]-.5*std::log1p(delta);
        for(int i=0;i<d;++i){
            w[i]=((1+pp)*v[i]-xp*v[d+i])/det;
            w[d+i]=((1+xx)*v[d+i]-xp*v[i])/det;
        }
        for(int i=0;i<d;++i)for(int j=0;j<d;++j)
            next[i*d+j]=prev[i*d+j]-v[i]*w[j]-v[d+i]*w[d+j];
    }
    return 0;
}

struct Gap {
    int m,d,spectra;
    const double *basis,*history,*vac;
    static constexpr int ld=2*limit;
    // Evaluate the small determinant on selected quadratures. Quad precision
    // preserves the alternating vacuum sum on the tested sparse-click panels.
    static Q inverse_root(const double* kernel,int mask) {
        int sel[ld],p=0;
        for(unsigned v=mask;v;v&=v-1){int i=__builtin_ctz(v);sel[p++]=2*i;sel[p++]=2*i+1;}
        Q l[ld*ld],diag[ld],det=1;
        for(int i=0;i<p;++i) {
            for(int j=0;j<i;++j) {
                Q z=-Q(kernel[sel[i]*ld+sel[j]]);
                for(int h=0;h<j;++h)z-=l[i*ld+h]*l[j*ld+h]*diag[h];
                l[i*ld+j]=z/diag[j];
            }
            Q z=1-Q(kernel[sel[i]*ld+sel[i]]);
            for(int h=0;h<i;++h)z-=l[i*ld+h]*l[i*ld+h]*diag[h];
            if(z<=0)return -1;
            diag[i]=z;det*=z;
        }
        return 1/sqrtq(det);
    }
    double probability(const std::vector<int>& clicked,int end)const {
        const int k=clicked.size();
        if(!k)return vac[end];
        std::vector<double> kernel(size_t(spectra)*ld*ld),vec(d);
        double singleton_log=0;
        for(int s=0;s<spectra;++s) {
            const double* matrix=history+((size_t(s)*(m+1)+end)*d)*d;
            double* a=kernel.data()+s*ld*ld;
            for(int i=0;i<2*k;++i) {
                const double* v=basis+(size_t(clicked[i/2])*2+i%2)*d;
                for(int h=0;h<d;++h) {
                    double z=0;for(int j=0;j<d;++j)z+=matrix[h*d+j]*v[j];vec[h]=z;
                }
                for(int j=0;j<=i;++j) {
                    const double* w=basis+(size_t(clicked[j/2])*2+j%2)*d;
                    double z=0;for(int h=0;h<d;++h)z+=vec[h]*w[h];
                    a[i*ld+j]=a[j*ld+i]=z;
                }
            }
            if(k==1) {
                double aa=a[0],bb=a[ld+1],cc=a[1];
                singleton_log-=.5*std::log1p(-aa-bb+aa*bb-cc*cc);
            }
        }
        if(k==1)return vac[end]*std::expm1(singleton_log);
        Q sum=0;
        for(int mask=0;mask<1<<k;++mask) {
            Q v=1;
            for(int s=0;s<spectra;++s)v*=inverse_root(kernel.data()+s*ld*ld,mask);
            if((k-__builtin_popcount((unsigned)mask))%2)sum-=v;else sum+=v;
        }
        return vac[end]*double(sum);
    }
};

extern "C" int gap115(int rows,int m,int d,int spectra,int stop,int threads,uint64_t seed,uint32_t offset,
                      const double* basis,const double* history,const double* vac,unsigned char* out,
                      uint64_t* hits) {
    if(m<1||d<1||rows<1||spectra<1||stop<1||stop>limit||threads<1||
       uint64_t(offset)+rows>(uint64_t(1)<<32))return 1;
    Gap model{m,d,spectra,basis,history,vac};
    int failed=0;uint64_t capped=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4) reduction(|:failed) reduction(+:capped)
    for(int row=0;row<rows;++row) {
        auto* record=out+size_t(row)*m;std::fill_n(record,m,0);
        DomainRng84 rng(seed,0,201,offset+row);
        std::vector<int> clicked;
        int after=0;double denominator=1;
        while(after<m) {
            double target=denominator*uniform84(rng);
            double last=model.probability(clicked,m);
            if(!std::isfinite(last)||last < -1e-9*denominator||last>denominator*(1+1e-9)){failed|=2;break;}
            if(target<=last)break;
            int lo=after,hi=m;double left=denominator,right=last;
            while(hi-lo>1) {
                int mid=(hi+lo)/2;double value=model.probability(clicked,mid);
                if(!std::isfinite(value)||value<right-1e-9*denominator||value>left+1e-9*denominator) {
                    failed|=4;break;
                }
                if(value>=target){lo=mid;left=value;}else{hi=mid;right=value;}
            }
            if(hi-lo>1)break;
            clicked.push_back(hi-1);record[hi-1]=1;
            if(int(clicked.size())==stop){++capped;break;}
            after=hi;denominator=left-right;
        }
    }
    *hits=capped;return failed;
}

extern "C" int gapcheck115(int m,int d,int spectra,int k,int end,const int* sites,const double* basis,
                           const double* history,const double* vac,double* p) {
    if(k<0||k>limit||end<0||end>m)return 1;
    std::vector<int> v(sites,sites+k);
    for(int i=0;i<k;++i)if(v[i]<0||v[i]>=end||(i&&v[i]<=v[i-1]))return 2;
    *p=Gap{m,d,spectra,basis,history,vac}.probability(v,end);
    return 0;
}
