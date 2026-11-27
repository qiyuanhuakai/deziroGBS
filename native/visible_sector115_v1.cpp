#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>
#include "gaussian_rng_domain84_v1.hpp"

using Z=std::complex<double>;

// Match the labeled single photons first, then pair the remaining output positions.
struct Partial {
    int n,k,l;
    const std::vector<Z>& pair;
    const std::vector<Z>& single;
    std::vector<Z> sum;
    std::vector<double> squares;
    std::vector<unsigned char> ready;
    Partial(int n_,int k_,const std::vector<Z>& p,const std::vector<Z>& s):
        n(n_),k(k_),l(n_-2*k_),pair(p),single(s),sum(1<<n),squares(1<<n),ready(1<<n) {
        sum[0]=1;squares[0]=1;ready[0]=1;
    }
    void visit(unsigned mask) {
        if(ready[mask])return;
        int free=__builtin_popcount(mask);
        if(free>2*k) {
            int label=n-free;
            for(unsigned a=mask;a;a&=a-1) {
                int i=__builtin_ctz(a);unsigned sub=mask^(1u<<i);
                visit(sub);Z z=single[label*n+i];
                sum[mask]+=z*sum[sub];squares[mask]+=std::norm(z)*squares[sub];
            }
        } else {
            int i=__builtin_ctz(mask);unsigned rest=mask^(1u<<i);
            for(unsigned a=rest;a;a&=a-1) {
                int j=__builtin_ctz(a);unsigned sub=rest^(1u<<j);
                visit(sub);Z z=pair[i*n+j];
                sum[mask]+=z*sum[sub];squares[mask]+=std::norm(z)*squares[sub];
            }
        }
        ready[mask]=1;
    }
};
static int choose(const double* cdf,int size,DomainRng84& rng) {
    return std::min<int>(size-1,std::upper_bound(cdf,cdf+size,uniform84(rng))-cdf);
}

extern "C" int visible115(int rows,int m,int rank,int cap,int threads,uint64_t seed,
        uint32_t offset,int spec,const double* energy,const double* sectors,const double* first,
        const double* partner,const Z* pair,const double* source,const double* site,const Z* mode,
        unsigned char* out,uint64_t* attempts) {
    if(m<1||rows<1||rank<1||cap<1||cap>16||threads<1||spec<0||spec>65535||
       uint64_t(offset)+rows>(uint64_t(1)<<32))return 1;
    const int stride=cap/2+1;
    int bad=0;uint64_t tries=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,8) reduction(|:bad) reduction(+:tries)
    for(int row=0;row<rows;++row) {
        DomainRng84 number(seed,spec,101,offset+row),kind(seed,spec,102,offset+row),
            left(seed,spec,103,offset+row),right(seed,spec,104,offset+row),
            axis(seed,spec,105,offset+row),single(seed,spec,106,offset+row),accept(seed,spec,107,offset+row);
        auto* dest=out+size_t(row)*m;std::fill_n(dest,m,0);
        const int n=choose(energy,cap+1,number);
        if(!n)continue;
        std::vector<int> locations(n),labels(n);
        std::vector<Z> b(n*n),v(n*n);
        for(;;) {
            ++tries;
            const int k=choose(sectors+n*stride,n/2+1,kind),l=n-2*k;
            for(int j=0;j<k;++j) {
                int i=choose(first,m,left);
                locations[2*j]=i;locations[2*j+1]=choose(partner+size_t(i)*m,m,right);
            }
            for(int j=0;j<l;++j) {
                int a=choose(source,rank,axis);labels[j]=a;
                locations[2*k+j]=choose(site+size_t(a)*m,m,single);
            }
            double ratio=1;
            if(n>1 && !(n==2 && k==1) && !(k==0 && std::all_of(labels.begin()+1,labels.begin()+l,[&](int a){return a==labels[0];}))) {
                double d=1;
                for(int j=1;j<=n;++j)d*=j;
                for(int j=1;j<=k;++j)d/=2*j;
                if(k) {
                    double scale=0;
                    for(int i=0;i<n;++i)for(int j=i+1;j<n;++j) {
                        Z z=pair[size_t(locations[i])*m+locations[j]];b[i*n+j]=z;scale=std::max(scale,std::abs(z));
                    }
                    if(!(scale>0)){bad|=2;break;}
                    for(auto& z:b)z/=scale;
                }
                for(int j=0;j<l;++j) {
                    double scale=0;
                    for(int i=0;i<n;++i) {
                        Z z=mode[size_t(labels[j])*m+locations[i]];v[j*n+i]=z;scale=std::max(scale,std::abs(z));
                    }
                    if(scale>0)for(int i=0;i<n;++i)v[j*n+i]/=scale;
                }
                Partial hf(n,k,b,v);unsigned mask=(1u<<n)-1;hf.visit(mask);
                ratio=std::norm(hf.sum[mask])/(d*hf.squares[mask]);
            }
            if(!(ratio>=0 && ratio<=1+1e-9)){bad|=4;break;}
            if(ratio>=1||uniform84(accept)<ratio) {
                for(int i:locations)++dest[i];
                break;
            }
        }
    }
    *attempts=tries;return bad;
}
