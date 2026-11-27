#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>
#include "gaussian_rng_domain84_v1.hpp"
using Z=std::complex<double>;

struct Haf {
    int n;
    const std::vector<Z>& b;
    std::vector<Z> sum;
    std::vector<double> squares;
    std::vector<unsigned char> known;
    Haf(int n_,const std::vector<Z>& b_):n(n_),b(b_),sum(1<<n),squares(1<<n),known(1<<n) {
        sum[0]=1;squares[0]=1;known[0]=1;
    }
    void eval(unsigned mask) {
        if(known[mask])return;
        int i=__builtin_ctz(mask);
        unsigned rest=mask^(1u<<i);
        for(unsigned rem=rest;rem;rem&=rem-1){
            int j=__builtin_ctz(rem);unsigned sub=rest^(1u<<j);
            eval(sub);
            sum[mask]+=b[i*n+j]*sum[sub];
            squares[mask]+=std::norm(b[i*n+j])*squares[sub];
        }
        known[mask]=1;
    }
};

static double prefix_mass(const Z* g,const Z* v,int r) {
    double out=0;
    for(int i=0;i<r;++i) {
        out+=g[i*r+i].real()*std::norm(v[i]);
        for(int j=0;j<i;++j)out+=2*(std::conj(v[i])*g[i*r+j]*v[j]).real();
    }
    return out;
}
static Z dot(const Z* a,const Z* b,int r) {
    Z z{};
    for(int i=0;i<r;++i)z+=a[i]*b[i];
    return z;
}

extern "C" int pairs114(int rows,int m,int all,int r,int cap,int block,int threads,
        uint64_t seed,uint32_t offset,int spectrum,const double* number,const Z* t,
        const double* lambda,const Z* prefix,const double* first,unsigned char* out,
        uint64_t* totals) {
    if(rows<1||m<1||all<m||r<1||cap<2||cap>14||block<1||threads<1||spectrum<0||spectrum>65535||
       uint64_t(offset)+rows>(uint64_t(1)<<32))return 1;
    int fail=0;uint64_t attempts=0,accepted=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,8) reduction(|:fail) reduction(+:attempts,accepted)
    for(int row=0;row<rows;++row) {
        DomainRng84 pick(seed,spectrum,11,offset+row),left(seed,spectrum,12,offset+row),
                    right(seed,spectrum,13,offset+row),accept(seed,spectrum,14,offset+row);
        auto* dest=out+size_t(row)*m;
        std::fill_n(dest,m,0);
        int n=std::upper_bound(number,number+cap+1,uniform84(pick))-number;
        if(n>cap||n%2){fail|=2;continue;}
        if(!n)continue;
        int blocks=(all+block-1)/block;
        std::vector<int> site(n);
        std::vector<Z> v(r),matrix(n*n);
        double matchings=1;
        for(int i=3;i<n;i+=2)matchings*=i;
        for(;;) {
            ++attempts;
            for(int k=0;k<n;k+=2) {
                int i=std::upper_bound(first,first+all,uniform84(left))-first;
                if(i==all){fail|=4;i=all-1;}
                site[k]=i;
                double norm=0;
                for(int j=0;j<r;++j){v[j]=t[size_t(i)*r+j]*lambda[j];norm+=std::norm(v[j]);}
                double target=uniform84(right)*norm;
                int lo=0,hi=blocks-1;
                while(lo<hi) {
                    int mid=(lo+hi)/2;
                    if(prefix_mass(prefix+size_t(mid)*r*r,v.data(),r)<=target)lo=mid+1;else hi=mid;
                }
                if(lo)target-=prefix_mass(prefix+size_t(lo-1)*r*r,v.data(),r);
                int loc=lo*block,end=std::min(all,loc+block);
                for(;loc<end-1;++loc) {
                    target-=std::norm(dot(t+size_t(loc)*r,v.data(),r));
                    if(target<0)break;
                }
                site[k+1]=loc;
            }
            double ratio=1;
            if(n>2) {
                double scale=0;
                for(int i=0;i<n;++i) {
                    for(int k=0;k<r;++k)v[k]=t[size_t(site[i])*r+k]*lambda[k];
                    for(int j=i+1;j<n;++j) {
                        Z val=dot(t+size_t(site[j])*r,v.data(),r);
                        matrix[i*n+j]=val;scale=std::max(scale,std::abs(val));
                    }
                }
                if(!(scale>0)){fail|=8;break;}
                for(auto& z:matrix)z/=scale;
                Haf hf(n,matrix);unsigned mask=(1u<<n)-1;hf.eval(mask);
                ratio=std::norm(hf.sum[mask])/(matchings*hf.squares[mask]);
            }
            if(!std::isfinite(ratio)||ratio<0||ratio>1+1e-10){fail|=16;break;}
            if(n==2||uniform84(accept)<ratio) {
                ++accepted;
                for(int loc:site)if(loc<m)++dest[loc];
                break;
            }
        }
    }
    totals[0]=attempts;totals[1]=accepted;
    return fail;
}
