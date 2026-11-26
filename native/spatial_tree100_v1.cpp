#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

using Word=std::uint64_t;
constexpr Word scale=Word(1)<<53;

// Coefficients of det(I-zK)^(-1/2), including the separate vacuum prefactor.
extern "C" __declspec(dllexport) int spatial100_series(int rows,int rank,int cap,double vacuum,
        const double* eig,double* out){
    if(rows<1||rank<1||cap<0||vacuum<=0||!eig||!out)return 1;
    std::vector<long double> sums(rank);
    for(int i=0;i<rows;++i){
        std::fill(sums.begin(),sums.end(),0);
        long double prev=vacuum;out[i*(cap+1)]=vacuum;
        for(int n=1;n<=cap;++n){
            long double value=0;
            for(int j=0;j<rank;++j){
                sums[j]=eig[i*rank+j]*(prev+sums[j]);value+=sums[j];
            }
            prev=value/(2*n);out[i*(cap+1)+n]=double(prev);
        }
    }
    return 0;
}

static int pick(const Word* cuts,int n,DomainRng84& rng){
    auto u=rng()>>11;
    return u<cuts[0]?0:int(std::upper_bound(cuts+1,cuts+n,u)-cuts);
}
extern "C" __declspec(dllexport) int spatial100_complex(int rows,int rank,int cap,const double* roots,
        const double* prefactor,double* output){
    if(rows<1||rank<1||cap<0||!roots||!prefactor||!output)return 1;
    using Z=std::complex<double>;
    std::vector<Z> sums(rank),eig(rank);
    for(int i=0;i<rows;++i){
        std::fill(sums.begin(),sums.end(),Z{});
        for(int j=0;j<rank;++j)eig[j]=Z(roots[2*(i*rank+j)],roots[2*(i*rank+j)+1]);
        Z prev(prefactor[2*i],prefactor[2*i+1]);
        for(int n=0;n<=cap;++n){
            if(n){
                Z v{};for(int j=0;j<rank;++j){sums[j]=eig[j]*(prev+sums[j]);v+=sums[j];}
                prev=v/(2.*n);
            }
            auto id=2*(i*(cap+1)+n);output[id]=prev.real();output[id+1]=prev.imag();
        }
    }
    return 0;
}
// Distribute photons conditioned on the terminal occupied subset, backwards
// through the multinomial occupancy recursion. The last draw can have hit a
// previously occupied port or introduced a new one.
static bool distribute(int n,unsigned mask,int width,const double* p,const double* previous,
                       DomainRng84& rng,std::uint16_t* output){
    int size=1<<width;
    while(n){
        if(!mask||std::popcount(mask)>n)return false;
        if(std::has_single_bit(mask)){output[std::countr_zero(mask)]+=n;return true;}
        if(std::popcount(mask)==n){
            while(mask){++output[std::countr_zero(mask)];mask&=mask-1;}return true;
        }
        const double* law=previous+std::size_t(n-1)*size;
        std::array<double,16> weights{};double total=0;
        for(unsigned bits=mask;bits;bits&=bits-1){
            int j=std::countr_zero(bits);
            weights[2*j]=p[j]*law[mask];
            weights[2*j+1]=p[j]*law[mask^(1u<<j)];
            total+=weights[2*j]+weights[2*j+1];
        }
        if(!(total>0))return false;
        double u=uniform84(rng)*total;int last=-1,chosen=-1;
        for(int k=0;k<2*width;++k)if(weights[k]>0){
            last=k;u-=weights[k];if(u<0){chosen=k;break;}
        }
        if(chosen<0)chosen=last;
        ++output[chosen/2];
        if(chosen&1)mask^=1u<<(chosen/2);
        --n;
    }
    return mask==0;
}
extern "C" __declspec(dllexport) int spatial100_draw(int leaves,int width,const int* children,
        const int* caps,const int* starts,const Word* root,const Word* splits,const int* leafstarts,
        const Word* masks,const double* prior,const double* ports,int shots,int route,
        Word seed,Word offset,int workers,std::uint16_t* out,unsigned char* flags){
    if(leaves<1||leaves>16384||width<1||width>8||!children||!caps||!starts||!root||!splits||
       !leafstarts||!masks||!prior||!ports||shots<1||route<0||route>1||workers<1||workers>64||
       offset>0xffffffffULL||offset+shots>0x100000000ULL||!out||!flags)return 1;
    int nodes=2*leaves-1,size=1<<width;
    #pragma omp parallel num_threads(workers)
    {
        std::vector<int> number(nodes);
        #pragma omp for schedule(static)
        for(int r=0;r<shots;++r){
            DomainRng84 total(seed,0,168,offset+r),branch(seed,0,169,offset+r),
                        clicks(seed,0,170,offset+r),pnr(seed,0,171,offset+r);
            number.back()=pick(root,caps[nodes-1]+1,total);unsigned char flag=0;
            for(int i=nodes-1;i>=leaves;--i){
                int n=number[i],a=children[2*i],b=children[2*i+1],k=0;
                if(n>caps[i]){flag|=1;k=n/2;}
                else if(n)k=pick(splits+starts[i]+n*(n+1)/2,n+1,branch);
                number[a]=k;number[b]=n-k;
            }
            auto row=out+std::size_t(r)*leaves*width;std::fill(row,row+leaves*width,0);
            for(int g=0;g<leaves;++g){
                int n=number[g];if(!n)continue;
                auto dst=row+g*width;
                if(n>caps[g]){
                    flag|=1;
                    for(int h=0;h<n;++h){
                        double u=uniform84(clicks);int j=0;
                        while(j+1<width&&u>=ports[g*width+j]){u-=ports[g*width+j];++j;}
                        if(route)++dst[j];else dst[j]=1;
                    }
                    continue;
                }
                auto base=std::size_t(leafstarts[g])*size;
                unsigned mask=pick(masks+base+std::size_t(n)*size,size,clicks);
                if(route){
                    if(!distribute(n,mask,width,ports+g*width,prior+base,pnr,dst))flag|=2;
                }else while(mask){dst[std::countr_zero(mask)]=1;mask&=mask-1;}
            }
            flags[r]=flag;
        }
    }
    return 0;
}
