#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

// Fock occupations on a simplex, in colex order of their cumulative sums.
// Earlier total sectors precede every state that depends on them.
struct Simplex110 {
    int modes,cut,states;
    std::vector<std::array<int64_t,9>> choose;
    std::vector<unsigned short> occupation;
    Simplex110(int d,int k,int n):modes(d),cut(k),states(n),choose(k+d+1),occupation(size_t(n)*d) {
        for(int i=0;i<=k+d;++i){
            choose[i][0]=1;
            for(int j=1;j<=std::min(i,d);++j)choose[i][j]=choose[i-1][j-1]+choose[i-1][j];
        }
    }
    void decode(int a) {
        int rest=a,upper=cut+modes-1;
        for(int j=modes-1;j>=0;--j){
            int lo=j,hi=upper;
            while(lo<hi){int mid=(lo+hi+1)/2;if(choose[mid][j+1]<=rest)lo=mid;else hi=mid-1;}
            occupation[size_t(a)*modes+j]=lo;
            rest-=int(choose[lo][j+1]);upper=lo-1;
        }
        auto v=occupation.data()+size_t(a)*modes;
        for(int j=modes-1;j>0;--j)v[j]-=v[j-1]+1;
    }
    int index(const unsigned short* v,int i,int j,int drop=1) const {
        int prefix=0;int64_t result=0;
        for(int s=0;s<modes;++s){
            prefix+=v[s]+1-(s==i?drop:0)-(s==j);
            result+=choose[prefix-1][s+1];
        }
        return int(result);
    }
};

static void transform110(const Simplex110& grid,const unsigned short* states,const double* emitted,const int* width,
    const double* efficiency,double* pnr,double* click,int workers){
    int size=grid.states,d=grid.modes,cut=grid.cut;
    for(int route=0;route<2;++route){
        // Transform one mode at a time. Terms only move down the simplex.
        std::vector<long double> source(emitted,emitted+size),dest(size);
        for(int j=0;j<d;++j){
            int stride=cut+1;std::vector<long double> kernel(size_t(stride)*stride);kernel[0]=1;
            for(int n=0;n<cut;++n)for(int k=0;k<=n;++k){
                long double p=kernel[n*stride+k];
                long double next=route ? efficiency[j]*(1-(long double)k/width[j]):efficiency[j];
                if(route && k>=width[j])next=0;
                kernel[(n+1)*stride+k]+=p*(1-next);
                kernel[(n+1)*stride+k+1]+=p*next;
            }
            // Each fiber has one member with coordinate j=0; fibers do not overlap.
            #pragma omp parallel for num_threads(workers) schedule(static)
            for(int a=0;a<size;++a){
                const auto* v=states+size_t(a)*d;if(v[j])continue;
                int top=cut;for(int i=0;i<d;++i)top-=v[i];
                std::array<int,513> index{};
                for(int n=0;n<=top;++n)index[n]=grid.index(v,j,-1,-n);
                for(int k=0;k<=top;++k){
                    long double val=0;
                    for(int n=k;n<=top;++n)val+=source[index[n]]*kernel[n*stride+k];
                    dest[index[k]]=val;
                }
            }
            source.swap(dest);
        }
        std::copy(source.begin(),source.end(),route?click:pnr);
    }
}
