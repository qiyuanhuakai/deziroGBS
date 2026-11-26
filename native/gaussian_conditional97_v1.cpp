// Marginalize uneventful ports or project them onto vacuum via a two-quadrature Schur update.
#include "gaussian_rng_domain84_v1.hpp"
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>
namespace {
using Real=long double;
constexpr int stride=36;
struct Sector {Real q[stride][stride],mu[stride];};
using Frame=std::vector<Sector>;
Real include_vacuum(const Frame& from,Frame& to,int pos,int modes){
    int a=2*pos,b=a+1,end=2*modes;
    Real mass=1;
    for(std::size_t s=0;s<from.size();++s){
        auto& x=from[s];auto& y=to[s];Real aa=x.q[a][a],bb=x.q[b][b],ab=x.q[b][a];
        Real det=aa*bb-ab*ab;if(!(det>0))return -1;
        Real k0=bb/det,k1=-ab/det,k2=aa/det;
        Real u=x.mu[a],v=x.mu[b];
        mass/=std::sqrt(det);
        if(u!=0||v!=0)mass*=std::exp(-.5L*(u*(k0*u+k1*v)+v*(k1*u+k2*v)));
        for(int i=b+1;i<end;++i){
            Real t=x.q[i][a]*k0+x.q[i][b]*k1,z=x.q[i][a]*k1+x.q[i][b]*k2;
            y.mu[i]=x.mu[i]-t*u-z*v;
            for(int j=b+1;j<=i;++j)y.q[i][j]=x.q[i][j]-t*x.q[j][a]-z*x.q[j][b];
        }
    }
    return mass;
}
// Vacuum, two one-photon marginals at joint vacuum, and their joint derivative.
std::array<Real,4> jets(const Frame& frame,int first,int modes){
    int d=2*modes;Real p=1,g[2]{},cross=0;
    for(const auto& x:frame){
        Real L[4][4]{},invL[4][4]{},inv[4][4]{},h[4]{};
        for(int i=0;i<d;++i){
            for(int j=0;j<=i;++j){
                Real z=x.q[2*first+i][2*first+j];
                for(int a=0;a<j;++a)z-=L[i][a]*L[j][a];
                L[i][j]=i==j?std::sqrt(z):z/L[j][j];
            }
            p/=L[i][i];
            for(int j=0;j<=i;++j){
                Real z=i==j?1:0;
                for(int a=j;a<i;++a)z-=L[i][a]*invL[a][j];
                invL[i][j]=z/L[i][i];
            }
        }
        for(int i=0;i<d;++i)for(int j=0;j<d;++j){
            for(int k=std::max(i,j);k<d;++k)inv[i][j]+=invL[k][i]*invL[k][j];
            h[i]+=inv[i][j]*x.mu[2*first+j];
        }
        Real disp=0;for(int i=0;i<d;++i){disp+=h[i]*x.mu[2*first+i];g[i/2]+=.5L*(1-inv[i][i]+h[i]*h[i]);}
        if(disp!=0)p*=std::exp(-disp/2);
        if(modes==2)for(int i=0;i<2;++i)for(int j=2;j<4;++j)
            cross+=.5L*inv[i][j]*inv[i][j]-h[i]*inv[i][j]*h[j];
    }
    return {p,p*g[0],p*g[1],p*(g[0]*g[1]+cross)};
}
void ternary(const Frame& f,int first,int w,Real mass,Real* out){
    auto u=jets(f,first,1);
    if(w==1){out[0]=mass*u[0];out[1]=mass*u[1];out[2]=mass*(1-u[0]-u[1]);return;}
    auto v=jets(f,first+1,1),j=jets(f,first,2);
    Real law[]={j[0],j[1],v[0]-j[0]-j[1],j[2],j[3],v[1]-j[2]-j[3],
        u[0]-j[0]-j[2],u[1]-j[1]-j[3],1-u[0]-u[1]-v[0]-v[1]+j[0]+j[1]+j[2]+j[3]};
    for(int k=0;k<9;++k)out[k]=mass*law[k];
}
struct Sweep{
    int bits,modes,width,base,leaves;
    std::vector<Frame> stack;
    std::vector<Real>& v;bool good=true;
    void walk(int pos,int mask,Real mass,const Frame& f){
        if(pos==bits){
            if(base==2)v[mask]=mass;
            else ternary(f,pos,width,mass,v.data()+std::size_t(mask)*leaves);
            return;
        }
        walk(pos+1,mask,mass,f);auto& next=stack[pos];
        Real factor=include_vacuum(f,next,pos,modes);
        if(factor<0){good=false;return;}
        walk(pos+1,mask|(1<<pos),mass*factor,next);
    }
};
}
extern "C" __declspec(dllexport) int cond97_law(int parents,int width,int base,int sectors,int workers,
        const double* qs,const double* means,double* out,double* diag){
    int modes=parents+width;
    if(parents<0||width<1||modes>18||(base!=2&&base!=3)||(base==3&&width>2)||
       sectors<1||sectors>8||workers<1||workers>64||!qs||!means||!out||!diag)return 1;
    try{
        Frame original(sectors);int dim=2*modes;
        for(int s=0;s<sectors;++s)for(int i=0;i<dim;++i){
            int a=i/2+(i%2)*modes;original[s].mu[i]=means[s*dim+a];
            for(int j=0;j<=i;++j)original[s].q[i][j]=qs[(s*dim+a)*dim+j/2+(j%2)*modes];
        }
        int bits=base==2?modes:parents,leaves=base==2?1:(width==1?3:9);
        int top=0;while((1<<top)<workers&&top<bits)++top;
        std::vector<Real> v((1<<bits)*leaves);std::atomic<bool> good{true};
        #pragma omp parallel num_threads(workers) if(workers>1)
        {
            Sweep sweep{bits,modes,width,base,leaves,std::vector<Frame>(bits,Frame(sectors)),v};
            #pragma omp for schedule(static)
            for(int mask=0;mask<(1<<top);++mask){
                Real mass=1;const Frame* f=&original;
                for(int i=0;i<top;++i)if(mask&(1<<i)){
                    Real x=include_vacuum(*f,sweep.stack[i],i,modes);
                    if(x<0)sweep.good=false;
                    mass*=x;f=&sweep.stack[i];
                }
                sweep.walk(top,mask,mass,*f);
            }
            if(!sweep.good)good=false;
        }
        if(!good)return 2;
        int size=1<<bits;
        for(int bit=1;bit<size;bit*=2)for(int mask=0;mask<size;++mask)if(!(mask&bit))
            for(int j=0;j<leaves;++j)v[mask*leaves+j]-=v[(mask|bit)*leaves+j];
        Real sum=0,neg=0;for(auto x:v){if(!std::isfinite(x))return 3;neg+=std::max(Real(0),-x);sum+=std::max(Real(0),x);}
        diag[0]=double(neg);diag[1]=double(sum);if(neg>1e-8L||!(sum>0))return 3;
        int pm=(1<<parents)-1,child=base==2?(1<<width):leaves;
        for(int p=0;p<=pm;++p)for(int z=0;z<child;++z){
            int k=base==2?(size-1-(p+(z<<parents))):(pm-p)*leaves+z;
            out[p*child+z]=double(std::max(Real(0),v[k])/sum);
        }
        return 0;
    }catch(...){return 4;}
}
namespace {
struct Group{
    std::vector<int> past,ports;int k;
    std::vector<std::uint64_t> cuts;
};
struct Net{int m,base,cap;std::vector<Group> groups;std::vector<std::uint64_t> emissions,splits;};
int choose(const std::uint64_t* c,int n,DomainRng84& rng){
    return int(std::upper_bound(c,c+n,rng()>>11)-c);
}
}
extern "C" __declspec(dllexport) void* cond97_prepare(int m,int groups,int base,int cap,const int* widths,
        const int* ports,const int* histories,const int* past,const std::uint64_t* cuts,
        const std::uint64_t* emissions,const std::uint64_t* splits){
    if(!widths||!ports||!histories||!past||!cuts||!emissions||!splits||m<1||groups<1||groups>m||
        (base!=2&&base!=3)||cap<1||cap>126)return nullptr;
    try{
        auto net=std::make_unique<Net>();net->m=m;net->base=base;net->cap=cap;std::vector<bool> seen(m);
        for(int g=0;g<groups;++g){
            int w=widths[g],h=histories[g];
            if(w<1||w>18||h<0||w+h>18||(base==3&&w>2))return nullptr;
            for(int j=0;j<h;++j)if(past[j]<0||past[j]>=m||!seen[past[j]])return nullptr;
            for(int j=0;j<w;++j){if(ports[j]<0||ports[j]>=m||seen[ports[j]])return nullptr;seen[ports[j]]=true;}
            int k=base==2?(1<<w):(w==1?3:9);auto& a=net->groups.emplace_back();
            a.k=k;a.ports.assign(ports,ports+w);a.past.assign(past,past+h);ports+=w;past+=h;
            std::size_t len=std::size_t(1<<h)*k;a.cuts.assign(cuts,cuts+len);cuts+=len;
            for(std::size_t j=0;j<len;j+=k)if(a.cuts[j+k-1]!=(1ULL<<53)||!std::is_sorted(a.cuts.begin()+j,a.cuts.begin()+j+k))return nullptr;
        }
        if(std::find(seen.begin(),seen.end(),false)!=seen.end())return nullptr;
        if(base==3){std::size_t n=std::size_t(groups)*9*(cap+2);net->emissions.assign(emissions,emissions+n);net->splits.assign(splits,splits+n*(cap+2));}
        return net.release();
    }catch(...){return nullptr;}
}
extern "C" __declspec(dllexport) void cond97_free(void* net){delete static_cast<Net*>(net);}
extern "C" __declspec(dllexport) int cond97_draw(const void* ptr,int records,std::uint64_t domain,std::uint64_t offset,int workers,unsigned char* out,unsigned char* tail){
    if(!ptr||records<1||workers<1||workers>64||!out||!tail||offset>0x100000000ULL||std::uint64_t(records)>0x100000000ULL-offset)return 1;
    auto& net=*static_cast<const Net*>(ptr);int stride=net.cap+2;
    #pragma omp parallel for num_threads(workers) if(workers>1) schedule(static)
    for(int r=0;r<records;++r){
        DomainRng84 mark(domain,0,150,offset+r),fine(domain,0,151,offset+r),split(domain,0,152,offset+r);
        auto row=out+std::size_t(r)*net.m;tail[r]=0;
        for(std::size_t i=0;i<net.groups.size();++i){
            auto& g=net.groups[i];unsigned h=0;
            for(int j=0;j<int(g.past.size());++j)h|=unsigned(row[g.past[j]]>0)<<j;
            int z=choose(g.cuts.data()+std::size_t(h)*g.k,g.k,mark);
            if(net.base==2){for(auto a:g.ports){row[a]=z&1;z>>=1;}continue;}
            auto idx=(i*9+z)*stride;
            int n=choose(net.emissions.data()+idx,stride,fine);tail[r]|=n==net.cap+1;
            if(g.ports.size()==1){row[g.ports[0]]=n;continue;}
            int a=z%3<2?z%3:(z/3<2?n-z/3:choose(net.splits.data()+(idx+n)*stride,n+1,split));
            row[g.ports[0]]=a;row[g.ports[1]]=n-a;
        }
    }
    return 0;
}
