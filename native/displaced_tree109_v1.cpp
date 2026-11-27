#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using Real = long double;
using Quad = std::array<Real,4>;
using Quadratic = std::array<Real,10>;

static Quadratic products(const Quad& d) {
    Quadratic result{};
    int k=0;
    for(int i=0;i<4;++i) for(int j=i;j<4;++j) result[k++]=d[i]*d[j];
    return result;
}
static Real dot(const Quadratic& d,const double* coefficients) {
    Real value=0;
    for(int i=0;i<10;++i) value+=d[i]*coefficients[i];
    return value;
}
extern "C" void normals109(int n,int spectra,int rank,uint64_t seed,uint32_t offset,int workers,double* z) {
    #pragma omp parallel for num_threads(workers) schedule(static)
    for(int row=0;row<n;++row) for(int s=0;s<spectra;++s) {
        DomainRng84 rng(seed,s,340,offset+row);
        std::normal_distribution<double> normal;
        double* dst=z+(size_t(s)*n+row)*rank;
        for(int k=0;k<rank;++k) dst[k]=normal(rng);
    }
}
struct Vacuum {
    std::array<Real,9> det,num;
    static Real eval(const std::array<Real,9>& p,Real x,Real y) {
        auto row=[&](int i){return (p[i+2]*y+p[i+1])*y+p[i];};
        return (row(6)*x+row(3))*x+row(0);
    }
    Real log_mass(Real x,Real y) const {
        Real d=eval(det,x,y);
        return -std::log(d)/2-eval(num,x,y)/d;
    }
};
struct ThresholdEdge {
    const std::vector<Vacuum>& laws;
    const double* eta_parent;
    const double* eta_child;
    int parent_word,parent_zeros;
    Real mass(int ones,int zeros) const {
        Real total=0;
        for(int a=parent_word;;a=(a-1)&parent_word) {
            Real x=eta_parent ? eta_parent[a|parent_zeros] : 0;
            for(int b=ones;;b=(b-1)&ones) {
                Real y=eta_child[b|zeros],logp=0;
                for(auto& spec:laws) logp+=spec.log_mass(x,y);
                Real v=std::exp(logp);
                total+=(std::popcount(unsigned(a))+std::popcount(unsigned(b)))%2 ? -v : v;
                if(!b) break;
            }
            if(!a) break;
        }
        return total;
    }
    int sample(int width,double u) const {
        int full=(1<<width)-1;
        Real norm=parent_word||parent_zeros ? mass(0,0):1;
        if(!(norm>0)) return -1;
        Real target=u*norm;
        if(target<mass(0,full)) return 0;
        int ones=0,zeros=0;
        for(int bit=0;bit<width;++bit) {
            int flag=1<<bit;
            Real zero=mass(ones,zeros|flag);
            if(zero < -1e-13L || zero>norm+1e-13L) return -1;
            zero=std::clamp(zero,Real(0),norm);
            if(target<zero) {zeros|=flag;norm=zero;}
            else {ones|=flag;target-=zero;norm-=zero;}
        }
        return ones;
    }
};
// Exact one-mode probability recurrence in the covariance eigenbasis.
struct Marginal {
    Real gamma[2],ell[2],s[2]{},h[2]{},p;
    Marginal(const double* pars,const Quad& d) {
        Real logp=0;
        for(int i=0;i<2;++i,pars+=6) {
            Real z=0;
            for(int j=0;j<4;++j) z+=pars[2+j]*d[j];
            Real inv=pars[0];
            gamma[i]=pars[1];ell[i]=z*z*inv*inv;
            logp+=std::log(inv)/2-z*z*inv;
        }
        p=std::exp(logp);
    }
    Real next(int n) {
        Real sum=0;
        for(int i=0;i<2;++i) {
            s[i]=p+gamma[i]*s[i];h[i]=s[i]+gamma[i]*h[i];
            sum+=gamma[i]*s[i]/2+ell[i]*h[i];
        }
        return p=sum/n;
    }
};
constexpr int count_guard=1024;
static int photon_count(int parent,Real u,const Quad& d,const double* one,const double* meta,const double* coeff) {
    if(!parent) {
        Marginal child(one+12,d);
        for(int b=0;b<=count_guard;++b) {
            if(b) child.next(b);
            if(child.p < -1e-14L) return -1;
            if(u<child.p) return b;
            u-=child.p;
        }
        return -1;
    }
    Marginal marginal(one,d);
    for(int a=1;a<=parent;++a) marginal.next(a);
    Real target=u*marginal.p;
    Quad shift{};
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) shift[i]+=meta[4*i+j]*d[j];
    Real logp=-meta[16];
    for(int j=0;j<4;++j) logp-=shift[j]*d[j];
    auto quad=products(shift);
    Real S[25],R[25];
    for(int k=0;k<25;++k) {
        S[k]=coeff[k*12];R[k]=coeff[k*12+1]+dot(quad,coeff+k*12+2);
    }
    std::vector<Real> buf(5*(parent+1));
    auto at=[&](int a,int b)->Real& {return buf[(b%5)*(parent+1)+a];};
    for(int b=0;b<=count_guard;++b) {
        for(int a=0;a<=parent;++a) {
            int degree=a+b;
            if(!degree) {at(a,b)=std::exp(logp);continue;}
            Real v=0;
            for(int i=0;i<=std::min(4,a);++i) for(int j=0;j<=std::min(4,b);++j) {
                if(i+j) v+=at(a-i,b-j)*(R[i*5+j]-(degree-i-j)*S[i*5+j]);
            }
            at(a,b)=v/degree;
        }
        Real p=at(parent,b);
        if(p < -1e-13L*marginal.p || !(marginal.p>0)) return -1;
        if(target<p) return b;
        target-=p;
    }
    return -1;
}

extern "C" int draw109(int n,int spectra,int groups,int fan,int pnr,int workers,
    uint64_t seed,uint32_t offset,const int* parent,const int* order,const double* eta,
    const double* fields,const double* vacuum,const double* photon,const double* meta,
    const double* single,uint32_t* out) {
    int errors=0,masks=1<<fan;
    #pragma omp parallel for num_threads(workers) schedule(static) reduction(+:errors)
    for(int row=0;row<n;++row) {
        std::vector<int> states(size_t(spectra)*groups);
        std::vector<Vacuum> laws(spectra);
        std::vector<DomainRng84> detector;
        for(int s=0;s<spectra;++s) detector.emplace_back(seed,s,342,offset+row);
        DomainRng84 clicks(seed,0,341,offset+row),split(seed,0,343,offset+row);
        for(int pos=0;pos<groups;++pos) {
            int g=order[pos],p=parent[g],total=0;
            bool bad=false;
            for(int s=0;s<spectra;++s) {
                size_t idx=size_t(s)*groups+g;
                const double* f=fields+(size_t(s)*n+row)*groups*2;
                Quad d{p<0?0:f[p*2],p<0?0:f[p*2+1],f[g*2],f[g*2+1]};
                if(pnr) {
                    int count=photon_count(p<0?0:states[s*groups+p],uniform84(detector[s]),d,single+idx*24,meta+idx*23,photon+idx*300);
                    if(count<0) {bad=true;break;}
                    states[idx]=count;total+=count;
                } else {
                    auto quad=products(d);
                    for(int k=0;k<9;++k) {
                        const double* pars=vacuum+(idx*9+k)*11;
                        laws[s].det[k]=pars[0];laws[s].num[k]=dot(quad,pars+1);
                    }
                }
            }
            auto* record=out+(size_t(row)*groups+g)*fan;
            std::fill_n(record,fan,0u);
            if(bad) {++errors;break;}
            const double* eff=eta+g*masks;
            if(pnr) {
                for(int j=0;j<total;++j) {
                    double v=uniform84(split);int k=0;
                    while(k+1<fan && v>=eff[(2<<k)-1]) ++k;
                    ++record[k];
                }
            } else {
                int w=p<0?0:states[p];
                ThresholdEdge edge{laws,p<0?nullptr:eta+p*masks,eff,w,p<0?0:(masks-1)^w};
                int word=edge.sample(fan,uniform84(clicks));
                if(word<0) {++errors;break;}
                states[g]=word;
                for(int j=0;j<fan;++j) record[j]=(word>>j)&1;
            }
        }
    }
    return errors;
}
