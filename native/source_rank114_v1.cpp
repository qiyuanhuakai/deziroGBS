#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>
#include "gaussian_rng_domain84_v1.hpp"

using Z = std::complex<double>;
using Vec = std::vector<Z>;
using Real = std::vector<double>;

// Each cumulative block contains sum conj(T_i) T_i. The last block includes loss outputs.
static double mass(const Z* gram, const Z* effect, int r) {
    double out=0;
    for(int j=0;j<r;++j) {
        out+=gram[j*r+j].real()*effect[j*r+j].real();
        for(int k=0;k<j;++k) out+=2*(gram[j*r+k]*effect[j*r+k]).real();
    }
    return out;
}
static double site_mass(const Z* gram,const Z* t,int r) {
    double out=0;
    for(int j=0;j<r;++j) {
        out+=gram[j*r+j].real()*std::norm(t[j]);
        for(int k=0;k<j;++k) out+=2*(gram[j*r+k]*std::conj(t[j])*t[k]).real();
    }
    return out;
}

extern "C" int pnr114(int rows,int m,int all,int r,int cap,int block,int threads,
        uint64_t seed,uint32_t offset,const double* cdf,const int* dims,const int* starts,
        const int* parent,const double* factor,const double* initial,const Z* transfer,
        const Z* prefix,unsigned char* records) {
    if(rows<1||r<1||all<m||m<1||block<1||cap>255||cap<1||threads<1||
       uint64_t(offset)+rows>(uint64_t(1)<<32))return 1;
    const int blocks=(all+block-1)/block;
    int failed=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,8) reduction(|:failed)
    for(int row=0;row<rows;++row) {
        DomainRng84 number(seed,0,1,offset+row),site(seed,0,2,offset+row);
        auto* record=records+size_t(row)*m;
        std::fill_n(record,m,0);
        int n=std::upper_bound(cdf,cdf+cap+1,uniform84(number))-cdf;
        if(n>cap){failed|=2;continue;}
        Vec psi(initial+starts[n],initial+starts[n]+dims[n]),a,gram(r*r);
        for(;n>0;--n) {
            const int dim=dims[n-1];
            a.resize(size_t(dim)*r);
            std::fill(gram.begin(),gram.end(),Z{});
            for(int j=0;j<dim;++j) {
                Z* v=a.data()+size_t(j)*r;
                for(int i=0;i<r;++i) {
                    size_t ix=size_t(starts[n-1]+j)*r+i;
                    v[i]=factor[ix]*psi[parent[ix]];
                    for(int k=0;k<=i;++k)gram[i*r+k]+=std::conj(v[i])*v[k];
                }
            }
            const double norm=mass(gram.data(),prefix+size_t(blocks-1)*r*r,r);
            if(!(norm>0)){failed|=4;break;}
            double u=uniform84(site)*norm;
            int lo=0,hi=blocks-1;
            while(lo<hi) {
                int mid=(lo+hi)/2;
                if(mass(gram.data(),prefix+size_t(mid)*r*r,r)<=u)lo=mid+1;else hi=mid;
            }
            if(lo)u-=mass(gram.data(),prefix+size_t(lo-1)*r*r,r);
            int idx=lo*block,end=std::min(all,idx+block);
            for(;idx<end-1;++idx) {
                u-=std::max(0.,site_mass(gram.data(),transfer+size_t(idx)*r,r));
                if(u<0)break;
            }
            if(idx<m)++record[idx];
            if(n==1)break;
            const Z* t=transfer+size_t(idx)*r;
            psi.assign(dim,Z{});
            double norm2=0;
            for(int j=0;j<dim;++j) {
                for(int i=0;i<r;++i)psi[j]+=a[size_t(j)*r+i]*t[i];
                norm2+=std::norm(psi[j]);
            }
            if(!(norm2>0)){failed|=8;break;}
            const double inv=1/std::sqrt(norm2);
            for(auto& v:psi)v*=inv;
        }
    }
    return failed;
}

static Z complex_expm1(Z z) {
    if(std::abs(z.real())+std::abs(z.imag())<1e-3)
        return z*(1.+z*(.5+z*(1./6+z*(1./24+z/120.))));
    const double h=std::sin(.5*z.imag());
    return {std::expm1(z.real())*std::cos(z.imag())-2*h*h,std::exp(z.real())*std::sin(z.imag())};
}

struct Click {
    int m,r,b,block;
    const Z* t;
    const double *pos,*proposal,*hazard,*diag;
    const Z* base;
    Z amplitude(int site,int branch)const {
        Z z{};
        for(int j=0;j<r;++j)z+=t[size_t(site)*r+j]*pos[branch*r+j];
        return z; // A common factor of i cancels from all inner products.
    }
    std::pair<double,double> probability(const std::vector<int>& sites)const {
        // Index a,b stores <b|E|a>, including the unobserved environment overlap.
        Vec product(base,base+size_t(b)*(b+1)/2),z(b);
        Real q(diag,diag+b);
        double scale_product=1;
        for(int site:sites) {
            double scale=0;
            for(int a=0;a<b;++a) {
                z[a]=amplitude(site,a);
                scale=std::max(scale,std::norm(z[a]));
            }
            if(!(scale>0))return {0,0};
            scale_product*=scale;
            int ix=0;
            for(int a=0;a<b;++a) {
                for(int j=0;j<a;++j)
                    product[ix++]*=complex_expm1(std::conj(z[j])*z[a])/scale;
                const double factor=std::expm1(std::norm(z[a]))/scale;
                product[ix++]*=factor;
                q[a]*=factor;
            }
        }
        double p=0,den=0;
        for(Z v:product)p+=v.real();
        for(double v:q)den+=v;
        return {p*scale_product,den>0?p/den:0};
    }
    void propose(int branch,DomainRng84& rng,std::vector<int>& sites)const {
        const int blocks=(m+block-1)/block;
        const double* h=hazard+size_t(branch)*blocks;
        int last=-1,current=-1;
        double cumulative=0;
        sites.clear();
        while(true) {
            const double target=cumulative-std::log1p(-uniform84(rng));
            int index=std::upper_bound(h+std::max(current,0),h+blocks,target)-h;
            if(index==blocks)return;
            int j=last+1;
            if(index!=current){j=index*block;cumulative=index?h[index-1]:0;}
            const int end=std::min(m,(index+1)*block);
            for(;j<end;++j) {
                cumulative+=std::norm(amplitude(j,branch));
                if(cumulative>target)break;
            }
            if(j==end){cumulative=h[index];current=index;last=end-1;continue;}
            sites.push_back(j);
            last=j;current=index;
        }
    }
};

extern "C" int click114(int rows,int m,int r,int b,int block,int threads,uint64_t seed,uint32_t offset,
        const Z* t,const double* pos,const double* proposal,const double* hazard,
        const double* diag,const Z* base,unsigned char* pass,uint64_t* attempts) {
    if(m<1||rows<1||r<1||b<1||block<1||threads<1||uint64_t(offset)+rows>(uint64_t(1)<<32))return 1;
    Click model{m,r,b,block,t,pos,proposal,hazard,diag,base};
    int failure=0;
    uint64_t tries=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,8) reduction(+:tries) reduction(|:failure)
    for(int row=0;row<rows;++row) {
        DomainRng84 branch(seed,0,3,offset+row), gaps(seed,0,4,offset+row), accept(seed,0,5,offset+row);
        auto* result=pass+size_t(row)*m;
        std::fill_n(result,m,0);
        std::vector<int> sites;
        for(;;) {
            ++tries;
            int a=std::upper_bound(proposal,proposal+b,uniform84(branch))-proposal;
            if(a==b){failure|=2;break;}
            model.propose(a,gaps,sites);
            double ratio=model.probability(sites).second;
            if(!std::isfinite(ratio)||ratio < -1e-10||ratio>1+1e-10){failure|=4;break;}
            if(uniform84(accept)<ratio) {
                for(int site:sites)result[site]=1;
                break;
            }
        }
    }
    *attempts=tries;
    return failure;
}

extern "C" int law114(int m,int r,int b,const Z* t,const double* pos,const double* diag,
                      const Z* base,double* out) {
    if(m<1||m>16)return 1;
    Click model{m,r,b,0,t,pos,nullptr,nullptr,diag,base};
    #pragma omp parallel for num_threads(8)
    for(int mask=0;mask<1<<m;++mask) {
        std::vector<int> sites;
        for(int j=0;j<m;++j)if(mask&(1<<j))sites.push_back(j);
        out[mask]=model.probability(sites).first;
    }
    return 0;
}
