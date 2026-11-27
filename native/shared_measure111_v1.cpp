#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#include "gaussian_rng_domain84_v1.hpp"

using Complex = std::complex<double>;
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

// Each record owns its proposal, acceptance and permutation streams even under OpenMP.
API int memory111(int rows, int cells, int alphabet, int k, int radix, int threads,
                  uint64_t seed, uint32_t offset, const double *branch_cdf,
                  const double *local_cdf, const double *den_weights, const double *num_weights,
                  const Complex *law, const Complex *log_law, const int *symbols, uint8_t *output, int *attempts) {
    if (rows<1 || cells<1 || k<1 || alphabet<2 || radix<2 || threads<1 ||
        uint64_t(offset)+rows>(uint64_t(1)<<32)) return 1;
    int bad=0;
    #pragma omp parallel for num_threads(threads) reduction(|:bad) schedule(static)
    for (int r=0;r<rows;++r) {
        DomainRng84 proposal(seed,0,1,offset+r), accept(seed,0,2,offset+r), perm(seed,0,3,offset+r);
        std::vector<int> words(cells);
        bool histogram=cells>64;
        std::vector<int> h(histogram?alphabet:0);
        int tries=0;
        for (;;) {
            if (++tries>100000) {bad=1;break;}
            int branch=std::upper_bound(branch_cdf,branch_cdf+k,uniform84(proposal))-branch_cdf;
            auto cdf=local_cdf+size_t(branch)*alphabet;
            if (histogram) {
                int left=cells;
                double prev=0;
                std::fill(h.begin(),h.end(),0);
                for (int y=0;y<alphabet-1 && left;++y) {
                    double p=prev<1?(cdf[y]-prev)/(1-prev):0;
                    h[y]=std::binomial_distribution<int>(left,std::clamp(p,0.,1.))(proposal);
                    left-=h[y];prev=cdf[y];
                }
                h.back()=left;
            } else {
                for (int &y:words) y=std::upper_bound(cdf,cdf+alphabet,uniform84(proposal))-cdf;
            }
            std::vector<Complex> logp;
            if(histogram) {
                const size_t stride=size_t(k)*k;
                logp.resize(stride);
                for(int y=0;y<alphabet;++y) if(h[y]) {
                    const Complex *term=log_law+y*stride;
                    for(int i=0;i<k;++i) for(int j=0;j<=i;++j) {
                        size_t ij=size_t(i)*k+j;
                        logp[ij]+=double(h[y])*term[ij];
                    }
                }
            }
            double numerator=0,denominator=0;
            for (int i=0;i<k;++i) for (int j=0;j<=i;++j) {
                size_t ij=size_t(i)*k+j, stride=size_t(k)*k;
                Complex v=1;
                if (histogram) {
                    v=std::exp(logp[ij]);
                } else {
                    for (int y:words) v*=law[y*stride+ij];
                }
                numerator+=num_weights[ij]*v.real()*(i==j?1:2);
                if(i==j) denominator+=den_weights[i]*v.real();
            }
            double ratio=numerator/denominator;
            if(!std::isfinite(ratio) || ratio < -1e-8 || ratio>1+1e-8) {bad=2;break;}
            if(uniform84(accept)>=ratio) continue;
            if(histogram) {
                auto it=words.begin();
                for(int y=0;y<alphabet;++y) it=std::fill_n(it,h[y],y);
                std::shuffle(words.begin(),words.end(),perm);
            }
            auto dst=output+size_t(r)*cells*4;
            for(int word:words) {
                int y=symbols[word];
                for(int m=0;m<4;++m) { *dst++=uint8_t(y%radix);y/=radix; }
            }
            break;
        }
        attempts[r]=tries;
    }
    return bad;
}

// Walker alias entries for a fully contracted small threshold register.
API void cached111(int rows, int threads, uint64_t seed, uint32_t offset,
                   const uint32_t *alias, const double *prob, uint8_t *out) {
    #pragma omp parallel for num_threads(threads)
    for (int r=0;r<rows;++r) {
        DomainRng84 rng(seed,0,4,offset+r);
        uint32_t w=uint32_t(rng() & 65535);
        if(uniform84(rng)>=prob[w]) w=alias[w];
        for(int j=0;j<16;++j) out[16*size_t(r)+j]=(w>>j)&1;
    }
}
