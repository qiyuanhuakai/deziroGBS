// Terminal couplers connect the right port of each Fourier cell to its successor's left port.
#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

using Complex98 = std::complex<long double>;
using Word98 = std::uint64_t;

struct Cycle98 {
    int cap, route;
    const double *amplitudes, *grams;
    const Word98 *cuts;
    int width() const { return cap + 1; }
    double d(int n, int m, int k) const {
        return amplitudes[(n * width() + m) * (2 * cap + 1) + k];
    }
    double projection(int n, int m, int shift, int other, int label) const {
        int total = n + m;
        auto outer = [&](int k) {
            return d(n+shift,m-shift,k) * d(n+other,m-other,k);
        };
        if (route) return outer(label);
        if (label == 0) return outer(0);
        if (label == 1) return outer(total);
        return double(shift == other) - outer(0) - outer(total);
    }
    int draw_edge(int n, int m, DomainRng84& rng) const {
        if (!n && !m) return 0;
        int count = route ? n+m+1 : 2;
        const auto c = cuts + (n*width()+m)*(route ? 2*cap+1 : 2);
        return int(std::upper_bound(c,c+count,rng()>>11)-c);
    }
    // All previously observed edge outcomes have the dephased marginal.
    int close(const std::vector<unsigned>& states, const std::vector<unsigned>& groups,
              const std::vector<int>& labels, int min_right, int min_left,
              double phase, DomainRng84& rng) const {
        int cells = int(states.size()), last = cells-1;
        int n = (states[last]>>12)&15, m = states[0]&15;
        int outcomes = route ? n+m+1 : 3;
        std::array<long double,31> weights{};
        for (int a=-min_right; a<=min_left; ++a) {
            for (int b=-min_right; b<=a; ++b) {
                Complex98 product = std::polar(1.L,static_cast<long double>((a-b)*phase));
                for (int i=0;i<cells;++i) {
                    int right=(states[i]>>12)&15, next_left=states[(i+1)%cells]&15;
                    auto ix = (std::size_t(groups[i])*width()+right+a)*width()+right+b;
                    product *= Complex98(grams[2*ix],grams[2*ix+1]);
                    if (i<last) product *= projection(right,next_left,a,b,labels[i]);
                }
                for (int k=0;k<outcomes;++k)
                    weights[k] += (a==b ? 1 : 2)*product.real()*projection(n,m,a,b,k);
            }
        }
        long double total=0,scale=0;
        for(int k=0;k<outcomes;++k) scale=std::max(scale,std::abs(weights[k]));
        if (!(scale>0) || !std::isfinite(scale)) return -1;
        for(int k=0;k<outcomes;++k) {
            if (weights[k]<-1e-12L*scale || !std::isfinite(weights[k])) return -1;
            total+=std::max(weights[k],0.L);
        }
        if (!(total>0)) return -1;
        long double u=uniform84(rng)*total;
        for(int k=0;k<outcomes;++k) {
            u-=std::max(weights[k],0.L);
            if (u<0) return k;
        }
        return -1;
    }
};

extern "C" __declspec(dllexport) int cycle98_draw(
        int cells,int route,int bits,int cap,int coherent,double phase,
        const Word98* local_cut,const unsigned* alias,const unsigned* keys,const unsigned* group_ids,
        const double* grams,const double* amplitudes,const Word98* cuts,
        int records,Word98 experiment,Word98 offset,int workers,unsigned char* output,unsigned char* ambiguous) {
    if(cells<2 || cells>16384 || route<0 || route>1 || bits<1 || bits>24 || cap<1 || cap>9 ||
       records<1 || workers<1 || workers>64 || offset+static_cast<Word98>(records)>0x100000000ULL ||
       offset>0xffffffffULL || !std::isfinite(phase) || !local_cut || !alias || !keys || !group_ids ||
       !grams || !amplitudes || !cuts || !output || !ambiguous) return 1;
    Cycle98 cycle{cap,route,amplitudes,grams,cuts};
    int failures=0;
    #pragma omp parallel num_threads(workers) if(workers>1) reduction(+:failures)
    {
        std::vector<unsigned> states(cells),groups(cells);
        std::vector<int> labels(cells);
        #pragma omp for schedule(static)
        for(int r=0;r<records;++r) {
            DomainRng84 local(experiment,0,157,offset+r), edges(experiment,0,158,offset+r), closure(experiment,0,159,offset+r);
            auto row=output+std::size_t(r)*8*cells;
            int min_right=cap,min_left=cap;
            for(int i=0;i<cells;++i) {
                auto u=local();unsigned column=u&((1u<<bits)-1);
                auto slot=(u>>bits)<local_cut[column] ? column : alias[column];
                auto key=keys[slot];states[i]=key;groups[i]=group_ids[slot];
                min_left=std::min(min_left,int(key&15));min_right=std::min(min_right,int((key>>12)&15));
                for(int j=0;j<4;++j) {
                    int v=(key>>(4*j))&15;
                    row[4*i+j]=route ? v : (v>0);
                    row[4*cells+4*i+j]=(key>>(16+4*j))&15;
                }
            }
            ambiguous[r]=(min_right+min_left)>0;
            for(int i=0;i<cells;++i) {
                int next=(i+1)%cells, n=(states[i]>>12)&15,m=states[next]&15;
                int label=(coherent && ambiguous[r] && i==cells-1)
                    ? cycle.close(states,groups,labels,min_right,min_left,phase,closure) : cycle.draw_edge(n,m,edges);
                if(label<0) { ++failures; break; }
                labels[i]=label;
                row[4*i+3]=route ? label : (n+m && label!=0);
                row[4*next]=route ? n+m-label : (n+m && label!=1);
            }
        }
    }
    return failures ? 2 : 0;
}
