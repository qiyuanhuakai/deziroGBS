#include <cmath>
#include <complex>
#include <cstdint>
#include <unordered_map>
#include <vector>

// Normalized Fock-amplitude recurrence for a zero-displacement pure Gaussian
// state. The caller supplies occupations in ascending even total degree.
extern "C" __declspec(dllexport) int audit95_fock(int m,int n,const double* b,double vacuum,
        const unsigned char* counts,double* prob) {
    if(m<1||m>10||n<1||!b||!counts||!prob||vacuum<=0) return 1;
    try {
        using C=std::complex<double>;
        std::unordered_map<std::uint64_t,C> previous;
        previous.reserve(n);
        for(int k=0;k<n;++k) {
            const auto* row=counts+std::size_t(k)*m;
            std::uint64_t code=0;int i=-1;
            for(int j=0;j<m;++j) {
                if(row[j]>63) return 2;
                code|=std::uint64_t(row[j])<<(6*j);
                if(row[j]&&i<0) i=j;
            }
            C amplitude=std::sqrt(vacuum);
            if(i>=0) {
                amplitude=0.;
                for(int j=0;j<m;++j) {
                    int multiplicity=int(row[j])-(i==j);
                    if(multiplicity<1) continue;
                    auto ancestor=code-(1ULL<<(6*i))-(1ULL<<(6*j));
                    auto it=previous.find(ancestor);
                    if(it==previous.end()) return 3;
                    int idx=2*(i*m+j);
                    amplitude+=C(b[idx],b[idx+1])*std::sqrt(double(multiplicity)/row[i])*it->second;
                }
            }
            previous.emplace(code,amplitude);prob[k]=std::norm(amplitude);
        }
        return 0;
    } catch(...) {return 4;}
}

#include "gaussian_loop_residual95_v1.cpp"
// Project eight bits in eight distinct spatial groups after global conditioning.
extern "C" __declspec(dllexport) int audit95_mask(void* ptr,const int* group,const double* chance,
        int weighted,int workers,double* law) {
    if(!ptr||!group||!chance||!law||workers<1||workers>64) return 1;
    const auto& plan=*static_cast<const TotalPlan95*>(ptr);const auto& p=*plan.base;
    std::vector<int> cover;std::vector<int> marked(2*plan.leaves);
    for(int j=0;j<8;++j) {
        if(group[j]<0||group[j]>=p.m||marked[plan.leaves+group[j]]) return 1;
        for(int i=plan.leaves+group[j];i>0;i/=2) marked[i]=1;
    }
    for(int i=2;i<2*plan.leaves;++i) if(marked[i/2]&&!marked[i]) cover.push_back(i);
    std::fill(law,law+256,0.);
    std::vector<double> weight(plan.limit+1,1.);
    if(weighted) for(int i=0;i<=plan.limit;++i) weight[i]=plan.q[i]>0?plan.target[i]/plan.q[i]:0.;
    #pragma omp parallel num_threads(workers)
    {
        std::array<double,256> partial{};
        std::vector<PolyFilter> patterns(256);
        #pragma omp for schedule(static)
        for(int c=0;c<p.centers;++c) {
            const auto& tree=plan.tree[c];
            PolyFilter outer{1.};
            for(int node:cover) outer=convolution95(outer,tree[node],plan.limit+1);
            patterns[0]={1.};int size=1;
            for(int j=0;j<8;++j) {
                auto& leaf=tree[plan.leaves+group[j]];
                PolyFilter yes(leaf.size()),no(leaf.size());
                for(int k=0;k<int(leaf.size());++k) {
                    double v=chance[j*(p.levels+1)+k];yes[k]=v*leaf[k];no[k]=(1-v)*leaf[k];
                }
                size+=int(leaf.size())-1;
                for(int mask=0;mask<(1<<j);++mask) {
                    patterns[mask+(1<<j)]=convolution95(patterns[mask],yes,plan.limit+1);
                    patterns[mask]=convolution95(patterns[mask],no,plan.limit+1);
                }
            }
            std::vector<double> integrated(size,0.);
            for(int k=0;k<size;++k) for(int t=0;t<std::min<int>(outer.size(),plan.limit+1-k);++t)
                integrated[k]+=outer[t]*weight[t+k];
            for(int mask=0;mask<256;++mask)
                for(int k=0;k<int(patterns[mask].size());++k) partial[mask]+=patterns[mask][k]*integrated[k];
        }
        #pragma omp critical
        for(int j=0;j<256;++j) law[j]+=partial[j]/p.centers;
    }
    return 0;
}
