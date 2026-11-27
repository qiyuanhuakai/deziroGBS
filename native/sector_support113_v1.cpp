// Coefficients of the zero-mean Gaussian number generating function.
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <omp.h>

using Matrix = Eigen::MatrixXd;
using Key = std::uint64_t;

extern "C" int sector113(int modes, int limit, int threads, const double* input,
                         double vacuum, double* output) {
    if (modes < 1 || modes > 16 || limit < 1 || limit > 128 || threads < 1) return 1;
    Eigen::Map<const Matrix> kernel(input, 2*modes, 2*modes);
    const int count = 1 << modes, stride = limit+2;
    std::vector<long double> table(count*stride, 0);
    int failed = 0;
    #pragma omp parallel for num_threads(threads) reduction(|:failed)
    for (int mask=0; mask<count; ++mask) {
        auto* row = &table[mask*stride];
        row[0] = vacuum;
        std::vector<int> idx;
        for (int j=0; j<2*modes; ++j) if (mask & (1<<(j%modes))) idx.push_back(j);
        if (idx.empty()) { row[limit+1] = vacuum; continue; }
        Eigen::SelfAdjointEigenSolver<Matrix> eig(kernel(idx,idx), Eigen::EigenvaluesOnly);
        if (eig.info()!=Eigen::Success) { failed = 1; continue; }
        std::vector<long double> trace(limit+1,0);
        long double logdet=0;
        for (double ev : eig.eigenvalues()) {
            if (std::abs(ev)>=1) { failed = 1; break; }
            logdet += std::log1p(-(long double)ev);
            long double power = ev;
            for (int n=1;n<=limit;++n) { trace[n]+=power; power*=ev; }
        }
        row[limit+1] = vacuum*std::exp(-logdet/2);
        for (int n=1; n<=limit; ++n) {
            for (int k=1; k<=n; ++k) row[n] += trace[k]*row[n-k];
            row[n] /= 2*n;
        }
    }
    if (failed) return 2;
    for (int bit=1; bit<count; bit*=2)
        for (int mask=bit; mask<count; ++mask) if (mask & bit)
            for (int n=0; n<stride; ++n) table[mask*stride+n] -= table[(mask-bit)*stride+n];
    std::copy(table.begin(),table.end(),output);
    return 0;
}

// D(z) = det(I - ZK). Each selected quadrature contributes one to its port degree.
static void minors(const Matrix& kernel, int modes, int common, int extended, int width,
                   int next, std::vector<int>& indices, int support, Key code,
                   int sign, std::vector<double>& coeff, const std::vector<int>& pow3) {
    for (int j=next; j<2*modes; ++j) {
        const int port=j/2, mask=support|(1<<port);
        const int degree=(int)indices.size()+1;
        if (degree>common && (degree>extended || std::popcount((unsigned)mask)>width)) continue;
        indices.push_back(port+(j%2)*modes);
        const Key key=code+pow3[port];
        coeff[key] += sign*kernel(indices,indices).eval().determinant();
        minors(kernel,modes,common,extended,width,j+1,indices,mask,key,-sign,coeff,pow3);
        indices.pop_back();
    }
}

using Index = std::unordered_map<Key,double>;
struct Term { int shift, ternary, maximum; };
static double convolution(const std::vector<Term>& terms, int pos, int degree, int used,
                          Key remainder, int dkey, const std::vector<double>& det,
                          const Index& previous) {
    if (pos==(int)terms.size()) {
        if (!used) return 0;
        return (degree-0.5*used)*det[dkey]*previous.at(remainder);
    }
    const auto& t=terms[pos];
    double sum=0;
    for (int k=0; k<=t.maximum; ++k)
        sum+=convolution(terms,pos+1,degree,used+k,remainder-(Key(k)<<t.shift),
                         dkey+k*t.ternary,det,previous);
    return sum;
}

// keys are all retained occupation vectors, degree-sorted and downward closed.
extern "C" int counts113(int modes, int common, int extended, int width, int threads,
                         const double* input, double vacuum, const Key* keys,
                         const int* offsets, double* values) {
    if (modes<1 || modes>14 || common<1 || extended<common || extended>15 ||
        width<1 || width>modes || threads<1) return 1;
    const Matrix kernel=Eigen::Map<const Matrix>(input,2*modes,2*modes);
    std::vector<int> pow3(modes+1,1);
    for (int j=0;j<modes;++j) pow3[j+1]=3*pow3[j];
    std::vector<double> det(pow3.back(),0);
    std::vector<int> indices;
    minors(kernel,modes,common,extended,width,0,indices,0,0,-1,det,pow3);
    Index previous;
    previous.reserve(offsets[extended+1]); previous.emplace(0,vacuum);
    values[0]=vacuum;
    for (int n=1; n<=extended; ++n) {
        #pragma omp parallel for num_threads(threads) schedule(dynamic,128)
        for (int k=offsets[n]; k<offsets[n+1]; ++k) {
            std::vector<Term> terms;
            for (int j=0;j<modes;++j) {
                const int digit=(keys[k]>>(4*j))&15;
                if (digit) terms.push_back({4*j,pow3[j],std::min(digit,2)});
            }
            values[k]=-convolution(terms,0,n,0,keys[k],0,det,previous)/n;
        }
        for (int k=offsets[n];k<offsets[n+1];++k) previous.emplace(keys[k],values[k]);
    }
    return 0;
}

static Key splitmix(Key& state) {
    Key x=(state+=0x9e3779b97f4a7c15ULL);
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}

// Consumes one uniform per independent inverse-CDF draw, returns every port.
extern "C" void draw113(int modes, int size, const double* cdf, const unsigned char* records,
                        int shots, Key seed, unsigned char* out) {
    for (int n=0;n<shots;++n) {
        const double u=(splitmix(seed)>>11)*0x1p-53;
        const auto i=std::upper_bound(cdf,cdf+size,u)-cdf;
        std::copy_n(records+std::min<int>(i,size-1)*modes,modes,out+n*modes);
    }
}
