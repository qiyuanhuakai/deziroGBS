#include "collective_simplex110_v1.hpp"

// Euler's derivative of P(z) = p0 / sqrt(D(z)); one polynomial may
// contain a product of determinants from distinguishable optical spectra.
extern "C" int mixed110(int d,int cut,int size,int terms,const int* powers,const double* coeff,
    double p0,const int* width,int workers,unsigned short* states,double* raw,double* pnr,double* clicks){
    if(d<1||d>8||cut<0||cut>512||size<1||size>8000000||terms<0||workers<1||!(p0>0&&p0<=1+1e-12))return 1;
    Simplex110 grid(d,cut,size);if(grid.choose[cut+d][d]!=size)return 1;
    std::vector<int> sums(size_t(terms)*d),degree(terms);
    for(int k=0;k<terms;++k){
        for(int j=0;j<d;++j){
            int p=powers[k*d+j];if(p<0)return 1;
            degree[k]+=p;sums[k*d+j]=degree[k];
        }
        if(!degree[k])return 1;
    }
    #pragma omp parallel for num_threads(workers) schedule(static)
    for(int n=0;n<size;++n)grid.decode(n);
    std::copy(grid.occupation.begin(),grid.occupation.end(),states);
    std::vector<long double> probability(size);probability[0]=p0;
    for(int n=1;n<=cut;++n){
        int lo=int(grid.choose[n+d-1][d]),hi=int(grid.choose[n+d][d]);
        std::vector<long double> factor(terms);
        for(int k=0;k<terms;++k)factor[k]=-coeff[k]*(1.L-degree[k]/(2.L*n));
        #pragma omp parallel for num_threads(workers) schedule(static)
        for(int a=lo;a<hi;++a){
            const auto* v=states+size_t(a)*d;std::array<int,8> bars{};
            for(int j=0,prefix=0;j<d;++j){prefix+=v[j]+1;bars[j]=prefix-1;}
            long double sum=0;
            for(int k=0;k<terms;++k){
                if(degree[k]>n)continue;
                int pred=0;bool valid=true;
                for(int j=0;j<d;++j){
                    if(v[j]<powers[k*d+j]){valid=false;break;}
                    pred+=int(grid.choose[bars[j]-sums[k*d+j]][j+1]);
                }
                if(valid)sum+=factor[k]*probability[pred];
            }
            probability[a]=sum;
        }
    }
    std::vector<double> nonnegative(size);
    for(int n=0;n<size;++n){raw[n]=double(probability[n]);nonnegative[n]=std::max(0.,raw[n]);}
    std::vector<double> eff(d,1.);transform110(grid,states,nonnegative.data(),width,eff.data(),pnr,clicks,workers);
    return 0;
}
