#include "collective_simplex110_v1.hpp"
using Complex110=std::complex<long double>;
extern "C" int collective110(int d,int cut,int size,const double* matrix,double p0,const int* width,
    const double* efficiency,unsigned short* states,double* emitted,double* pnr,double* click,int workers){
    if(d<1||d>8||cut<0||cut>512||size<1||size>8000000||workers<1||!(p0>0&&p0<=1))return 1;
    for(int i=0;i<d;++i)if(width[i]<1||!(efficiency[i]>=0&&efficiency[i]<=1))return 1;
    Simplex110 grid(d,cut,size);
    if(grid.choose[cut+d][d]!=size)return 1;
    std::vector<Complex110> amplitude(size);amplitude[0]=std::sqrt(p0);
    std::vector<long double> root(cut+1);for(int n=0;n<=cut;++n)root[n]=std::sqrt((long double)n);
    #pragma omp parallel for num_threads(workers) schedule(static)
    for(int n=0;n<size;++n)grid.decode(n);
    std::copy(grid.occupation.begin(),grid.occupation.end(),states);
    for(int sector=2;sector<=cut;sector+=2){
        int lo=int(grid.choose[sector+d-1][d]),hi=int(grid.choose[sector+d][d]);
        #pragma omp parallel for num_threads(workers) schedule(static)
        for(int n=lo;n<hi;++n){
            const auto* v=states+size_t(n)*d;
            int mark=0;while(!v[mark])++mark;
            Complex110 sum=0;
            for(int j=0;j<d;++j){
                int available=v[j]-(mark==j);if(!available)continue;
                auto b=matrix+2*(mark*d+j);
                sum+=Complex110(b[0],b[1])*root[available]*amplitude[grid.index(v,mark,j)];
            }
            amplitude[n]=sum/root[v[mark]];
        }
    }
    for(int n=0;n<size;++n)emitted[n]=double(std::norm(amplitude[n]));
    transform110(grid,states,emitted,width,efficiency,pnr,click,workers);
    return 0;
}
