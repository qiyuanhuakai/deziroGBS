#include <algorithm>
#include <cmath>
#include <vector>

// Two-variable optical PGF P=p0/sqrt(D), with D degree <=4 per bank for two spectra.
// Compare full fired-bin multiplicities after arbitrary unequal eight-way splitters.
extern "C" int joint115(int pairs,int cap,int threads,const double* polynomial,const double* vacuum,
                        const int* sites,const double* occupancy,double* words,double* loss) {
    if(pairs<1||cap<1||threads<1)return 1;
    int bad=0;const int side=cap+1;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,4) reduction(|:bad)
    for(int index=0;index<pairs;++index) {
        std::vector<double> p(side*side),partial(9*side,0.);
        const double* d=polynomial+25*index;
        p[0]=vacuum[index];
        for(int n=0;n<side;++n)for(int m=0;m<side;++m) {
            if(n+m==0)continue;
            double z=0;int degree=n+m;
            for(int a=0;a<=std::min(4,n);++a)
                for(int b=(a==0);b<=std::min(4,m);++b)
                    z+=(degree-.5*(a+b))*d[a*5+b]*p[(n-a)*side+m-b];
            p[n*side+m]=-z/degree;
        }
        double norm=0,neg=0;
        for(double& v:p){neg+=std::max(-v,0.);v=std::max(0.,v);norm+=v;}
        if(!(norm>0)||norm>1+1e-9||neg>1e-10)bad=2;
        loss[index]=1-norm;
        const double* a=occupancy+sites[2*index]*9*side;
        const double* b=occupancy+sites[2*index+1]*9*side;
        for(int i=0;i<9;++i)for(int n=0;n<side;++n)for(int m=0;m<side;++m)
            partial[i*side+m]+=a[i*side+n]*p[n*side+m];
        for(int i=0;i<9;++i)for(int j=0;j<9;++j) {
            double z=0;for(int m=0;m<side;++m)z+=partial[i*side+m]*b[j*side+m];
            words[81*index+i*9+j]=z/norm;
        }
    }
    return bad;
}
