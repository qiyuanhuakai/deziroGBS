#include <algorithm>
#include <cmath>

// Covariance of normalized bivariate Hermite basis functions of equal degree.
// Store degree k as a dense (k+1)^2 matrix at the next consecutive offset.
extern "C" __declspec(dllexport) int chaos103_transfer(int degree,const double* r,double* out){
    if(degree<0 || degree>64 || !r || !out)return 1;
    double* prev=out;
    prev[0]=1;
    double* m=prev+1;
    for(int k=1;k<=degree;++k){
        for(int i=0;i<=k;++i){
            int a=(i>k-i);
            int ii=i-a;
            int multiplicity=a?i:k-i;
            for(int j=0;j<=k;++j){
                double value=0;
                if(j<k)value+=r[2*a]*std::sqrt(double(k-j)/multiplicity)*prev[ii*k+j];
                if(j>0)value+=r[2*a+1]*std::sqrt(double(j)/multiplicity)*prev[ii*k+j-1];
                m[i*(k+1)+j]=value;
            }
        }
        prev=m;
        m+=(k+1)*(k+1);
    }
    return 0;
}
