#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <bit>
#include <algorithm>
#include <numeric>

// Count generating function for d observed modes in independent Gaussian spectra.
// With H=I-Q^{-1}, each determinant det(I-ZH) is a small polynomial.
// Euler differentiation of P(z)^2 D(z)=P(0)^2 gives the coefficient recurrence.
extern "C" __declspec(dllexport) int counts107(int d,int spectra,const double* q,const int* sizes,double* out){
    using L=long double;
    using Mat=Eigen::Matrix<L,Eigen::Dynamic,Eigen::Dynamic>;
    if(d<1||d>4||spectra<1||spectra>4)return 1;
    int m=2*d,degree=2*spectra+1,poly_size=1,cells=1;
    std::vector<int> stride(d),ps(d);
    for(int j=0;j<d;++j){
        if(sizes[j]<1||cells>2000000/sizes[j])return 2;
        stride[j]=cells;cells*=sizes[j];ps[j]=poly_size;poly_size*=degree;
    }
    std::vector<L> poly(poly_size,0);poly[0]=1;
    L vacuum=1;
    for(int s=0;s<spectra;++s){
        Mat a(m,m);
        for(int i=0;i<m;++i)for(int j=0;j<m;++j)a(i,j)=q[(s*m+i)*m+j];
        if(a.determinant()<=0)return 3;
        vacuum/=std::sqrt(a.determinant());
        Mat h=Mat::Identity(m,m)-a.inverse();
        std::vector<L> next(poly_size,0);
        for(int mask=0;mask<(1<<m);++mask){
            std::vector<int> idx;int shift=0;
            for(int j=0;j<m;++j)if(mask&(1<<j)){idx.push_back(j);shift+=ps[j%d];}
            Mat sub(idx.size(),idx.size());
            for(int i=0;i<int(idx.size());++i)for(int j=0;j<int(idx.size());++j)sub(i,j)=h(idx[i],idx[j]);
            L value=idx.empty()?1:sub.determinant();
            if(idx.size()%2)value=-value;
            for(int k=0;k+shift<poly_size;++k)if(poly[k]!=0)next[k+shift]+=value*poly[k];
        }
        poly.swap(next);
    }
    struct Term {int index,order;std::array<int,4> exp;L coefficient;};
    std::vector<Term> terms;
    for(int k=1;k<poly_size;++k)if(poly[k]!=0){
        Term t{};t.coefficient=poly[k];int v=k;
        for(int j=0;j<d;++j){t.exp[j]=v%degree;v/=degree;t.order+=t.exp[j];t.index+=t.exp[j]*stride[j];}
        terms.push_back(t);
    }
    std::vector<L> p(cells);p[0]=vacuum;
    for(int k=1;k<cells;++k){
        std::array<int,4> e{};int v=k,n=0;
        for(int j=0;j<d;++j){e[j]=v%sizes[j];v/=sizes[j];n+=e[j];}
        L acc=0;
        for(const auto& t:terms){
            bool valid=true;
            for(int j=0;j<d;++j)if(t.exp[j]>e[j]){valid=false;break;}
            if(valid)acc+=(2*n-t.order)*t.coefficient*p[k-t.index];
        }
        p[k]=-acc/(2*n);
    }
    for(int k=0;k<cells;++k)out[k]=double(p[k]);
    return 0;
}
