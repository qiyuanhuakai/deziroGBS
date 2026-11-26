#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <omp.h>

static double joint106(int m,int i,int j,const double* raw,double gain){
    int ids[]={i,j,m+i,m+j};
    Eigen::Matrix4d a;
    for(int x=0;x<4;++x)for(int y=0;y<4;++y)
        a(x,y)=double(x==y)+gain*raw[std::size_t(ids[x])*2*m+ids[y]];
    return 1./std::sqrt(a.determinant());
}
static double single106(int m,int i,const double* raw,double v){
    int dim=2*m,b=i+m;
    return 1./std::sqrt((1+v*raw[std::size_t(i)*dim+i])*(1+v*raw[std::size_t(b)*dim+b])
        -v*v*raw[std::size_t(i)*dim+b]*raw[std::size_t(b)*dim+i]);
}
extern "C" __declspec(dllexport) int census106(int m,int spectra,int atoms,int workers,
    const double* x,const double* gain,const double* weight,double* marg,double* pair){
    if(m<2||spectra<1||atoms<1||workers<1)return 1;
    const std::size_t stride=std::size_t(2*m)*2*m;
    #pragma omp parallel for num_threads(workers) schedule(dynamic,4)
    for(int i=0;i<m;++i){
        double u=0;
        for(int k=0;k<atoms;++k){
            double val=weight[k];
            for(int s=0;s<spectra;++s)val*=single106(m,i,x+s*stride,gain[k]);
            u+=val;
        }
        marg[i]=u;pair[std::size_t(i)*m+i]=u;
        for(int j=0;j<i;++j){
            double p=0;
            for(int k=0;k<atoms;++k){
                double val=weight[k];
                for(int s=0;s<spectra;++s)val*=joint106(m,i,j,x+s*stride,gain[k]);
                p+=val;
            }
            pair[std::size_t(i)*m+j]=pair[std::size_t(j)*m+i]=p;
        }
    }
    return 0;
}
