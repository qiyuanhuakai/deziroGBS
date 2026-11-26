#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <algorithm>
#include <omp.h>

using C106=std::complex<double>;
using Matrix106=Eigen::Matrix<C106,Eigen::Dynamic,Eigen::Dynamic>;
struct LawNode106 {int left,right,offset,dim,cutoff,order;};

// The generating function is E[prod z_j^n_j]. G=I+T^dagger(diag(z)-I)T.
// B is real, with lambda on the within-source exchange entries. For each
// coherent subtree the total-input-pair generating function is proportional
// to det(I-t B^dagger G B G^T)^(-1/2).
extern "C" __declspec(dllexport) int group106_law(int cases,int ports,int columns,int count,int workers,
    const C106* optics,const C106* points,const LawNode106* nodes,const int* ids,
    const double* lambda,const double* vacuum,const int* root,C106* result) {
    if(cases<1||ports<1||columns<2||count<1||workers<1)return 1;
    #pragma omp parallel num_threads(workers)
    {
        std::vector<C106> value(count);
        std::vector<std::vector<C106>> coefficients(count);
        for(int b=0;b<count;++b)coefficients[b].resize(nodes[b].order+1);
        Matrix106 gram(columns,columns),local,exchanged,h,power;
        int previous=-1;
        #pragma omp for schedule(static)
        for(int s=0;s<cases;++s){
            const bool reset=(s%64==0 || s!=previous+1 || previous<0);
            if(reset)gram.setIdentity();
            for(int j=0;j<ports;++j){
                C106 z=points[std::size_t(s)*ports+j]-(reset?C106(1):points[std::size_t(s-1)*ports+j]);
                if(z==C106(0))continue;
                const auto* u=optics+std::size_t(j)*columns;
                for(int a=0;a<columns;++a)for(int b=0;b<columns;++b)gram(a,b)+=std::conj(u[a])*z*u[b];
            }
            C106 total=1;
            for(int b=0;b<count;++b){
                const auto& n=nodes[b]; const int* cols=ids+n.offset;
                local.resize(n.dim,n.dim);exchanged.resize(n.dim,n.dim);
                for(int i=0;i<n.dim;++i)for(int j=0;j<n.dim;++j) {
                    local(i,j)=gram(cols[i],cols[j]);
                    exchanged(i,j)=lambda[cols[i]]*lambda[cols[j]]*gram(cols[i^1],cols[j^1]);
                }
                h.noalias()=exchanged*local.transpose();
                auto& c=coefficients[b];
                c[0]=vacuum[b];
                if(n.order){
                    std::vector<C106> traces(n.order+1);
                    power=h;
                    for(int k=1;k<=n.order;++k){
                        traces[k]=power.trace();
                        c[k]=0;
                        for(int j=1;j<=k;++j)c[k]+=traces[j]*c[k-j];
                        c[k]/=(2*k);
                        if(k<n.order)power=(power*h).eval();
                    }
                }
                if(n.left<0){
                    C106 det=(C106(1)-h(0,0))*(C106(1)-h(1,1))-h(1,0)*h(0,1);
                    value[b]=vacuum[b]/std::sqrt(det);
                }else{
                    value[b]=value[n.left]*value[n.right];
                    for(int k=0;k<=n.cutoff;++k){
                        value[b]+=c[k];
                        for(int j=0;j<=k;++j)value[b]-=coefficients[n.left][j]*coefficients[n.right][k-j];
                    }
                }
                if(root[b])total*=value[b];
            }
            result[s]=total; previous=s;
        }
    }
    return 0;
}
