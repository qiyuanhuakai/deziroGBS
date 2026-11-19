// Independent binary128 threshold oracle using incremental Cholesky factorizations.
#include <quadmath.h>
#include <cmath>
#include <vector>
namespace {
using Q=__float128;
struct Factor {
    const double *cov,*mean;
    int m;
    Q ch[36][36]{},shift[36]{};
    int ix[36]{};
    std::vector<Q>& vacuum;
    Q extend(int n,int id,int& error) {
        ix[n]=id;
        for(int k=0;k<n;++k) {
            Q v=cov[id*(2*m)+ix[k]];
            for(int j=0;j<k;++j)v-=ch[n][j]*ch[k][j];
            ch[n][k]=v/ch[k][k];
        }
        Q diag=cov[id*(2*m)+id],displacement=mean[id];
        for(int j=0;j<n;++j){diag-=ch[n][j]*ch[n][j];displacement-=ch[n][j]*shift[j];}
        if(!(diag>0)){error=1;return 0;}
        ch[n][n]=sqrtq(diag);shift[n]=displacement/ch[n][n];
        return expq(-shift[n]*shift[n]/2)/ch[n][n];
    }
    void visit(int next,int count,int mask,Q mass,int& error) {
        vacuum[mask]*=mass;
        for(int i=next;i<m&&!error;++i) {
            Q x=extend(count,i,error);
            Q y=extend(count+1,i+m,error);
            visit(i+1,count+2,mask|(1<<i),mass*x*y,error);
        }
    }
};
}
extern "C" __declspec(dllexport) int oracle96(int m,int sectors,const double* q,const double* mu,
        double* pmf,double* diag) {
    if(m<1||m>18||sectors<1||sectors>8||!q||!mu||!pmf||!diag)return 1;
    int size=1<<m,error=0;std::vector<Q> before(size,1);
    for(int s=0;s<sectors&&!error;++s){
        Factor f{q+s*4*m*m,mu+s*2*m,m,{},{},{},before};f.visit(0,0,0,1,error);
    }
    if(error)return 2;
    std::vector<Q> law(size);
    for(int i=0;i<size;++i)law[i]=before[size-1-i];
    for(int bit=1;bit<size;bit*=2)for(int i=0;i<size;++i)if(i&bit)law[i]-=law[i^bit];
    Q neg=0,total=0;
    for(int i=0;i<size;++i){
        if(law[i]<0)neg-=law[i];total+=law[i];pmf[i]=double(law[i]);
    }
    diag[0]=double(neg);diag[1]=double(total);return 0;
}
