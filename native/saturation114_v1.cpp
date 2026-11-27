#include <algorithm>
#include <cmath>
#include <vector>

// Raw column/quadrature order x[ports], p[ports]. Return detector means and
// joint covariances NN, HH, NH summed into disjoint banks of eight bins.
extern "C" int saturation114(int m,int spectra,int fan,const double* x,double* means,double* cov) {
    if(m<1||spectra<1||fan<1||m%fan)return 1;
    int d=2*m,g=m/fan;
    auto read=[&](int s,int i,int j,int a,int b){return x[(size_t(s)*d+i+a*m)*d+j+b*m];};
    std::vector<double> mu(m),p0(m,1),inverse(size_t(m)*spectra*4);
    for(int i=0;i<m;++i) {
        for(int s=0;s<spectra;++s) {
            double a=1+read(s,i,i,0,0),b=read(s,i,i,0,1),c=1+read(s,i,i,1,1);
            double det=a*c-b*b;
            if(!(det>0))return 2;
            double* q=&inverse[(size_t(i)*spectra+s)*4];
            q[0]=c/det;q[1]=q[2]=-b/det;q[3]=a/det;
            p0[i]/=std::sqrt(det);mu[i]+=(a+c-2)/2;
        }
    }
    for(int i=0;i<m;++i){means[i]=mu[i];means[m+i]=1-p0[i];}
    int bad=0;
    #pragma omp parallel for num_threads(8) reduction(|:bad)
    for(int bi=0;bi<g;++bi)for(int bj=0;bj<g;++bj) {
        double nn=0,nh=0,hh=0;
        for(int i=bi*fan;i<(bi+1)*fan;++i)for(int j=bj*fan;j<(bj+1)*fan;++j) {
            double log_ratio=0,drop=0;
            for(int s=0;s<spectra;++s) {
                double b[2][2],w[2][2]{};
                const double* inv=&inverse[(size_t(j)*spectra+s)*4];
                for(int a=0;a<2;++a)for(int k=0;k<2;++k){
                    b[a][k]=read(s,i,j,a,k);nn+=.5*b[a][k]*b[a][k];
                }
                if(i==j)continue;
                for(int a=0;a<2;++a)for(int k=0;k<2;++k)
                    for(int u=0;u<2;++u)for(int v=0;v<2;++v)
                        w[a][k]+=b[a][u]*inv[2*u+v]*b[k][v];
                drop+=.5*(w[0][0]+w[1][1]);
                double a=1+read(s,i,i,0,0),z=read(s,i,i,0,1),c=1+read(s,i,i,1,1);
                double det=a*c-z*z;
                double change=(-c*w[0][0]-a*w[1][1]+w[0][0]*w[1][1]+2*z*w[0][1]-w[0][1]*w[0][1])/det;
                if(change<=-1){bad=4;continue;}
                log_ratio-=.5*std::log1p(change);
            }
            if(i==j){nn+=mu[i];nh+=p0[i]*mu[i];hh+=p0[i]*(1-p0[i]);}
            else {nh+=p0[j]*drop;hh+=p0[i]*p0[j]*std::expm1(log_ratio);}
        }
        int index=bi*g+bj;
        cov[index]=nn;cov[g*g+index]=hh;cov[2*g*g+index]=nh;
    }
    return bad;
}
