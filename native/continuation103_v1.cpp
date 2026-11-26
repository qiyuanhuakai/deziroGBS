#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace continuation103 {
using Matrix=std::array<std::array<double,4>,4>;
using Vector=std::array<double,4>;
constexpr double invtau=1/(2*std::numbers::pi);

static double phi(double x){return .5*std::erfc(-x/std::sqrt(2.));}
static double density(double x,double y,double r){
    double d=(1-r)*(1+r);
    return invtau*std::exp(-.5*(x*x+y*y-2*r*x*y)/d)/std::sqrt(d);
}

struct Rule {
    int n;
    const double *nodes,*weights;
    double two(double x,double y,double r) const {
        double result=phi(x)*phi(y);
        if(r==0)return result;
        double sum=0;
        for(int k=0;k<n;++k)sum+=weights[k]*density(x,y,r*nodes[k]);
        return result+r*sum;
    }
    double derivative(const Vector& x,const Matrix& m,int size,int a,int b) const {
        const double r=m[a][b], d=1-r*r;
        double f=density(x[a],x[b],r);
        if(size==2 || f==0)return f;
        Vector z{},wa{},wb{},sd{};
        int ids[2],k=0;
        for(int i=0;i<size;++i)if(i!=a && i!=b){
            ids[k++]=i;
            wa[i]=(m[i][a]-r*m[i][b])/d;
            wb[i]=(m[i][b]-r*m[i][a])/d;
            sd[i]=std::sqrt(std::max(1e-16,1-wa[i]*m[i][a]-wb[i]*m[i][b]));
            z[i]=(x[i]-wa[i]*x[a]-wb[i]*x[b])/sd[i];
        }
        int i=ids[0];
        if(size==3)return f*phi(z[i]);
        int j=ids[1];
        double c=(m[i][j]-wa[i]*m[j][a]-wb[i]*m[j][b])/(sd[i]*sd[j]);
        return f*two(z[i],z[j],std::clamp(c,-.999999999999,.999999999999));
    }
    double cdf(const Vector& x,const Matrix& target,int size) const {
        double base=1.;
        for(int i=0;i<size;++i)base*=phi(x[i]);
        // Straight-line homotopy stays inside the positive definite cone.
        for(int k=0;k<n;++k){
            Matrix m{};
            for(int i=0;i<size;++i)for(int j=0;j<i;++j)m[i][j]=m[j][i]=nodes[k]*target[i][j];
            for(int i=0;i<size;++i)for(int j=0;j<i;++j)if(target[i][j]!=0)
                base+=weights[k]*target[i][j]*derivative(x,m,size,i,j);
        }
        return base;
    }
    double fit(const Vector& x,Matrix& m,int size,int a,int b,double goal) const {
        // Feasible interval by Schur complement on the other zero, one or two variables.
        int rest[2],k=0;
        for(int i=0;i<size;++i)if(i!=a && i!=b)rest[k++]=i;
        double va=1,vb=1,center=0;
        if(k){
            int i=rest[0],j=k==2?rest[1]:i;
            double r=k==2?m[i][j]:0;
            double u=m[a][i],v=k==2?m[a][j]:0,s=m[b][i],t=k==2?m[b][j]:0;
            va-=(u*u+v*v-2*r*u*v)/(1-r*r);
            vb-=(s*s+t*t-2*r*s*t)/(1-r*r);
            center=(u*s+v*t-r*(u*t+v*s))/(1-r*r);
        }
        double radius=std::sqrt(std::max(0.,va*vb))*.999999;
        double lo=center-radius,hi=center+radius;
        double base=cdf(x,m,size),value=0.;
        for(int step=0;step<25;++step){
            double q=base;
            if(value!=0){
                double sum=0;
                for(int j=0;j<n;++j){
                    m[a][b]=m[b][a]=value*nodes[j];
                    sum+=weights[j]*derivative(x,m,size,a,b);
                }
                q+=value*sum;
            }
            m[a][b]=m[b][a]=value;
            if(std::abs(q-goal)<2e-15)break;
            if(q>goal)hi=value;else lo=value;
            double next=value+(goal-q)/derivative(x,m,size,a,b);
            value=(next>lo && next<hi)?next:(lo+hi)/2;
        }
        m[a][b]=m[b][a]=value;
        return value;
    }
};

static void subcase(const double* x,const double* r,int a,int b,Vector& z,Matrix& m){
    z[0]=x[0];
    if(a==2)z[1]=x[1];
    z[a]=x[2];
    if(b==2)z[a+1]=x[3];
    for(int i=0;i<a;++i)for(int j=0;j<b;++j)m[i][a+j]=m[a+j][i]=r[2*i+j];
}
}
extern "C" __declspec(dllexport) void continuation103_fit(int pairs,const double* thresholds,
        const double* target,int n,const double* nodes,const double* weights,int workers,double* output){
    using namespace continuation103;
    Rule rule{n,nodes,weights};
    #pragma omp parallel for num_threads(workers) schedule(static)
    for(int k=0;k<pairs;++k){
        double* r=output+4*k;
        std::fill(r,r+4,0.);
        for(int a=1;a<=2;++a)for(int b=1;b<=2;++b){
            Vector x{};Matrix m{};
            subcase(thresholds+4*k,r,a,b,x,m);
            int idx=2*(a-1)+b-1;
            r[idx]=rule.fit(x,m,a+b,a-1,a+b-1,target[4*k+idx]);
        }
    }
}
extern "C" __declspec(dllexport) void continuation103_eval(int count,const double* thresholds,
        const double* corr,const int* dims,int n,const double* nodes,const double* weights,
        int workers,double* output){
    using namespace continuation103;
    Rule rule{n,nodes,weights};
    #pragma omp parallel for num_threads(workers) schedule(static)
    for(int i=0;i<count;++i){
        Vector x{};Matrix m{};
        int a=dims[2*i],b=dims[2*i+1];
        subcase(thresholds+4*i,corr+4*i,a,b,x,m);
        output[i]=rule.cdf(x,m,a+b);
    }
}
