#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

using C = std::complex<double>;
struct Port { C a,b,d,inv00,inv01,inv11,v; };

// Spectrum-specific H = I + z A X (I + (1-z)X)^(-1) A^T, interleaved x,p.
extern "C" __declspec(dllexport) void moments104(int m,const C* first,const C* second,double* output){
    const int w=2*m;
    const C* matrix[2]={first,second};
    std::vector<Port> port[2];
    for(int s=0;s<2;++s){
        port[s].resize(m);
        auto h=matrix[s];
        for(int i=0;i<m;++i){
            const int x=2*i,y=x+1;
            C a=h[x*w+x],b=h[x*w+y],d=h[y*w+y],det=a*d-b*b;
            port[s][i]={a,b,d,d/det,-b/det,a/det,1./(std::sqrt(a)*std::sqrt(det/a))};
        }
    }
    C single=0.;
    for(int i=0;i<m;++i)single+=1.-port[0][i].v*port[1][i].v;
    double real=0.,imag=0.;
    #pragma omp parallel for num_threads(8) schedule(dynamic,8) reduction(+:real,imag)
    for(int i=0;i<m;++i){
        C subtotal=0.;
        for(int j=i+1;j<m;++j){
            C both=1.;
            for(int s=0;s<2;++s){
                const auto &a=port[s][i], &b=port[s][j];
                auto h=matrix[s]+2*i*w+2*j;
                C u=h[0],v=h[1],t=h[w],r=h[w+1];
                C f=a.inv00*u+a.inv01*t,g=a.inv01*u+a.inv11*t;
                C k=a.inv00*v+a.inv01*r,l=a.inv01*v+a.inv11*r;
                C aa=b.a-u*f-t*g,ab=b.b-v*f-r*g,dd=b.d-v*k-r*l;
                both*=a.v/(std::sqrt(aa)*std::sqrt(dd-ab*ab/aa));
            }
            subtotal+=1.-port[0][i].v*port[1][i].v-port[0][j].v*port[1][j].v+both;
        }
        real+=subtotal.real();imag+=subtotal.imag();
    }
    output[0]=single.real();output[1]=single.imag();output[2]=2*real;output[3]=2*imag;
}
