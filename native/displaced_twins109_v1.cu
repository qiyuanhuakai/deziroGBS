#include "continuation_gpu103_v1.cu"

__device__ double roll109(U64 seed,U32 row,U32 pair,U32 use) {
    // Disjoint domains for threshold word, twin number, and displaced number.
    return (double(low103(block103(seed,row,pair&65535,370+4*(pair>>16)+use,0))>>12)+.5)*0x1.0p-52;
}
__device__ int displaced_number(int n,double rate,double u) {
    if(rate==0)return n;
    // Fock matrix element of D(sqrt(rate)), evaluated using associated Laguerre
    // polynomials. Taking the log of the factorial prefactor avoids overflow.
    if(n>256||rate>150)return -1;
    double lograte=log(rate);
    for(int m=0;m<=1024;++m) {
        int k=min(n,m),delta=abs(n-m);
        double prev=0,laguerre=1;
        for(int j=1;j<=k;++j) {
            double next=((2*j-1+delta-rate)*laguerre-(j-1+delta)*prev)/j;
            prev=laguerre;laguerre=next;
        }
        double p=exp(-rate+lgamma(1.+k)-lgamma(1.+k+delta)+delta*lograte+2*log(fabs(laguerre)));
        if(!isfinite(p)||p>1.+1.e-12)return -1;
        if(u<p)return m;
        u-=p;
    }
    return -1;
}
template<class T>
__global__ void twins109(int m,int n,int pnr,U64 seed,U32 off,const double* lambda,const float* f,T* out,int* error) {
    int pairs=(m+1)/2,i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pairs*n)return;
    int row=i/pairs,k=i%pairs;bool tail=(2*k+1==m);
    auto field=f+size_t(row)*pairs*2;
    double x=field[k],y=field[pairs+k],rate=x*x+y*y;
    double l=lambda[k],vac=1-l;
    int a=0,b=0;
    if(pnr) {
        if(l>0)a=int(floor(log(roll109(seed,off+row,k,1))/log(l)));
        b=displaced_number(a,rate,roll109(seed,off+row,k,2));
        if(b<0){atomicMax(error,2);return;}
        if(sizeof(T)==1 && max(a,b)>255){atomicMax(error,1);return;}
    }else{
        double va=vac,vb=vac*exp(-vac*rate),both=va*exp(-rate);
        double u=roll109(seed,off+row,k,0);
        if(u>=both){
            if(u<va)b=1;
            else{a=1;b=(u>=va+vb-both);}
        }
    }
    T* dest=out+size_t(row)*m+2*k;
    if(tail)dest[0]=b;else{dest[0]=a;dest[1]=b;}
}
extern "C" __declspec(dllexport) int twin109(int m,int n,int pnr,int wide,U64 seed,U32 off,
    const double* l,const float* f,void* out,int* error,cudaStream_t stream) {
    int blocks=(((m+1)/2)*n+255)/256;
    if(wide)twins109<<<blocks,256,0,stream>>>(m,n,pnr,seed,off,l,f,(U32*)out,error);
    else twins109<<<blocks,256,0,stream>>>(m,n,pnr,seed,off,l,f,(unsigned char*)out,error);
    return int(cudaGetLastError());
}
