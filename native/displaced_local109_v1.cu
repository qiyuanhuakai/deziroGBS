#include "continuation_gpu103_v1.cu"

// Each index owns two counter words, each purpose a separate field.
// The round occupies the high counter word and supports the full 32-bit index.
__device__ double uniform109(U64 seed,U32 row,U32 index,U32 purpose,U32 draw) {
    uint4 c=make_uint4(draw/2,109,row,index);
    U32 a=U32(seed),b=U32(seed>>32);
    // Encode purpose in the high counter word; no addition to the seed.
    c.y|=purpose<<16;
    for(int i=0;i<10;++i) {
        U32 x=0xD2511F53u*c.x,y=0xCD9E8D57u*c.z;
        c=make_uint4(__umulhi(0xCD9E8D57u,c.z)^c.y^a,y,__umulhi(0xD2511F53u,c.x)^c.w^b,x);
        a+=0x9E3779B9u;b+=0xBB67AE85u;
    }
    return (double((draw&1?high103(c):low103(c))>>12)+.5)*0x1.0p-52;
}
__device__ double vacancy(int mask,int g,int groups,int spectra,int masks,int n,int row,
                          const double* lut,const float* field) {
    double logp=lut[mask*groups+g];
    int slab=groups*masks;
    for(int s=0;s<spectra;++s) {
        auto v=field+(size_t(s)*n+row)*2*groups;
        double x=v[g],y=v[g+groups];
        auto f=lut+(1+2*s)*size_t(slab)+mask*groups+g;
        logp-=x*x*f[0]+y*y*f[slab];
    }
    return exp(logp);
}
__device__ int count109(double e0,double e1,double x,double y,double u) {
    double q0=1/(1+e0),q1=1/(1+e1);
    double p=sqrt(q0*q1)*exp(-x*x*q0-y*y*q1);
    double a0=e0*q0,a1=e1*q1,b0=x*x*q0*q0,b1=y*y*q1*q1;
    double s0=0,s1=0,h0=0,h1=0;
    for(int k=0;k<=1024;++k) {
        if(p < -1e-13 || !isfinite(p)) return -1;
        if(u<p)return k;
        u-=p;s0=p+a0*s0;s1=p+a1*s1;h0=s0+a0*h0;h1=s1+a1*h1;
        p=(.5*(a0*s0+a1*s1)+b0*h0+b1*h1)/(k+1);
    }
    return -1;
}
template<class T>
__global__ void read109(int groups,int width,int spectra,int n,int pnr,U64 seed,U32 off,
                        const float* field,const double* eig,const double* lut,const double* cdf,T* out,int* error) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=groups*n)return;
    int g=id%groups,row=id/groups,masks=1<<width;
    T* dst=out+size_t(id)*width;
    for(int j=0;j<width;++j)dst[j]=0;
    if(pnr) {
        int total=0;
        for(int s=0;s<spectra;++s) {
            auto v=field+(size_t(s)*n+row)*2*groups;
            auto e=eig+2*s*groups+g;
            int count=count109(e[0],e[groups],v[g],v[g+groups],uniform109(seed,off+row,s*groups+g,2,0));
            if(count<0){atomicMax(error,2);return;}
            total+=count;
        }
        for(int i=0;i<total;++i) {
            double u=uniform109(seed,off+row,g,3,i);
            int j=0;while(j+1<width&&u>=cdf[j*groups+g])++j;
            if(sizeof(T)==1 && dst[j]==255){atomicMax(error,1);return;}
            ++dst[j];
        }
        return;
    }
    double u=uniform109(seed,off+row,g,1,0),norm=1;
    if(u<vacancy(masks-1,g,groups,spectra,masks,n,row,lut,field))return;
    int ones=0,zeros=0;
    for(int j=0;j<width;++j) {
        int bit=1<<j;double p=0;
        for(int sub=ones;;sub=(sub-1)&ones) {
            double v=vacancy(sub|zeros|bit,g,groups,spectra,masks,n,row,lut,field);
            p+=__popc(sub)%2 ? -v:v;
            if(!sub)break;
        }
        if(p < -1e-12 || p>norm+1e-12 || !isfinite(p)){atomicMax(error,2);return;}
        p=fmax(0.,fmin(p,norm));
        if(u<p){zeros|=bit;norm=p;}else{u-=p;norm-=p;ones|=bit;dst[j]=1;}
    }
}
extern "C" __declspec(dllexport) int local109(int groups,int width,int spectra,int n,int pnr,int wide,
        U64 seed,U32 offset,const float* field,const double* eig,const double* lut,const double* cdf,
        void* output,int* error,cudaStream_t stream) {
    int blocks=(groups*n+255)/256;
    if(wide)read109<<<blocks,256,0,stream>>>(groups,width,spectra,n,pnr,seed,offset,field,eig,lut,cdf,(U32*)output,error);
    else read109<<<blocks,256,0,stream>>>(groups,width,spectra,n,pnr,seed,offset,field,eig,lut,cdf,(unsigned char*)output,error);
    return int(cudaGetLastError());
}
