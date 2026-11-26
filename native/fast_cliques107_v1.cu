#include "continuation_gpu103_v1.cu"
#include <cuda_fp16.h>
struct Block107 {int ids[16],radix[4],dim,start,size;};

__device__ uint4 tile_rng107(U64 seed,U32 row,U32 index,U32 purpose,U32 block){
    return block103(seed,row,index&65535,purpose+8*(index>>16),block);
}
__device__ double open107(U64 u){return (double(u>>12)+.5)*0x1.0p-52;}

template<class F>
__global__ void normals107(int k,int ns,int n,U64 seed,U32 off,int atoms,const double* g,const double* cdf,F* out,double* gain){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n*k*ns)return;
    int j=id%k,s=id/(n*k),row=(id/k)%n;
    uint4 v=tile_rng107(seed,off+row,s*k+j,221,0);
    double v0=sqrt(-2.*log(open107(low103(v)))),si,co;sincos(6.2831853071795864769*open107(high103(v)),&si,&co);
    out[std::size_t(2)*id]=F(float(v0*co));out[std::size_t(2)*id+1]=F(float(v0*si));
    if(s==0&&j==0){
        double u=open107(low103(block103(seed,off+row,0,222,0)));int i=0;while(i+1<atoms&&u>=cdf[i])++i;gain[row]=g[i];
    }
}
template<class F,class T>
__global__ void fast107(int blocks,int m,int spectra,int n,int pnr,U64 seed,U32 off,const Block107* layout,const F* field,
    const double* gain,const float* rule,const U64* cdf,T* out,int* status){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n*blocks)return;
    int row=id/blocks,block=id%blocks;const auto& tile=layout[block];T* dest=out+std::size_t(row)*m;
    bool interior=true;int state=0,stride=1;
    for(int j=0;j<tile.dim;++j){
        int port=tile.ids[j];double intensity=0;
        for(int s=0;s<spectra;++s){
            const F* z=field+(std::size_t(s)*n+row)*2*m;
            double x=float(z[port]),y=float(z[m+port]);intensity+=x*x+y*y;
        }
        double rate=gain[row]*intensity;
        if(!(rate>=0.)||rate>1.e8){atomicMax(status,2);return;}
        uint4 v=tile_rng107(seed,off+row,port,241,0);
        double u=open107(low103(v));
        unsigned ct=0;
        if(!pnr)ct=(u>exp(-rate));
        else if(rate<=8.){
            double p=exp(-rate),cum=p;
            while(u>cum){p*=rate/(++ct);cum+=p;if(p<1e-20 && cum<1.-2.e-15){atomicMax(status,2);return;}if(p<1.e-20)break;}
        }else{
            // Arrival times provide a stable sampler at large optical rates.
            double t=-log(u);
            while(t<rate){++ct;if(!(ct&1))v=tile_rng107(seed,off+row,port,241,ct/2);t-=log(open107((ct&1)?high103(v):low103(v)));}
        }
        if(sizeof(T)==1 && ct>255)atomicMax(status,1);
        dest[port]=T(ct);
        if(pnr){
            int radix=tile.radix[j];if(ct>=radix)interior=false;
            if(interior)state+=stride*ct;stride*=radix;
        }else state|=ct<<j;
    }
    if(!interior)return;
    double r=rule[tile.start+state];if(!(r>0))return;
    uint4 v=tile_rng107(seed,off+row,block,243+pnr,0);
    if(unit103(low103(v))>=r)return;
    U64 draw=high103(v)>>12;int lo=0,hi=tile.size;
    while(lo<hi){int mid=(lo+hi)/2;if(cdf[tile.start+mid]<=draw)lo=mid+1;else hi=mid;}
    for(int j=0;j<tile.dim;++j){
        int base=pnr?tile.radix[j]:2;
        unsigned count=lo%base;
        if(sizeof(T)==1 && count>255)atomicMax(status,1);
        dest[tile.ids[j]]=T(count);lo/=base;
    }
}
extern "C" __declspec(dllexport) int normals107_launch(int k,int ns,int n,U64 seed,U32 off,int atoms,const double* g,const double* cdf,
    int half,void* out,double* gain,void* stream){
    int grid=(k*ns*n+255)/256;
    if(half)normals107<<<grid,256,0,(cudaStream_t)stream>>>(k,ns,n,seed,off,atoms,g,cdf,(__half*)out,gain);
    else normals107<<<grid,256,0,(cudaStream_t)stream>>>(k,ns,n,seed,off,atoms,g,cdf,(float*)out,gain);
    return cudaGetLastError();
}
extern "C" __declspec(dllexport) int fast107_launch(int blocks,int m,int ns,int n,int pnr,U64 seed,U32 off,int half,int wide,const Block107* layout,
    const void* field,const double* gain,const float* rule,const U64* cdf,void* out,int* status,void* stream){
    int grid=(blocks*n+255)/256;
    #define RUN107(F,T) fast107<<<grid,256,0,(cudaStream_t)stream>>>(blocks,m,ns,n,pnr,seed,off,layout,(const F*)field,gain,rule,cdf,(T*)out,status)
    if(half){if(wide){RUN107(__half,U32);}else{RUN107(__half,unsigned char);}}
    else{if(wide){RUN107(float,U32);}else{RUN107(float,unsigned char);}}
    #undef RUN107
    return cudaGetLastError();
}
