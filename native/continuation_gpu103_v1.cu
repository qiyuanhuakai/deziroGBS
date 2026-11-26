#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

using U64=unsigned long long;
using U32=unsigned int;

// Same Philox-4x32-10 round and counter layout as gaussian_rng_domain84_v1.hpp.
// Group occupies the sector field and the three uses have distinct purpose codes.
__device__ uint4 block103(U64 seed,U32 row,U32 group,U32 purpose,U32 block){
    uint4 c=make_uint4(block,0,row,(group<<16)|purpose);
    U32 lo=U32(seed),hi=U32(seed>>32);
    #pragma unroll
    for(int i=0;i<10;++i){
        U32 a=0xD2511F53u*c.x,b=0xCD9E8D57u*c.z;
        c=make_uint4(__umulhi(0xCD9E8D57u,c.z)^c.y^lo,b,
                     __umulhi(0xD2511F53u,c.x)^c.w^hi,a);
        lo+=0x9E3779B9u;hi+=0xBB67AE85u;
    }
    return c;
}
__device__ U64 low103(uint4 c){return U64(c.x)|(U64(c.y)<<32);}
__device__ U64 high103(uint4 c){return U64(c.z)|(U64(c.w)<<32);}
__device__ double unit103(U64 u){return double(u>>11)*0x1.0p-53;}

template<class F> __global__ void normals103(int groups,int n,U64 seed,U32 offset,F* out){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n*groups)return;
    uint4 b=block103(seed,offset+id/groups,id%groups,194,0);
    double rad=sqrt(-2.*log1p(-unit103(low103(b))));
    double s,c;
    sincospi(2.*unit103(high103(b)),&s,&c);
    out[2*id]=F(rad*c);
    out[2*id+1]=F(rad*s);
}

template<class T,class V> __device__ int upper103(const T* arr,int size,V val){
    int lo=0,hi=size;
    while(lo<hi){
        int m=(lo+hi)/2;
        if(val<arr[m])hi=m;else lo=m+1;
    }
    return lo;
}

template<class F> __global__ void emit103(int groups,int rows,int route,U64 seed,U32 off,const F* z,
        const int* cap,const int* start,const double* cuts,const int* leaves,const U64* masks,
        const double* prior,const double* weights,const U64* shortcut,const unsigned char* direct,
        const unsigned char* alternate,unsigned short* out,unsigned short* ns,int* error){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=rows*groups)return;
    int g=id%groups,row=id/groups;
    const double* t=cuts+start[g];
    unsigned short dst[8]={};
    int n=0;
    if(z[2*id]>=t[0]){
        double u=z[2*id+1];
        n=u<t[1]?1:u<t[2]?2:3+upper103(t+3,cap[g]-3,u);
    }
    ns[id]=n;
    if(n){
        U64 u=low103(block103(seed,off+row,g,195,0));
        unsigned mask;
        std::size_t base=std::size_t(leaves[g])*256;
        if(n<=2){
            int bits=n==1?3:6;
            int k=g*72+(n==1?0:8)+int(u&((1u<<bits)-1));
            mask=(u>>bits)<shortcut[k]?direct[k]:alternate[k];
        }else{
            mask=upper103(masks+base+n*256,256,u>>11);
        }
        if(mask==0 || mask>255 || __popc(mask)>n){atomicExch(error,1);return;}
        if(!route){
            for(unsigned s=mask;s;s&=s-1)dst[__ffs(s)-1]=1;
        }else{
            unsigned draw=0;
            uint4 cache{};
            while(n && mask){
                if((mask&(mask-1))==0){dst[__ffs(mask)-1]+=n;break;}
                if(__popc(mask)==n){
                    for(unsigned s=mask;s;s&=s-1)++dst[__ffs(s)-1];
                    break;
                }
                const double* law=prior+base+(n-1)*256;
                double w[16]={},total=0;
                for(unsigned s=mask;s;s&=s-1){
                    int j=__ffs(s)-1;
                    w[2*j]=weights[g*8+j]*law[mask];
                    w[2*j+1]=weights[g*8+j]*law[mask^(1u<<j)];
                    total+=w[2*j]+w[2*j+1];
                }
                if(!(total>0)){atomicExch(error,2);return;}
                if(!(draw&1))cache=block103(seed,off+row,g,196,draw/2);
                double roll=unit103((draw++&1)?high103(cache):low103(cache))*total;
                int last=-1;
                for(int j=0;j<16;++j)if(w[j]>0){last=j;roll-=w[j];if(roll<0)break;}
                ++dst[last/2];
                if(last&1)mask^=1u<<(last/2);
                --n;
            }
        }
    }
    // Each thread emits one complete eight-way detector group.
    for(int j=0;j<8;++j)out[8*id+j]=dst[j];
}
extern "C" __declspec(dllexport) int normal103_launch(int g,int n,U64 seed,U32 off,int wide,void* output,void* stream){
    int len=(g*n+127)/128;
    if(wide)normals103<<<len,128,0,(cudaStream_t)stream>>>(g,n,seed,off,(double*)output);
    else normals103<<<len,128,0,(cudaStream_t)stream>>>(g,n,seed,off,(float*)output);
    return int(cudaGetLastError());
}
extern "C" __declspec(dllexport) int emit103_launch(int g,int n,int route,U64 seed,U32 off,int wide,
        const void* z,const int* cap,const int* start,const double* cuts,const int* leaves,const U64* masks,
        const double* prior,const double* weights,const U64* shortcut,const unsigned char* direct,
        const unsigned char* alt,unsigned short* out,unsigned short* counts,int* error,void* stream){
    int len=(g*n+127)/128;
    if(wide)emit103<<<len,128,0,(cudaStream_t)stream>>>(g,n,route,seed,off,(const double*)z,cap,start,cuts,leaves,masks,prior,weights,shortcut,direct,alt,out,counts,error);
    else emit103<<<len,128,0,(cudaStream_t)stream>>>(g,n,route,seed,off,(const float*)z,cap,start,cuts,leaves,masks,prior,weights,shortcut,direct,alt,out,counts,error);
    return int(cudaGetLastError());
}
