#include "continuation_gpu103_v1.cu"

struct CollectiveStream110 {
    U64 seed;U32 record,cell,draw,route;
    __device__ U64 next(){
        U32 pos=draw++;uint4 c=make_uint4(pos/2,11000+route,record,cell);
        U32 a=U32(seed),b=U32(seed>>32);
        for(int i=0;i<10;++i){
            U32 x=0xD2511F53u*c.x,y=0xCD9E8D57u*c.z;
            c=make_uint4(__umulhi(0xCD9E8D57u,c.z)^c.y^a,y,__umulhi(0xD2511F53u,c.x)^c.w^b,x);
            a+=0x9E3779B9u;b+=0xBB67AE85u;
        }
        return pos&1?high103(c):low103(c);
    }
    __device__ U32 bounded(U32 limit){
        U32 threshold=(0u-limit)%limit;
        while(true){
            U64 v=U64(U32(next()))*limit;
            if(U32(v)>=threshold)return U32(v>>32);
        }
    }
};
template<class T>
__global__ void collective_draw110(int n,int cells,int d,int span,int size,int pnr,U64 seed,U32 off,
    const int* end,const U64* cdf,const unsigned short* values,T* output){
    int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=n*cells)return;
    int row=id/cells,cell=id%cells;CollectiveStream110 rng{seed,off+row,U32(cell),0,U32(pnr)};
    U64 cut=rng.next()>>12;int lo=0,hi=size;
    while(lo<hi){int mid=lo+(hi-lo)/2;if(cdf[mid]<=cut)lo=mid+1;else hi=mid;}
    auto v=values+size_t(lo)*d;
    T* x=output+size_t(id)*span;
    for(int j=0;j<d;++j){
        int begin=j?end[j-1]:0,width=end[j]-begin,count=v[j];
        if(width==1){x[begin]=T(count);continue;}
        if(pnr){
            for(int k=0;k<count;++k)++x[begin+rng.bounded(width)];
        }else{
            // Floyd sampling of a uniform subset of the precomputed cardinality.
            for(int k=width-count;k<width;++k){
                int h=rng.bounded(k+1);
                x[begin+(x[begin+h]?k:h)]=1;
            }
        }
    }
}
extern "C" __declspec(dllexport) int collective_emit110(int n,int cells,int d,int span,int size,
    int pnr,int wide,U64 seed,U32 offset,const int* end,const U64* cdf,const unsigned short* values,
    void* output,cudaStream_t stream){
    size_t length=size_t(n)*cells*span;
    auto err=cudaMemsetAsync(output,0,length*(wide?2:1),stream);
    if(err!=cudaSuccess)return int(err);
    int blocks=(n*cells+255)/256;
    if(wide)collective_draw110<<<blocks,256,0,stream>>>(n,cells,d,span,size,pnr,seed,offset,end,cdf,values,(unsigned short*)output);
    else collective_draw110<<<blocks,256,0,stream>>>(n,cells,d,span,size,pnr,seed,offset,end,cdf,values,(unsigned char*)output);
    return int(cudaGetLastError());
}
