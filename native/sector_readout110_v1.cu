#include "continuation_gpu103_v1.cu"

// High counter word 110 separates repair draws from the proposal sampler.
__device__ U64 roll110(U64 seed,U32 row,U32 purpose) {
    uint4 c=make_uint4(0,110,row,purpose);
    U32 a=U32(seed),b=U32(seed>>32);
    for(int r=0;r<10;++r) {
        U32 x=0xD2511F53u*c.x,y=0xCD9E8D57u*c.z;
        c=make_uint4(__umulhi(0xCD9E8D57u,c.z)^c.y^a,y,__umulhi(0xD2511F53u,c.x)^c.w^b,x);
        a+=0x9E3779B9u;b+=0xBB67AE85u;
    }
    return low103(c);
}

template<class T>
__global__ void sector110(int n,int m,int pn,int bits,int bins,U64 seed,U32 off,
    const int* ids,const int* anchor_bit,const U32* weights,const U64* move,const U64* cdf,
    T* output,int* error) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x,row=tid/32,lane=tid%32;
    if(row>=n)return;
    T* x=output+size_t(row)*m;int words=1<<bits;
    U64 old_total=0;
    if(pn) {
        for(int i=lane;i<m;i+=32)old_total+=x[i];
        for(int sh=16;sh;sh/=2)old_total+=__shfl_down_sync(0xffffffff,old_total,sh);
    }
    unsigned word=__ballot_sync(0xffffffff,lane<bits && x[ids[lane]]>0);
    if(lane || old_total>=bins)return;
    int key=int(old_total)*words+word;
    if((roll110(seed,off+row,1)>>12)>=move[key])return;
    U64 cut=roll110(seed,off+row,2)>>12;int lo=0,hi=bins*words;
    while(lo<hi){int mid=lo+(hi-lo)/2;if(cdf[mid]<=cut)lo=mid+1;else hi=mid;}
    if(lo>=bins*words){atomicMax(error,2);return;}
    int desired=lo/words;unsigned new_word=unsigned(lo%words);
    int now=int(old_total);
    for(int j=0;j<bits;++j){
        int i=ids[j],previous=x[i];
        int value=((new_word>>j)&1) ? (pn?max(1,previous):1):0;
        now+=value-previous;x[i]=T(value);
    }
    if(!pn || desired==now)return;
    bool add=desired>now;int delta=abs(desired-now);U64 mass=0;
    for(int i=0;i<m;++i) {
        int bit=anchor_bit[i];bool lit=(bit>=0) && ((new_word>>bit)&1);
        mass+=add ? ((bit<0||lit)?weights[i]:0) : (x[i]-int(lit));
    }
    bool uniform=add && mass==0;
    if(uniform)mass=m-bits+__popc(new_word);
    if(!mass){atomicMax(error,2);return;}
    U64 phase=__umul64hi(roll110(seed,off+row,3),mass),prefix=0,assigned=0;
    for(int i=0;i<m;++i){
        int bit=anchor_bit[i];bool lit=(bit>=0) && ((new_word>>bit)&1);
        U64 w=add ? ((bit<0||lit)?(uniform?1:weights[i]):0) : (x[i]-int(lit));
        if(!w)continue;
        prefix+=w;U64 next=(prefix*delta+phase)/mass;
        int change=int(next-assigned);assigned=next;
        int value=int(x[i])+(add?change:-change);
        if(value<0){atomicMax(error,2);return;}
        if(sizeof(T)==1 && value>255){atomicMax(error,1);return;}
        x[i]=T(value);
    }
    if(assigned!=delta)atomicMax(error,2);
}
extern "C" __declspec(dllexport) int transport110(int n,int m,int pn,int wide,int bits,int bins,
    U64 seed,U32 offset,const int* ids,const int* anchor_bit,const U32* weights,const U64* move,
    const U64* cdf,void* out,int* error,cudaStream_t stream) {
    int blocks=(n+7)/8;
    if(wide)sector110<<<blocks,256,0,stream>>>(n,m,pn,bits,bins,seed,offset,ids,anchor_bit,weights,move,cdf,(U32*)out,error);
    else sector110<<<blocks,256,0,stream>>>(n,m,pn,bits,bins,seed,offset,ids,anchor_bit,weights,move,cdf,(unsigned char*)out,error);
    return int(cudaGetLastError());
}
