#include "continuation_gpu103_v1.cu"

__device__ U64 fused_rng112(U64 seed, U32 row, U32 window, U32 draw) {
    uint4 c=make_uint4(draw,11170,row,window);
    U32 a=U32(seed), b=U32(seed>>32);
    for(int i=0;i<10;i++){
        U32 x=0xD2511F53u*c.x,y=0xCD9E8D57u*c.z;
        c=make_uint4(__umulhi(0xCD9E8D57u,c.z)^c.y^a,y,__umulhi(0xD2511F53u,c.x)^c.w^b,x);
        a+=0x9E3779B9u;b+=0xBB67AE85u;
    }
    return low103(c);
}

__device__ int source112(unsigned tile,int lane,int window,U32 row,U64 seed,U32 active,
        const U32* cdf,const U64* move){
    int oldmask=(__ballot_sync(tile,active>0)>>(threadIdx.x%32-lane))&255;
    int desired=oldmask, shift=0;
    if(lane==0){
        const U32* h=cdf+(window*9+__popc(oldmask))*256;
        if(h[255] && (!move || (fused_rng112(seed,row,window,2)>>12)<move[256*window+oldmask])){
            U64 random=fused_rng112(seed,row,window,0);
            U32 cut=U32(__umul64hi(random,U64(h[255])));
            int lo=0,hi=255;
            while(lo<hi){int mid=(lo+hi)/2;if(h[mid]<=cut)lo=mid+1;else hi=mid;}
            desired=lo;
            shift=fused_rng112(seed,row,window,1)&7;
        }
    }
    desired=__shfl_sync(tile,desired,0,8);
    shift=__shfl_sync(tile,shift,0,8);
    int from=lane;
    if(((oldmask^desired)>>lane)&1){
        int donors=oldmask & ~desired, targets=desired & ~oldmask;
        if(!(desired&(1<<lane))){int t=donors;donors=targets;targets=t;}
        int rank=__popc(targets&((1<<lane)-1));
        for(int i=0;i<8;++i){
            int idx=(shift+i)&7;
            if((donors&(1<<idx)) && rank--==0){from=idx;break;}
        }
    }
    return from;
}

template<class T> __global__ void fused112(int n,U64 seed,U32 offset,
        const int* groups1,const U32* cdf1,const U64* move1,
        const int* groups2,const U32* cdf2,const U64* move2,T* records){
    // Both stages cover exactly the same non-protected sites. Each warp uses
    // four disjoint eight-lane masks; the final block barrier covers every lane.
    int row=blockIdx.x,tid=threadIdx.x,lane=tid%8,window=tid/8;
    T* data=records+size_t(row)*1152;
    __shared__ U32 banks[144][8];
    unsigned tile=0xffu<<(tid%32-lane);
    if(tid<144){
        int bank=groups1[tid];U32 vals[8],active=0;
        for(int j=0;j<8;++j){vals[j]=bank<0?0:data[bank*8+j];active|=vals[j];}
        int from=source112(tile,lane,window,offset+row,seed,active,cdf1,move1);
        for(int j=0;j<8;++j){
            U32 val=__shfl_sync(tile,vals[j],from,8);
            if(bank>=0)banks[bank][j]=val;
        }
    }
    __syncthreads();
    if(tid<144){
        int bank=groups2[tid];U32 vals[8],active=0;
        for(int j=0;j<8;++j){vals[j]=bank<0?0:banks[bank][j];active|=vals[j];}
        int from=source112(tile,lane,window,offset+row,seed^(U64(1)<<62),active,cdf2,move2);
        for(int j=0;j<8;++j){
            U32 val=__shfl_sync(tile,vals[j],from,8);
            if(bank>=0)data[bank*8+j]=T(val);
        }
    }
}

extern "C" __declspec(dllexport) int fused_emit112(int n,int wide,U64 seed,U32 offset,
        const int* g1,const U32* c1,const U64* m1,const int* g2,const U32* c2,const U64* m2,
        void* records,cudaStream_t stream){
    if(n<1 || U64(n)+offset>(U64(1)<<32))return 1;
    if(wide)fused112<<<n,160,0,stream>>>(n,seed,offset,g1,c1,m1,g2,c2,m2,(U32*)records);
    else fused112<<<n,160,0,stream>>>(n,seed,offset,g1,c1,m1,g2,c2,m2,(unsigned char*)records);
    return int(cudaGetLastError());
}
