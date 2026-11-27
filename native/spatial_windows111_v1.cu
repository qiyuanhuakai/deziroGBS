#include "continuation_gpu103_v1.cu"

__device__ U64 window_rng111(U64 seed, U32 row, U32 window, U32 draw) {
    uint4 c=make_uint4(draw,11170,row,window);
    U32 a=U32(seed), b=U32(seed>>32);
    for(int i=0;i<10;i++){
        U32 x=0xD2511F53u*c.x,y=0xCD9E8D57u*c.z;
        c=make_uint4(__umulhi(0xCD9E8D57u,c.z)^c.y^a,y,__umulhi(0xD2511F53u,c.x)^c.w^b,x);
        a+=0x9E3779B9u;b+=0xBB67AE85u;
    }
    return low103(c);
}

template<class T> __global__ void windows111(int n,int windows,U64 seed,U32 offset,
        const int* groups,const U32* cdf,const U64* move,T* records) {
    int gid=(blockIdx.x*blockDim.x+threadIdx.x)/8, lane=threadIdx.x%8;
    int row=gid/windows, window=gid%windows;
    if(row>=n)return;
    unsigned tile=0xffu<<(threadIdx.x%32-lane);
    int bank=groups[8*window+lane];
    T* data=records+size_t(row)*1152;
    U32 values[8],active=0;
    for(int j=0;j<8;++j){values[j]=bank<0?0:data[8*bank+j];active|=values[j];}
    int oldmask=(__ballot_sync(tile,active>0)>>(threadIdx.x%32-lane))&255;
    int desired=oldmask, shift=0;
    if(lane==0){
        const U32* h=cdf+(window*9+__popc(oldmask))*256;
        if(h[255] && (!move || (window_rng111(seed,row+offset,window,2)>>12)<move[256*window+oldmask])){
            U64 random=window_rng111(seed,row+offset,window,0);
            U32 cut=U32(__umul64hi(random,U64(h[255])));
            int lo=0,hi=255;
            while(lo<hi){int mid=(lo+hi)/2;if(h[mid]<=cut)lo=mid+1;else hi=mid;}
            desired=lo;
            shift=window_rng111(seed,row+offset,window,1)&7;
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
    for(int j=0;j<8;++j){
        U32 v=__shfl_sync(tile,values[j],from,8);
        if(bank>=0)data[8*bank+j]=T(v);
    }
}

extern "C" __declspec(dllexport) int window_emit111(int n,int windows,int wide,
        U64 seed,U32 offset,const int* groups,const U32* cdf,const U64* move,void* records,cudaStream_t stream){
    if(n<=0 || windows<=0 || windows>18)return 1;
    int blocks=(n*windows*8+255)/256;
    if(wide)windows111<<<blocks,256,0,stream>>>(n,windows,seed,offset,groups,cdf,move,(U32*)records);
    else windows111<<<blocks,256,0,stream>>>(n,windows,seed,offset,groups,cdf,move,(unsigned char*)records);
    return int(cudaGetLastError());
}
