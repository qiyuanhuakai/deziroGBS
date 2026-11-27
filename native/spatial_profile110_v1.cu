#include "continuation_gpu103_v1.cu"

__device__ U64 profile_roll110(U64 seed,U32 row,U32 site,U32 draw){
    uint4 c=make_uint4(draw,11060,row,site);U32 a=U32(seed),b=U32(seed>>32);
    for(int i=0;i<10;++i){
        U32 x=0xD2511F53u*c.x,y=0xCD9E8D57u*c.z;
        c=make_uint4(__umulhi(0xCD9E8D57u,c.z)^c.y^a,y,__umulhi(0xD2511F53u,c.x)^c.w^b,x);
        a+=0x9E3779B9u;b+=0xBB67AE85u;
    }
    return low103(c);
}
template<class T>
__global__ void profile110(int groups,int width,U64 seed,U32 off,const int* frozen,const int* starts,
    const U32* freq,const unsigned short* profiles,const U32* weights,T* out,int* audit){
    int row=blockIdx.x,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    __shared__ int cumulative[9],warp_count[9][8],base[9],choice,mask_seed;
    __shared__ int2 scans[8];__shared__ U32 packet[2048];
    if(!tid)mask_seed=U32(profile_roll110(seed,row+off,0,0));
    __syncthreads();
    int g=((2*int(unsigned(mask_seed)&127)+1)*tid+int((unsigned(mask_seed)>>8)&255))&255;
    bool live=g<groups && !frozen[g];T* rec=out+size_t(row)*groups*width;int occupied=0;
    if(live)for(int j=0;j<width;++j)occupied|=(rec[g*width+j]>0)<<j;
    int k=live?__popc(occupied):9,within=0;
    for(int j=0;j<=width;++j){
        U32 members=__ballot_sync(0xffffffff,k==j);
        if(!lane)warp_count[j][warp]=__popc(members);
        if(k==j)within=__popc(members&((1u<<lane)-1));
    }
    __syncthreads();
    if(tid<=width){
        int sum=0;for(int w=0;w<8;++w){int here=warp_count[tid][w];warp_count[tid][w]=sum;sum+=here;}base[tid]=sum;
    }
    __syncthreads();
    if(!tid){
        int sum=0,free_clicks=0;
        for(int j=0;j<=width;++j){int cnt=base[j];free_clicks+=j*cnt;base[j]=sum;sum+=cnt;}
        int lo=starts[free_clicks],hi=starts[free_clicks+1];choice=-1;
        if(lo<hi){
            U32 beg=lo?freq[lo-1]:0,lim=freq[hi-1];
            U32 cut=beg+U32(__umul64hi(profile_roll110(seed,row+off,0,1),U64(lim-beg)));
            while(lo<hi){int mid=lo+(hi-lo)/2;if(freq[mid]<=cut)lo=mid+1;else hi=mid;}
            choice=lo;int seen=0;
            for(int j=0;j<=width;++j){seen+=profiles[size_t(lo)*(width+1)+j];cumulative[j]=seen;}
        }
        if(audit)audit[row]=choice>=0;
    }
    __syncthreads();if(choice<0)return;
    int wanted=0;
    if(live){
        int rank=base[k]+warp_count[k][warp]+within;
        while(wanted<width && cumulative[wanted]<=rank)++wanted;
    }
    int diff=live?wanted-k:0,send=max(0,-diff),receive=max(0,diff);
    int sx=send,rx=receive;
    for(int j=1;j<32;j*=2){
        int a=__shfl_up_sync(0xffffffff,sx,j),b=__shfl_up_sync(0xffffffff,rx,j);
        if(lane>=j){sx+=a;rx+=b;}
    }
    if(lane==31)scans[warp]=make_int2(sx,rx);
    __syncthreads();
    if(!tid)for(int w=1;w<8;++w){scans[w].x+=scans[w-1].x;scans[w].y+=scans[w-1].y;}
    __syncthreads();int os=sx-send+(warp?scans[warp-1].x:0),orr=rx-receive+(warp?scans[warp-1].y:0);
    for(int pass=0;pass<2;++pass){
        int count=pass?receive:send;
        for(int n=0;n<count;++n){
            U64 sum=0,c[8];for(int j=0;j<width;++j){
                bool active=(occupied>>j)&1;
                if(pass?!active:active)sum+=pass?1u+weights[g*width+j]:1u;
                c[j]=sum;
            }
            U64 cut=__umul64hi(profile_roll110(seed,row+off,g,2+pass*8+n),sum);
            int at=0;while(c[at]<=cut)++at;
            T& x=rec[g*width+at];
            if(pass){x=T(packet[orr+n]);occupied|=1<<at;}
            else{packet[os+n]=x;x=0;occupied&=~(1<<at);}
        }
        __syncthreads();
    }
}
extern "C" __declspec(dllexport) int profile_emit110(int n,int g,int width,int wide,U64 seed,U32 offset,
    const int* frozen,const int* starts,const U32* freq,const unsigned short* profiles,const U32* weights,
    void* output,int* audit,cudaStream_t stream){
    if(g<1||g>256||width<1||width>8)return 1;
    if(wide)profile110<<<n,256,0,stream>>>(g,width,seed,offset,frozen,starts,freq,profiles,weights,(U32*)output,audit);
    else profile110<<<n,256,0,stream>>>(g,width,seed,offset,frozen,starts,freq,profiles,weights,(unsigned char*)output,audit);
    return int(cudaGetLastError());
}
