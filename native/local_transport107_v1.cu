#include "continuation_gpu103_v1.cu"

__device__ int lookup107(const U64* end,int length,U64 roll){
    int lo=0,hi=length;
    U64 u=roll>>12;
    while(lo<hi){int mid=(lo+hi)/2;if(end[mid]<=u)lo=mid+1;else hi=mid;}
    return lo;
}
__device__ int move107(int state,int length,int group,U64 seed,U32 row,U32 purpose,
                      const float* move,const U64* end){
    double chance=move[std::size_t(group)*length+state];
    if(chance<=0.)return state;
    uint4 word=block103(seed,row,group,purpose,0);
    if(unit103(low103(word))>=chance)return state;
    return lookup107(end+std::size_t(group)*length,length,high103(word));
}
__global__ void clicks107(int groups,int n,int packed,U64 seed,U32 off,const int* pairs,
    const float* move,const U64* end,const unsigned char* raw,unsigned char* out){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n*groups)return;
    int row=id/groups,g=id%groups;
    int a=2*groups*row+pairs[2*g],b=2*groups*row+pairs[2*g+1];
    int s=move107(raw[a]+256*raw[b],65536,g,seed,off+row,231,move,end);
    if(packed){out[a]=s&255;out[b]=s>>8;}
    else{
        #pragma unroll
        for(int k=0;k<8;++k){out[8*a+k]=(s>>k)&1;out[8*b+k]=(s>>(k+8))&1;}
    }
}
template<class T>
__device__ void fan107(T* out,int new_count,const double* cdf,U64 seed,U32 row,int site){
    for(int j=0;j<8;++j)out[j]=0;
    for(int k=0;k<new_count;k+=2){
        uint4 v=block103(seed,row,site,233,k/2);
        double u=unit103(low103(v)),w=unit103(high103(v));
        int a=0,b=0;
        while(a<7&&u>=cdf[a])++a;out[a]++;
        if(k+1<new_count){while(b<7&&w>=cdf[b])++b;out[b]++;}
    }
}
template<class T>
__global__ void count107(int groups,int n,int cap,U64 seed,U32 off,const int* pairs,const double* weights,
    const float* move,const U64* end,T* out){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n*groups)return;
    int row=id/groups,g=id%groups;
    int a=pairs[2*g],b=pairs[2*g+1];
    T* left=out+8*(2*groups*row+a);T* right=out+8*(2*groups*row+b);
    U64 x=0,y=0;for(int k=0;k<8;++k){x+=left[k];y+=right[k];}
    if(x>=cap||y>=cap)return;
    int given=int(x+cap*y),s=move107(given,cap*cap,g,seed,off+row,232,move,end);
    if(s!=given){
        if(s%cap!=x)fan107(left,s%cap,weights+8*a,seed,off+row,a);
        if(s/cap!=y)fan107(right,s/cap,weights+8*b,seed,off+row,b);
    }
}
extern "C" __declspec(dllexport) int local107_launch(int groups,int n,int cap,int mode,U64 seed,U32 off,const int* pairs,const double* weights,
    const float* move,const U64* end,void* raw,void* out,void* stream){
    int blocks=(groups*n+255)/256;
    if(mode==0||mode==1)clicks107<<<blocks,256,0,(cudaStream_t)stream>>>(groups,n,mode,seed,off,pairs,move,end,(const unsigned char*)raw,(unsigned char*)out);
    else if(mode==2)count107<<<blocks,256,0,(cudaStream_t)stream>>>(groups,n,cap,seed,off,pairs,weights,move,end,(unsigned char*)raw);
    else if(mode==3)count107<<<blocks,256,0,(cudaStream_t)stream>>>(groups,n,cap,seed,off,pairs,weights,move,end,(U32*)raw);
    else return 1;
    return cudaGetLastError();
}
