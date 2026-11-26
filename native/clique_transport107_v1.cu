#include "continuation_gpu103_v1.cu"
struct Clique107 {int ids[4],radix[4],dim,click_offset,number_offset,states;};
__global__ void clique107(int tiles,int m,int n,int pnr,U64 seed,U32 off,const Clique107* plan,
    const float* rule,const U64* cdf,int* output){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n*tiles)return;
    int row=id/tiles,g=id%tiles;
    const auto &c=plan[g];int* rec=output+std::size_t(row)*m;
    int state=0,stride=1;
    for(int j=0;j<c.dim;++j){
        unsigned count=rec[c.ids[j]],range=pnr?c.radix[j]:2;
        if(count>=range)return;
        state+=count*stride;stride*=range;
    }
    int start=pnr?c.number_offset:c.click_offset;
    double chance=rule[start+state];
    if(chance<=0.)return;
    // Higher tile indices use successive pairs of distinct purpose values.
    uint4 v=block103(seed,off+row,g&65535,236+2*(g>>16)+pnr,0);
    if(unit103(low103(v))>=chance)return;
    int lo=0,hi=stride;U64 u=high103(v)>>12;
    while(lo<hi){int middle=(lo+hi)/2;if(cdf[start+middle]<=u)lo=middle+1;else hi=middle;}
    for(int j=0;j<c.dim;++j){int range=pnr?c.radix[j]:2;rec[c.ids[j]]=lo%range;lo/=range;}
}
extern "C" __declspec(dllexport) int clique107_launch(int tiles,int m,int n,int pnr,U64 seed,U32 off,const Clique107* plan,
    const float* rule,const U64* cdf,int* output,void* stream){
    if(tiles<1||m<1||n<1||236+2*(tiles>>16)+1>65535)return 1;
    int threads=n*tiles;
    clique107<<<(threads+255)/256,256,0,(cudaStream_t)stream>>>(tiles,m,n,pnr,seed,off,plan,rule,cdf,output);
    return cudaGetLastError();
}
