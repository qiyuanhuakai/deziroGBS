#include "continuation_gpu103_v1.cu"

// Corrections never change zero or one photon. For a changed positive count,
// put its second normal strictly inside the desired quantile interval.
template<class F> __global__ void tail103(int pairs,int groups,int shots,U64 seed,U32 off,
        const int* edges,const int* base,const U64* keep,const int* ptr,const U64* cuts,
        const U32* labels,const int* caps,const int* starts,const double* levels,
        const double* representative,F* field){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=shots*pairs)return;
    int e=id%pairs,row=id/pairs;
    auto z=field+2*groups*row;
    int g[2]={edges[2*e],edges[2*e+1]},n[2];
    for(int k=0;k<2;++k){
        int j=g[k];
        auto t=levels+starts[j];
        double v=z[2*j+1];
        n[k]=z[2*j]<t[0]?0:v<t[1]?1:v<t[2]?2:3+upper103(t+3,caps[j]-3,v);
    }
    if(n[0]<2 && n[1]<2)return;
    int cell=base[e]+n[0]*(caps[g[1]]+1)+n[1];
    if(keep[cell]==(1ULL<<53))return;
    uint4 rng=block103(seed,off+row,e,197,0);
    if((low103(rng)>>11)<keep[cell])return;
    int first=ptr[cell],size=ptr[cell+1]-first;
    U32 code=labels[first+upper103(cuts+first,size,high103(rng)>>11)];
    for(int k=0;k<2;++k){
        int desired=(code>>(16*k))&65535;
        if(desired!=n[k])z[2*g[k]+1]=F(representative[starts[g[k]]+desired]);
    }
}
extern "C" __declspec(dllexport) int tail103_launch(int pairs,int groups,int n,U64 seed,U32 off,int wide,
        const int* edges,const int* base,const U64* keep,const int* ptr,const U64* cuts,const U32* labels,
        const int* cap,const int* start,const double* thresholds,const double* repr,void* latent,void* stream){
    int len=(pairs*n+127)/128;
    if(wide)tail103<<<len,128,0,(cudaStream_t)stream>>>(pairs,groups,n,seed,off,edges,base,keep,ptr,cuts,labels,cap,start,thresholds,repr,(double*)latent);
    else tail103<<<len,128,0,(cudaStream_t)stream>>>(pairs,groups,n,seed,off,edges,base,keep,ptr,cuts,labels,cap,start,thresholds,repr,(float*)latent);
    return int(cudaGetLastError());
}
