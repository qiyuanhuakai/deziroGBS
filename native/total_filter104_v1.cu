// Reuse the same Philox permutation and its non-overlapping purpose layout.
#include "continuation_gpu103_v1.cu"

__global__ void total104(int groups,int shots,U64 seed,U32 first,const unsigned short* out,
                         const unsigned short* counts,const double* probabilities,int nc,int nn,
                         int* totals,int* accepted){
    int row=blockIdx.x;
    int lane=threadIdx.x,click=0,photon=0;
    for(int i=lane;i<groups;i+=blockDim.x)photon+=counts[row*groups+i];
    for(int i=lane;i<8*groups;i+=blockDim.x)click+=out[8*row*groups+i]!=0;
    __shared__ int clicks[128],photons[128];
    clicks[lane]=click;
    photons[lane]=photon;
    __syncthreads();
    for(int stride=64;stride;stride>>=1){
        if(lane<stride){clicks[lane]+=clicks[lane+stride];photons[lane]+=photons[lane+stride];}
        __syncthreads();
    }
    if(lane==0){
        click=clicks[0];photon=photons[0];
        totals[2*row]=click;totals[2*row+1]=photon;
        double p=probabilities[min(photon,nn-1)*nc+min(click,nc-1)];
        accepted[row]=unit103(low103(block103(seed,first+row,0,198,0)))<p;
    }
}
extern "C" __declspec(dllexport) int total104_launch(int groups,int shots,U64 seed,U32 first,
    const unsigned short* out,const unsigned short* counts,const double* p,int nc,int nn,
    int* totals,int* accepted,void* stream){
    total104<<<shots,128,0,(cudaStream_t)stream>>>(groups,shots,seed,first,out,counts,p,nc,nn,totals,accepted);
    return int(cudaGetLastError());
}
