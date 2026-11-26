#include "continuation_gpu103_v1.cu"

// Each thread owns both eight-bin outputs of one spatial pair in one record.
template<class T>
__global__ void refine107(int groups,int n,int states,U64 seed,U32 offset,
    const int* pairs,const int* starts,const int* sizes,const int* choose,
    const U64* words,const float* move,const U64* cdf,T* records){
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=groups*n)return;
    int g=index%groups,row=index/groups;
    T* record=records+std::size_t(row)*16*groups;
    T* bins[2]={record+8*pairs[2*g],record+8*pairs[2*g+1]};
    int total[2]={0,0},rank[2]={0,0};
    for(int side=0;side<2;++side){
        for(int j=0;j<8;++j){
            unsigned v=bins[side][j];
            if(v>unsigned(5-total[0]-total[1]))return;
            for(unsigned k=0;k<v;++k){
                int degree=++total[side];
                rank[side]+=choose[(j+degree-1)*6+degree];
            }
        }
    }
    int a=total[0],b=total[1],start=starts[a*6+b];
    std::size_t base=std::size_t(g)*states+start;
    double chance=move[base+rank[0]+sizes[a]*rank[1]];
    if(chance<=0.)return;
    uint4 v=block103(seed,offset+row,g,251,0);
    if(unit103(low103(v))>=chance)return;
    int lo=0,hi=sizes[a]*sizes[b];
    U64 draw=high103(v)>>12;
    while(lo<hi){
        int mid=(lo+hi)/2;
        if(cdf[base+mid]<=draw)lo=mid+1;else hi=mid;
    }
    U64 state=words[start+lo];
    for(int k=0;k<16;++k){bins[k/8][k%8]=T(state&15);state>>=4;}
}

extern "C" __declspec(dllexport) int refine107_launch(int groups,int n,int states,int wide,U64 seed,U32 offset,
    const int* pairs,const int* starts,const int* sizes,const int* choose,const U64* words,
    const float* move,const U64* cdf,void* out,void* stream){
    int blocks=(groups*n+255)/256;
    if(wide)refine107<<<blocks,256,0,(cudaStream_t)stream>>>(groups,n,states,seed,offset,pairs,starts,sizes,choose,words,move,cdf,(U32*)out);
    else refine107<<<blocks,256,0,(cudaStream_t)stream>>>(groups,n,states,seed,offset,pairs,starts,sizes,choose,words,move,cdf,(unsigned char*)out);
    return cudaGetLastError();
}
