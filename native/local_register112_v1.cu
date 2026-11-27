#include "continuation_gpu103_v1.cu"

__global__ void local_words112(int n,int cells,int mask,U32 route,U64 seed,U32 off,
        const U64* threshold,const U32* alias,const U32* words,U32* output){
    unsigned long long id=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=(unsigned long long)n*cells)return;
    U32 row=U32(id/cells),cell=U32(id%cells);
    uint4 c=make_uint4(0,11200+route,off+row,cell);
    U32 a=U32(seed),b=U32(seed>>32);
    for(int j=0;j<10;++j){
        U32 x=0xD2511F53u*c.x,y=0xCD9E8D57u*c.z;
        c=make_uint4(__umulhi(0xCD9E8D57u,c.z)^c.y^a,y,__umulhi(0xD2511F53u,c.x)^c.w^b,x);
        a+=0x9E3779B9u;b+=0xBB67AE85u;
    }
    U32 slot=c.x&mask;
    U64 ticket=((U64(c.y)<<32)|c.z)>>11;
    output[id]=words[ticket<threshold[slot]?slot:alias[slot]];
}

extern "C" __declspec(dllexport) int local_emit112(int n,int cells,int size,U32 route,U64 seed,U32 offset,
        const U64* threshold,const U32* alias,const U32* words,U32* out,cudaStream_t stream){
    if(n<1||cells<1||size<1||(size&(size-1))||(U64(n)*cells)>2147483392u||
            U64(offset)+n>(U64(1)<<32))return 1;
    local_words112<<<(n*cells+255)/256,256,0,stream>>>(n,cells,size-1,route,seed,offset,threshold,alias,words,out);
    return int(cudaGetLastError());
}
