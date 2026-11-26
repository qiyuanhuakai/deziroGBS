#include "continuation_gpu103_v1.cu"


// Use the high counter word for large port or source indices. The original
// Philox addresses are preserved for the first 65,536 indices.
__device__ uint4 field_block106(U64 seed,U32 row,U32 index,U32 purpose,U32 block){
    if(index<=65535u)return block103(seed,row,index,purpose,block);
    uint4 c=make_uint4(block,index>>16,row,((index&65535u)<<16)|purpose);
    U32 lo=U32(seed),hi=U32(seed>>32);
    #pragma unroll
    for(int i=0;i<10;++i){
        U32 a=0xD2511F53u*c.x,b=0xCD9E8D57u*c.z;
        c=make_uint4(__umulhi(0xCD9E8D57u,c.z)^c.y^lo,b,
                     __umulhi(0xD2511F53u,c.x)^c.w^hi,a);
        lo+=0x9E3779B9u;hi+=0xBB67AE85u;
    }
    return c;
}

__device__ double open106(U64 value){
    return (double(value>>12)+0.5)*0x1.0p-52;
}

template<class F>
__global__ void noise106(int sources,int spectra,int rows,U64 seed,U32 offset,F* output){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=spectra*rows*sources)return;
    int j=id%sources,row=(id/sources)%rows,s=id/(rows*sources);
    uint4 words=field_block106(seed,offset+row,s*sources+j,221,0);
    double a=sqrt(-2.*log(open106(low103(words))));
    double sine,cosine;
    sincos(6.2831853071795864769*open106(high103(words)),&sine,&cosine);
    output[2*id]=F(a*cosine);output[2*id+1]=F(a*sine);
}

template<class F>
__global__ void detect106(int m,int spectra,int rows,int pnr,U64 seed,U32 off,const F* field,
    int atoms,const double* brightness,const double* cdf,unsigned* output,int* error){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=rows*m)return;
    int row=id/m,port=id%m;
    double rate=0;
    for(int s=0;s<spectra;++s){
        auto base=field+(std::size_t(s)*rows+row)*2*m;
        double a=base[port],b=base[m+port];
        rate+=a*a+b*b;
    }
    if(atoms>1){
        double u=open106(low103(field_block106(seed,off+row,0,222,0)));
        int k=0;while(k+1<atoms && u>=cdf[k])++k;
        rate*=brightness[k];
    }else rate*=brightness[0];
    if(!(rate>=0.) || rate>1.e8){atomicExch(error,1);return;}
    uint4 word=field_block106(seed,off+row,port,223,0);
    double arrival=-log(open106(low103(word)));
    unsigned n=0;
    while(arrival<rate){
        ++n;
        if(!pnr)break;
        if((n&1)==0)word=field_block106(seed,off+row,port,223,n/2);
        arrival-=log(open106((n&1)?high103(word):low103(word)));
    }
    output[id]=n;
}

extern "C" __declspec(dllexport) int noise106_launch(int sources,int spectra,int rows,U64 seed,U32 off,int wide,void* output,void* stream){
    int count=rows*sources*spectra;
    if(wide)noise106<<<(count+255)/256,256,0,(cudaStream_t)stream>>>(sources,spectra,rows,seed,off,(double*)output);
    else noise106<<<(count+255)/256,256,0,(cudaStream_t)stream>>>(sources,spectra,rows,seed,off,(float*)output);
    return cudaGetLastError();
}
extern "C" __declspec(dllexport) int detect106_launch(int m,int spectra,int rows,int pnr,U64 seed,U32 off,int wide,
    const void* field,int atoms,const double* brightness,const double* cdf,unsigned* output,int* error,void* stream){
    int count=rows*m;
    if(wide)detect106<<<(count+255)/256,256,0,(cudaStream_t)stream>>>(m,spectra,rows,pnr,seed,off,(const double*)field,atoms,brightness,cdf,output,error);
    else detect106<<<(count+255)/256,256,0,(cudaStream_t)stream>>>(m,spectra,rows,pnr,seed,off,(const float*)field,atoms,brightness,cdf,output,error);
    return cudaGetLastError();
}
