#include "continuation_gpu103_v1.cu"

__global__ void normal_packed106(int k,int spectra,int n,U64 seed,U32 off,float* z){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=n*k*spectra)return;
    int j=id%k,row=(id/k)%n,s=id/(k*n);
    uint4 v=block103(seed,off+row,s*k+j,221,0);
    double a=sqrt(-2.*log((double(low103(v)>>12)+.5)*0x1.0p-52));
    double sine,cosine;
    sincos(6.2831853071795864769*(double(high103(v)>>12)+.5)*0x1.0p-52,&sine,&cosine);
    z[2*id]=float(a*cosine);z[2*id+1]=float(a*sine);
}

template<int Rank>
__global__ void packed_field106(int groups,int width,int spectra,int n,int pnr,U64 seed,U32 off,
    const float* field,const double2* transfer,int atoms,const double* gain,const double* cdf,unsigned short* out,int* error){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    int m=groups*width;
    if(id>=n*m)return;
    int port=id%m,row=id/m,g=port/width;
    double rate=0;
    for(int s=0;s<spectra;++s){
        auto base=field+(std::size_t(s)*n+row)*2*groups*Rank;
        double re=0,im=0;
        #pragma unroll
        for(int h=0;h<Rank;++h){
            double x=base[g*Rank+h],y=base[groups*Rank+g*Rank+h];
            double2 c=transfer[port*Rank+h];
            re+=c.x*x-c.y*y;im+=c.x*y+c.y*x;
        }
        rate+=re*re+im*im;
    }
    uint4 b=block103(seed,off+row,0,222,0);
    double u=(double(low103(b)>>12)+.5)*0x1.0p-52;
    int a=0;while(a+1<atoms && u>=cdf[a])++a;
    rate*=gain[a];if(!(rate>=0.)||rate>1.e8){atomicExch(error,1);return;}
    uint4 v=block103(seed,off+row,port,223,0);
    double wait=-log((double(low103(v)>>12)+.5)*0x1.0p-52);
    unsigned k=0;
    while(wait<rate){
        ++k;if(!pnr)break;
        if(k>=65536){atomicExch(error,2);return;}
        if(!(k&1))v=block103(seed,off+row,port,223,k/2);
        wait-=log((double(((k&1)?high103(v):low103(v))>>12)+.5)*0x1.0p-52);
    }
    out[id]=static_cast<unsigned short>(k);
}
extern "C" __declspec(dllexport) int normal_packed106_launch(int k,int spectra,int n,U64 seed,U32 off,float* z,void* stream){
    int threads=k*spectra*n;
    normal_packed106<<<(threads+255)/256,256,0,(cudaStream_t)stream>>>(k,spectra,n,seed,off,z);
    return cudaGetLastError();
}
extern "C" __declspec(dllexport) int packed_field106_launch(int groups,int width,int rank,int spectra,int n,int pnr,U64 seed,U32 off,
    const float* field,const double2* tr,int atoms,const double* gain,const double* cdf,unsigned short* out,int* err,void* stream){
    int blocks=(groups*width*n+255)/256;
    if(rank==1)packed_field106<1><<<blocks,256,0,(cudaStream_t)stream>>>(groups,width,spectra,n,pnr,seed,off,field,tr,atoms,gain,cdf,out,err);
    else if(rank==2)packed_field106<2><<<blocks,256,0,(cudaStream_t)stream>>>(groups,width,spectra,n,pnr,seed,off,field,tr,atoms,gain,cdf,out,err);
    else return 1;
    return cudaGetLastError();
}
