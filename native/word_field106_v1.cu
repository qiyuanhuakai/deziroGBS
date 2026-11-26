#include "continuation_gpu103_v1.cu"

__global__ void normal_word106(int k,int spectra,int n,U64 seed,U32 off,int atoms,const double* g,const double* cdf,float* z,double* scale){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=k*spectra*n)return;
    int j=id%k,s=id/(n*k),row=(id/k)%n;
    uint4 v=block103(seed,off+row,s*k+j,221,0);
    double a=sqrt(-2.*log((double(low103(v)>>12)+.5)*0x1.0p-52));
    double sine,cosine;
    sincos(6.2831853071795864769*(double(high103(v)>>12)+.5)*0x1.0p-52,&sine,&cosine);
    z[2*id]=float(a*cosine);z[2*id+1]=float(a*sine);
    if(!s&&!j){
        double u=(double(low103(block103(seed,off+row,0,222,0))>>12)+.5)*0x1.0p-52;
        int i=0;while(i+1<atoms&&u>=cdf[i])++i;scale[row]=g[i];
    }
}

__device__ double detector_word106(uint4 word,int b){
    U32 v=(b==0)?word.x:(b==1)?word.y:(b==2)?word.z:word.w;
    return (double(v)+.5)*0x1.0p-32;
}
__device__ unsigned poisson_word106(double rate,double u){
    double p=exp(-rate),cum=p;
    unsigned n=0;
    while(u>cum){++n;p*=rate/n;cum+=p;}
    return n;
}
template<int Mode>
__global__ void word106(int groups,int spectra,int n,U64 seed,U32 off,const float* field,
    const double* weight,const double* scale,void* out,int* status){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=groups*n)return;
    int row=id/groups,g=id%groups;
    double intensity=0;
    for(int s=0;s<spectra;++s){
        auto p=field+(std::size_t(s)*n+row)*2*groups;
        double re=p[g],im=p[groups+g];
        intensity+=re*re+im*im;
    }
    intensity*=scale[row];
    unsigned mask=0;
    for(int half=0;half<2;++half){
        uint4 first=block103(seed,off+row,g,225,half);
        #pragma unroll
        for(int b=0;b<4;++b){
            int j=half*4+b;
            double rate=intensity*weight[8*g+j],u=detector_word106(first,b);
            if(!(rate>=0.)||rate>1.e8){atomicMax(status,2);return;}
            if(Mode==0 || Mode==3){
                const bool clicked=u>exp(-rate);
                if(Mode==3)reinterpret_cast<unsigned char*>(out)[std::size_t(id)*8+j]=clicked;
                else if(clicked)mask|=1u<<j;
            }else{
                // Split high rates into equal Poisson increments. The first
                // chunk's u is shared with the click route only for rate<=8.
                // Decide zero with the full-rate conditional first, then invert
                // the zero-truncated tail for high rates via conditional chunks.
                unsigned count;
                if(rate<=8.) count=poisson_word106(rate,u);
                else if(u<=exp(-rate)) count=0;
                else {
                    int chunks=int(ceil(rate/8.));
                    // The conditional positive Poisson is exactly a compound
                    // of independent chunks conditioned on a nonzero sum.
                    U32 serial=0;
                    do{
                        count=0;
                        for(int k=0;k<chunks;++k){
                            if(serial>=0x80000000u){atomicMax(status,2);return;}
                            uint4 v=block103(seed,off+row,g,226,2*serial+half);
                            ++serial;
                            count+=poisson_word106(rate/chunks,detector_word106(v,b));
                        }
                    }while(!count);
                }
                if(Mode==1){
                    if(count>255)atomicMax(status,1);
                    reinterpret_cast<unsigned char*>(out)[std::size_t(id)*8+j]=static_cast<unsigned char>(count);
                }else reinterpret_cast<U32*>(out)[std::size_t(id)*8+j]=count;
            }
        }
    }
    if(Mode==0)reinterpret_cast<unsigned char*>(out)[id]=static_cast<unsigned char>(mask);
}
extern "C" __declspec(dllexport) int noise_word106_launch(int k,int spectra,int n,U64 seed,U32 off,int atoms,const double* g,const double* cdf,float* z,double* scale,void* stream){
    int count=k*spectra*n;
    normal_word106<<<(count+255)/256,256,0,(cudaStream_t)stream>>>(k,spectra,n,seed,off,atoms,g,cdf,z,scale);
    return cudaGetLastError();
}
extern "C" __declspec(dllexport) int word106_launch(int groups,int spectra,int n,int mode,U64 seed,U32 off,const float* field,const double* weight,const double* scale,void* out,int* status,void* stream){
    int blocks=(groups*n+255)/256;
    if(mode==0)word106<0><<<blocks,256,0,(cudaStream_t)stream>>>(groups,spectra,n,seed,off,field,weight,scale,out,status);
    else if(mode==1)word106<1><<<blocks,256,0,(cudaStream_t)stream>>>(groups,spectra,n,seed,off,field,weight,scale,out,status);
    else if(mode==2)word106<2><<<blocks,256,0,(cudaStream_t)stream>>>(groups,spectra,n,seed,off,field,weight,scale,out,status);
    else if(mode==3)word106<3><<<blocks,256,0,(cudaStream_t)stream>>>(groups,spectra,n,seed,off,field,weight,scale,out,status);
    else return 1;
    return cudaGetLastError();
}
