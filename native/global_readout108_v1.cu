#include "continuation_gpu103_v1.cu"
#include <cuda_runtime.h>
#include <cstdint>

__device__ uint4 rng108(U64 seed,U32 row,U32 index,U32 purpose,U32 word){
    // Each purpose owns a distinct residue modulo 16. The upper bits of the
    // index extend the sector range while retaining those residues.
    uint4 c=make_uint4(word,108,row,((index&65535)<<16)|(purpose+16*(index>>16)));
    U32 a=U32(seed),b=U32(seed>>32);
    for(int i=0;i<10;++i){
        U32 x=0xD2511F53u*c.x,y=0xCD9E8D57u*c.z;
        c=make_uint4(__umulhi(0xCD9E8D57u,c.z)^c.y^a,y,__umulhi(0xD2511F53u,c.x)^c.w^b,x);
        a+=0x9E3779B9u;b+=0xBB67AE85u;
    }
    return c;
}
__device__ double uniform108(U64 x){return (double(x>>12)+.5)*0x1.0p-52;}
__device__ double draw108(U64 seed,U32 row,U32 index,U32 purpose,U32 j){
    uint4 v=rng108(seed,row,index,purpose,j/2);
    return uniform108(j&1?high103(v):low103(v));
}
__device__ int quant108(const U64* cdf,int size,U64 r){
    int left=0,right=size;
    while(left<right){int mid=left+(right-left)/2;if(cdf[mid]<=(r>>12))left=mid+1;else right=mid;}
    return left;
}
__device__ int position108(const double* sums,int lo,int hi,double cut){
    while(lo<hi){int mid=lo+(hi-lo)/2;if(sums[mid]<=cut)lo=mid+1;else hi=mid;}
    return lo;
}
__device__ unsigned poisson108(double rate,U64 seed,U32 row,U32 index,int* error){
    if(!(rate>=0.)||rate>1.e6){atomicMax(error,2);return 0;}
    if(rate==0)return 0;
    if(rate<12.){
        double u=draw108(seed,row,index,274,0),p=exp(-rate),sum=p;
        unsigned n=0;
        while(sum<u){p*=rate/(++n);sum+=p;if(p<1.e-20)break;}
        return n;
    }
    // Count exponential arrivals. The full range remains stable at high rate.
    unsigned n=0;double t=-log(draw108(seed,row,index,274,0));
    while(t<rate){++n;t-=log(draw108(seed,row,index,274,n));}
    return n;
}
__global__ void rates108(int m,int spectra,int n,const float* field,const double* gain,double* rate,int* error){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m*n)return;
    int j=i%m,row=i/m;double sum=0;
    for(int s=0;s<spectra;++s){
        const float* f=field+(size_t(s)*n+row)*2*m;
        double x=f[j],y=f[m+j];sum+=x*x+y*y;
    }
    sum*=gain[row];
    if(!(sum>=0.)||!isfinite(sum)){atomicMax(error,2);sum=0.;}
    rate[i]=sum;
}
__device__ int move108(int state,int size,const float* move,const U64* cdf,U64 seed,U32 row,U32 purpose){
    if(move[state]<=0.)return state;
    uint4 roll=rng108(seed,row,0,purpose,0);
    return uniform108(low103(roll))<move[state]?quant108(cdf,size,high103(roll)):state;
}
__global__ void flags108(int m,int h,int groups,int n,const int* cuts,const double* rate,const double* sums,
    const float* move,const U64* cdf,U64 seed,U32 off,int* flags){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
    const double* lam=rate+size_t(row)*m;const double* cum=sums+size_t(row)*m;
    int word=0;
    for(int j=0;j<h+groups;++j){
        double mean=j<h?lam[j]:cum[cuts[j-h+1]-1]-cum[cuts[j-h]-1];
        if(draw108(seed,off+row,j,275,0)<-expm1(-mean))word|=1<<j;
    }
    word=move108(word,1<<(h+groups),move,cdf,seed,off+row,276);
    int* out=flags+row*(groups+1);out[0]=word;
    for(int g=0;g<groups;++g){
        int lo=cuts[g],hi=cuts[g+1],first=hi;
        if(word&(1<<(h+g))){
            double start=cum[lo-1],mass=cum[hi-1]-start;
            double u=draw108(seed,off+row,g,277,0);
            if(mass>0.){
                double pos=fmin(nextafter(cum[hi-1],start),start-log1p(-u*(-expm1(-mass))));
                first=position108(cum,lo,hi,pos);
            } else first=lo+int(u*(hi-lo)); // Exactly dark conditional fallback.
        }
        out[g+1]=first;
    }
}
__global__ void click_output108(int m,int h,int groups,int n,const int* order,const int* region,const double* rate,
    const int* flags,U64 seed,U32 off,unsigned char* out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m*n)return;
    int row=i/m,j=i%m;const int* rule=flags+row*(1+groups);
    int flag;
    if(j<h)flag=(rule[0]>>j)&1;
    else{
        int first=rule[region[j]+1];
        flag=(j==first)||(j>first && draw108(seed,off+row,j,278,0)<-expm1(-rate[i]));
    }
    out[size_t(row)*m+order[j]]=flag;
}
template<class T>
__global__ void pnr108(int m,int n,int ca,int cb,int cc,const int* order,const double* rate,const double* sums,
    const float* move,const U64* cdf,U64 seed,U32 off,T* out,int* error){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
    const double* lam=rate+size_t(row)*m;const double* cum=sums+size_t(row)*m;
    double begin=cum[1],bulk=fmax(0.,cum[m-1]-begin);
    unsigned a=poisson108(lam[0],seed,off+row,0,error),b=poisson108(lam[1],seed,off+row,1,error),c=poisson108(bulk,seed,off+row,2,error);
    if(a<ca&&b<cb&&c<cc){
        int state=move108((a*cb+b)*cc+c,ca*cb*cc,move,cdf,seed,off+row,279);
        c=state%cc;state/=cc;b=state%cb;a=state/cb;
    }
    T* x=out+size_t(row)*m;
    if(sizeof(T)==1&&(a>255||b>255)){atomicMax(error,1);return;}
    x[order[0]]=T(a);x[order[1]]=T(b);
    for(unsigned k=0;k<c;++k){
        if(m<=2){atomicMax(error,2);return;}
        double u=draw108(seed,off+row,0,280,k);
        int j=bulk>0?position108(cum,2,m,fmin(nextafter(cum[m-1],begin),begin+u*bulk)):2+int(u*(m-2));
        auto& cell=x[order[j]];
        if(sizeof(T)==1&&cell==255){atomicMax(error,1);return;}++cell;
    }
}
extern "C" __declspec(dllexport) int rates108_launch(int m,int s,int n,const float* f,const double* gain,double* lam,int* status,void* stream){
    rates108<<<(m*n+255)/256,256,0,(cudaStream_t)stream>>>(m,s,n,f,gain,lam,status);
    return cudaGetLastError();
}
extern "C" __declspec(dllexport) int read108_launch(int m,int n,int h,int groups,int ca,int cb,int cc,int route,U64 seed,U32 off,
    const int* order,const int* bounds,const int* region,const double* rate,const double* cum,const float* move,const U64* cdf,
    int* flags,void* buffer,int* error,void* stream){
    cudaStream_t s=(cudaStream_t)stream;
    if(!route){
        flags108<<<(n+127)/128,128,0,s>>>(m,h,groups,n,bounds,rate,cum,move,cdf,seed,off,flags);
        if(cudaGetLastError()!=cudaSuccess)return 1;
        click_output108<<<(n*m+255)/256,256,0,s>>>(m,h,groups,n,order,region,rate,flags,seed,off,(unsigned char*)buffer);
    }else{
        if(cudaMemsetAsync(buffer,0,size_t(n)*m*(route==2?4:1),s)!=cudaSuccess)return 2;
        if(route==1)pnr108<<<(n+127)/128,128,0,s>>>(m,n,ca,cb,cc,order,rate,cum,move,cdf,seed,off,(unsigned char*)buffer,error);
        else pnr108<<<(n+127)/128,128,0,s>>>(m,n,ca,cb,cc,order,rate,cum,move,cdf,seed,off,(U32*)buffer,error);
    }
    return cudaGetLastError();
}
