#include "continuation_gpu103_v1.cu"

template<class F> __device__ double blank115(const double* gate,const F* a) {
    double x=0;for(int j=0;j<4;++j){double v=a[j];x+=gate[1+j]*v*v;}
    return gate[0]*exp(-x);
}

template<class F,bool Tilt> __global__ void banks115(int rows,int groups,int fan,int pn,U64 seed,U32 offset,
        const F* field,const double* x,const double* gates,const double* split,unsigned char* out,int* error,
        const int* labels,const double* scales) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=rows*groups)return;
    int g=id%groups,row=id/groups;
    int param=g;
    const F* src=field+size_t(id)*4;
    double a[4];
    if constexpr(Tilt) {
        param+=labels[row]*groups;
        for(int j=0;j<4;++j)a[j]=double(src[j])*scales[param*4+j];
    } else for(int j=0;j<4;++j)a[j]=src[j];
    const double* gate=gates+size_t(param)*(1<<fan)*5;
    auto* dst=out+size_t(id)*fan;
    for(int j=0;j<fan;++j)dst[j]=0;
    double u=unit103(low103(block103(seed,offset+row,g,321+pn,0)));
    double p0=blank115(gate+((1<<fan)-1)*5,a);
    if(u<p0)return;
    if(!pn) {
        if(fan==1){dst[0]=1;return;}
        int zero=0,hit=0;double mass=1;
        for(int j=0;j<fan;++j) {
            int next=zero|(1<<j);
            double p=0;
            for(int sub=hit;;sub=(sub-1)&hit) {
                double term=blank115(gate+(next|sub)*5,a);
                p+=(__popc(sub)&1)?-term:term;
                if(!sub)break;
            }
            if(p < -1e-10||p>mass+1e-10||!isfinite(p)){atomicOr(error,1);return;}
            p=fmin(mass,fmax(0.,p));
            if(u<p){zero=next;mass=p;}
            else {dst[j]=1;hit|=1<<j;u-=p;mass-=p;}
        }
        return;
    }
    double coeff[4],v[4],power[4]={1,1,1,1};
    for(int j=0;j<4;++j){
        double e=x[param*4+j],w=1/(1+e);coeff[j]=e*w;v[j]=double(a[j])*double(a[j])*w*w;
    }
    constexpr int cap=64;
    double prob[cap+1],trace[cap+1];prob[0]=p0;u-=p0;
    int count=1;
    for(;count<=cap;++count) {
        double t=0,p=0;
        for(int j=0;j<4;++j){t+=power[j]*(.5*coeff[j]+count*v[j]);power[j]*=coeff[j];}
        trace[count]=t;
        for(int j=1;j<=count;++j)p+=trace[j]*prob[count-j];
        prob[count]=p/count;
        if(prob[count]<-1e-12||!isfinite(p)){atomicOr(error,2);return;}
        if(u<prob[count])break;
        u-=prob[count];
    }
    if(count>cap){atomicOr(error,4);return;}
    if(fan==1){dst[0]=count;return;}
    for(int j=0;j<count;++j){
        uint4 z=block103(seed,offset+row,g,325,j/2);
        double rnd=unit103((j&1)?high103(z):low103(z));
        int site=0;while(site<fan-1&&rnd>=split[g*fan+site])++site;
        ++dst[site];
    }
}
extern "C" __declspec(dllexport) int bank115_launch(int rows,int groups,int fan,int pn,int wide,U64 seed,U32 offset,
        const void* field,const double* x,const double* gates,const double* split,unsigned char* out,
        int* error,cudaStream_t stream) {
    if(rows<1||groups<1||groups>65535||(fan!=1&&fan!=8)||U64(rows)*groups>=2147483392||
        U64(offset)+rows>(U64(1)<<32))return 1;
    int n=(rows*groups+255)/256;
    if(wide)banks115<double,false><<<n,256,0,stream>>>(rows,groups,fan,pn,seed,offset,(const double*)field,x,gates,split,out,error,nullptr,nullptr);
    else banks115<float,false><<<n,256,0,stream>>>(rows,groups,fan,pn,seed,offset,(const float*)field,x,gates,split,out,error,nullptr,nullptr);
    return int(cudaGetLastError());
}


template<class F> __global__ void latent115(int dim,int components,U64 seed,U32 offset,const F* noise,
        const double* cdf,const F* transforms,int* labels,F* latent) {
    int row=blockIdx.x,j=threadIdx.x;
    __shared__ F z[128];
    __shared__ int component;
    if(!j) {
        double u=unit103(low103(block103(seed,offset+row,0,331,0)));
        int k=0;while(k<components-1&&cdf[k]<=u)++k;
        labels[row]=k;component=k;
    }
    if(j<dim)z[j]=noise[size_t(row)*dim+j];
    __syncthreads();
    if(j<dim) {
        const F* t=transforms+(size_t(component)*dim+j)*dim;
        F v=0;for(int i=0;i<dim;++i)v+=t[i]*z[i];
        latent[size_t(row)*dim+j]=v;
    }
}
extern "C" __declspec(dllexport) int tilt115_latent(int n,int dim,int components,int wide,U64 seed,U32 off,
        const void* noise,const double* cdf,const void* transforms,int* labels,void* latent,cudaStream_t stream) {
    if(n<1||dim<1||dim>128||components<1||U64(off)+n>(U64(1)<<32))return 1;
    if(wide)latent115<<<n,128,0,stream>>>(dim,components,seed,off,(const double*)noise,cdf,(const double*)transforms,labels,(double*)latent);
    else latent115<<<n,128,0,stream>>>(dim,components,seed,off,(const float*)noise,cdf,(const float*)transforms,labels,(float*)latent);
    return int(cudaGetLastError());
}
extern "C" __declspec(dllexport) int tilt115_banks(int n,int g,int fan,int pn,int wide,U64 seed,U32 off,
        const void* field,const double* x,const double* gates,const double* split,const int* labels,
        const double* scales,unsigned char* out,int* error,cudaStream_t stream) {
    if(n<1||g<1||g>65535||(fan!=1&&fan!=8)||U64(n)*g>=2147483392||U64(off)+n>(U64(1)<<32))return 1;
    int blocks=(n*g+255)/256;
    if(wide)banks115<double,true><<<blocks,256,0,stream>>>(n,g,fan,pn,seed,off,(const double*)field,x,gates,split,out,error,labels,scales);
    else banks115<float,true><<<blocks,256,0,stream>>>(n,g,fan,pn,seed,off,(const float*)field,x,gates,split,out,error,labels,scales);
    return int(cudaGetLastError());
}
