// FP64 CUDA C++ prototype: persistent tensors, batched cuBLAS contraction,
// fused per-trajectory POVM refinement/selection, and device-resident state.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#define API extern "C" __declspec(dllexport)
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void cb(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(e));}
static void need(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
static void message(char* out,int n,const std::string& text){if(out&&n>0){auto count=std::min(std::size_t(n-1),text.size());std::memcpy(out,text.data(),count);out[count]=0;}}
template<class T> struct Device {T* p=nullptr;std::size_t count;explicit Device(std::size_t n):count(n){if(n)ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;Device& operator=(const Device&)=delete;};
struct Site {int l,d,r;cuDoubleComplex* a=nullptr;};
struct Model {std::vector<Site> sites;int bond=1,width=1;cublasHandle_t blas=nullptr;std::size_t elements=0;~Model(){for(auto& s:sites)if(s.a)cudaFree(s.a);if(blas)cublasDestroy(blas);}};
__device__ double norm2(cuDoubleComplex a){return a.x*a.x+a.y*a.y;}
__global__ void initial(cuDoubleComplex* env,int batch,int bond){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b<batch)env[std::size_t(b)*bond]=make_cuDoubleComplex(1.,0.);}
__global__ void refine(cuDoubleComplex* amp,const cuDoubleComplex* coherent,const double* split,const double* uniforms,
                       cuDoubleComplex* next,int* labels,double* selected,double* vacuum,double* drift,int* status,
                       int d,int r,int bond,int modes,int site){
    const int b=blockIdx.x,t=threadIdx.x;cuDoubleComplex* a=amp+std::size_t(b)*d*r;const cuDoubleComplex* c=coherent+std::size_t(b)*d;
    extern __shared__ double shared[];double* pr=shared;double* pi=pr+r;double* reduction=pi+r;
    __shared__ double mass[32],weights[33],total;__shared__ int support[32],order[33],chosen,physical,bad;
    if(t==0){bad=0;double cn=0.;for(int n=0;n<d;++n)cn+=norm2(c[n]);double w=split[2*b],tail=split[2*b+1];if(!isfinite(cn)||fabs(cn-1.)>1e-12||c[0].x<0.||c[0].y!=0.||!isfinite(w)||!isfinite(tail)||w<0.||tail<0.||fabs(w+tail-1.)>2e-14)bad=1;}
    __syncthreads();
    for(int n=0;n<d;++n){double nz=0;for(int j=t;j<r;j+=blockDim.x){auto v=a[n*r+j];nz+=(v.x!=0.||v.y!=0.);}reduction[t]=nz;__syncthreads();for(int stride=blockDim.x/2;stride;stride/=2){if(t<stride)reduction[t]+=reduction[t+stride];__syncthreads();}if(t==0)support[n]=reduction[0]>0.;__syncthreads();}
    for(int j=t;j<r;j+=blockDim.x){double xr=0.,xi=0.;for(int n=0;n<d;++n){auto v=a[n*r+j];xr+=c[n].x*v.x+c[n].y*v.y;xi+=c[n].x*v.y-c[n].y*v.x;}pr[j]=xr;pi[j]=xi;}
    __syncthreads();
    for(int j=t;j<r;j+=blockDim.x){auto first=a[j];double xr=(first.x+pr[j])/(1.+c[0].x),xi=(first.y+pi[j])/(1.+c[0].x);for(int n=1;n<d;++n){auto& v=a[n*r+j];v.x-=c[n].x*xr-c[n].y*xi;v.y-=c[n].x*xi+c[n].y*xr;}a[j]=make_cuDoubleComplex(-pr[j],-pi[j]);}
    __syncthreads();
    for(int n=0;n<d;++n){double sum=0.;for(int j=t;j<r;j+=blockDim.x)sum+=norm2(a[n*r+j]);reduction[t]=sum;__syncthreads();for(int stride=blockDim.x/2;stride;stride/=2){if(t<stride)reduction[t]+=reduction[t+stride];__syncthreads();}if(t==0)mass[n]=reduction[0];__syncthreads();
        if(mass[n]==0.){double nz=0.;for(int j=t;j<r;j+=blockDim.x){auto v=a[n*r+j];nz+=(v.x!=0.||v.y!=0.);}reduction[t]=nz;__syncthreads();for(int stride=blockDim.x/2;stride;stride/=2){if(t<stride)reduction[t]+=reduction[t+stride];__syncthreads();}
            if(t==0){if(reduction[0]>0.)bad=2;bool reachable=false;for(int k=0;k<d;++k){double rr,ri;if(n==0){rr=-c[k].x;ri=c[k].y;}else if(k==0){rr=-c[n].x;ri=-c[n].y;}else{rr=-(c[n].x*c[k].x+c[n].y*c[k].y)/(1.+c[0].x);ri=-(c[n].y*c[k].x-c[n].x*c[k].y)/(1.+c[0].x);if(k==n)rr+=1.;}reachable|=support[k]&&(rr!=0.||ri!=0.);}if(reachable)bad=3;}
        }__syncthreads();
    }
    if(t==0){weights[0]=split[2*b]*mass[0];weights[1]=split[2*b+1]*mass[0];for(int n=1;n<d;++n)weights[n+1]=mass[n];
        for(int k=0;k<2;++k)if(split[2*b+k]>0.&&mass[0]>0.&&weights[k]==0.)bad=4;
        for(int k=0;k<=d;++k){order[k]=k;if(!isfinite(weights[k])||weights[k]<0.)bad=5;}
        for(int k=1;k<=d;++k){int key=order[k],j=k-1;while(j>=0&&weights[order[j]]>weights[key]){order[j+1]=order[j];--j;}order[j+1]=key;}
        double cumulative[33];total=0.;for(int k=0;k<=d;++k){double next_total=total+weights[order[k]];if(weights[order[k]]>0.&&next_total<=total)bad=6;total=next_total;cumulative[k]=total;}
        if(!isfinite(total)||total<=0.)bad=7;const std::size_t out=std::size_t(b)*modes+site;drift[out]=fabs(total-1.);if(drift[out]>1e-9)bad=8;double u=uniforms[out],target=u*total;if(!isfinite(u)||u<0.||u>=1.||(u>0.&&target==0.))bad=9;int k=0;while(k<=d&&cumulative[k]<=target)++k;if(k>d){bad=10;k=0;}chosen=order[k];physical=chosen<=1?0:chosen-1;if(mass[physical]<=0.)bad=11;
        labels[out]=chosen;selected[out]=weights[chosen]/total;vacuum[out]=weights[0]/total;if(bad)atomicCAS(status,0,100*site+bad);
    }__syncthreads();
    if(!bad){double scale=sqrt(mass[physical]);for(int j=t;j<r;j+=blockDim.x){auto v=a[physical*r+j];next[std::size_t(b)*bond+j]=make_cuDoubleComplex(v.x/scale,v.y/scale);}}
}

__global__ void prepare_coherent(const cuDoubleComplex* beta,cuDoubleComplex* coherent,double* split,const int* dimensions,const std::size_t* offsets,int batch,int modes,int* status){
    const std::size_t task=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(task>=std::size_t(batch)*modes)return;int site=static_cast<int>(task/batch),b=static_cast<int>(task%batch),d=dimensions[site];auto z=beta[std::size_t(b)*modes+site];auto* c=coherent+offsets[site]+std::size_t(b)*d;double* p=split+2*task;double t=z.x*z.x+z.y*z.y;
    if(!isfinite(z.x)||!isfinite(z.y)||!isfinite(t)||((z.x!=0.||z.y!=0.)&&t==0.)){atomicCAS(status,0,900001+100*site);return;}
    if(t==0.){c[0]=make_cuDoubleComplex(1.,0.);for(int n=1;n<d;++n)c[n]=make_cuDoubleComplex(0.,0.);p[0]=1.;p[1]=0.;return;}
    double logarithms[32],largest=0.,logt=log(t),phase=atan2(z.y,z.x);logarithms[0]=0.;for(int n=1;n<d;++n){logarithms[n]=.5*(n*logt-lgamma(double(n+1)));largest=fmax(largest,logarithms[n]);}double squared=0.;
    for(int n=0;n<d;++n){double a=exp(logarithms[n]-largest);if(a==0.){atomicCAS(status,0,900002+100*site);return;}double angle=n*phase;c[n]=make_cuDoubleComplex(a*cos(angle),a*sin(angle));squared+=a*a;}double length=sqrt(squared);for(int n=0;n<d;++n){c[n].x/=length;c[n].y/=length;}
    double retained,tail;
    if(t<double(d+1)){double term=1.,sum=1.;int k=1;for(;k<=512;++k){term*=t/(d+k);sum+=term;if(term<=sum*1.1102230246251565e-16)break;}if(k>512){atomicCAS(status,0,900003+100*site);return;}tail=exp(-t+d*logt-lgamma(double(d+1)))*sum;retained=1.-tail;}
    else {retained=exp(-t+2*largest)*squared;tail=1.-retained;}
    if(!(retained>0.&&tail>0.)||!isfinite(retained)||!isfinite(tail)||fabs(retained+tail-1.)>2e-14){atomicCAS(status,0,900004+100*site);return;}p[0]=retained;p[1]=tail;
}
API void* gds_create(int m,const int* dims,const double* const* tensors,char* error,int cap){Model* model=nullptr;try{need(m>0&&dims&&tensors,"invalid model input");model=new Model;cb(cublasCreate(&model->blas));cb(cublasSetAtomicsMode(model->blas,CUBLAS_ATOMICS_NOT_ALLOWED));int previous=1;
    for(int i=0;i<m;++i){int l=dims[3*i],d=dims[3*i+1],r=dims[3*i+2];need(l==previous&&d>0&&d<=32&&r>0&&tensors[i],"invalid dimensions; prototype requires d<=32");std::size_t size=std::size_t(l)*d*r;for(std::size_t k=0;k<2*size;++k)need(std::isfinite(tensors[i][k]),"nonfinite tensor");model->sites.push_back(Site{l,d,r,nullptr});auto& s=model->sites.back();ck(cudaMalloc(reinterpret_cast<void**>(&s.a),size*sizeof(cuDoubleComplex)));ck(cudaMemcpy(s.a,tensors[i],size*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));model->bond=std::max({model->bond,l,r});model->width=std::max(model->width,d*r);model->elements+=size;previous=r;}need(previous==1,"invalid right boundary");return model;
}catch(const std::exception& e){delete model;message(error,cap,e.what());return nullptr;}}
API void gds_destroy(void* h){delete static_cast<Model*>(h);}
API int gds_sample_beta(void* h,int batch,const double* beta,const double* uniforms,
                   int* labels,double* selected,double* vacuum,double* residual,char* error,int cap){try{
    need(h&&batch>=0&&beta&&uniforms&&labels&&selected&&vacuum&&residual,"invalid sample input");auto& model=*static_cast<Model*>(h);const int m=static_cast<int>(model.sites.size()),bond=model.bond;*residual=0.;if(!batch)return 0;
    std::vector<std::size_t> offsets;std::size_t count_c=0;for(auto& s:model.sites){offsets.push_back(count_c);count_c+=std::size_t(batch)*s.d;}std::vector<int> dimensions;for(auto& s:model.sites)dimensions.push_back(s.d);
    Device<cuDoubleComplex> dc(count_c),env(std::size_t(batch)*bond),next(std::size_t(batch)*bond),amp(std::size_t(batch)*model.width);Device<double> dp(std::size_t(batch)*m*2),du(std::size_t(batch)*m),ds(std::size_t(batch)*m),dv(std::size_t(batch)*m),dr(std::size_t(batch)*m);Device<int> dl(std::size_t(batch)*m),status(1);
    Device<cuDoubleComplex> db(std::size_t(batch)*m);Device<int> dd(m);Device<std::size_t> offsets_device(m);ck(cudaMemcpy(db.p,beta,db.count*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));ck(cudaMemcpy(dd.p,dimensions.data(),m*sizeof(int),cudaMemcpyHostToDevice));ck(cudaMemcpy(offsets_device.p,offsets.data(),m*sizeof(std::size_t),cudaMemcpyHostToDevice));ck(cudaMemset(dc.p,0,dc.count*sizeof(cuDoubleComplex)));ck(cudaMemset(dp.p,0,dp.count*sizeof(double)));ck(cudaMemcpy(du.p,uniforms,du.count*sizeof(double),cudaMemcpyHostToDevice));ck(cudaMemset(env.p,0,env.count*sizeof(cuDoubleComplex)));ck(cudaMemset(next.p,0,next.count*sizeof(cuDoubleComplex)));ck(cudaMemset(status.p,0,sizeof(int)));prepare_coherent<<<(std::size_t(batch)*m+255)/256,256>>>(db.p,dc.p,dp.p,dd.p,offsets_device.p,batch,m,status.p);initial<<<(batch+255)/256,256>>>(env.p,batch,bond);const cuDoubleComplex one=make_cuDoubleComplex(1.,0.),zero=make_cuDoubleComplex(0.,0.);
    for(int i=0;i<m;++i){const auto& s=model.sites[i];cb(cublasZgemm(model.blas,CUBLAS_OP_N,CUBLAS_OP_N,s.d*s.r,batch,s.l,&one,s.a,s.d*s.r,env.p,bond,&zero,amp.p,s.d*s.r));refine<<<batch,256,std::size_t(2*s.r+256)*sizeof(double)>>>(amp.p,dc.p+offsets[i],dp.p+std::size_t(i)*batch*2,du.p,next.p,dl.p,ds.p,dv.p,dr.p,status.p,s.d,s.r,bond,m,i);ck(cudaGetLastError());std::swap(env.p,next.p);}
    ck(cudaDeviceSynchronize());int bad;ck(cudaMemcpy(&bad,status.p,sizeof(int),cudaMemcpyDeviceToHost));need(bad==0,("GPU numerical rejection code "+std::to_string(bad)).c_str());std::vector<double> drift(dr.count);ck(cudaMemcpy(drift.data(),dr.p,dr.count*sizeof(double),cudaMemcpyDeviceToHost));*residual=*std::max_element(drift.begin(),drift.end());ck(cudaMemcpy(labels,dl.p,dl.count*sizeof(int),cudaMemcpyDeviceToHost));ck(cudaMemcpy(selected,ds.p,ds.count*sizeof(double),cudaMemcpyDeviceToHost));ck(cudaMemcpy(vacuum,dv.p,dv.count*sizeof(double),cudaMemcpyDeviceToHost));return 0;
}catch(const std::exception& e){message(error,cap,e.what());return 1;}}
