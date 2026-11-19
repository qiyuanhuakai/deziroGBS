// FP64 CUDA signed Gaussian threshold trajectories, no fast math or clipping.
// Same real-arithmetic bounded-branch rule as iteration71; different rounding.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#define API extern "C" __declspec(dllexport)
constexpr int MAX_D72=256,MAX_B72=128;
__device__ void add72(double value,double& sum,double& correction){double y=value-correction,t=sum+y;correction=(t-sum)-y;sum=t;}
__device__ double u72(int i,int j,const double* xa,const double* xb,const double* p){
    double ij=xa[i]*(p[2]*xa[j]+p[3]*xb[j])+xb[i]*(p[3]*xa[j]+p[4]*xb[j]);
    double ji=xa[j]*(p[2]*xa[i]+p[3]*xb[i])+xb[j]*(p[3]*xa[i]+p[4]*xb[i]);return .5*(ij+ji);
}
__global__ void click72(int m,int d,int count,int budget,const double* initial,const double* aa,const double* bb,const double* input_means,const double* draws,double* matrices,double* means,double* vectors,unsigned char* output,double* probabilities,int* errors,double* diagnostics){
    int row=blockIdx.x,t=threadIdx.x;if(row>=count)return;int slots=budget;std::size_t dd=std::size_t(d)*d;
    double* X=matrices+std::size_t(row)*slots*dd;double* W=means+std::size_t(row)*slots*d;double* V=vectors+std::size_t(row)*budget*3*d;
    __shared__ double weights[2*MAX_B72],params[8*MAX_B72],a[MAX_D72],b[MAX_D72],average[MAX_D72];
    __shared__ int n,clicked,err,where,merges,peak;__shared__ double p0,p1,mass,maximum_condition;
    if(t==0){n=1;weights[0]=1;err=0;where=-1;merges=0;peak=1;maximum_condition=1;}
    for(int i=t;i<d;i+=blockDim.x)W[i]=input_means[std::size_t(row)*d+i];
    for(std::size_t index=t;index<dd;index+=blockDim.x)X[index]=initial[(index%d)*d+index/d];
    __syncthreads();
    for(int mode=0;mode<m;++mode){
        for(int i=t;i<d;i+=blockDim.x){a[i]=aa[std::size_t(mode)*d+i];b[i]=bb[std::size_t(mode)*d+i];}__syncthreads();
        for(int k=0;k<n;++k){double* xa=V+std::size_t(k)*3*d;double* xb=xa+d;double* delta=xb+d;double* p=params+8*k;
            for(int i=t;i<d;i+=blockDim.x){double sa=0,sb=0,ca=0,cb=0;for(int j=0;j<d;++j){double x=X[std::size_t(k)*dd+std::size_t(j)*d+i];add72(x*a[j],sa,ca);add72(x*b[j],sb,cb);}xa[i]=sa;xb[i]=sb;}__syncthreads();
            if(t==0){double av=0,bv=0,ab=0,ba=0,mq=0,mp=0,c[6]={};for(int i=0;i<d;++i){add72(a[i]*xa[i],av,c[0]);add72(b[i]*xb[i],bv,c[1]);add72(a[i]*xb[i],ab,c[2]);add72(b[i]*xa[i],ba,c[3]);add72(a[i]*W[k*d+i],mq,c[4]);add72(b[i]*W[k*d+i],mp,c[5]);}
                double b00=1+av,b11=1+bv,b01=.5*(ab+ba),det=b00*b11-b01*b01;
                if(!(b00>0&&b11>0&&det>0&&isfinite(det)))err=1;
                p[2]=b11/det;p[3]=-b01/det;p[4]=b00/det;p[5]=p[2]*mq+p[3]*mp;p[6]=p[3]*mq+p[4]*mp;double logp=-.5*(log(det)+mq*p[5]+mp*p[6]);
                if(!(isfinite(logp)&&logp<=0))err=2;p[0]=exp(logp);p[1]=-expm1(logp);
                if(!(p[0]>0&&p[1]>=0&&isfinite(p[0])&&isfinite(p[1])))err=3;
            }__syncthreads();
            for(int i=t;i<d;i+=blockDim.x)delta[i]=p[5]*xa[i]+p[6]*xb[i];__syncthreads();
        }
        if(t==0){double z=0,o=0,cz=0,co=0;for(int k=0;k<n;++k){add72(weights[k]*params[8*k],z,cz);add72(weights[k]*params[8*k+1],o,co);}mass=z+o;
            if(!(isfinite(z)&&isfinite(o)&&z>=0&&o>=0&&mass>0))err=4;p0=z/mass;p1=o/mass;double draw=draws[std::size_t(row)*m+mode];if(!(isfinite(draw)&&draw>=0&&draw<1))err=5;
            clicked=draw<p1;probabilities[std::size_t(row)*m+mode]=p1;output[std::size_t(row)*m+mode]=clicked;
        }__syncthreads();if(err){if(t==0)where=mode;break;}if(mode+1==m)break;
        if(budget==1){double* xa=V,*xb=V+d,*delta=V+2*d;double* p=params;
            for(std::size_t index=t;index<dd;index+=blockDim.x){int i=index%d,j=index/d;if(i>j)continue;double u=u72(i,j,xa,xb,p),value=X[index];if(clicked){double q=p[0]/p[1];value+=q*u-(q/p[1])*delta[i]*delta[j];}else value-=u;X[index]=X[std::size_t(i)*d+j]=value;if(!isfinite(value))atomicExch(&err,6);}
            for(int i=t;i<d;i+=blockDim.x){W[i]+=clicked?(p[0]/p[1])*delta[i]:-delta[i];if(!isfinite(W[i]))atomicExch(&err,6);}if(t==0&&clicked)++merges;__syncthreads();
        }else if(clicked && 2*n>budget){
            // All 2*n children will immediately be merged. Preserve the same
            // pair ordering/normalization without materializing child matrices.
            if(t==0){
                for(int k=n-1;k>=0;--k){double w=weights[k];weights[2*k]=w/p1;weights[2*k+1]=-w*params[8*k]/p1;}
                double z=0,c=0;for(int k=0;k<2*n;++k)add72(weights[k],z,c);mass=z;
                if(!(isfinite(mass)&&mass>0))err=7;double condition=0;
                for(int k=0;k<2*n;++k){weights[k]/=mass;condition+=fabs(weights[k]);if(!isfinite(weights[k])||weights[k]==0)err=8;}
                maximum_condition=fmax(maximum_condition,condition);peak=max(peak,2*n);
            }
            __syncthreads();if(err){if(t==0)where=mode;break;}
            for(int i=t;i<d;i+=blockDim.x){double value=0,c=0;
                for(int k=0;k<n;++k){double before=W[k*d+i],after=before-V[std::size_t(k)*3*d+2*d+i];
                    add72(weights[2*k]*before,value,c);add72(weights[2*k+1]*after,value,c);}
                average[i]=value;
            }__syncthreads();
            for(std::size_t index=t;index<dd;index+=blockDim.x){int i=index%d,j=index/d;if(i>j)continue;double value=0,c=0;
                for(int k=0;k<n;++k){double* xa=V+std::size_t(k)*3*d,*xb=xa+d,*delta=xb+d,*p=params+8*k;
                    double before=X[std::size_t(k)*dd+index],after=before-u72(i,j,xa,xb,p);
                    double wi=W[k*d+i],wj=W[k*d+j],vi=wi-delta[i],vj=wj-delta[j];
                    add72(weights[2*k]*(before+(wi-average[i])*(wj-average[j])),value,c);
                    add72(weights[2*k+1]*(after+(vi-average[i])*(vj-average[j])),value,c);
                }
                X[index]=X[std::size_t(i)*d+j]=value;if(!isfinite(value))atomicExch(&err,6);
            }
            __syncthreads();for(int i=t;i<d;i+=blockDim.x)W[i]=average[i];
            if(t==0){n=1;weights[0]=1;++merges;}__syncthreads();
        }else{
            // Descending fork preserves old slots until each has been consumed.
            for(int k=n-1;k>=0;--k){double* xa=V+std::size_t(k)*3*d;double* xb=xa+d;double* delta=xb+d;double* p=params+8*k;int original=clicked?2*k:k,vacuum=clicked?2*k+1:k;
                for(std::size_t index=t;index<dd;index+=blockDim.x){int i=index%d,j=index/d;if(i>j)continue;double before=X[std::size_t(k)*dd+index],after=before-u72(i,j,xa,xb,p);
                    if(clicked){X[std::size_t(original)*dd+index]=before;X[std::size_t(original)*dd+std::size_t(i)*d+j]=before;}
                    X[std::size_t(vacuum)*dd+index]=after;X[std::size_t(vacuum)*dd+std::size_t(i)*d+j]=after;if(!isfinite(after))atomicExch(&err,6);
                }
                for(int i=t;i<d;i+=blockDim.x){double before=W[k*d+i];if(clicked)W[original*d+i]=before;W[vacuum*d+i]=before-delta[i];if(!isfinite(W[vacuum*d+i]))atomicExch(&err,6);}
                if(t==0){double weight=weights[k];if(clicked){weights[original]=weight/p1;weights[vacuum]=-weight*p[0]/p1;}else weights[k]=weight*p[0]/p0;}__syncthreads();
            }
            if(t==0){if(clicked)n*=2;peak=max(peak,n);double s=0,c=0;for(int k=0;k<n;++k)add72(weights[k],s,c);mass=s;if(!(isfinite(mass)&&mass>0))err=7;double condition=0;for(int k=0;k<n;++k){weights[k]/=mass;condition+=fabs(weights[k]);if(!isfinite(weights[k])||weights[k]==0)err=8;}maximum_condition=fmax(maximum_condition,condition);}
            __syncthreads();if(err){if(t==0)where=mode;break;}
            if(n>budget){
                for(int i=t;i<d;i+=blockDim.x){double value=0,c=0;for(int k=0;k<n;++k)add72(weights[k]*W[k*d+i],value,c);average[i]=value;}__syncthreads();
                for(std::size_t index=t;index<dd;index+=blockDim.x){int i=index%d,j=index/d;if(i>j)continue;double value=0,c=0;for(int k=0;k<n;++k)add72(weights[k]*(X[std::size_t(k)*dd+index]+(W[k*d+i]-average[i])*(W[k*d+j]-average[j])),value,c);X[index]=X[std::size_t(i)*d+j]=value;if(!isfinite(value))atomicExch(&err,6);}
                __syncthreads();for(int i=t;i<d;i+=blockDim.x)W[i]=average[i];if(t==0){n=1;weights[0]=1;++merges;}__syncthreads();
            }
        }
        if(err){if(t==0)where=mode;break;}
    }
    if(t==0){errors[2*row]=err;errors[2*row+1]=where;diagnostics[3*row]=merges;diagnostics[3*row+1]=peak;diagnostics[3*row+2]=maximum_condition;}
}
static void require72(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
static void cuda72(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
template<class T>struct Buffer72{T* p=nullptr;explicit Buffer72(std::size_t n){cuda72(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}~Buffer72(){if(p)cudaFree(p);}Buffer72(const Buffer72&)=delete;};
API int gc72_sample(int m,int d,int count,int budget,const double* initial,const double* a,const double* b,const double* means,const double* draws,unsigned char* output,double* probabilities,int* errors,double* diagnostics,double* timings,char* message,int cap){try{
    using Clock=std::chrono::steady_clock;auto start=Clock::now();require72(m>=1&&m<=65536&&d>=2&&d<=MAX_D72&&count>=0&&count<=1048576&&budget>=1&&budget<=MAX_B72&&initial&&a&&b&&means&&draws&&output&&probabilities&&errors&&diagnostics&&timings,"Invalid CUDA branch arguments");std::fill(timings,timings+6,0);if(count==0)return 0;
    std::size_t dd=std::size_t(d)*d,md=std::size_t(m)*d,nd=std::size_t(count)*d,nm=std::size_t(count)*m,slots=std::size_t(count)*budget;
    std::size_t bytes=(slots*dd+slots*d+std::size_t(count)*budget*3*d+dd+2*md+nd+2*nm+3*count)*sizeof(double)+nm+2*count*sizeof(int);std::size_t free,total;cuda72(cudaMemGetInfo(&free,&total));require72(bytes<free*3/4,"Explicit CUDA branch workspace exceeds 75 percent of free device memory");timings[4]=double(bytes);
    Buffer72<double> X(slots*dd),W(slots*d),V(std::size_t(count)*budget*3*d),in(dd),aa(md),bb(md),mm(nd),uu(nm),pp(nm),diag(3*count);Buffer72<unsigned char> out(nm);Buffer72<int> err(2*count);timings[0]=std::chrono::duration<double>(Clock::now()-start).count();
    start=Clock::now();cuda72(cudaMemcpy(in.p,initial,dd*sizeof(double),cudaMemcpyHostToDevice));cuda72(cudaMemcpy(aa.p,a,md*sizeof(double),cudaMemcpyHostToDevice));cuda72(cudaMemcpy(bb.p,b,md*sizeof(double),cudaMemcpyHostToDevice));cuda72(cudaMemcpy(mm.p,means,nd*sizeof(double),cudaMemcpyHostToDevice));cuda72(cudaMemcpy(uu.p,draws,nm*sizeof(double),cudaMemcpyHostToDevice));cuda72(cudaMemset(out.p,0,nm));cuda72(cudaMemset(pp.p,0,nm*sizeof(double)));timings[1]=std::chrono::duration<double>(Clock::now()-start).count();
    start=Clock::now();click72<<<count,128>>>(m,d,count,budget,in.p,aa.p,bb.p,mm.p,uu.p,X.p,W.p,V.p,out.p,pp.p,err.p,diag.p);cuda72(cudaGetLastError());cuda72(cudaDeviceSynchronize());timings[2]=std::chrono::duration<double>(Clock::now()-start).count();
    start=Clock::now();cuda72(cudaMemcpy(output,out.p,nm,cudaMemcpyDeviceToHost));cuda72(cudaMemcpy(probabilities,pp.p,nm*sizeof(double),cudaMemcpyDeviceToHost));cuda72(cudaMemcpy(errors,err.p,2*count*sizeof(int),cudaMemcpyDeviceToHost));cuda72(cudaMemcpy(diagnostics,diag.p,3*count*sizeof(double),cudaMemcpyDeviceToHost));timings[3]=std::chrono::duration<double>(Clock::now()-start).count();int first=-1;for(int row=0;row<count;++row)if(errors[2*row]){++timings[5];if(first<0)first=row;}
    if(first>=0)throw std::runtime_error("CUDA branch failure: row="+std::to_string(first)+" mode="+std::to_string(errors[2*first+1])+" code="+std::to_string(errors[2*first]));return 0;
}catch(const std::exception& e){if(message&&cap>0){auto n=std::min(std::strlen(e.what()),std::size_t(cap-1));std::memcpy(message,e.what(),n);message[n]=0;}return 1;}}
