#include "continuation_gpu103_v1.cu"

__device__ U64 collision_roll110(U64 seed,U32 row,U32 group,U32 purpose,U32 draw){
    uint4 c=make_uint4(draw,11020+purpose,row,group);
    U32 a=U32(seed),b=U32(seed>>32);
    for(int i=0;i<10;++i){
        U32 x=0xD2511F53u*c.x,y=0xCD9E8D57u*c.z;
        c=make_uint4(__umulhi(0xCD9E8D57u,c.z)^c.y^a,y,__umulhi(0xD2511F53u,c.x)^c.w^b,x);
        a+=0x9E3779B9u;b+=0xBB67AE85u;
    }
    return low103(c);
}
template<class T>
__device__ int3 capacity110(const T* x,int mask,int width,bool pnr){
    int k=0,active=0,free=0,photons=0;
    for(int j=0;j<width;++j){
        k+=x[j]>0;
        if(!((mask>>j)&1)){++free;active+=x[j]>0;photons+=x[j];}
    }
    int maximum=pnr?min(free,photons):(active?free:0);
    return make_int3(k,maximum-active,active-int(active>0));
}
__device__ int select110(const int* x,const U32* weights,int width,int mask,int role,U64 rv){
    U64 c[8],total=0;
    for(int j=0;j<width;++j){
        U64 w=0;
        if(!((mask>>j)&1)){
            if(role==0 && !x[j])w=1u+weights[j];
            if(role==1 && x[j]>1)w=x[j]-1;
            if(role==2 && x[j])w=1;
            if(role==3 && x[j])w=1u+weights[j];
        }
        total+=w;c[j]=total;
    }
    if(!total)return -1;
    U64 at=__umul64hi(rv,total);int j=0;while(c[j]<=at)++j;return j;
}
template<class T>
__global__ void collision110(int groups,int width,int pn,U64 seed,U32 offset,const int* frozen,
    const U32* weights,const U64* source,const U64* target,T* out,int* audit,int* error){
    int row=blockIdx.x,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    int first=int((size_t(tid)*groups)/blockDim.x),last=int((size_t(tid+1)*groups)/blockDim.x);
    T* record=out+size_t(row)*groups*width;
    int3 sum=make_int3(0,0,0);
    for(int g=first;g<last;++g){
        auto c=capacity110(record+g*width,frozen[g],width,pn);sum.x+=c.x;sum.y+=c.y;sum.z+=c.z;
    }
    for(int d=1;d<32;d*=2){
        int x=__shfl_up_sync(0xffffffff,sum.x,d),y=__shfl_up_sync(0xffffffff,sum.y,d),z=__shfl_up_sync(0xffffffff,sum.z,d);
        if(lane>=d){sum.x+=x;sum.y+=y;sum.z+=z;}
    }
    __shared__ int3 ends[8];__shared__ int delta,mass;__shared__ U64 phase;
    if(lane==31)ends[warp]=sum;__syncthreads();
    if(!tid){
        for(int w=1;w<blockDim.x/32;++w){
            ends[w].x+=ends[w-1].x;ends[w].y+=ends[w-1].y;ends[w].z+=ends[w-1].z;
        }
        auto all=ends[blockDim.x/32-1];int k=all.x,limit=width*groups;
        U64 lower=k?source[k-1]:0,upper=source[k];
        U64 q=lower+__umul64hi(collision_roll110(seed,row+offset,0,0,0),upper-lower);
        q=min(q,(1ULL<<52)-1);
        int lo=0,hi=limit;while(lo<hi){int mid=lo+(hi-lo)/2;if(target[mid]<=q)lo=mid+1;else hi=mid;}
        delta=max(-all.z,min(all.y,lo-k));mass=delta>0?all.y:all.z;
        phase=__umul64hi(collision_roll110(seed,row+offset,0,1,0),U64(mass));
        if(audit){audit[2*row]=lo;audit[2*row+1]=k+delta;}
    }
    __syncthreads();if(!delta)return;
    bool add=delta>0;
    int last_cap=add?sum.y:sum.z;
    int prev=__shfl_up_sync(0xffffffff,last_cap,1);
    int at=(lane?prev:0)+(warp?(add?ends[warp-1].y:ends[warp-1].z):0);
    for(int g=first;g<last;++g){
        auto c=capacity110(record+g*width,frozen[g],width,pn);int next=at+(add?c.y:c.z);
        int changes=int((U64(next)*abs(delta)+phase)/mass-(U64(at)*abs(delta)+phase)/mass);at=next;
        if(!changes)continue;
        int values[8],mask=frozen[g];for(int j=0;j<width;++j)values[j]=record[g*width+j];
        for(int j=0;j<changes;++j){
            int from=-1,to=-1;
            if(add){
                to=select110(values,weights+g*width,width,mask,0,collision_roll110(seed,row+offset,g,2,2*j));
                if(pn)from=select110(values,weights+g*width,width,mask,1,collision_roll110(seed,row+offset,g,2,2*j+1));
                if(to<0||(pn&&from<0)){atomicMax(error,2);return;}
                ++values[to];if(pn)--values[from];
            }else{
                from=select110(values,weights+g*width,width,mask,2,collision_roll110(seed,row+offset,g,3,2*j));
                if(from<0){atomicMax(error,2);return;}
                int v=values[from];values[from]=0;
                if(pn){
                    to=select110(values,weights+g*width,width,mask,3,collision_roll110(seed,row+offset,g,3,2*j+1));
                    if(to<0){atomicMax(error,2);return;}values[to]+=v;
                }
            }
        }
        for(int j=0;j<width;++j){
            if(sizeof(T)==1 && values[j]>255){atomicMax(error,1);return;}
            record[g*width+j]=T(values[j]);
        }
    }
}
extern "C" __declspec(dllexport) int collision_emit110(int n,int groups,int width,int pn,int wide,U64 seed,U32 offset,
    const int* frozen,const U32* weights,const U64* source,const U64* target,void* out,int* audit,int* error,cudaStream_t stream){
    if(width<1||width>8||groups<=0||groups>1000000)return 1;
    int threads=groups<128?64:256;
    if(wide)collision110<<<n,threads,0,stream>>>(groups,width,pn,seed,offset,frozen,weights,source,target,(U32*)out,audit,error);
    else collision110<<<n,threads,0,stream>>>(groups,width,pn,seed,offset,frozen,weights,source,target,(unsigned char*)out,audit,error);
    return int(cudaGetLastError());
}
