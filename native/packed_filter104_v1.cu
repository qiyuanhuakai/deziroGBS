#include "continuation_gpu103_v1.cu"

__global__ void decide104(int groups,U64 seed,U32 off,const float* latent,const int* caps,
        const int* starts,const double* levels,const int* leaves,const U64* masks,
        const U64* shortcut,const unsigned char* direct,const unsigned char* alternate,
        const double* table,int nr,int nc,unsigned* packed,int* pass,int* error){
    const int row=blockIdx.x,lane=threadIdx.x;
    int click=0,ph=0;
    for(int g=lane;g<groups;g+=blockDim.x){
        const int id=row*groups+g;
        const double* cuts=levels+starts[g];
        int n=0;
        unsigned mask=0;
        if(latent[2*id]>=cuts[0]){
            double z=latent[2*id+1];
            n=z<cuts[1]?1:z<cuts[2]?2:3+upper103(cuts+3,caps[g]-3,z);
            U64 u=low103(block103(seed,off+row,g,195,0));
            if(n<=2){
                int bits=n==1?3:6;
                int k=g*72+(n==1?0:8)+int(u&((1u<<bits)-1));
                mask=(u>>bits)<shortcut[k]?direct[k]:alternate[k];
            }else mask=upper103(masks+(std::size_t(leaves[g])+n)*256,256,u>>11);
            if(!mask||mask>255||__popc(mask)>n)atomicExch(error,1);
        }
        packed[id]=(unsigned(n)<<8)|mask;
        click+=__popc(mask);ph+=n;
    }
    __shared__ int c[128],p[128];
    c[lane]=click;p[lane]=ph;
    __syncthreads();
    for(int stride=64;stride;stride>>=1){
        if(lane<stride){c[lane]+=c[lane+stride];p[lane]+=p[lane+stride];}
        __syncthreads();
    }
    if(!lane)pass[row]=unit103(low103(block103(seed,off+row,0,198,0)))<table[min(p[0],nr-1)*nc+min(c[0],nc-1)];
}

__global__ void expand104(int groups,int limit,int route,U64 seed,U32 off,const unsigned* packed,
        const int* pass,const int* prefix,const int* leaves,const double* prior,const double* weights,
        unsigned short* output,unsigned short* counts,int* error){
    int row=blockIdx.x;
    if(!pass[row]||prefix[row]>limit)return;
    int dest=prefix[row]-1;
    for(int g=threadIdx.x;g<groups;g+=blockDim.x){
        unsigned word=packed[row*groups+g],mask=word&255;
        int n=word>>8;
        counts[dest*groups+g]=n;
        unsigned short dst[8]={};
        if(!route){
            for(unsigned s=mask;s;s&=s-1)dst[__ffs(s)-1]=1;
        }else{
            unsigned draw=0;
            uint4 cache{};
            while(n && mask){
                if((mask&(mask-1))==0){dst[__ffs(mask)-1]+=n;break;}
                if(__popc(mask)==n){
                    for(unsigned s=mask;s;s&=s-1)++dst[__ffs(s)-1];
                    break;
                }
                const double* law=prior+(std::size_t(leaves[g])+n-1)*256;
                double w[16]={},total=0;
                for(unsigned s=mask;s;s&=s-1){
                    int j=__ffs(s)-1;
                    w[2*j]=weights[g*8+j]*law[mask];
                    w[2*j+1]=weights[g*8+j]*law[mask^(1u<<j)];
                    total+=w[2*j]+w[2*j+1];
                }
                if(!(total>0)){atomicExch(error,2);return;}
                if(!(draw&1))cache=block103(seed,off+row,g,196,draw/2);
                double roll=unit103((draw++&1)?high103(cache):low103(cache))*total;
                int last=-1;
                for(int j=0;j<16;++j)if(w[j]>0){last=j;roll-=w[j];if(roll<0)break;}
                ++dst[last/2];
                if(last&1)mask^=1u<<(last/2);
                --n;
            }
        }
        for(int k=0;k<8;++k)output[8*(dest*groups+g)+k]=dst[k];
    }
}
extern "C" __declspec(dllexport) int decide104_launch(int groups,int n,U64 seed,U32 off,const float* field,
    const int* caps,const int* starts,const double* levels,const int* leaves,const U64* masks,const U64* shortcut,
    const unsigned char* direct,const unsigned char* alternate,const double* table,int nr,int nc,
    unsigned* packed,int* pass,int* error,void* stream){
    decide104<<<n,128,0,(cudaStream_t)stream>>>(groups,seed,off,field,caps,starts,levels,leaves,masks,shortcut,direct,alternate,table,nr,nc,packed,pass,error);
    return int(cudaGetLastError());
}
extern "C" __declspec(dllexport) int expand104_launch(int groups,int n,int limit,int route,U64 seed,U32 off,
    const unsigned* packed,const int* pass,const int* prefix,const int* leaves,const double* prior,const double* weights,
    unsigned short* output,unsigned short* counts,int* error,void* stream){
    expand104<<<n,128,0,(cudaStream_t)stream>>>(groups,limit,route,seed,off,packed,pass,prefix,leaves,prior,weights,output,counts,error);
    return int(cudaGetLastError());
}
