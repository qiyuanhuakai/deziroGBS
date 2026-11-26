#include "spatial_forest102_v1.cpp"

// First normal decides vacuum. The second samples the positive count quantile.
// Within-group orthogonality keeps both uniforms independent given ancestors.
extern "C" __declspec(dllexport) int continuation103_draw(
        int groups,int shots,int pnr,int threads,Word seed,Word offset,
        const int* order,const int* parent,const int* second,const int* third,int classes,
        const Word* markers,const int* caps,const int* starts,const double* cuts,
        const double* factor,Gemm102 gemm,const int* leaves,const Word* masks,
        const double* prior,const double* weights,const Word* fastcut,const unsigned char* fastmask,
        const unsigned char* fallback,std::uint16_t* output,std::uint16_t* counts){
    if(groups<1 || shots<1 || threads<1 || threads>64 || pnr<0 || pnr>1 ||
       offset>0xffffffffULL || Word(shots)>0x100000000ULL-offset || !gemm)return 1;
    static const NormalLayers102 normal;
    int size=2*groups;
    constexpr int batch=64;
    int failed=0;
    #pragma omp parallel num_threads(threads) reduction(|:failed)
    {
        std::vector<double> noise(batch*size),latent(batch*size);
        #pragma omp for schedule(static)
        for(int off=0;off<shots;off+=batch){
            int rows=std::min(batch,shots-off);
            for(int r=0;r<rows;++r){
                DomainRng84 random(seed,0,190,offset+off+r);
                for(int j=0;j<size;++j)noise[r*size+j]=normal.draw(random);
            }
            const double one=1,zero=0;
            gemm("T","N",&size,&rows,&size,&one,factor,&size,noise.data(),&size,&zero,latent.data(),&size);
            for(int r=0;r<rows;++r){
                auto record=offset+off+r;
                DomainRng84 click(seed,0,191,record),photon(seed,0,192,record),mark(seed,0,193,record);
                auto ns=counts+std::size_t(off+r)*groups;
                auto dst=output+std::size_t(off+r)*groups*8;
                std::fill(dst,dst+groups*8,0);
                for(int k=0;k<groups;++k){
                    int g=order[k],p=parent[g],bins=caps[g]+1;
                    int row=p<0?0:ns[p];
                    if(second[g]>=0)row+=ns[second[g]]*(caps[p]+1);
                    if(third[g]>=0){
                        int x=ns[third[g]];
                        int b=x<=1?x:pick(markers+x*classes,classes,mark);
                        row+=b*(caps[second[g]]+1)*(caps[p]+1);
                    }
                    const double* t=cuts+starts[g]+row*bins;
                    const double* z=latent.data()+r*size+2*k;
                    if(z[0]<t[0]){ns[g]=0;continue;}
                    int n=bins==2 || z[1]<t[1]?1:bins==3 || z[1]<t[2]?2:
                        int(std::upper_bound(t+3,t+bins-1,z[1])-t);
                    ns[g]=n;
                    unsigned mask;
                    if(n<=2){
                        Word u=click();unsigned bits=n==1?3:6;
                        auto j=std::size_t(g)*72+(n==1?0:8)+(u&((1u<<bits)-1));
                        mask=(u>>bits)<fastcut[j]?fastmask[j]:fallback[j];
                        if(pnr && n==2 && std::has_single_bit(mask)){
                            dst[g*8+std::countr_zero(mask)]=2;
                            continue;
                        }
                    }else{
                        auto base=std::size_t(leaves[g])*256;
                        mask=pick(masks+base+n*256,256,click);
                        if(pnr){
                            if(!distribute(n,mask,8,weights+g*8,prior+base,photon,dst+g*8))failed=2;
                            continue;
                        }
                    }
                    while(mask){dst[g*8+std::countr_zero(mask)]=1;mask&=mask-1;}
                }
            }
        }
    }
    return failed;
}
