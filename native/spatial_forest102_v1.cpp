#include "spatial_tree100_v1.cpp"
#include <numbers>

using Gemm102=void(*)(const char*,const char*,const int*,const int*,const int*,const double*,
                     const double*,const int*,const double*,const int*,const double*,double*,const int*);

// Equal-area horizontal layers for a standard normal. A fast rectangle draw
// uses disjoint bits for layer, sign, and horizontal coordinate.
struct NormalLayers102 {
    static constexpr int layers=256;
    static constexpr double tail=3.6541528853610088;
    std::array<double,layers> width{},height{},inner{};
    NormalLayers102(){
        double v=tail*std::exp(-tail*tail/2)+std::sqrt(std::numbers::pi/2)*std::erfc(tail/std::sqrt(2.));
        width.back()=tail;
        height.back()=std::exp(-tail*tail/2);
        width[0]=v/height.back();
        height[0]=1.;
        for(int i=layers-2;i>0;--i){
            height[i]=height[i+1]+v/width[i+1];
            width[i]=std::sqrt(-2*std::log(height[i]));
        }
        inner[0]=tail;
        for(int i=2;i<layers;++i) inner[i]=width[i-1];
    }
    double draw(DomainRng84& rng) const {
        for(;;){
            Word w=rng();
            int layer=w&255;
            double x=double(w>>11)*0x1.0p-53*width[layer];
            if(x>=inner[layer]){
                if(!layer){
                    double excess;
                    do {
                        excess=-std::log1p(-uniform84(rng))/tail;
                    } while(-2*std::log1p(-uniform84(rng))<excess*excess);
                    x=tail+excess;
                }else if(height[layer]+uniform84(rng)*(height[layer-1]-height[layer])>=std::exp(-.5*x*x)){
                    continue;
                }
            }
            return (w&256)?-x:x;
        }
    }
};

extern "C" __declspec(dllexport) int forest102_draw(
        int groups,int width,int shots,int route,int field,int workers,Word seed,Word offset,
        const int* order,const int* parent,const int* second,const int* third,int class_count,const Word* classes,const int* caps,const int* starts,const Word* cdf,
        const double* thresholds,const double* factor,Gemm102 gemm,const int* leafstarts,
        const Word* masks,const double* prior,const double* weights,const Word* shortcut_cut,
        const unsigned char* shortcut_mask,const unsigned char* shortcut_fallback,std::uint16_t* out,std::uint16_t* groupcounts){
    if(groups<1 || width<1 || width>8 || shots<1 || workers<1 || workers>64 ||
       route<0 || route>1 || field<0 || field>1 || (field && !gemm) ||
       offset>0xffffffffULL || Word(shots)>0x100000000ULL-offset) return 1;
    static const NormalLayers102 normal;
    constexpr int batch=64;
    int failed=0;
    #pragma omp parallel num_threads(workers) reduction(|:failed)
    {
        std::vector<double> noise(field?batch*groups:0),latent(noise.size());
        #pragma omp for schedule(static)
        for(int off=0;off<shots;off+=batch){
            int rows=std::min(batch,shots-off);
            if(field){
                for(int r=0;r<rows;++r){
                    DomainRng84 rng(seed,0,180,offset+off+r);
                    for(int i=0;i<groups;++i) noise[r*groups+i]=normal.draw(rng);
                }
                // C-order lower Cholesky times a column-major batch of innovations.
                const double one=1,zero=0;
                gemm("T","N",&groups,&rows,&groups,&one,factor,&groups,noise.data(),&groups,&zero,latent.data(),&groups);
            }
            for(int r=0;r<rows;++r){
                auto record=offset+off+r;
                DomainRng84 count(seed,0,181,record),click(seed,0,182,record),photon(seed,0,183,record),
                            marker(seed,0,184,record);
                auto ns=groupcounts+std::size_t(off+r)*groups;
                auto dst=out+std::size_t(off+r)*groups*width;
                std::fill(dst,dst+groups*width,0);
                for(int k=0;k<groups;++k){
                    int g=order[k],p=parent[g],bins=caps[g]+1;
                    int condition=p<0?0:ns[p];
                    if(second[g]>=0) condition+=ns[second[g]]*(caps[p]+1);
                    if(third[g]>=0){
                        int n0=ns[third[g]];
                        int mark=n0<=1?n0:pick(classes+n0*class_count,class_count,marker);
                        condition+=mark*(caps[second[g]]+1)*(caps[p]+1);
                    }
                    auto row=starts[g]+condition*bins;
                    int n;
                    if(field){
                        const double* first=thresholds+row;
                        double u=latent[r*groups+k];
                        n=bins==1 || u<first[0]?0:bins==2 || u<first[1]?1:
                          bins==3 || u<first[2]?2:int(std::upper_bound(first+3,first+bins-1,u)-first);
                    }else n=pick(cdf+row,bins,count);
                    ns[g]=n;
                    if(!n) continue;
                    unsigned words=1u<<width;
                    std::size_t base=std::size_t(leafstarts[g])*words;
                    unsigned mask;
                    if(n<=2 && width==8){
                        Word u=click();
                        unsigned bits=n==1?3:6;
                        auto id=std::size_t(g)*72+(n==1?0:8)+(u&((1u<<bits)-1));
                        mask=(u>>bits)<shortcut_cut[id]?shortcut_mask[id]:shortcut_fallback[id];
                        if(route && n==2 && std::has_single_bit(mask)){
                            dst[g*width+std::countr_zero(mask)]=2;
                            continue;
                        }
                        while(mask){
                            dst[g*width+std::countr_zero(mask)]=1;
                            mask&=mask-1;
                        }
                        continue;
                    }
                    mask=pick(masks+base+n*words,words,click);
                    if(route){
                        if(!distribute(n,mask,width,weights+g*width,prior+base,photon,dst+g*width)) failed=2;
                    }else while(mask){
                        dst[g*width+std::countr_zero(mask)]=1;
                        mask&=mask-1;
                    }
                }
            }
        }
    }
    return failed;
}
