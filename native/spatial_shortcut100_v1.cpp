#include "spatial_tree100_v1.cpp"

// Every nonzero subtree is visited at most once. Low-total tables integrate
// all unused intermediate choices, using separate physical payloads for clicks and photons.
extern "C" __declspec(dllexport) int shortcut100_draw(int leaves,int width,const int* children,
        const int* caps,const int* starts,const Word* root,const Word* splits,const int* leafstarts,
        const Word* masks,const double* prior,const double* ports,const int* offsets,const int* bits,
        const Word* cuts,const unsigned* alias,const unsigned* clicks,const unsigned* photons,
        int shots,int route,Word seed,Word offset,int workers,std::uint16_t* out,unsigned char* flags){
    if(leaves<1||leaves*width>=65535||width<1||width>8||!children||!caps||!starts||!root||!splits||
       !leafstarts||!masks||!prior||!ports||!offsets||!bits||!cuts||!alias||!clicks||!photons||
       !out||!flags||shots<1||workers<1||workers>64||route<0||route>1||offset>0xffffffffULL||
       offset+shots>0x100000000ULL)return 1;
    int top=2*leaves-2,size=1<<width;
    auto payload=route?photons:clicks;
    #pragma omp parallel num_threads(workers)
    {
        std::vector<std::pair<int,int>> active;active.reserve(64);
        #pragma omp for schedule(static)
        for(int r=0;r<shots;++r){
            DomainRng84 total(seed,0,168,offset+r),branch(seed,0,169,offset+r),det(seed,0,170,offset+r),
                        pnr(seed,0,171,offset+r),shorts(seed,0,172,offset+r);
            auto row=out+std::size_t(r)*leaves*width;std::fill(row,row+leaves*width,0);
            auto push=[&](int i,int n){if(n)active.emplace_back(i,n);};
            push(top,pick(root,caps[top]+1,total));unsigned char flag=0;
            while(!active.empty()){
                auto [i,n]=active.back();active.pop_back();
                if(n<=2 && offsets[2*i+n-1]>=0){
                    int id=2*i+n-1,pos=offsets[id],b=bits[id];
                    auto u=shorts();unsigned col=u&((1u<<b)-1);
                    unsigned k=(u>>b)<cuts[pos+col]?col:alias[pos+col];
                    unsigned code=payload[pos+k];++row[(code&65535)-1];
                    if(code>>16)++row[(code>>16)-1];
                }else if(i>=leaves){
                    int k=0;
                    if(n>caps[i]){flag|=1;k=n/2;}
                    else k=pick(splits+starts[i]+n*(n+1)/2,n+1,branch);
                    push(children[2*i+1],n-k);push(children[2*i],k);
                }else{
                    auto dst=row+i*width;
                    if(n>caps[i]){
                        flag|=1;
                        for(int h=0;h<n;++h){
                            double u=uniform84(det);int j=0;
                            while(j+1<width&&u>=ports[i*width+j]){u-=ports[i*width+j];++j;}
                            if(route)++dst[j];else dst[j]=1;
                        }
                        continue;
                    }
                    auto base=std::size_t(leafstarts[i])*size;
                    unsigned mask=pick(masks+base+std::size_t(n)*size,size,det);
                    if(route){if(!distribute(n,mask,width,ports+i*width,prior+base,pnr,dst))flag|=2;}
                    else while(mask){dst[std::countr_zero(mask)]=1;mask&=mask-1;}
                }
            }
            flags[r]=flag;
        }
    }
    return 0;
}
