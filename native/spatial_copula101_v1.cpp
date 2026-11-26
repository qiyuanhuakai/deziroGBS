#include "spatial_tree100_v1.cpp"
#include <queue>
#include <random>

// cuts[g,n] is the normal boundary between counts n and n+1. The last
// count has no upper boundary. Each route emits physically ordered ports.
extern "C" __declspec(dllexport) int spatial101_draw(
        int groups, int width, int shots, int route, int lock_total, int workers,
        Word seed, Word offset, const double* lower, const double* cuts, int stride,
        const double* direction, const double* score, double mean, double sigma,
        const int* caps, const Word* root, int root_size, const int* starts,
        const Word* masks, const double* prior, const double* weights,
        std::uint16_t* out, std::uint16_t* totals){
    if(groups<1 || width<1 || width>8 || shots<1 || workers<1 ||
       offset>0xffffffffULL || Word(shots)>0x100000000ULL-offset ||
       route<0 || route>1 || lock_total<0 || lock_total>1 || !(sigma>0)) return 1;
    int failed=0;
    const int words=1<<width;
    #pragma omp parallel num_threads(workers) reduction(|:failed)
    {
        std::vector<double> normals(groups), z(groups);
        std::vector<int> counts(groups);
        using Event=std::pair<double,int>;
        std::vector<Event> heap;
        heap.reserve(groups);
        #pragma omp for schedule(static)
        for(int r=0;r<shots;++r){
            DomainRng84 field(seed,0,175,offset+r), root_rng(seed,0,176,offset+r),
                        click_rng(seed,0,177,offset+r), photon_rng(seed,0,178,offset+r);
            std::normal_distribution<double> normal;
            for(double& x:normals) x=normal(field);
            double projection=0;
            for(int i=0;i<groups;++i){
                const double* row=lower+std::size_t(i)*groups;
                double value=0;
                for(int j=0;j<=i;++j) value+=row[j]*normals[j];
                z[i]=value;
                projection+=score[i]*value;
            }
            int target=lock_total?pick(root,root_size,root_rng):0;
            // Translation along direction changes the starting point only.
            double shift=lock_total?(target-mean)/sigma-projection:0.;
            int sum=0;
            for(int i=0;i<groups;++i){
                z[i]+=direction[i]*shift;
                const double* threshold=cuts+std::size_t(i)*stride;
                counts[i]=int(std::upper_bound(threshold,threshold+caps[i],z[i])-threshold);
                sum+=counts[i];
            }
            if(lock_total && sum!=target){
                const int step=sum<target?1:-1;
                auto next=[&](int i){
                    int boundary=counts[i]-(step<0);
                    if(boundary>=0 && boundary<caps[i]){
                        double distance=step*(cuts[std::size_t(i)*stride+boundary]-z[i])/direction[i];
                        if(std::isfinite(distance)) heap.emplace_back(distance,i);
                    }
                };
                heap.clear();
                for(int i=0;i<groups;++i) next(i);
                std::make_heap(heap.begin(),heap.end(),std::greater<Event>());
                while(sum!=target && !heap.empty()){
                    std::pop_heap(heap.begin(),heap.end(),std::greater<Event>());
                    int i=heap.back().second;
                    heap.pop_back();
                    counts[i]+=step;
                    sum+=step;
                    auto old=heap.size();
                    next(i);
                    if(heap.size()>old) std::push_heap(heap.begin(),heap.end(),std::greater<Event>());
                }
                if(sum!=target) failed=1;
            }
            totals[r]=lock_total?target:sum;
            auto dst=out+std::size_t(r)*groups*width;
            std::fill(dst,dst+groups*width,0);
            for(int i=0;i<groups;++i){
                int n=counts[i];
                if(!n) continue;
                std::size_t base=std::size_t(starts[i])*words;
                unsigned mask=pick(masks+base+std::size_t(n)*words,words,click_rng);
                if(route){
                    if(!distribute(n,mask,width,weights+i*width,prior+base,photon_rng,dst+i*width)) failed=1;
                }else while(mask){
                    dst[i*width+std::countr_zero(mask)]=1;
                    mask&=mask-1;
                }
            }
        }
    }
    return failed;
}
