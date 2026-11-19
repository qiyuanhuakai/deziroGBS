// Count-calibrated sampling. A rejected trial continues in the same output
// record's three independent counter domains. All retries have fresh words.
#include "gaussian_pair_table95_v1.cpp"

extern "C" __declspec(dllexport) int filter95_draw(void* ptr,int n,std::uint64_t experiment,
        std::uint64_t offset,int pnr,int workers,int bins,const std::uint64_t* thresholds,
        unsigned char* output,std::uint32_t* attempts) {
    if(!ptr||!thresholds||!output||!attempts||n<0||workers<1||workers>64||
       bins<1||(pnr!=0&&pnr!=1)||offset>0x100000000ULL||std::uint64_t(n)>0x100000000ULL-offset) return 1;
    const auto& p=*static_cast<const Table95*>(ptr);
    std::atomic<bool> failed{false};
    auto draw=[&](int begin,int end) {
        for(int i=begin;i<end;++i) {
            DomainRng84 component(experiment,0,123,offset+i),ports(experiment,0,124,offset+i),
                        selection(experiment,0,126,offset+i);
            auto* data=output+std::size_t(i)*p.m;
            for(std::uint32_t trial=1;trial<=2048;++trial) {
                auto start=std::size_t(component()&(p.centers-1))*p.m;
                auto* click=p.above.front().data()+start;
                int total=0;
                for(int j=0;j<p.m;j+=2) {
                    auto word=ports();
                    for(int h=0;h<std::min(2,p.m-j);++h) {
                        std::uint32_t u=static_cast<std::uint32_t>(word>>(h*32));
                        int count=u<click[j+h];
                        if(pnr&&count)
                            while(count<p.levels&&u<p.above[count][start+j+h]) ++count;
                        data[j+h]=static_cast<unsigned char>(count); total+=count;
                    }
                }
                auto uniform=static_cast<std::uint32_t>(selection());
                if(total<bins&&uniform<thresholds[total]) {attempts[i]=trial;break;}
                if(trial==2048) {attempts[i]=0;failed=true;}
            }
        }
    };
    if(n<2||workers==1) {draw(0,n);return failed?2:0;}
    std::vector<std::thread> pool;
    int threads=std::min(n,workers);
    try {
        for(int t=1;t<threads;++t) pool.emplace_back(draw,int(std::int64_t(t)*n/threads),int(std::int64_t(t+1)*n/threads));
        draw(0,n/threads);
    } catch(const std::exception&) {for(auto& t:pool)t.join();return 3;}
    for(auto& t:pool)t.join();
    return failed?2:0;
}

namespace {
using PolyFilter=std::vector<double>;
PolyFilter product95(const PolyFilter& left,const PolyFilter& right,int bins) {
    int size=std::min<int>(bins,int(left.size()+right.size()-1));
    PolyFilter result(size,0.);
    for(int k=0;k<std::min<int>(size,left.size());++k)
        for(int j=0;j<std::min<int>(size-k,right.size());++j) result[k+j]+=left[k]*right[j];
    while(result.size()>1&&result.back()<1e-19) result.pop_back();
    return result;
}
PolyFilter port95(const Table95& plan,int center,int port,bool pnr) {
    auto pos=std::size_t(center)*plan.m+port;
    double previous=1.;
    PolyFilter coeff;
    for(int level=0;level<(pnr?plan.levels:1)&&previous>0.;++level) {
        double upper=plan.above[level][pos]/word95;
        coeff.push_back(previous-upper);previous=upper;
    }
    if(previous>0.) coeff.push_back(previous);
    return coeff;
}
}
// For each center sum out the unobserved ports, retain click masks and the
// subtotal in the observed ports, then apply the same global count acceptance.
extern "C" __declspec(dllexport) int filter95_local(void* ptr,int pnr,int bins,
        const std::uint64_t* cut,int group,const int* ports,int workers,double* law) {
    if(!ptr||!cut||!ports||!law||group<1||group>8||workers<1||workers>64||
       bins<1||bins>2049||(pnr!=0&&pnr!=1)) return 1;
    const auto& plan=*static_cast<const Table95*>(ptr);
    std::vector<int> other;
    for(int j=0;j<group;++j) if(ports[j]<0||ports[j]>=plan.m||
        std::find(ports,ports+j,ports[j])!=ports+j) return 1;
    for(int j=0;j<plan.m;++j) if(std::find(ports,ports+group,j)==ports+group) other.push_back(j);
    try {
        int masks=1<<group;
        Dense rows=Dense::Zero(plan.centers,masks);
        std::vector<double> accept(bins);
        for(int j=0;j<bins;++j) accept[j]=double(cut[j])/word95;
        std::atomic<int> next{0};std::atomic<bool> ok{true};
        auto worker=[&]() {
            try {
                std::vector<PolyFilter> ext(std::max<std::size_t>(1,other.size())),inside(masks);
                for(int c=next++;c<plan.centers;c=next++) {
                    ext[0]={1.};
                    for(std::size_t j=0;j<other.size();++j) ext[j]=port95(plan,c,other[j],pnr);
                    for(std::size_t stride=1;stride<other.size();stride*=2)
                        for(std::size_t j=0;j+stride<other.size();j+=2*stride)
                            ext[j]=product95(ext[j],ext[j+stride],bins);
                    inside[0]={1.};
                    int max_sum=0;
                    for(int j=0;j<group;++j) {
                        auto clicked=port95(plan,c,ports[j],pnr);
                        double zero=clicked[0];clicked[0]=0.;
                        max_sum+=int(clicked.size()-1);
                        for(int k=0;k<(1<<j);++k) {
                            inside[k+(1<<j)]=product95(inside[k],clicked,bins);
                            for(double& v:inside[k]) v*=zero;
                        }
                    }
                    std::vector<double> integrated(max_sum+1,0.);
                    for(int s=0;s<=max_sum&&s<bins;++s)
                        for(int k=0;k<std::min<int>(ext[0].size(),bins-s);++k)
                            integrated[s]+=ext[0][k]*accept[k+s];
                    for(int k=0;k<masks;++k)
                        for(std::size_t s=0;s<inside[k].size();++s) rows(c,k)+=inside[k][s]*integrated[s];
                }
            } catch(const std::exception&) {ok=false;}
        };
        std::vector<std::thread> pool;
        try {for(int t=1;t<std::min(workers,plan.centers);++t) pool.emplace_back(worker);worker();}
        catch(const std::exception&) {for(auto& t:pool)t.join();return 3;}
        for(auto& t:pool)t.join();
        if(!ok) return 2;
        Eigen::Map<Vector95>(law,masks)=rows.colwise().mean().transpose();
        return 0;
    } catch(const std::exception&) {return 2;}
}
