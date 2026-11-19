// Finite mixture of conditionally independent detector outputs; one shared
// normal center per record. Conditional cumulative thresholds use integer words.
#include "gaussian_pair_copula95_v1.cpp"
#include <atomic>
namespace {
struct Table95 {
    int m,levels,centers;
    std::vector<std::vector<std::uint32_t>> above;
};
constexpr double word95=4294967296.;
void hadamard95(Vector95& v) {
    for(int stride=1;stride<v.size();stride*=2)
        for(int start=0;start<v.size();start+=2*stride)
            for(int k=0;k<stride;++k) {
                double a=v[start+k],b=v[start+stride+k];
                v[start+k]=a+b;v[start+stride+k]=a-b;
            }
}
double upper95(double z) {return .5*std::erfc(z/std::sqrt(2.));}
double calibrate95(const Vector95& center,double sigma,double original,double target) {
    if(!std::isfinite(original)||target<1e-14||target>1-1e-14) return original;
    double lo=center.minCoeff()+sigma*original,hi=center.maxCoeff()+sigma*original;
    double point=std::clamp(original,lo,hi);
    for(int i=0;i<28;++i) {
        double p=0,density=0;
        for(double c:center) {
            double z=(point-c)/sigma;
            p+=upper95(z);density+=std::exp(-.5*z*z);
        }
        double diff=p/center.size()-target;
        if(std::abs(diff)<2e-13) break;
        if(diff>0) lo=point;else hi=point;
        double proposed=point+diff*center.size()*sigma*std::sqrt(2*pi95)/density;
        point=proposed>lo&&proposed<hi?proposed:(lo+hi)/2;
    }
    return point;
}
}
extern "C" __declspec(dllexport) int cube95_centers(int m,int width,int frames,const double* loadings,
        std::uint64_t seed,double* centers) {
    if(!loadings||!centers||m<1||m>4096||width<m||width>8192||(width&(width-1))||
       frames<1||frames>16||(frames&(frames-1))) return 1;
    try {
        Eigen::Map<const Dense> input(loadings,m,m);
        Eigen::Map<Dense> output(centers,2*width*frames,m);
        for(int f=0;f<frames;++f) {
            DomainRng84 rng(seed,0,125,f);
            Eigen::Matrix<double,3,-1> sign(3,width);
            for(int t=0;t<3;++t) for(int j=0;j<width;++j) sign(t,j)=(rng()&1)?1.:-1.;
            for(int i=0;i<m;++i) {
                Vector95 v=Vector95::Zero(width);v.head(m)=input.row(i).transpose();
                for(int t=0;t<3;++t) {v.array()*=sign.row(t).transpose().array();hadamard95(v);}
                v/=width;
                output.col(i).segment(2*f*width,width)=v;
                output.col(i).segment((2*f+1)*width,width)=-v;
            }
        }
        return 0;
    } catch(const std::exception&) {return 2;}
}
extern "C" __declspec(dllexport) void* cube95_prepare(int m,int n,int cap,const double* centers,
        double sigma,const double* original,int calibrate,int workers,double* marginal,double* shifts) {
    if(m<1||m>4096||n<1||(n&(n-1))||n>131072||cap<1||cap>64||!(sigma>0&&sigma<=1)||
       !centers||!original||!marginal||!shifts||workers<1||workers>64) return nullptr;
    try {
        auto plan=std::make_unique<Table95>();plan->m=m;plan->centers=n;plan->levels=cap+1;
        const auto cells=std::size_t(n)*m;
        for(int k=0;k<=cap;++k) plan->above.emplace_back(cells);
        Eigen::Map<const Dense> locations(centers,n,m);
        std::atomic<int> next{0};std::atomic<bool> valid{true};
        auto build=[&]() {
            try {
                for(int j=next++;j<m;j=next++) {
                    Vector95 column=locations.col(j);
                    double previous=-std::numeric_limits<double>::infinity();
                    for(int k=0;k<=cap;++k) {
                        auto idx=std::size_t(j)*(cap+1)+k;
                        double z=original[idx],target=upper95(z);
                        double cut=calibrate?calibrate95(column,sigma,z,target):z;
                        // Monotonicity is required for all PNR quantiles of a port.
                        cut=std::max(previous,cut);previous=cut;
                        shifts[idx]=std::isfinite(cut)&&std::isfinite(z)?cut-z:0;
                        std::uint64_t total=0;
                        for(int c=0;c<n;++c) {
                            double prob=upper95((cut-column[c])/sigma);
                            if(!(prob>=0&&prob<=1)) {valid=false;return;}
                            auto quant=static_cast<std::uint32_t>(std::min(word95-1,std::floor(word95*prob+.5)));
                            plan->above[k][std::size_t(c)*m+j]=quant;total+=quant;
                        }
                        marginal[idx]=double(total)/(word95*n);
                    }
                }
            } catch(const std::exception&) {valid=false;}
        };
        std::vector<std::thread> pool;
        try {for(int j=1;j<std::min(workers,m);++j) pool.emplace_back(build);build();}
        catch(const std::exception&) {for(auto& t:pool)t.join();return nullptr;}
        for(auto& t:pool)t.join();
        return valid?plan.release():nullptr;
    } catch(const std::exception&) {return nullptr;}
}
extern "C" __declspec(dllexport) void cube95_destroy(void* p) {delete static_cast<Table95*>(p);}
extern "C" __declspec(dllexport) int cube95_sample(void* ptr,int n,std::uint64_t exp,std::uint64_t offset,
        int pnr,int workers,unsigned char* out) {
    if(!ptr||!out||n<0||workers<1||workers>64||(pnr!=0&&pnr!=1)||offset>0x100000000ULL||
       std::uint64_t(n)>0x100000000ULL-offset) return 1;
    const auto& p=*static_cast<const Table95*>(ptr);
    auto run=[&](int begin,int end) {
        for(int i=begin;i<end;++i) {
            DomainRng84 component(exp,0,123,offset+i),ports(exp,0,124,offset+i);
            auto start=std::size_t(component()&(p.centers-1))*p.m;
            auto* first=p.above[0].data()+start;
            auto* data=out+std::size_t(i)*p.m;
            for(int j=0;j<p.m;j+=2) {
                auto u=ports();
                data[j]=std::uint32_t(u)<first[j];
                if(j+1<p.m) data[j+1]=std::uint32_t(u>>32)<first[j+1];
                if(pnr) for(int a=j;a<std::min(p.m,j+2);++a) if(data[a]) {
                    auto v=std::uint32_t(a==j?u:u>>32);
                    while(data[a]<p.levels&&v<p.above[data[a]][start+a]) ++data[a];
                }
            }
        }
    };
    if(n<2||workers==1) {run(0,n);return 0;}
    std::vector<std::thread> pool;int threads=std::min(n,workers);
    try {
        for(int t=1;t<threads;++t) pool.emplace_back(run,int(std::int64_t(t)*n/threads),int(std::int64_t(t+1)*n/threads));
        run(0,n/threads);
    } catch(const std::exception&) {for(auto& t:pool)t.join();return 3;}
    for(auto& t:pool)t.join();
    return 0;
}
extern "C" __declspec(dllexport) int cube95_pair_stats(void* ptr,int pnr,double* first,double* second) {
    if(!ptr||!first||!second||!multiply95||(pnr!=0&&pnr!=1)) return 1;
    try {
        const auto& p=*static_cast<const Table95*>(ptr);
        Dense mean(p.centers,p.m);Vector95 diag=Vector95::Zero(p.m);
        int levels=pnr?p.levels:1;
        for(int c=0;c<p.centers;++c) for(int j=0;j<p.m;++j) {
            auto pos=std::size_t(c)*p.m+j;
            double expected=0,squared=0;
            for(int k=0;k<levels;++k) {
                double above=p.above[k][pos]/word95;
                expected+=above;squared+=(2*k+1)*above;
            }
            mean(c,j)=expected;diag[j]+=squared;
        }
        multiply95(101,112,111,p.m,p.m,p.centers,1./p.centers,mean.data(),p.m,mean.data(),p.m,0,second,p.m);
        Eigen::Map<Dense>(second,p.m,p.m).diagonal()=diag/p.centers;
        Eigen::Map<Vector95>(first,p.m)=mean.colwise().mean().transpose();
        return 0;
    } catch(const std::exception&) {return 2;}
}
extern "C" __declspec(dllexport) int cube95_local(void* ptr,int size,const int* ports,double* law) {
    if(!ptr||!ports||!law||size<1||size>12) return 1;
    const auto& p=*static_cast<const Table95*>(ptr);
    for(int j=0;j<size;++j) {
        if(ports[j]<0||ports[j]>=p.m) return 1;
        for(int k=0;k<j;++k) if(ports[j]==ports[k]) return 1;
    }
    std::fill(law,law+(1<<size),0.);
    std::vector<double> temp(1<<size);
    for(int c=0;c<p.centers;++c) {
        temp[0]=1./p.centers;
        for(int j=0;j<size;++j) {
            double chance=p.above[0][std::size_t(c)*p.m+ports[j]]/word95;
            for(int mask=0;mask<(1<<j);++mask) {
                temp[mask+(1<<j)]=temp[mask]*chance;temp[mask]*=1-chance;
            }
        }
        for(int i=0;i<(1<<size);++i) law[i]+=temp[i];
    }
    return 0;
}
extern "C" __declspec(dllexport) int cube95_counts(void* ptr,int limit,int workers,double* law) {
    if(!ptr||!law||workers<1||workers>64) return 1;
    const auto& p=*static_cast<Table95*>(ptr);
    if(limit<0||limit>p.m) return 1;
    try {
        Dense partial=Dense::Zero(p.centers,limit+1);
        std::atomic<int> next{0};
        auto work=[&](){
            for(int c=next++;c<p.centers;c=next++) {
                auto* v=partial.data()+std::size_t(c)*(limit+1);v[0]=1.;
                for(int j=0;j<p.m;++j) {
                    double click=p.above[0][std::size_t(c)*p.m+j]/word95;
                    for(int k=std::min(j+1,limit);k>0;--k) v[k]=(1-click)*v[k]+click*v[k-1];
                    v[0]*=1-click;
                }
            }
        };
        std::vector<std::thread> pool;
        try {for(int j=1;j<std::min(workers,p.centers);++j) pool.emplace_back(work);work();}
        catch(const std::exception&) {for(auto& t:pool)t.join();return 3;}
        for(auto& t:pool)t.join();
        Eigen::Map<Vector95>(law,limit+1)=partial.colwise().mean().transpose();return 0;
    } catch(const std::exception&) {return 2;}
}

// Positive-coefficient convolutions keep the mass dropped by truncation visible
// as one minus the returned sum. Index cap+1 represents an overflow observation.
extern "C" __declspec(dllexport) int cube95_photon_law(void* ptr,int limit,int workers,double* law) {
    if(!ptr||!law||limit<1||limit>2048||workers<1||workers>64) return 1;
    const auto& plan=*static_cast<const Table95*>(ptr);
    try {
        Dense rows=Dense::Zero(plan.centers,limit+1);
        std::atomic<int> next{0};
        std::atomic<bool> ok{true};
        auto worker=[&]() {
            try {
                std::vector<std::vector<double>> polys(plan.m);
                for(int c=next++;c<plan.centers;c=next++) {
                    for(int j=0;j<plan.m;++j) {
                        auto pos=std::size_t(c)*plan.m+j;
                        auto& poly=polys[j];poly.clear();
                        double previous=1.;
                        for(int k=0;k<plan.levels&&previous>0.;++k) {
                            double above=plan.above[k][pos]/word95;
                            poly.push_back(previous-above);
                            previous=above;
                        }
                        if(previous>0.) poly.push_back(previous);
                    }
                    for(int stride=1;stride<plan.m;stride*=2)
                        for(int j=0;j+stride<plan.m;j+=2*stride) {
                            const auto& left=polys[j];const auto& right=polys[j+stride];
                            int size=std::min<int>(limit+1,int(left.size()+right.size()-1));
                            std::vector<double> product(size,0.);
                            for(int k=0;k<std::min<int>(size,left.size());++k) {
                                int n=std::min<int>(size-k,right.size());
                                for(int t=0;t<n;++t) product[k+t]+=left[k]*right[t];
                            }
                            while(product.size()>1&&product.back()<1e-19) product.pop_back();
                            polys[j].swap(product);
                        }
                    for(int k=0;k<std::min<int>(limit+1,polys[0].size());++k) rows(c,k)=polys[0][k];
                }
            } catch(const std::exception&) {ok=false;}
        };
        std::vector<std::thread> threads;
        try {for(int t=1;t<std::min(workers,plan.centers);++t) threads.emplace_back(worker);worker();}
        catch(const std::exception&) {for(auto& thread:threads) thread.join();return 3;}
        for(auto& thread:threads) thread.join();
        if(!ok) return 2;
        Eigen::Map<Vector95>(law,limit+1)=rows.colwise().mean().transpose();
        return 0;
    } catch(const std::exception&) {return 2;}
}

// Flat index uses the first physical port as the least significant digit.
extern "C" __declspec(dllexport) int cube95_box(void* ptr,int cutoff,double* law) {
    if(!ptr||!law||cutoff<0) return 1;
    const auto& plan=*static_cast<const Table95*>(ptr);
    if(plan.m>4||cutoff>=plan.levels) return 1;
    int size=1,base=cutoff+1;
    for(int j=0;j<plan.m;++j) {
        if(size>2000000/base) return 1;
        size*=base;
    }
    try {
        std::fill(law,law+size,0.);
        std::vector<double> row(size);
        for(int c=0;c<plan.centers;++c) {
            int width=1;row[0]=1./plan.centers;
            for(int j=0;j<plan.m;++j) {
                auto pos=std::size_t(c)*plan.m+j;
                for(int k=cutoff;k>=0;--k) {
                    double prev=k?plan.above[k-1][pos]/word95:1.;
                    double mass=prev-plan.above[k][pos]/word95;
                    for(int t=0;t<width;++t) row[k*width+t]=row[t]*mass;
                }
                width*=base;
            }
            for(int i=0;i<size;++i) law[i]+=row[i];
        }
        return 0;
    } catch(const std::exception&) {return 2;}
}
