// Contiguous overlapping Gaussian click marginals, shared by all records.
#include "gaussian_rng_domain84_v1.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace {
using Dense = Eigen::Matrix<double,-1,-1,Eigen::RowMajor>;
using Extended = Eigen::Matrix<long double,-1,-1>;
using Column = Eigen::Matrix<long double,-1,1>;
struct Tile {
    int first, width, history;
    std::vector<double> cdf;
};
struct Plan { int modes; std::vector<Tile> tiles; };
long double vacuum(const Extended& q, const Column& mean, int mask, int n) {
    if (!mask) return 1;
    int indices[24], count=0;
    for (int i=0;i<n;++i) if (mask&(1<<i)) {
        indices[count++]=2*i;
        indices[count++]=2*i+1;
    }
    Extended sub(count,count);
    Column v(count);
    for (int i=0;i<count;++i) {
        v[i]=mean[indices[i]];
        for(int j=0;j<count;++j) sub(i,j)=q(indices[i],indices[j]);
    }
    Eigen::LLT<Extended> factor(sub);
    if (factor.info()!=Eigen::Success) throw std::domain_error("nonpositive covariance");
    Column whitened=factor.matrixL().solve(v);
    long double exponent=-whitened.squaredNorm()/2;
    for(int i=0;i<count;++i) exponent-=std::log(factor.matrixLLT()(i,i));
    return std::exp(exponent);
}
}

extern "C" __declspec(dllexport) int window95_moments(int m,int n,const double* optical,
        const double* t,const double* alpha,const double* noise,double* q,double* mu) {
    if (m<1||m>4096||n<1||n>m||!optical||!t||!alpha||!noise||!q||!mu) return 1;
    try {
        Dense transfer(2*m,2*n);
        for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
            auto z=optical+2*(i*n+j);
            transfer(i,j)=transfer(i+m,j+n)=z[0];
            transfer(i+m,j)=z[1];transfer(i,j+n)=-z[1];
        }
        Eigen::Map<const Dense> l(noise,2*n,2*n);
        Dense core=2*l*l.transpose();
        Eigen::VectorXd displacement(2*n);
        for(int j=0;j<n;++j) {
            if (!(std::abs(t[j])<1)) return 1;
            core(j,j)+=t[j]/(1-t[j]);core(j+n,j+n)-=t[j]/(1+t[j]);
            displacement[j]=std::sqrt(2.)*alpha[2*j];
            displacement[j+n]=std::sqrt(2.)*alpha[2*j+1];
        }
        Eigen::Map<Dense> output(q,2*m,2*m);
        Eigen::Map<Eigen::VectorXd> mean(mu,2*m);
        output.noalias()=(transfer*core)*transfer.transpose();
        output.diagonal().array()+=1;
        mean.noalias()=transfer*displacement;
        return output.allFinite()&&mean.allFinite()?0:2;
    } catch(const std::exception&) {return 3;}
}

extern "C" __declspec(dllexport) void* window95_prepare(int m,int width,int history,int sectors,
        const double* q,const double* mu,double* diagnostics) {
    if(m<1||m>4096||width<1||history<0||width>12||history>12-width||sectors<1||sectors>8||
       !q||!mu||!diagnostics) return nullptr;
    std::fill(diagnostics,diagnostics+4,0.);
    try {
        auto plan=std::make_unique<Plan>();plan->modes=m;
        for(int first=0;first<m;first+=width) {
            Tile tile{first,std::min(width,m-first),std::min(history,first),{}};
            int n=tile.width+tile.history, size=1<<n;
            std::vector<long double> mass(size,1);
            for(int s=0;s<sectors;++s) {
                Eigen::Map<const Dense> all(q+std::size_t(s)*4*m*m,2*m,2*m);
                Extended sub(2*n,2*n); Column v(2*n);
                for(int i=0;i<2*n;++i) {
                    int a=first-tile.history+i/2+(i%2)*m;
                    v[i]=mu[2*m*s+a];
                    for(int j=0;j<2*n;++j) {
                        int b=first-tile.history+j/2+(j%2)*m;
                        sub(i,j)=(static_cast<long double>(all(a,b))+all(b,a))/2;
                    }
                }
                for(int mask=1;mask<size;++mask) mass[mask]*=vacuum(sub,v,mask,n);
            }
            for(int bit=1;bit<size;bit*=2)
                for(int mask=0;mask<size;++mask)
                    if (!(mask&bit)) mass[mask]-=mass[mask|bit];
            int outcomes=1<<tile.width;
            tile.cdf.resize(size);
            for(int prior=0;prior<(1<<tile.history);++prior) {
                long double sum=0;
                auto* row=tile.cdf.data()+prior*outcomes;
                for(int bits=0;bits<outcomes;++bits) {
                    int click=prior|(bits<<tile.history);
                    long double p=mass[size-1-click];
                    if(!std::isfinite(p)||p < -2e-13L) return nullptr;
                    if(p<0) diagnostics[0]+=static_cast<double>(-p);
                    sum+=std::max(0.L,p);
                    row[bits]=static_cast<double>(sum);
                }
                if(sum==0) {++diagnostics[1];continue;}
                for(int bits=0;bits<outcomes;++bits) row[bits]/=static_cast<double>(sum);
                row[outcomes-1]=1;
            }
            diagnostics[2]+=size;
            plan->tiles.push_back(std::move(tile));
        }
        diagnostics[3]=plan->tiles.size();
        return plan.release();
    } catch(const std::exception&) {return nullptr;}
}
extern "C" __declspec(dllexport) void window95_destroy(void* p) { delete static_cast<Plan*>(p); }
extern "C" __declspec(dllexport) int window95_sample(void* p,void* alternate,int count,std::uint64_t exp,
        std::uint64_t offset,unsigned char* bits) {
    if(!p||count<0||!bits||offset>0x100000000ULL||std::uint64_t(count)>0x100000000ULL-offset) return 1;
    const auto* first=static_cast<const Plan*>(p);
    const auto* second=static_cast<const Plan*>(alternate);
    if(second && second->modes!=first->modes) return 1;
    for(int r=0;r<count;++r) {
        DomainRng84 rng(exp,0,Click84,offset+r);
        DomainRng84 component(exp,0,Resplitting84,offset+r);
        const auto& plan=*(second && (component()&1) ? second:first);
        auto* record=bits+std::size_t(r)*plan.modes;
        for(const auto& tile:plan.tiles) {
            int context=0;
            for(int j=0;j<tile.history;++j) context|=record[tile.first-tile.history+j]<<j;
            int outcomes=1<<tile.width;
            const double* row=tile.cdf.data()+outcomes*context;
            int draw=static_cast<int>(std::upper_bound(row,row+outcomes,uniform84(rng))-row);
            if(draw==outcomes) return 2;
            for(int j=0;j<tile.width;++j) record[tile.first+j]=(draw>>j)&1;
        }
    }
    return 0;
}

// Exponential change of measure by total count. The backward messages sum
// future factors and retain only the separator. Common rescaling cancels.
extern "C" __declspec(dllexport) void* window95_tilt(void* ptr,int history,double slope) {
    if(!ptr||history<0||history>11||!std::isfinite(slope)||std::abs(slope)>4) return nullptr;
    try {
        auto plan=std::make_unique<Plan>(*static_cast<Plan*>(ptr));
        std::vector<double> future(1<<history,1.),prior(future.size());
        for(auto it=plan->tiles.rbegin();it!=plan->tiles.rend();++it) {
            auto& tile=*it;
            if(tile.history!=std::min(history,tile.first)) return nullptr;
            int size=1<<tile.width;
            std::fill(prior.begin(),prior.end(),0.);
            for(int ctx=0;ctx<(1<<tile.history);++ctx) {
                auto* row=tile.cdf.data()+ctx*size;
                double previous=0,sum=0;
                for(int word=0;word<size;++word) {
                    double chance=row[word]-previous;
                    previous=row[word];
                    int suffix=(ctx|(word<<tile.history))>>std::max(0,tile.history+tile.width-history);
                    sum+=chance*std::exp(slope*__builtin_popcount(unsigned(word)))*future[suffix];
                    row[word]=sum;
                }
                prior[ctx]=sum;
                if(sum>0) for(int word=0;word<size;++word) row[word]/=sum;
            }
            double scale=*std::max_element(prior.begin(),prior.end());
            if(!(scale>0 && std::isfinite(scale))) return nullptr;
            for(double& x:prior) x/=scale;
            future.swap(prior);
        }
        return plan.release();
    } catch(const std::exception&) {return nullptr;}
}
extern "C" __declspec(dllexport) int window95_law(void* p,double* out) {
    if(!p||!out) return 1;
    const auto& plan=*static_cast<const Plan*>(p);
    if(plan.modes>16) return 1;
    for(int x=0;x<(1<<plan.modes);++x) {
        double mass=1;
        for(const auto& tile:plan.tiles) {
            int size=1<<tile.width;
            int prior=(x>>(tile.first-tile.history))&((1<<tile.history)-1);
            int word=(x>>tile.first)&(size-1);
            const auto* row=tile.cdf.data()+prior*size;
            mass*=row[word]-(word?row[word-1]:0.);
        }
        out[x]=mass;
    }
    return 0;
}

// Propagate (last h clicks, total clicks) to obtain the model count law.
extern "C" __declspec(dllexport) int window95_counts(void* ptr,int history,double* out) {
    if(!ptr||!out||history<0||history>11) return 1;
    const auto& plan=*static_cast<const Plan*>(ptr);
    const int m=plan.modes, contexts=1<<history;
    try {
        std::vector<double> current(std::size_t(contexts)*(m+1)),next(current.size());
        current[0]=1;
        for(const auto& tile:plan.tiles) {
            if(tile.history!=std::min(tile.first,history)) return 1;
            std::fill(next.begin(),next.end(),0.);
            int outcomes=1<<tile.width;
            for(int prior=0;prior<(1<<tile.history);++prior) {
                auto* src=current.data()+prior*(m+1);
                auto* row=tile.cdf.data()+prior*outcomes;
                for(int word=0;word<outcomes;++word) {
                    double chance=row[word]-(word?row[word-1]:0.);
                    if(!chance) continue;
                    int combined=prior|(word<<tile.history);
                    int suffix=combined>>std::max(0,tile.width+tile.history-history);
                    auto* dest=next.data()+suffix*(m+1)+__builtin_popcount(unsigned(word));
                    for(int count=0;count<=tile.first;++count) dest[count]+=chance*src[count];
                }
            }
            current.swap(next);
        }
        std::fill(out,out+m+1,0.);
        for(int prior=0;prior<contexts;++prior)
            for(int count=0;count<=m;++count) out[count]+=current[prior*(m+1)+count];
        return 0;
    } catch(const std::exception&) {return 2;}
}

// All one- and two-port vacuum events suffice for the true total-count variance.
extern "C" __declspec(dllexport) int window95_true_moments(int m,int sectors,const double* q,
        const double* mu,double* moments) {
    if(m<1||m>4096||sectors<1||sectors>8||!q||!mu||!moments) return 1;
    std::vector<double> singles(m,1.);
    long double first=0, variance=0;
    for(int i=0;i<m;++i) {
        for(int s=0;s<sectors;++s) {
            Eigen::Map<const Dense> all(q+std::size_t(s)*4*m*m,2*m,2*m);
            Eigen::Matrix2d sub;
            sub<<all(i,i),all(i,i+m),all(i+m,i),all(i+m,i+m);
            Eigen::LLT<Eigen::Matrix2d> chol(sub);
            if(chol.info()!=Eigen::Success) return 2;
            Eigen::Vector2d v(mu[2*s*m+i],mu[2*s*m+i+m]);
            singles[i]*=std::exp(-.5*chol.matrixL().solve(v).squaredNorm())/
                chol.matrixLLT().diagonal().prod();
        }
        first+=1-singles[i];variance+=singles[i]*(1-singles[i]);
        for(int j=0;j<i;++j) {
            int index[4]={i,i+m,j,j+m};
            double joint=1;
            for(int s=0;s<sectors;++s) {
                Eigen::Map<const Dense> all(q+std::size_t(s)*4*m*m,2*m,2*m);
                Eigen::Matrix4d sub;
                Eigen::Vector4d v;
                for(int a=0;a<4;++a) {
                    v[a]=mu[2*s*m+index[a]];
                    for(int b=0;b<4;++b) sub(a,b)=all(index[a],index[b]);
                }
                Eigen::LLT<Eigen::Matrix4d> chol(sub);
                if(chol.info()!=Eigen::Success) return 2;
                joint*=std::exp(-.5*chol.matrixL().solve(v).squaredNorm())/
                    chol.matrixLLT().diagonal().prod();
            }
            variance+=2*(joint-singles[i]*singles[j]);
        }
    }
    moments[0]=double(first);moments[1]=double(variance);
    return std::isfinite(moments[0])&&std::isfinite(moments[1])?0:2;
}
