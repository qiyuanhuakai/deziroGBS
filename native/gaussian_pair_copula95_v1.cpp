// Optical one/two-port targets, exact window pair laws, and latent Gaussian sampling.
#include "gaussian_click_window95_v2.cpp"
#include <random>

namespace {
using Matrix95=Eigen::MatrixXd;
using Vector95=Eigen::VectorXd;
constexpr double pi95=3.14159265358979323846;
using Gemm95=void(*)(int,int,int,int,int,int,double,const double*,int,const double*,int,double,double*,int);
Gemm95 multiply95=nullptr;
// Integrate the derivative of the bivariate normal probability from independence.
// rho=sin(angle) cancels the endpoint Jacobian.
struct Quadrature95 {
    Vector95 nodes,weights;
    Quadrature95() {
        Matrix95 j=Matrix95::Zero(24,24);
        for(int i=1;i<24;++i) j(i,i-1)=j(i-1,i)=i/std::sqrt(4.*i*i-1.);
        Eigen::SelfAdjointEigenSolver<Matrix95> eig(j);
        nodes=(eig.eigenvalues().array()+1)/2;
        weights=eig.eigenvectors().row(0).array().square().transpose();
    }
    double covariance(double a,double b,double angle) const {
        if(angle==0||!std::isfinite(a)||!std::isfinite(b)) return 0;
        double sum=0;
        for(int k=0;k<nodes.size();++k) {
            double theta=angle*nodes[k],s=std::sin(theta),c=std::cos(theta);
            double v=(a-b*s)/c;
            sum+=weights[k]*std::exp(-.5*(v*v+b*b));
        }
        return angle*sum/(2*pi95);
    }
};
double inverse_cov95(const Quadrature95& quad,double a,double b,double pi,double pj,double target) {
    double upper=std::min(pi,pj)-pi*pj,lower=std::max(0.,pi+pj-1)-pi*pj;
    if(std::abs(target)<2e-16) return 0.;
    if(target>upper+1e-12||target<lower-1e-12) throw std::domain_error("infeasible binary marginal");
    if(upper-target<2e-15) return 1.;
    if(target-lower<2e-15) return -1.;
    double lo=-pi95/2,hi=pi95/2;
    double theta=0;
    for(int step=0;step<54;++step) {
        double error=quad.covariance(a,b,theta)-target;
        if(std::abs(error)<2e-14) break;
        if(error<0) lo=theta;else hi=theta;
        double v=(a-b*std::sin(theta))/std::cos(theta);
        double derivative=std::exp(-.5*(v*v+b*b))/(2*pi95);
        double next=theta-error/derivative;
        theta=next>lo && next<hi ? next:(hi+lo)/2;
    }
    return std::sin(theta);
}
double quantile95(long double p) {
    if(p<=0) return -std::numeric_limits<double>::infinity();
    if(p>=1) return std::numeric_limits<double>::infinity();
    long double lo=-40,hi=40;
    // Evaluate the smaller tail directly.
    bool reflect=p>.5L;
    long double target=reflect?1-p:p;
    for(int i=0;i<64;++i) {
        long double x=(lo+hi)/2;
        if(std::erfc(-x/std::sqrt(2.L))/2<target) lo=x;else hi=x;
    }
    return double(reflect?-(lo+hi)/2:(lo+hi)/2);
}
}

extern "C" __declspec(dllexport) int pair95_window(void* handle,double* first,double* joint) {
    if(!handle||!first||!joint) return 1;
    try {
        const auto& plan=*static_cast<const Plan*>(handle);
        int m=plan.modes;
        Eigen::Map<Dense> output(joint,m,m);output.setZero();
        Vector95 context=Vector95::Ones(1);
        Matrix95 past(0,1);
        for(std::size_t idx=0;idx<plan.tiles.size();++idx) {
            const auto& t=plan.tiles[idx];
            int rows=1<<t.history,words=1<<t.width;
            if(rows!=context.size()) return 2;
            int next_h=idx+1<plan.tiles.size()?plan.tiles[idx+1].history:0;
            Matrix95 transition=Matrix95::Zero(rows,1<<next_h),emission=Matrix95::Zero(rows,t.width);
            Matrix95 added=Matrix95::Zero(t.width,1<<next_h);
            for(int ctx=0;ctx<rows;++ctx) for(int word=0;word<words;++word) {
                const auto* row=t.cdf.data()+ctx*words;
                double chance=row[word]-(word?row[word-1]:0);
                if(!chance) continue;
                int suffix=(ctx|(word<<t.history))>>(t.history+t.width-next_h);
                transition(ctx,suffix)+=chance;
                double prob=context[ctx]*chance;
                for(int a=0;a<t.width;++a) if(word&(1<<a)) {
                    emission(ctx,a)+=chance;added(a,suffix)+=prob;
                    for(int b=0;b<t.width;++b) if(word&(1<<b)) output(t.first+a,t.first+b)+=prob;
                }
            }
            if(t.first) {
                output.block(0,t.first,t.first,t.width).noalias()=past*emission;
                output.block(t.first,0,t.width,t.first)=output.block(0,t.first,t.first,t.width).transpose();
            }
            Matrix95 next(t.first+t.width,1<<next_h);
            if(t.first) next.topRows(t.first).noalias()=past*transition;
            next.bottomRows(t.width)=added;past.swap(next);
            context=(transition.transpose()*context).eval();
        }
        Eigen::Map<Vector95>(first,m)=output.diagonal();
        return std::abs(context.sum()-1)<1e-10 ? 0:2;
    } catch(const std::exception&) {return 3;}
}

extern "C" __declspec(dllexport) int pair95_optical(int m,int sectors,int cap,const double* q,const double* mu,
        double* single,double* joint,double* photon_law,double* thresholds,double* photon_cov) {
    if(m<1||m>4096||sectors<1||sectors>8||cap<1||cap>128||!q||!mu||!single||!joint||
       !photon_law||!thresholds||!photon_cov) return 1;
    try {
        std::vector<double> vac(m,1.);
        Eigen::Map<Dense> cov(photon_cov,m,m);cov.setZero();
        for(int i=0;i<m;++i) {
            std::vector<long double> derivative(cap+1,0);
            long double p0=1;
            for(int s=0;s<sectors;++s) {
                Eigen::Map<const Dense> all(q+std::size_t(s)*4*m*m,2*m,2*m);
                Eigen::Vector2d u(mu[2*s*m+i],mu[2*s*m+i+m]);
                Eigen::Matrix2d local;
                local<<all(i,i),all(i,i+m),all(i+m,i),all(i+m,i+m);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(local);
                auto mean=(eig.eigenvectors().transpose()*u).eval();
                auto excess=(local-Eigen::Matrix2d::Identity()).eval();
                cov(i,i)+=(excess.trace()+u.squaredNorm()+excess.squaredNorm())/2+u.dot(excess*u);
                for(int a=0;a<2;++a) {
                    long double lambda=eig.eigenvalues()[a],beta=(lambda-1)/lambda;
                    if(!(lambda>0)) return 2;
                    long double shift=mean[a]*mean[a]/(2*lambda);
                    p0*=std::exp(-shift)/std::sqrt(lambda);
                    long double power=1;
                    for(int k=1;k<=cap;++k) {
                        derivative[k]+=power*(beta/2+k*shift*(1-beta));
                        power*=beta;
                    }
                }
            }
            vac[i]=double(p0);single[i]=1-vac[i];
            std::vector<long double> mass(cap+1);mass[0]=p0;long double cumulative=0;
            for(int k=0;k<=cap;++k) {
                if(k) {
                    for(int j=1;j<=k;++j) mass[k]+=derivative[j]*mass[k-j];
                    mass[k]/=k;
                }
                if(mass[k]<-2e-13L||!std::isfinite(mass[k])) return 2;
                mass[k]=std::max(0.L,mass[k]);
                cumulative+=mass[k];
                if(cumulative>1+1e-12L) return 2;
                auto pos=std::size_t(i)*(cap+1)+k;
                photon_law[pos]=double(mass[k]);thresholds[pos]=quantile95(cumulative);
            }
            joint[std::size_t(i)*m+i]=single[i];
            for(int j=0;j<i;++j) {
                double v=1;
                int ix[4]={i,i+m,j,j+m};
                for(int s=0;s<sectors;++s) {
                    Eigen::Map<const Dense> all(q+std::size_t(s)*4*m*m,2*m,2*m);
                    Eigen::Vector4d mean;Eigen::Matrix4d sub;
                    for(int a=0;a<4;++a) {
                        mean[a]=mu[2*s*m+ix[a]];
                        for(int b=0;b<4;++b) sub(a,b)=all(ix[a],ix[b]);
                    }
                    Eigen::LLT<Eigen::Matrix4d> chol(sub);
                    if(chol.info()!=Eigen::Success) return 2;
                    v*=std::exp(-.5*chol.matrixL().solve(mean).squaredNorm())/chol.matrixLLT().diagonal().prod();
                    auto block=sub.topRightCorner<2,2>();
                    cov(i,j)+=block.squaredNorm()/2+mean.head<2>().dot(block*mean.tail<2>());
                }
                joint[std::size_t(i)*m+j]=joint[std::size_t(j)*m+i]=1-vac[i]-vac[j]+v;
                cov(j,i)=cov(i,j);
            }
        }
        return 0;
    } catch(const std::exception&) {return 3;}
}

extern "C" __declspec(dllexport) int pair95_invert(int m,const double* threshold,const double* single,
        const double* joint,double* correlation) {
    if(m<1||m>4096||!threshold||!single||!joint||!correlation) return 1;
    try {
        Quadrature95 quad;
        for(int i=0;i<m;++i) {
            correlation[std::size_t(i)*m+i]=1;
            for(int j=0;j<i;++j) {
                double r=inverse_cov95(quad,threshold[i],threshold[j],single[i],single[j],
                    joint[std::size_t(i)*m+j]-single[i]*single[j]);
                correlation[std::size_t(i)*m+j]=correlation[std::size_t(j)*m+i]=r;
            }
        }
        return 0;
    } catch(const std::exception&) {return 3;}
}

extern "C" __declspec(dllexport) int pair95_forward(int m,const double* thresholds,const double* single,
        const double* corr,double* joint) {
    if(!thresholds||!single||!corr||!joint||m<1||m>4096) return 1;
    Quadrature95 quad;
    for(int i=0;i<m;++i) {
        joint[std::size_t(i)*m+i]=single[i];
        for(int j=0;j<i;++j) {
            double r=corr[std::size_t(i)*m+j];
            if(!std::isfinite(r)||std::abs(r)>1+1e-10) return 2;
            double p=r>=1?std::min(single[i],single[j]):r<=-1?std::max(0.,single[i]+single[j]-1):
                single[i]*single[j]+quad.covariance(thresholds[i],thresholds[j],std::asin(r));
            joint[std::size_t(i)*m+j]=joint[std::size_t(j)*m+i]=p;
        }
    }
    return 0;
}
extern "C" __declspec(dllexport) int pair95_gemm(void* fn) {
    if(!fn) return 1;multiply95=reinterpret_cast<Gemm95>(fn);return 0;
}
extern "C" __declspec(dllexport) int pair95_eigen(int m,const double* correlation,double* values,double* vectors) {
    if(m<1||m>4096||!correlation||!values||!vectors) return 1;
    try {
        Eigen::Map<const Dense> input(correlation,m,m);
        Eigen::SelfAdjointEigenSolver<Matrix95> eig(input);
        if(eig.info()!=Eigen::Success) return 2;
        Eigen::Map<Vector95>(values,m)=eig.eigenvalues();
        Eigen::Map<Dense>(vectors,m,m)=eig.eigenvectors();
        return 0;
    } catch(const std::exception&) {return 3;}
}
// Separate experiment/record/purpose addresses supply the shared and independent normals.
extern "C" __declspec(dllexport) int pair95_sample(int m,int rank,int cap,const double* factors,
        const double* residual,const double* thresholds,int records,std::uint64_t experiment,std::uint64_t offset,
        int pnr,unsigned char* output) {
    if(m<1||m>4096||rank<1||rank>m||cap<1||cap>128||!factors||!residual||!thresholds||records<0||
       std::uint64_t(records)+offset<offset||std::uint64_t(records)+offset>0x100000000ULL||!output||
       (pnr!=0&&pnr!=1)||!multiply95) return 1;
    try {
        for(int first=0;first<records;first+=128) {
            int batch=std::min(128,records-first);
            Dense z(batch,rank),values(batch,m);
            for(int i=0;i<batch;++i) {
                DomainRng84 rng(experiment,0,121,offset+first+i);
                std::normal_distribution<double> normal;
                for(int k=0;k<rank;++k) z(i,k)=normal(rng);
            }
            multiply95(101,111,112,batch,m,rank,1.,z.data(),rank,factors,rank,0.,values.data(),m);
            for(int i=0;i<batch;++i) {
                DomainRng84 rng(experiment,0,122,offset+first+i);
                std::normal_distribution<double> normal;
                for(int j=0;j<m;++j) {
                    double g=values(i,j)+(residual[j]>0?normal(rng)*residual[j]:0);
                    const double* cuts=thresholds+std::size_t(j)*(cap+1);
                    unsigned char count=g>cuts[0];
                    if(pnr && count) count=static_cast<unsigned char>(std::lower_bound(cuts+1,cuts+cap+1,g)-cuts);
                    output[std::size_t(first+i)*m+j]=count;
                }
            }
        }
        return 0;
    } catch(const std::exception&) {return 3;}
}
