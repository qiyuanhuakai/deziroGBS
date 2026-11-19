// Sequential threshold sampling in a pure Gaussian core with classical displacement.
#include "gaussian_click_proxy95_v1.cpp"
#include "gaussian_rng_domain84_v1.hpp"
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <memory>

namespace {
using Matrix=Eigen::MatrixXd;
using Vector=Eigen::VectorXd;
using Pair=Eigen::Matrix2d;
using Mean=Eigen::Vector2d;
struct Port95 {
    Matrix optical,gain;
    Pair g,inverse;
    double logdet;
};
struct Plan95 {
    std::vector<Port95> ports;
    Vector mean;
    Matrix noise;
    bool random_mean;
};
double probability95(const Port95& port,const Mean& mu) {
    double logp=-(port.logdet+mu.dot(port.inverse*mu))/2;
    if (!std::isfinite(logp) || logp>2e-13) throw std::domain_error("invalid vacuum probability");
    return -std::expm1(std::min(0.,logp));
}
// Returns the charge to apply on this branch. The last port has no future state.
double update95(const Port95& port,const Mean& mu,bool click,Vector& mean) {
    Mean observation=Mean::Zero();
    double charge=0;
    if(click) {
        double v[6];
        if(coherent95_fidelity(port.g(0,0),port.g(0,1),port.g(1,1),mu.x(),mu.y(),0,0,v))
            throw std::domain_error("conditional fidelity calculation failed");
        observation=Mean(v[2],v[3]);
        // Small positive allowance avoids assigning zero charge to a tiny negative
        // floating result. This is a numerical guard, not interval arithmetic.
        charge=std::sqrt(std::min(1.,std::max(0.,v[4])+2e-10));
    }
    mean.noalias() += port.gain*(observation-mu);
    return charge;
}
void enumerate95(const Plan95& plan,int j,std::size_t mask,Vector mean,double mass,double paid,double budget,
                 double* output,double& weighted) {
    if(j==int(plan.ports.size())) {output[mask]+=mass; weighted+=mass*paid;return;}
    const auto& port=plan.ports[j];
    Mean mu=port.optical*mean;
    const double p=probability95(port,mu);
    for(int bit=0;bit<=1;++bit) {
        const double weight=mass*(bit?p:1-p);
        if(weight==0)continue;
        Vector next=mean;
        try {
            const double charge=j+1<int(plan.ports.size()) ? update95(port,mu,bit,next) : 0.;
            if(paid+charge<=budget) {
                enumerate95(plan,j+1,mask|(std::size_t(bit)<<j),next,weight,paid+charge,budget,output,weighted);
                continue;
            }
        } catch(const std::domain_error&) {}
        output[std::size_t(1)<<plan.ports.size()]+=weight;
        weighted+=weight*paid;
    }
}
}
extern "C" __declspec(dllexport) void* trajectory95_prepare(int m,int n,const double* optical,
        const double* t,const double* alpha,const double* noise) {
    if(m<1||n<1||n>m||m>4096||n>256||!optical||!t||!alpha||!noise)return nullptr;
    try {
        auto plan=std::make_unique<Plan95>();
        Matrix core=Matrix::Zero(2*n,2*n);
        plan->mean.resize(2*n);
        Eigen::Map<const Eigen::Matrix<double,-1,-1,Eigen::RowMajor>> factor(noise,2*n,2*n);
        plan->noise=std::sqrt(2.)*factor; plan->random_mean=plan->noise.cwiseAbs().maxCoeff()!=0;
        for(int i=0;i<n;++i) {
            if(!(std::abs(t[i])<1))return nullptr;
            core(i,i)=t[i]/(1-t[i]);core(n+i,n+i)=-t[i]/(1+t[i]);
            plan->mean[i]=std::sqrt(2.)*alpha[2*i];plan->mean[n+i]=std::sqrt(2.)*alpha[2*i+1];
        }
        if(!plan->mean.allFinite()||!plan->noise.allFinite())return nullptr;
        // The optical columns represent orthogonal inputs embedded in output vacuum.
        Eigen::MatrixXcd u(m,n);
        for(int j=0;j<m;++j)for(int i=0;i<n;++i){
            const auto* z=optical+2*(j*n+i);u(j,i)={z[0],z[1]};
        }
        if(!u.allFinite()||(u.adjoint()*u-Eigen::MatrixXcd::Identity(n,n)).cwiseAbs().maxCoeff()>1e-10)return nullptr;
        for(int j=0;j<m;++j) {
            Port95 port;port.optical.resize(2,2*n);
            for(int i=0;i<n;++i) {
                const auto z=u(j,i);
                port.optical(0,i)=port.optical(1,n+i)=z.real();
                port.optical(0,n+i)=-z.imag();port.optical(1,i)=z.imag();
            }
            Matrix cross=core*port.optical.transpose();
            Pair q=Pair::Identity()+port.optical*cross;
            q=((q+q.transpose())*.5).eval();
            Eigen::LLT<Pair> chol(q);
            if(chol.info()!=Eigen::Success)return nullptr;
            port.g=2*q-Pair::Identity();port.logdet=std::log(q.determinant());
            port.inverse=chol.solve(Pair::Identity());
            port.gain=cross*port.inverse;
            core.noalias()-=port.gain*cross.transpose();core=((core+core.transpose())*.5).eval();
            plan->ports.push_back(std::move(port));
        }
        return plan.release();
    } catch(const std::exception&) {return nullptr;}
}
extern "C" __declspec(dllexport) void trajectory95_destroy(void* ptr) {delete static_cast<Plan95*>(ptr);}
extern "C" __declspec(dllexport) int trajectory95_sample(void* ptr,int count,std::uint64_t exp,
        unsigned sector,std::uint64_t offset,double budget,unsigned char* bits,double* stats,int* errors) {
    if(!ptr||count<0||!bits||!stats||!errors||!std::isfinite(budget)||budget<0||sector>65535||
       offset>0x100000000ULL||std::uint64_t(count)>0x100000000ULL-offset)return 1;
    const auto& plan=*static_cast<Plan95*>(ptr);
    const int m=int(plan.ports.size());
    for(int row=0;row<count;++row) {
        auto trace=stats+6*row;auto record=bits+std::size_t(row)*m;
        std::fill(trace,trace+6,0.);std::fill(record,record+m,0);errors[row]=0;
        DomainRng84 click(exp,sector,Click84,offset+row),disp(exp,sector,Displacement84,offset+row);
        try {
            Vector mean=plan.mean;
            if(plan.random_mean) {
                std::normal_distribution<double> normal;
                Vector z(mean.size());for(auto& v:z)v=normal(disp);
                mean.noalias()+=plan.noise*z;
            }
            for(int j=0;j<m;++j) {
                const auto& port=plan.ports[j];
                const Mean mu=port.optical*mean;
                const double p=probability95(port,mu);
                const bool bit=uniform84(click)<p;
                record[j]=bit;trace[0]=j+1;
                const double charge=j+1<m ? update95(port,mu,bit,mean) : 0.;
                trace[2]=std::max(trace[2],charge);
                if(trace[1]+charge>budget) {errors[row]=2;break;}
                trace[1]+=charge;
            }
        } catch(const std::exception&) {errors[row]=4;}
        trace[3]=click.used;trace[4]=disp.used;trace[5]=errors[row]==0;
    }
    return 0;
}
extern "C" __declspec(dllexport) int trajectory95_tree(void* ptr,double budget,double* law,double* weighted) {
    if(!ptr||!law||!weighted||!std::isfinite(budget)||budget<0)return 1;
    const auto& plan=*static_cast<Plan95*>(ptr);
    if(plan.random_mean||plan.ports.size()>10)return 1;
    try {
        std::fill(law,law+(1ULL<<plan.ports.size())+1,0.);*weighted=0;
        enumerate95(plan,0,0,plan.mean,1.,0.,budget,law,*weighted);return 0;
    } catch(const std::exception&) {return 2;}
}
