// Count cumulants from every physical port pair and triple at zero displacement.
#include "gaussian_click_window95_v2.cpp"
#include <atomic>

namespace {
struct Cross95 {
    Eigen::Matrix2d cross,conditional;
    double det;
};
struct Sector95 {
    std::vector<Cross95> pairs;
};
// For three whitened 2-by-2 diagonal blocks, first eliminate i, then j.
double triple_det95(const Cross95& ij,const Eigen::Matrix2d& inverse,
                    const Cross95& ik,const Cross95& jk) {
    Eigen::Matrix2d cross=jk.cross-ij.cross.transpose()*ik.cross;
    Eigen::Matrix2d last=ik.conditional-cross.transpose()*inverse*cross;
    return ij.det*last.determinant();
}
}

extern "C" __declspec(dllexport) int count95_cumulants(int m,int sectors,const double* q,
        int threads,double* out,double* contributions) {
    if(m<1||m>2048||sectors<1||sectors>2||threads<1||threads>64||!q||!out||!contributions) return 1;
    try {
        std::vector<Sector95> models(sectors);
        std::vector<double> vacuum(m,1.),ratio(std::size_t(m)*m,1.);
        for(int s=0;s<sectors;++s) {
            Eigen::Map<const Dense> full(q+std::size_t(s)*4*m*m,2*m,2*m);
            if(!full.allFinite()) return 2;
            std::vector<Eigen::Matrix2d> whiten(m);
            auto& pairs=models[s].pairs;
            pairs.resize(std::size_t(m)*m);
            for(int i=0;i<m;++i) {
                Eigen::Matrix2d sub;
                sub<<full(i,i),full(i,i+m),full(i+m,i),full(i+m,i+m);
                Eigen::LLT<Eigen::Matrix2d> chol(sub);
                if(chol.info()!=Eigen::Success) return 2;
                whiten[i]=chol.matrixL().solve(Eigen::Matrix2d::Identity());
                vacuum[i]*=whiten[i].determinant();
            }
            for(int i=0;i<m;++i) for(int j=i+1;j<m;++j) {
                auto& p=pairs[std::size_t(i)*m+j];
                Eigen::Matrix2d cross;
                cross<<full(i,j),full(i,j+m),full(i+m,j),full(i+m,j+m);
                p.cross=whiten[i]*cross*whiten[j].transpose();
                p.conditional=Eigen::Matrix2d::Identity()-p.cross.transpose()*p.cross;
                p.det=p.conditional.determinant();
                if(!(p.det>0)) return 2;
                ratio[std::size_t(i)*m+j]/=std::sqrt(p.det);
            }
        }
        std::vector<long double> triple_by_i(m);
        std::atomic<int> next{0};
        std::atomic<bool> valid{true};
        auto worker=[&]() {
            for(int i=next++;i<m;i=next++) {
                long double sum=0;
                for(int j=i+1;j<m;++j) {
                    auto ij=std::size_t(i)*m+j;
                    Eigen::Matrix2d inverses[2];
                    for(int s=0;s<sectors;++s) inverses[s]=models[s].pairs[ij].conditional.inverse();
                    for(int k=j+1;k<m;++k) {
                        auto ik=std::size_t(i)*m+k,jk=std::size_t(j)*m+k;
                        double det=1;
                        for(int s=0;s<sectors;++s) {
                            const auto& p=models[s].pairs;
                            det*=triple_det95(p[ij],inverses[s],p[ik],p[jk]);
                        }
                        if(!(det>0)) {valid=false;return;}
                        long double normalized=(1.L/std::sqrt(det)-ratio[ij])-
                            (static_cast<long double>(ratio[ik])-1)-(static_cast<long double>(ratio[jk])-1);
                        sum-=vacuum[i]*vacuum[j]*vacuum[k]*normalized;
                    }
                }
                triple_by_i[i]=sum;
            }
        };
        std::vector<std::thread> pool;
        try {
            for(int t=1;t<std::min(m,threads);++t) pool.emplace_back(worker);
            worker();
        } catch(const std::exception&) {for(auto& t:pool)t.join();return 3;}
        for(auto& t:pool)t.join();
        if(!valid) return 2;
        long double mean=0,var=0,third=0;
        for(int i=0;i<m;++i) {
            double p=1-vacuum[i];
            long double vi=p*vacuum[i],ci=vi*(1-2*p);
            for(int j=i+1;j<m;++j) {
                long double cov=vacuum[i]*vacuum[j]*(ratio[std::size_t(i)*m+j]-1);
                vi+=2*cov;
                ci+=6*(vacuum[i]+vacuum[j]-1)*cov;
            }
            mean+=p;var+=vi;third+=ci+6*triple_by_i[i];
            contributions[3*i]=double(vi);contributions[3*i+1]=double(ci);
            contributions[3*i+2]=double(6*triple_by_i[i]);
        }
        out[0]=double(mean);out[1]=double(var);out[2]=double(third);
        return 0;
    } catch(const std::exception&) {return 3;}
}
