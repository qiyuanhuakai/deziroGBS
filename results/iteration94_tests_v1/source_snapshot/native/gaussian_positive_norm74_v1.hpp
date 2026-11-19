// Normal-ordered Gaussian moments as a Bargmann polynomial squared norm.
// N = L L^* is supplied as a factor; no eigenvalue clipping is performed.
#include <unordered_map>
#include <complex>
using Complex74=std::complex<long double>;
using CMatrix74=Eigen::Matrix<Complex74,Eigen::Dynamic,Eigen::Dynamic>;
using CVector74=Eigen::Matrix<Complex74,Eigen::Dynamic,1>;
using Poly74=std::unordered_map<std::uint64_t,Complex74>;
static long double positive_norm74(const CVector74& alpha,const CMatrix74& anomalous,const CMatrix74& factor){
    int k=alpha.size(),rank=std::min<int>(k,factor.cols());need(k<=8&&factor.rows()==k&&anomalous.rows()==k&&anomalous.cols()==k,"Positive norm dimensions");if(!k)return 1;
    // Compress only the linear creation span. QR preserves a supplied PSD
    // Gram factor even at exact rank zero, without subtracting eigenvalues.
    Eigen::HouseholderQR<CMatrix74> qr(factor.adjoint());CMatrix74 r=CMatrix74::Zero(rank,k);
    for(int i=0;i<rank;++i)for(int j=i;j<k;++j)r(i,j)=qr.matrixQR()(i,j);
    CMatrix74 linear=r.adjoint();std::vector<Poly74> cache(std::size_t(1)<<k);cache[0][0]=1;
    for(unsigned mask=1;mask<(1U<<k);++mask){int i=__builtin_ctz(mask);unsigned rest=mask&~(1U<<i);auto& out=cache[mask];
        for(const auto& item:cache[rest]){out[item.first]+=alpha[i]*item.second;
            for(int j=0;j<rank;++j)if(linear(i,j)!=Complex74(0))out[item.first+(1ULL<<(4*j))]+=linear(i,j)*item.second;}
        for(unsigned scan=rest;scan;scan&=scan-1){int j=__builtin_ctz(scan);if(anomalous(i,j)!=Complex74(0))for(const auto& item:cache[rest&~(1U<<j)])out[item.first]+=anomalous(i,j)*item.second;}
    }
    long double norm=0;for(const auto& item:cache.back()){long double weight=1;for(int j=0;j<rank;++j){int n=(item.first>>(4*j))&15;for(int h=2;h<=n;++h)weight*=h;}norm+=weight*std::norm(item.second);}
    need(std::isfinite(norm)&&norm>=0,"Positive norm overflow");return norm;
}
