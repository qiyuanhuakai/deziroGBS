// One shared complex loop-Wick cache replaces 2^k real-quadrature expansions.
// All first/second derivatives are contractions of the same cache. No clipping.
struct ComplexWick75 {
    using Z=std::complex<LD73>;
    int length;Eigen::Matrix<Z,Eigen::Dynamic,1> mean;
    Eigen::Matrix<Z,Eigen::Dynamic,Eigen::Dynamic> cov;
    std::vector<Z> cache;std::vector<unsigned char> known;
    ComplexWick75(const LVec73& mu,const LMat73& sigma):length(mu.size()),mean(length),cov(length,length),cache(std::size_t(1)<<length),known(cache.size(),0){
        const LD73 scale=1/std::sqrt(2.L);
        for(int a=0;a<length;++a){int i=2*(a/2);Z u(0,(a%2?-1:1));mean[a]=(mu[i]+u*mu[i+1])*scale;
            for(int b=0;b<length;++b){int j=2*(b/2);Z v(0,(b%2?-1:1));cov(a,b)=(sigma(i,j)+u*sigma(i+1,j)+v*sigma(i,j+1)+u*v*sigma(i+1,j+1))/2.L;}}
        cache[0]=1;known[0]=1;
    }
    Z get(unsigned mask){if(known[mask])return cache[mask];int i=__builtin_ctz(mask);unsigned rest=mask&~(1U<<i);Z value=mean[i]*get(rest);for(unsigned scan=rest;scan;scan&=scan-1){int j=__builtin_ctz(scan);value+=cov(i,j)*get(rest&~(1U<<j));}known[mask]=1;return cache[mask]=value;}
};
static Moments73 polynomial73(const LVec73& mean,const LMat73& cov,bool derivatives){
    int length=mean.size();need(length%2==0&&length<=16,"Complex polynomial degree limit");
    ComplexWick75 w(mean,cov);unsigned full=(1U<<length)-1;auto value=w.get(full);
    need(std::abs(value.imag())<=1e-12L*(1+std::abs(value.real())),"Complex moment imaginary residual");
    Moments73 out{value.real(),LVec73::Zero(length),LMat73::Zero(length,length)};
    if(!derivatives)return out;
    using Z=ComplexWick75::Z;LD73 scale=1/std::sqrt(2.L);
    auto coefficient=[&](int coordinate,int half){return coordinate%2?Z(0,(half?-1:1)*scale):Z(scale,0);};
    for(int a=0;a<length;++a){Z gradient=0;for(int u=0;u<2;++u){int i=2*(a/2)+u;gradient+=coefficient(a,u)*w.get(full&~(1U<<i));}out.gradient[a]=gradient.real();
        for(int b=0;b<=a;++b){Z hessian=0;for(int u=0;u<2;++u)for(int v=0;v<2;++v){int i=2*(a/2)+u,j=2*(b/2)+v;if(i!=j)hessian+=coefficient(a,u)*coefficient(b,v)*w.get(full&~(1U<<i)&~(1U<<j));}out.hessian(a,b)=out.hessian(b,a)=hessian.real();}}
    return out;
}
