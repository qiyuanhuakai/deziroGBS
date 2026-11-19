// Conditional allocation inside groups of at most eight consecutive detectors.
#include "gaussian_count_filter95_v1.cpp"
#include <omp.h>

// Compute homogeneous coefficients [z^t] G(x*z,z) at roots x of unity.
// Inverting the Fourier transform in x recovers the joint child-total law.
extern "C" __declspec(dllexport) int alloc95_radial(int m,int left,int sectors,int cap,int freq,
        const double* qs,const double* means,double* values) {
    if(m<1||m>8||left<0||left>m||sectors<1||sectors>8||cap<1||cap>96||
       freq<=cap||freq>256||!qs||!means||!values) return 1;
    using Complex=std::complex<double>;
    using Matrix=Eigen::Matrix<Complex,-1,-1>;
    int dim=2*m;
    try {
        std::vector<Eigen::MatrixXd> b;
        std::vector<Eigen::VectorXd> shifted;
        double vacuum=1.;
        for(int s=0;s<sectors;++s) {
            Eigen::Map<const Dense> q(qs+std::size_t(s)*dim*dim,dim,dim);
            Eigen::Map<const Vector95> mu(means+s*dim,dim);
            Eigen::LLT<Eigen::MatrixXd> f(q);
            if(f.info()!=Eigen::Success) return 2;
            auto id=Eigen::MatrixXd::Identity(dim,dim);
            b.push_back(id-f.solve(id));
            shifted.emplace_back(f.solve(mu));
            vacuum*=std::exp(-.5*mu.dot(shifted.back())-f.matrixLLT().diagonal().array().log().sum());
        }
        for(int k=0;k<freq;++k) {
            Complex x=std::polar(1.,2*pi95*k/freq);
            std::vector<Complex> derivative(cap+1),coeff(cap+1);
            for(int s=0;s<sectors;++s) {
                Matrix a=b[s].cast<Complex>();
                Eigen::VectorXcd h=shifted[s].cast<Complex>(),dh=h;
                for(int j=0;j<dim;++j) if(j%m<left) {a.col(j)*=x;dh[j]*=x;}
                Matrix power=a;
                bool displ=shifted[s].squaredNorm()>0;
                for(int j=1;j<=cap;++j) {
                    derivative[j]+=.5*power.trace();
                    if(displ) {
                        derivative[j]+=.5*double(j)*(dh.transpose()*h).value();
                        h=(a*h).eval();
                    }
                    if(j<cap) power=(power*a).eval();
                }
            }
            coeff[0]=vacuum;
            for(int t=1;t<=cap;++t) {
                for(int j=1;j<=t;++j) coeff[t]+=derivative[j]*coeff[t-j];
                coeff[t]/=double(t);
            }
            for(int t=0;t<=cap;++t) {
                auto pos=2*(k*(cap+1)+t);values[pos]=coeff[t].real();values[pos+1]=coeff[t].imag();
            }
        }
        return 0;
    } catch(const std::exception&) {return 3;}
}

namespace {
void allocate95(int width,int begin,int node,int n,int stride,const std::uint64_t* cut,
                DomainRng84& rng,unsigned char* dest) {
    if(width==1) {dest[begin]=static_cast<unsigned char>(n);return;}
    if(!n) {std::fill(dest+begin,dest+begin+width,0);return;}
    const auto* row=cut+(std::size_t(node)*stride+n)*stride;
    auto word=static_cast<std::uint32_t>(rng());
    int a=int(std::upper_bound(row,row+n+1,word)-row);
    int half=width/2;
    allocate95(half,begin,2*node,a,stride,cut,rng,dest);
    allocate95(width-half,begin+half,2*node+1,n-a,stride,cut,rng,dest);
}
}
// Homogeneous terms of the multivariate count PGF at real interpolation nodes.
extern "C" __declspec(dllexport) int alloc95_low(int m,int sectors,int degree,int points,const double* qs,
        const double* means,const double* directions,double* value) {
    if(m<1||m>8||sectors<1||sectors>8||degree<1||degree>15||points<1||
       !qs||!means||!directions||!value) return 1;
    int dim=2*m;
    try {
        using Mat=Eigen::Matrix<long double,-1,-1>;
        using Vec=Eigen::Matrix<long double,-1,1>;
        std::vector<Mat> b;
        std::vector<Vec> shifted;
        long double vacuum=1;
        for(int s=0;s<sectors;++s) {
            Mat q=Eigen::Map<const Dense>(qs+std::size_t(s)*dim*dim,dim,dim).cast<long double>();
            Vec mu=Eigen::Map<const Vector95>(means+s*dim,dim).cast<long double>();
            Eigen::LLT<Mat> f(q);
            if(f.info()!=Eigen::Success) return 2;
            Mat id=Mat::Identity(dim,dim);
            b.push_back(id-f.solve(id));shifted.emplace_back(f.solve(mu));
            vacuum*=std::exp(-.5L*mu.dot(shifted.back())-f.matrixLLT().diagonal().array().log().sum());
        }
        for(int i=0;i<points;++i) {
            std::array<long double,16> derivative{},coeff{};
            for(int s=0;s<sectors;++s) {
                Mat a=b[s];Vec h=shifted[s],dh=h;
                for(int j=0;j<dim;++j) {
                    double x=directions[i*m+j%m];a.col(j)*=x;dh[j]*=x;
                }
                Mat power=a;
                bool displaced=shifted[s].squaredNorm()>0;
                for(int t=1;t<=degree;++t) {
                    derivative[t]+=.5L*power.trace();
                    if(displaced) {derivative[t]+=.5L*t*dh.dot(h);h=(a*h).eval();}
                    if(t<degree) power=(power*a).eval();
                }
            }
            coeff[0]=vacuum;
            for(int t=1;t<=degree;++t) {
                for(int j=1;j<=t;++j) coeff[t]+=derivative[j]*coeff[t-j];
                coeff[t]/=t;
            }
            for(int t=0;t<=degree;++t) value[i*(degree+1)+t]=double(coeff[t]);
        }
        return 0;
    } catch(const std::exception&) {return 3;}
}
extern "C" __declspec(dllexport) int alloc95_draw(void* ptr,int n,std::uint64_t experiment,std::uint64_t offset,
        int kind,int workers,int bins,const std::uint64_t* accept,int width,int last,int stride,
        const std::uint64_t* split,const std::uint64_t* masks,unsigned char* out,std::uint32_t* attempts,
        unsigned char* group_counts,int degree,const int* offsets,const std::uint64_t* low_cdf,
        const std::uint32_t* low_patterns) {
    if(!ptr||!out||!attempts||!group_counts||!split||!masks||width<1||width>8||last<1||last>width||
       stride<3||n<1||workers<1||workers>64||kind<0||kind>1||degree<0||degree>15||
       !offsets||!low_cdf||!low_patterns) return 1;
    auto& plan=*static_cast<Table95*>(ptr);
    if(kind&&stride<=plan.levels) return 1;
    int rc;
    if(bins) rc=filter95_draw(ptr,n,experiment,offset,1,workers,bins,accept,group_counts,attempts);
    else {
        rc=cube95_sample(ptr,n,experiment,offset,1,workers,group_counts);
        std::fill(attempts,attempts+n,1);
    }
    if(rc) return rc;
    int modes=(plan.m-1)*width+last;
    #pragma omp parallel for num_threads(workers) schedule(static) if(n>1 && workers>1)
    for(int i=0;i<n;++i) {
        DomainRng84 rng(experiment,0,kind?131:130,offset+i);
        auto* dest=out+std::size_t(i)*modes;
        for(int g=0;g<plan.m;++g) {
            int size=g+1==plan.m?last:width;
            int count=group_counts[std::size_t(i)*plan.m+g];
            if(!kind) {
                int mask=0;
                if(count>=size) mask=(1<<size)-1;
                else if(count) {
                    auto* row=masks+(std::size_t(g)*9+count)*256;
                    mask=int(std::upper_bound(row,row+(1<<size),static_cast<std::uint32_t>(rng()))-row);
                }
                for(int j=0;j<size;++j) dest[g*width+j]=(mask>>j)&1;
            } else if(count>0&&count<=degree) {
                int index=g*(degree+2)+count;
                int k=int(std::upper_bound(low_cdf+offsets[index],low_cdf+offsets[index+1],
                    static_cast<std::uint32_t>(rng()))-low_cdf);
                auto pattern=low_patterns[k];
                for(int j=0;j<size;++j) dest[g*width+j]=(pattern>>(4*j))&15;
            } else allocate95(size,g*width,1,count,stride,split+std::size_t(g)*16*stride*stride,rng,dest);
        }
    }
    return 0;
}

// Adjoint of the positive-coefficient convolution tree. For all loop totals
// at once, sum the acceptance probability over the other loops in that center.
extern "C" __declspec(dllexport) int alloc95_marginals(void* ptr,int bins,const std::uint64_t* cut,
        int workers,double* result) {
    if(!ptr||!cut||!result||bins<1||bins>2049||workers<1||workers>64) return 1;
    const auto& p=*static_cast<Table95*>(ptr);
    int leaves=1;
    while(leaves<p.m) leaves*=2;
    int levels=p.levels+1;
    std::fill(result,result+std::size_t(p.m)*levels,0.);
    #pragma omp parallel num_threads(workers)
    {
        std::vector<double> partial(std::size_t(p.m)*levels);
        std::vector<PolyFilter> tree(2*leaves),reward(2*leaves);
        #pragma omp for schedule(static)
        for(int c=0;c<p.centers;++c) {
            for(int g=0;g<leaves;++g) tree[leaves+g]=g<p.m?port95(p,c,g,true):PolyFilter{1.};
            for(int i=leaves-1;i>0;--i) tree[i]=product95(tree[2*i],tree[2*i+1],bins);
            reward[1].resize(bins);
            for(int j=0;j<bins;++j) reward[1][j]=cut[j]/word95;
            for(int i=1;i<leaves;++i) {
                for(int side=0;side<2;++side) {
                    int child=2*i+side;
                    const auto& sibling=tree[2*i+1-side];
                    auto& v=reward[child];v.assign(tree[child].size(),0.);
                    for(int k=0;k<int(v.size());++k)
                        for(int t=0;t<std::min<int>(sibling.size(),int(reward[i].size())-k);++t)
                            v[k]+=sibling[t]*reward[i][k+t];
                }
            }
            for(int g=0;g<p.m;++g)
                for(int k=0;k<int(tree[leaves+g].size());++k)
                    partial[std::size_t(g)*levels+k]+=tree[leaves+g][k]*reward[leaves+g][k];
        }
        #pragma omp critical
        for(std::size_t i=0;i<partial.size();++i) result[i]+=partial[i]/p.centers;
    }
    return 0;
}
