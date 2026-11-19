// Expose one selected jump for independent dense-state projection tests.
#include "gaussian_mps_adaptive95_v1.cpp"
API93 int adaptive95_control(int n,int d,const int* bonds,const double* sites,
        const double* t,const double* alpha,const double* weight,int base_rank,
        int max_rank,int max_dim,double threshold,double* output,double* trace) {
    if(n<1||n>4||d<2||d>4||!bonds||!sites||!t||!alpha||!weight||!trace||!output||
       bonds[0]!=1||bonds[n]!=1||max_dim<d||max_dim>d+1||base_rank<1||max_rank<base_rank) return 1;
    try {
        Guard93 guard;
        Mps93 prior(n);
        Gaussian93 bg;
        std::vector<Z93> weights(n);
        bg.t.assign(t,t+n);bg.alpha.resize(n);
        int cursor=0;
        for(int i=0;i<n;++i) {
            if(bonds[i]<1||bonds[i+1]<1||bonds[i]>16||bonds[i+1]>16) return 1;
            for(int s=0;s<d;++s) {
                Mat93 a(bonds[i],bonds[i+1]);
                for(int l=0;l<bonds[i];++l) for(int r=0;r<bonds[i+1];++r) {
                    a(l,r)={sites[2*cursor],sites[2*cursor+1]};++cursor;
                }
                prior[i].push_back(std::move(a));
            }
            bg.alpha[i]={alpha[2*i],alpha[2*i+1]};weights[i]={weight[2*i],weight[2*i+1]};
        }
        canonical93(prior);
        auto rates=rates93(prior,bg,false);
        auto state=jump_mpo93(prior,bg,weights);
        long double inside=canonical93(state),outside=0;
        for(int i=0;i<n;++i) outside+=std::norm(weights[i])*rates.outside[i];
        std::fill(trace,trace+14,0.);
        double stats[18]{};
        finish_jump95(prior,state,bg,weights,rates.outside,inside+outside,outside,0,
            base_rank,1.,stats,Truncation95{max_dim,max_rank,threshold,trace});
        int size=1;
        for(int i=0;i<n;++i) size*=d+1;
        for(int pattern=0;pattern<size;++pattern) {
            int code=pattern;
            Mat93 product=Mat93::Ones(1,1);
            for(int i=n-1;i>=0;--i) {
                int level=code%(d+1);code/=d+1;
                if(level>=int(prior[i].size())) {product=Mat93::Zero(1,1);break;}
                product=(prior[i][level]*product).eval();
            }
            output[2*pattern]=product(0,0).real();output[2*pattern+1]=product(0,0).imag();
        }
        return 0;
    } catch(const std::exception&) {return 2;}
}
