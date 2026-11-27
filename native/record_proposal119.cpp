#include "displaced_quartic117.cpp"

// A click resolves a^dagger int_0^1 t^N dt a using a continuous heterodyne
// after the attenuation. The auxiliary outcome introduces one linear monomer.
static int continuous_click(const Local& loc,Z alpha,DomainRng84& detect,DomainRng84& aux,
                            Five& row,Z& field,double& probability) {
    const double a=std::abs(alpha);
    field=-std::conj(alpha);
    row[0]=std::exp(-a*a/2.);
    for(int j=1;j<5;++j)row[j]=row[j-1]*field/std::sqrt(double(j));
    probability=loc.mass(row);
    double trace=0.;
    for(int j=0;j<5;++j)trace+=loc.gram[j][j].real();
    if(uniform84(detect)*trace<probability)return 0;
    const int choose[5][5]={{1},{1,1},{1,2,1},{1,3,3,1},{1,4,6,4,1}};
    double coefficients[5]{},weight[5]{},total=0.;
    for(int k=0;k<5;++k) {
        double diag=std::sqrt(std::max(0.,loc.gram[k][k].real()));
        for(int j=0;j<=k;++j)
            coefficients[j]+=diag*a*choose[k][j]*std::pow(a,k-j)/fac[k];
        for(int j=0;j<k;++j)
            coefficients[j]+=diag*std::sqrt(double(k))*choose[k-1][j]*std::pow(a,k-1-j)/fac[k-1];
    }
    for(int j=0;j<5;++j){weight[j]=coefficients[j]*fac[j];total+=weight[j];}
    if(!(total>0.))return -1;
    for(;;) {
        double threshold=uniform84(aux)*total;int order=0;
        for(;order<4 && threshold>=weight[order];++order)threshold-=weight[order];
        double radius2=0.;
        for(int j=0;j<=order;++j)radius2-=std::log1p(-uniform84(aux));
        Z z=std::polar(std::sqrt(radius2),6.283185307179586*uniform84(aux));
        double t=uniform84(aux);
        field=-(1.-t)*std::conj(alpha)+std::sqrt(t)*std::conj(z);
        Z power=1.,last=0.;
        for(int k=0;k<5;++k) {
            row[k]=alpha*power/fac[k]+(k?double(k)*last/fac[k]:Z{});
            last=power;power*=field;
        }
        probability=loc.mass(row);
        double majorant=0.;
        for(int j=4;j>=0;--j)majorant=majorant*radius2+coefficients[j]/fac[j];
        double ratio=std::exp(-(1.-t)*a*a)*probability/(total*majorant);
        if(!std::isfinite(ratio)||ratio>1.+1e-8||ratio< -1e-10)return -1;
        if(uniform84(aux)<ratio)return 1;
    }
}

extern "C" int proposal119(int rows,int m,int r,int direct,int threads,uint64_t seed,uint32_t offset,
        unsigned spectrum,const Z* q,const Z* S,const Z* s,const Z* g,const Z* w,const Z* cubic,
        const double* pn,double norm,const Z* alpha,unsigned char* output,Z* linear,double* log_ratio) {
    if(!rows||m<1||r<1||threads<1||spectrum>=65536||uint64_t(offset)+rows>(uint64_t(1)<<32))return 1;
    int failed=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1) reduction(|:failed)
    for(int n=0;n<rows;++n) {
        DomainRng84 detect(seed,spectrum,direct?412:411,offset+n),aux(seed,spectrum,413,offset+n);
        State state(r,S,norm);Scratch sc(r);double logw=0.,defect=0.;
        for(int i=0;i<m;++i) {
            const size_t v=size_t(i)*r,mat=v*r,idx=size_t(n)*m+i;
            Site site{r,q+v,s+v,g+mat,w+mat,cubic+mat,pn[2*i],pn[2*i+1]};
            const Local loc=site.factor(state,sc);
            Five f;double prob;Z field=-std::conj(alpha[idx]);
            int outcome=direct ? continuous_click(loc,alpha[idx],detect,aux,f,field,prob)
                               : measurement(loc,alpha[idx],false,detect,f,prob,defect);
            if(outcome<0||!(prob>0.)||std::norm(f[0])==0.){failed|=2;break;}
            logw+=std::log(prob/std::norm(f[0]));
            output[idx]=static_cast<unsigned char>(outcome);linear[idx]=field;
            site.update(state,sc,S,f,prob);
        }
        log_ratio[n]=logw;
        if(std::abs(std::norm(state.c)-1.)>1e-7)failed|=4;
    }
    return failed;
}
