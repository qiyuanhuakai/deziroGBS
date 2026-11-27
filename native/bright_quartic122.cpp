// Displaced finite-Fock readout with wide photon counts. The local state and
// conditional updates are shared with the previously tested quartic engine.
#include "displaced_quartic117.cpp"

namespace {
constexpr int arcs = 8;
constexpr double pi122 = 3.14159265358979323846;

double log1p_remainder(double x) {
    if (std::abs(x) > .001) return std::log1p(x) - x;
    return x*x*(-.5 + x*(1./3 + x*(-.25 + x*(.2 - x/6))));
}

// Use a discrete Laplace envelope for a Poisson law. On either side of the
// mode the ratio is a tilted Poisson sequence, so its maximum is known.
double log_poisson(uint64_t k,double mu) {
    if (k<24) return -mu+double(k)*std::log(mu)-std::lgamma(double(k)+1.);
    const double n=double(k), x=(n-mu)/mu;
    double dev;
    if (std::abs(x)<.01) {
        double sum=0.,term=x*x;
        for (int j=2;j<=12;++j) {sum+=term/(j*(j-1));term*=-x;}
        dev=mu*sum;
    } else dev=n*std::log(n/mu)-(n-mu);
    const double z=1./n, z2=z*z;
    const double correction=z*(1./12+z2*(-1./360+z2*(1./1260-z2/1680)));
    return -dev-.5*std::log(2*pi122*n)-correction;
}

struct PoissonEnvelope {
    double mu,h,log_peak;
    int64_t mode;
    void initialize(double mean) {
        mu=mean;h=1./std::sqrt(mu);mode=static_cast<int64_t>(mu);
        log_peak=-std::numeric_limits<double>::infinity();
        for (double tilt : {-h,h}) {
            const auto k=static_cast<int64_t>(mu*std::exp(tilt));
            log_peak=std::max(log_peak,log_poisson(k,mu)+h*std::abs(k-mode));
        }
        log_peak+=2e-12;
    }
    uint64_t draw(DomainRng84& rng) const {
        while (true) {
            // 1-U lies in (0,1]; independent geometric differences give
            // the two-sided exponential mass exp(-h |k-mode|).
            const auto a=static_cast<int64_t>(-std::log1p(-uniform84(rng))/h);
            const auto b=static_cast<int64_t>(-std::log1p(-uniform84(rng))/h);
            const auto k=mode+a-b;
            if (k>=0 && std::log1p(-uniform84(rng)) <
                  log_poisson(k,mu)+h*std::abs(k-mode)-log_peak)
                return static_cast<uint64_t>(k);
        }
    }
};

bool number122(const Local& loc, Z alpha, DomainRng84& rng, Five& row,
               double& mass, uint64_t& count, uint64_t& trials, double& defect) {
    const double u = std::norm(alpha);
    if (!std::isfinite(u) || u >= 0x1p48) return false;
    double trace=0.;
    for (int j=0;j<5;++j) trace += loc.gram[j][j].real();
    defect=std::max(defect,std::abs(trace-1.));
    if (u<16.) {
        row[0]=std::exp(-u/2);
        for (int j=1;j<5;++j) row[j]=-std::conj(alpha)*row[j-1]/std::sqrt(double(j));
        double target=uniform84(rng)*trace;
        for (count=0;;++count) {
            mass=loc.mass(row);
            if (target<mass) return true;
            target-=mass;
            Five previous=row;
            const double root=std::sqrt(double(count+1));
            row[0]=previous[0]*alpha/root;
            for (int j=1;j<5;++j) row[j]=(root*previous[j-1]-std::conj(alpha)*row[j-1])/std::sqrt(double(j));
            // Signal exhaustion of finite-precision mass rather than silently
            // mapping an unassigned outcome to the last represented count.
            if (row==Five{} || !std::isfinite(target)) return false;
        }
    }
    const double amplitude=std::sqrt(u);
    double ring=0., envelope=std::numeric_limits<double>::infinity();
    for (double x : {.125,.50,1.25,std::pow(24.,.25)}) {
        double prob=std::exp(-x), score=0., minprob=prob;
        for (int j=0;j<5;++j) {
            if (j) prob*=x/j;
            score+=std::max(0.,loc.gram[j][j].real())/prob;
            minprob=std::min(minprob,prob);
        }
        score=std::min(score,trace/minprob);
        if (score<envelope) {envelope=score;ring=x;}
    }
    double delta[arcs+1],slope[arcs+1],constant[arcs+1],means[arcs+1];
    PoissonEnvelope grid[arcs+1];
    double gap=0.;
    for (int j=0;j<=arcs;++j) {
        delta[j]=(ring+2.*amplitude*std::sqrt(ring)*std::cos(pi122*j/arcs))/u;
        means[j]=u*(1.+delta[j]);
        grid[j].initialize(means[j]);
        slope[j]=std::log1p(delta[j]);
        constant[j]=u*log1p_remainder(delta[j]);
        if (j) gap=std::max(gap,means[j-1]*std::pow(slope[j]-slope[j-1],2)/8.);
    }
    const double factor=2*envelope*std::exp(gap)*(1+2e-12);
    const Z phase=std::conj(alpha)/amplitude;
    while (true) {
        ++trials;
        const unsigned pick=unsigned(uniform84(rng)*(2*arcs));
        const unsigned endpoint=pick/2+pick%2;
        count=grid[endpoint].draw(rng);
        const double d=double(count)-u, x=d/amplitude;
        double pol[5]={1.,x,0.,0.,0.};
        row[0]=1.; Z phase_power=1.;
        for (int k=1;k<5;++k) {
            if (k>=2) pol[k]=((x-(k-1)/amplitude)*pol[k-1]-std::sqrt(double(k-1))*pol[k-2])/std::sqrt(double(k));
            phase_power*=phase;
            row[k]=phase_power*pol[k];
        }
        mass=loc.mass(row);
        double relative=0.;
        for (int j=0;j<=arcs;++j) {
            double weight=(j==0 || j==arcs) ? .5/arcs : 1./arcs;
            relative+=weight*std::exp(d*slope[j]+constant[j]);
        }
        const double acceptance=mass/(factor*relative);
        if (!std::isfinite(acceptance) || acceptance>1+1e-10 || acceptance < -1e-10) return false;
        if (uniform84(rng)<acceptance) return true;
    }
}
}

extern "C" int quartic122(int rows,int m,int r,int direct,int threads,uint64_t seed,uint32_t offset,
        unsigned spectrum,const Z* q,const Z* S,const Z* s,const Z* g,const Z* w,const Z* cubic,
        const double* pn,double norm,const Z* alpha,uint64_t* output,double* diag,uint64_t* counters) {
    if (rows<1 || m<1 || r<1 || threads<1 || spectrum>=65536 || uint64_t(offset)+rows>(uint64_t(1)<<32)) return 1;
    double defect=0.,closing=0.; int failed=0;
    uint64_t trials=0,bright=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1) reduction(|:failed) reduction(+:trials,bright) reduction(max:defect,closing)
    for (int n=0;n<rows;++n) {
        DomainRng84 rng(seed,spectrum,direct?522:521,offset+n);
        State state(r,S,norm); Scratch sc(r);
        for (int i=0;i<m;++i) {
            const size_t v=size_t(i)*r,mat=v*r;
            Site site{r,q+v,s+v,g+mat,w+mat,cubic+mat,pn[2*i],pn[2*i+1]};
            Local loc=site.factor(state,sc);
            Five row{}; double mass=0.; uint64_t outcome=0;
            Z field=alpha[size_t(n)*m+i];
            if (direct) {
                int result=measurement(loc,field,true,rng,row,mass,defect);
                if (result<0) {failed|=2;break;}
                outcome=static_cast<uint64_t>(result);
            } else {
                bright+=std::norm(field)>=16.;
                if (!number122(loc,field,rng,row,mass,outcome,trials,defect)) {failed|=4;break;}
            }
            if (!(mass>0.) || !std::isfinite(mass)) {failed|=8;break;}
            output[size_t(n)*m+i]=outcome;
            site.update(state,sc,S,row,mass);
        }
        closing=std::max(closing,std::abs(std::norm(state.c)-1.));
    }
    diag[0]=defect;diag[1]=closing;counters[0]=trials;counters[1]=bright;
    return failed;
}

// Independent validation supplies a full local Gram matrix, including coherences.
extern "C" int local122(int n,const Z* matrix,const Z* alpha,uint64_t seed,uint64_t* out,uint64_t* trials) {
    Local loc;
    for (int i=0;i<5;++i) for (int j=0;j<5;++j) loc.gram[i][j]=matrix[5*i+j];
    double defect=0.,mass=0.;
    uint64_t t=0;
    for (int i=0;i<n;++i) {
        DomainRng84 rng(seed,0,523,i);
        Five row{};
        if (!number122(loc,*alpha,rng,row,mass,out[i],t,defect)) return 1;
    }
    *trials=t;
    return 0;
}
