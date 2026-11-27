#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>
#include "gaussian_rng_domain84_v1.hpp"
using Z=std::complex<double>;
using Five=std::array<Z,5>;
constexpr double fac[5]={1.,1.,1.41421356237309505,2.44948974278317810,4.89897948556635620};
struct Local {
    Z gram[5][5]{};
    double mass(const Five& f) const {
        Z total{};
        for(int i=0;i<5;++i) for(int j=0;j<5;++j)
            total+=std::conj(f[i])*gram[i][j]*f[j];
        return total.real();
    }
};
static Z inner(const Z* a,const Z* b,int r) {
    Z val{};
    for(int i=0;i<r;++i) val+=std::conj(a[i])*b[i];
    return val;
}
struct State {
    Z c,d;
    std::vector<Z> u,b,H,metricH;
    State(int r,const Z* S,double norm):c(norm),d(norm),u(r),b(r),H(S,S+r*r) {
        for(auto& z:H) z*=norm;
        metricH=H;
    }
};
struct Scratch {
    std::vector<Z> v,gb,gs,cb,cs,tailH;
    std::array<std::vector<Z>,4> linear,metric;
    Scratch(int r):v(r),gb(r),gs(r),cb(r),cs(r),tailH(r*r) {
        for(auto& a:linear)a.resize(r);
        for(auto& a:metric)a.resize(r);
    }
};
static void matvec(const Z* mat,const Z* x,Z* y,int r) {
    for(int i=0;i<r;++i) {
        Z a{};
        for(int j=0;j<r;++j)a+=mat[i*r+j]*x[j];
        y[i]=a;
    }
}
struct Site {
    int r;
    const Z *q,*s,*g,*w,*cubic;
    double pair_norm,quartic_norm;
    Z h,uj,bj,hjj;
    Five constant;
    Local factor(const State& st,Scratch& sc) {
        // Coefficients of exp(x S x/2) truncated through four core photons.
        matvec(st.H.data(),q,sc.v.data(),r);
        h=uj=bj=hjj=Z{};
        for(int i=0;i<r;++i) {
            h+=q[i]*s[i];uj+=q[i]*st.u[i];bj+=q[i]*st.b[i];hjj+=q[i]*sc.v[i];
        }
        constant={st.c,uj,hjj*.5,bj*h*.5,st.d*h*h*.125};
        Local loc;
        for(int i=0;i<5;++i) for(int j=i;j<5;++j)
            loc.gram[i][j]=std::conj(constant[i])*constant[j];
        for(int i=0;i<r;++i) {
            sc.linear[0][i]=st.u[i];sc.linear[1][i]=sc.v[i];
            sc.linear[2][i]=bj*s[i]+.5*h*st.b[i];
            sc.linear[3][i]=.5*st.d*h*s[i];
        }
        for(int j=1;j<4;++j) {
            matvec(g,sc.linear[j].data(),sc.metric[j].data(),r);
            for(int i=0;i<=j;++i)
                loc.gram[i][j]+=inner(sc.linear[i].data(),sc.metric[j].data(),r);
        }
        matvec(g,st.u.data(),sc.metric[0].data(),r);
        loc.gram[0][0]+=inner(st.u.data(),sc.metric[0].data(),r);
        matvec(cubic,st.b.data(),sc.cb.data(),r);
        loc.gram[0][0]+=inner(st.b.data(),sc.cb.data(),r)+std::norm(st.d)*quartic_norm;
        matvec(g,st.b.data(),sc.gb.data(),r);matvec(g,s,sc.gs.data(),r);
        matvec(cubic,s,sc.cs.data(),r);
        loc.gram[0][1]+=st.d*inner(st.b.data(),sc.cs.data(),r);
        loc.gram[1][1]+=std::norm(st.d)*inner(s,sc.cs.data(),r);
        const Z bs=inner(st.b.data(),sc.gs.data(),r);
        const double bb=inner(st.b.data(),sc.gb.data(),r).real();
        const double ss=inner(s,sc.gs.data(),r).real();
        Z HS{},HB{},HT{},SB{},ST{};
        for(int i=0;i<r;++i) for(int j=0;j<r;++j) {
            const Z Hbar=std::conj(st.H[i*r+j]),W=w[i*r+j];
            sc.tailH[i*r+j]=st.metricH[i*r+j]-std::conj(q[i])*sc.metric[1][j]
                -sc.metric[1][i]*std::conj(q[j])-hjj*std::conj(q[i]*q[j]);
            loc.gram[0][0]+=.5*Hbar*sc.tailH[i*r+j];
            HS+=.5*Hbar*W;HB+=Hbar*sc.gb[i]*sc.gs[j];
            HT+=.5*Hbar*sc.gs[i]*sc.gs[j];
            SB+=std::conj(W)*st.b[i]*s[j];ST+=.5*std::conj(W)*s[i]*s[j];
        }
        // Quadratic basis H, S, b*s'+s*b', s*s'.
        const Z dh=st.d*h*.5;
        loc.gram[0][1]+=bj*HS+HB;
        loc.gram[0][2]+=dh*HS+st.d*HT;
        loc.gram[1][1]+=std::norm(bj)*pair_norm+bb*ss+std::norm(bs)+2.*(std::conj(bj)*SB).real();
        loc.gram[1][2]+=std::conj(bj)*dh*pair_norm+std::conj(bj)*st.d*ST+dh*std::conj(SB)+st.d*bs*ss;
        loc.gram[2][2]+=std::norm(dh)*pair_norm+.5*std::norm(st.d)*ss*ss+2.*(std::conj(dh)*st.d*ST).real();
        for(int i=0;i<5;++i) for(int j=i;j<5;++j) {
            loc.gram[i][j]*=fac[i]*fac[j];
            if(i==j) {
                loc.gram[i][i]=loc.gram[i][i].real();
            } else loc.gram[j][i]=std::conj(loc.gram[i][j]);
        }
        return loc;
    }
    void update(State& st,const Scratch& sc,const Z* S,Five f,double prob)const {
        for(int i=0;i<5;++i)f[i]*=fac[i]/std::sqrt(prob);
        Z z{};
        for(int j=0;j<5;++j)z+=f[j]*constant[j];
        const Z addS=f[1]*bj+f[2]*st.d*h*.5;
        for(int i=0;i<r;++i) for(int j=0;j<r;++j) {
            const int ij=i*r+j;
            st.metricH[ij]=f[0]*sc.tailH[ij]+addS*w[ij]
                +f[1]*(sc.gb[i]*sc.gs[j]+sc.gs[i]*sc.gb[j])+f[2]*st.d*sc.gs[i]*sc.gs[j];
            st.H[ij]=f[0]*st.H[ij]+addS*S[ij]+f[1]*(st.b[i]*s[j]+s[i]*st.b[j])+f[2]*st.d*s[i]*s[j];
        }
        for(int i=0;i<r;++i) {
            Z val{};
            for(int j=0;j<4;++j)val+=f[j]*sc.linear[j][i];
            st.u[i]=val;st.b[i]=f[0]*st.b[i]+f[1]*st.d*s[i];
        }
        st.c=z;st.d*=f[0];
    }
};
static int measurement(const Local& loc,Z alpha,bool click,DomainRng84& rng,Five& row,double& prob,double& defect,int wanted=-1) {
    const double incident=std::norm(alpha);
    Five vacuum{};
    vacuum[0]=std::exp(-.5*incident);
    for(int k=1;k<5;++k)vacuum[k]=-std::conj(alpha)*vacuum[k-1]/std::sqrt(double(k));
    row=vacuum;prob=loc.mass(row);
    defect=std::max({defect,-prob,prob-1.});
    double trace=0.;for(int i=0;i<5;++i)trace+=loc.gram[i][i].real();
    defect=std::max(defect,std::abs(trace-1.));
    double target=uniform84(rng)*trace;
    if(wanted==0||(wanted<0&&target<prob))return 0;
    target-=prob;
    if(click) {
        double norm=0.;for(auto z:vacuum)norm+=std::norm(z);
        const double den=1.+std::sqrt(std::max(0.,1.-norm));
        for(int k=0;k<5;++k) {
            for(int j=0;j<5;++j)row[j]=(k==j?1.:0.)-std::conj(vacuum[k])*vacuum[j]/den;
            prob=loc.mass(row);defect=std::max(defect,-prob);
            if(wanted==k+1||(wanted<0&&target<prob))return 1;
            target-=prob;
        }
    } else {
        for(int n=1;n<=255;++n) {
            const Five prev=row;
            double root=std::sqrt(double(n));
            row[0]=prev[0]*alpha/root;
            for(int k=1;k<5;++k)row[k]=(root*prev[k-1]-std::conj(alpha)*row[k-1])/std::sqrt(double(k));
            prob=loc.mass(row);defect=std::max(defect,-prob);
            if(wanted==n||(wanted<0&&target<prob))return n;
            target-=prob;
        }
    }
    return -1;
}
extern "C" int quartic117(int rows,int m,int r,int click,int threads,uint64_t seed,uint32_t offset,
        unsigned spectrum,const Z* q,const Z* S,const Z* s,const Z* g,const Z* w,const Z* cubic,
        const double* pn,double norm,const Z* alpha,unsigned char* output,double* diagnostics,const unsigned char* replay,double* likelihood) {
    if(rows<1||m<1||r<1||threads<1||spectrum>=65536||uint64_t(offset)+rows>(uint64_t(1)<<32))return 1;
    double defect=0.,closing=0.;int failed=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1) reduction(|:failed) reduction(max:defect,closing)
    for(int n=0;n<rows;++n) {
        DomainRng84 rng(seed,spectrum,click?317:318,offset+n);
        State state(r,S,norm);Scratch sc(r);
        double path=1.;
        for(int i=0;i<m;++i) {
            const size_t v=size_t(i)*r,mat=v*r;
            Site site{r,q+v,s+v,g+mat,w+mat,cubic+mat,pn[2*i],pn[2*i+1]};
            const Local loc=site.factor(state,sc);
            Five f;double prob;
            const int result=measurement(loc,alpha[size_t(n)*m+i],click,rng,f,prob,defect,replay?replay[size_t(n)*m+i]:-1);
            if(replay){path*=prob;if(prob<1e-25 && prob>=-1e-12){path=0.;break;}}
            if(result<0 || !(prob>0.) || !std::isfinite(prob)){failed|=2;break;}
            output[size_t(n)*m+i]=static_cast<unsigned char>(result);
            site.update(state,sc,S,f,prob);
        }
        if(replay) likelihood[n]=path;
        if(!replay||path>1e-15) closing=std::max(closing,std::abs(std::norm(state.c)-1.));
    }
    diagnostics[0]=defect;diagnostics[1]=closing;
    return failed;
}
