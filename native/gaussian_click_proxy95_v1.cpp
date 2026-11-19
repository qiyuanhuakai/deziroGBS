// Conditional fidelity of a coherent-projection Gaussian proxy after a click.
// Coordinates x,p have vacuum covariance Gamma=I and alpha=(x+ip)/sqrt(2).
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
using Real = long double;
struct Pair95 { Real a,b,d; };
Real det95(Pair95 m) { return m.a*m.d-m.b*m.b; }
Real quad95(Pair95 m,Real x,Real y) { return (m.d*x*x-2*m.b*x*y+m.a*y*y)/det95(m); }
struct Fidelity95 {
    Pair95 q,q2;
    Real mx,my,logp0,p0,p1,coherence,log_scale;
    Real loss(Real x,Real y) const {
        const Real dx=x-mx,dy=y-my;
        const Real log_a=log_scale-(quad95(q2,dx,dy)-quad95(q,dx,dy))/2;
        const Real log_overlap=-coherence*(x*x+y*y)/2;
        return (-std::expm1(log_a)+p0*std::expm1(log_overlap))/p1;
    }
};
}

// Pointwise probe. 'out' contains click probability, squared fidelity loss at
// requested point, chosen x,p, chosen loss and vacuum-proxy loss, respectively.
extern "C" __declspec(dllexport) int coherent95_fidelity(
    double ga,double gb,double gd,double mux,double muy,double x,double y,double* out) {
    if (!out || !std::isfinite(ga+gb+gd+mux+muy+x+y)) return 1;
    Pair95 g{ga,gb,gd}; const Real nu2=det95(g);
    if (!(ga>0 && gd>0 && nu2>=1-2e-13L)) return 2;
    Fidelity95 f{};
    f.q={(g.a+1)/2,g.b/2,(g.d+1)/2};
    const Real multiplier=(1+1/nu2)/2;
    f.q2={(1+multiplier*g.a)/2,multiplier*g.b/2,(1+multiplier*g.d)/2};
    f.mx=mux; f.my=muy;
    f.logp0=-(std::log(det95(f.q))+quad95(f.q,mux,muy))/2;
    f.p0=std::exp(f.logp0); f.p1=-std::expm1(f.logp0);
    if (!(f.p1>1e-12L && f.p1<=1)) return 3;
    f.log_scale=(std::log(det95(f.q))-std::log(det95(f.q2))-std::log(nu2))/2;
    f.coherence=(nu2-1)/(4*det95(f.q));
    Real best=f.loss(0,0),bx=0,by=0;
    auto try_point=[&](Real xx,Real yy) {
        const Real v=f.loss(xx,yy);
        if(v<best) {best=v;bx=xx;by=yy;}
    };
    // Candidate search only. Any chosen coherent projection gives a positive
    // Gaussian state. Its attained fidelity, rather than search optimality,
    // determines the trace-distance upper bound.
    for(Real factor:{.5L,1.L,1.5L,2.L,1/f.p1})
        try_point(factor*mux,factor*muy);
    for(Real radius:{.5L,1.L,2.L,4.L}) {
        try_point(mux+radius,muy);try_point(mux-radius,muy);
        try_point(mux,muy+radius);try_point(mux,muy-radius);
    }
    for(Real step:{1.L,.5L,.25L,.125L,.0625L,.03125L}) {
        const Real center_x=bx,center_y=by;
        for(int i=-1;i<=1;++i) for(int j=-1;j<=1;++j)
            try_point(center_x+i*step,center_y+j*step);
    }
    const Real loss=f.loss(x,y),vac=f.loss(0,0);
    if (!(std::isfinite(loss+best+vac) && std::min({loss,best,vac})>=-2e-10L &&
          std::max({loss,best,vac})<=1+2e-10L)) return 4;
    out[0]=double(f.p1);out[1]=double(loss);out[2]=double(bx);out[3]=double(by);
    out[4]=double(best);out[5]=double(vac);return 0;
}
