"""Construct a new version without changing the executed iteration71 source."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'native/gaussian_pnr_branches_v2.cpp').read_text(encoding='utf-8')
start=s.index('static bool flow_pnr71(');end=s.index('static int select_pnr71(',start)
replacement=r'''
struct RootStats72 {double calls=0,iterations=0,fallbacks=0,residual=0;};
static thread_local RootStats72 root_stats72;
static bool flow_pnr71(std::vector<Branch71>& branches,double u,Trace71& trace){
    ++root_stats72.calls;
    std::vector<Spectrum71> spectra;for(auto& b:branches){Eigen::SelfAdjointEigenSolver<Real> solver(b.x);need(solver.info()==Eigen::Success,"Signed-flow eigensolver failure");Spectrum71 e{solver.eigenvalues(),solver.eigenvectors().transpose()*b.w,solver.eigenvectors()};need(e.values.minCoeff()>-1,"Signed-flow determinant domain");spectra.push_back(std::move(e));}
    struct Eval {long double p,derivative,absolute;};
    std::vector<long double> terms(branches.size()),derivatives(branches.size()),absolute(branches.size());
    auto evaluate=[&](long double s,bool derivative){
        for(std::size_t k=0;k<branches.size();++k){auto& e=spectra[k];terms[k]=branches[k].weight*std::exp(log_survival71(e,s));absolute[k]=std::abs(terms[k]);long double slope=0;
            if(derivative)for(int j=0;j<e.values.size();++j){long double value=e.values[j],a=e.rotated[j],den=1+s*value;slope-=.5L*(value/den+a*a/(den*den));}derivatives[k]=terms[k]*slope;
        }trace.evaluations+=branches.size();return Eval{sum71(terms),sum71(derivatives),sum71(absolute)};
    };
    auto vacuum=evaluate(1,false).p;need(std::isfinite(vacuum)&&vacuum>=0&&vacuum<=1,"Signed-flow invalid vacuum; no clipping");if(u<=vacuum)return false;
    long double lo=0,hi=1,s=.5L;auto origin=evaluate(0,true);auto guess=std::log((long double)u)/origin.derivative;if(std::isfinite(guess)&&guess>0&&guess<1)s=guess;
    bool converged=false;
    for(int j=0;j<32;++j){auto e=evaluate(s,true);++root_stats72.iterations;need(std::isfinite(e.p)&&e.p>=0&&e.p<=1&&std::isfinite(e.derivative),"Signed Newton evaluation invalid; no clipping");auto f=e.p-u;
        // Arithmetic-scale stopping tolerance; not a global distribution bound.
        auto tolerance=64.L*std::numeric_limits<long double>::epsilon()*(1+std::abs((long double)u)+e.absolute);
        if(std::abs(f)<=tolerance){converged=true;break;}if(f>0)lo=s;else hi=s;auto proposed=s-f/e.derivative;
        if(!(e.derivative<0&&std::isfinite(proposed)&&proposed>lo&&proposed<hi)||proposed==s)proposed=(lo+hi)/2;s=proposed;
    }
    if(!converged){++root_stats72.fallbacks;for(int j=0;j<64;++j){s=(lo+hi)/2;auto p=evaluate(s,false).p;++root_stats72.iterations;need(std::isfinite(p)&&p>=0&&p<=1,"Signed fallback survival invalid; no clipping");if(p>u)lo=s;else hi=s;}s=(lo+hi)/2;}
    double rounded=double(s);need(rounded>=0&&rounded<1,"Signed Newton root rounded to endpoint");long double total=evaluate(rounded,false).p;need(total>0,"Signed-flow normalization vanished");root_stats72.residual=std::max(root_stats72.residual,double(std::abs(total-u)));
    for(std::size_t k=0;k<branches.size();++k){auto& b=branches[k];auto& e=spectra[k];b.weight*=std::exp(log_survival71(e,rounded))/total;Vector values=e.values,rot=e.rotated;for(int j=0;j<values.size();++j){double den=1+rounded*e.values[j];values[j]=(1-rounded)*e.values[j]/den;rot[j]*=std::sqrt(1-rounded)/den;}b.x=e.basis*values.asDiagonal()*e.basis.transpose();b.x=(b.x+b.x.transpose()).eval()/2;b.w=e.basis*rot;}
    audit71(branches,trace);return true;
}
'''
s=s[:start]+replacement+s[end:]
s=s.replace('Version 2: condition-aware analytic moment compression.','Iteration72: safeguarded Newton survival inversion; condition-aware jump compression.',1)
s=s.replace('API int pnr72_sample(', 'API int pnr73_sample(',1)
s=s.replace('stats+7','stats+11').replace('std::array<double,7>','std::array<double,11>')
s=s.replace('auto worker=[&](int id){try{PrecisionGuard precision;', 'auto worker=[&](int id){try{PrecisionGuard precision;root_stats72=RootStats72{};',1)
s=s.replace('++completed;','local[id][7]=root_stats72.calls;local[id][8]=root_stats72.iterations;local[id][9]=root_stats72.fallbacks;local[id][10]=root_stats72.residual;++completed;',1)
s=s.replace('for(int j=0;j<7;++j){if(j==2||j==3||j==4)','for(int j=0;j<11;++j){if(j==2||j==3||j==4||j==10)',1)
s=s.replace('// maximum absolute weight sum, compressions, survival/component evaluations.','// maximum absolute weight sum, compressions, survival/component evaluations,\n// flow calls, root iterations, fallback roots, maximum absolute root residual.',1)
with (ROOT/'native/gaussian_pnr_newton72_v1.cpp').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
