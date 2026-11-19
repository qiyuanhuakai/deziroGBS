// Iteration72: safeguarded Newton survival inversion; condition-aware jump compression.
// Experimental PNR trajectories with finite-tap signed Gaussian jump states.
// Flow and detector rates are for the current signed state. A jump is replaced
// by a threshold herald in a tap of strength epsilon, then global attenuation.
// This is approximate at finite epsilon, additionally approximate on merging.
// There is no photon-count cutoff; storage/numerical failures abort the batch.
#include "gaussian_click_branches_v1.cpp"
struct Spectrum71 {Vector values,rotated;Real basis;};
static long double log_survival71(const Spectrum71& e,long double s){
    long double logp=0;for(int j=0;j<e.values.size();++j){long double v=e.values[j],a=e.rotated[j],den=1+s*v;logp-=.5L*(std::log1p(s*v)+s*a*a/den);}return logp;
}

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
static int select_pnr71(const ClickModel70& model,const std::vector<Branch71>& branches,double u){
    std::vector<long double> cumulative;long double total=0;
    for(int mode=0;mode<model.model->m;++mode){Vector a=model.a.col(mode),b=model.b.col(mode);std::vector<long double> terms;
        for(auto& state:branches){Vector xa=state.x*a,xb=state.x*b;auto ma=dot70(a,state.w),mb=dot70(b,state.w);terms.push_back(state.weight*.5L*(dot70(a,xa)+dot70(b,xb)+ma*ma+mb*mb));}
        auto rate=sum71(terms);need(std::isfinite(rate)&&rate>=0,"Signed PNR detector rate invalid; no clipping");auto next=total+rate;need(!(rate>0&&next<=total),"Signed PNR rate interval lost");total=next;cumulative.push_back(total);
    }return categorical(cumulative,total,u);
}

// Analytically merge each component's click moments before multiplying the
// signed mixture weights. This avoids differencing nearly equal covariances.
// The condition limit is an explicit approximation control, not clipping.
static void stable_jump72(std::vector<Branch71>& branches,const Step71& step,int budget,double condition_limit,Trace71& trace){
    need(step.p1>0,"Zero finite-tap herald probability");
    long double predicted=0;for(std::size_t k=0;k<branches.size();++k)predicted+=std::abs(branches[k].weight)*(1+step.parts[k].p0)/step.p1;
    if(branches.size()*2<=std::size_t(budget)&&predicted<=condition_limit){update71(branches,step,true,budget,trace);return;}
    std::vector<Branch71> moments;for(std::size_t k=0;k<branches.size();++k){auto& p=step.parts[k];if(p.p1==0)continue;auto b=branches[k];click_update70(b.x,b.w,p,true);b.weight*=p.p1/step.p1;moments.push_back(std::move(b));}
    need(!moments.empty(),"Empty finite-tap conditional moments");auto merged=merge71(moments);branches.clear();branches.push_back(std::move(merged));++trace.merges;
}

// Probe finite-tap conditional first/second moments for independent checks.
API int pnr71_jump_probe(void* handle,const double* x,const double* mean,int mode,double epsilon,double* out,double* outmean,double* probability,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&x&&mean&&out&&outmean&&probability&&epsilon>0&&epsilon<1,"Invalid finite-tap probe");auto& model=*static_cast<ClickModel70*>(handle);int d=2*model.model->n;need(mode>=0&&mode<model.model->m,"Invalid finite-tap mode");
    ClickModel70 tap;tap.a=std::sqrt(epsilon)*model.a;tap.b=std::sqrt(epsilon)*model.b;Real state(d,d);Vector w(d);for(int i=0;i<d;++i){w[i]=mean[i];for(int j=0;j<d;++j)state(i,j)=x[std::size_t(i)*d+j];}
    auto p=click_parameters70(tap,state,w,mode);*probability=double(p.p1);click_update70(state,w,p,true);state*=1-epsilon;w*=std::sqrt(1-epsilon);for(int i=0;i<d;++i){outmean[i]=w[i];for(int j=0;j<d;++j)out[std::size_t(i)*d+j]=state(i,j);}return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
// stats: photons, completed, peak photons, peak transient branches,
// maximum absolute weight sum, compressions, survival/component evaluations,
// flow calls, root iterations, fallback roots, maximum absolute root residual.
API int pnr73_sample(void* handle,int count,std::uint64_t seed,int threads,int budget,double epsilon,double condition_limit,const double* mean,const double* variance,std::uint32_t* output,double* stats,char* message,int cap){try{
    PrecisionGuard precision;need(handle&&count>=0&&threads>=1&&threads<=256&&budget>=1&&budget<=65536&&epsilon>0&&epsilon<1&&std::isfinite(condition_limit)&&condition_limit>=1&&mean&&variance&&output&&stats,"Invalid signed PNR batch");auto& model=*static_cast<ClickModel70*>(handle);int d=2*model.model->n,m=model.model->m;Vector center(d),deviation(d);
    for(int j=0;j<d;++j){need(std::isfinite(mean[j])&&std::isfinite(variance[j])&&variance[j]>=0,"Invalid signed PNR displacement");center[j]=mean[j];deviation[j]=std::sqrt(variance[j]);}
    ClickModel70 tap;tap.a=std::sqrt(epsilon)*model.a;tap.b=std::sqrt(epsilon)*model.b;
    std::fill(stats,stats+11,0);std::fill(output,output+std::size_t(count)*m,0);if(!count)return 0;int workers=std::min(count,threads);
    std::atomic<std::int64_t> next{0},completed{0};std::atomic<bool> failed{false};std::vector<std::array<double,11>> local(workers);std::vector<std::exception_ptr> failures(workers);std::vector<std::thread> pool;pool.reserve(workers);
    auto worker=[&](int id){try{PrecisionGuard precision;root_stats72=RootStats72{};for(;;){if(failed.load())break;auto row=next.fetch_add(1);if(row>=count)break;
        std::mt19937_64 waits(mix(seed+2ULL*row)),detector(mix(seed+2ULL*row+1)),noise(mix(seed+0x81c6f2d54a970e31ULL+row));std::normal_distribution<double> normal;Vector w=center;for(int j=0;j<d;++j)if(deviation[j]>0)w[j]+=deviation[j]*normal(noise);
        std::vector<Branch71> branches{{model.model->initial,w,1}};Trace71 trace;std::uint64_t photons=0;
        while(flow_pnr71(branches,uniform(waits),trace)){
            int mode=select_pnr71(model,branches,uniform(detector));auto step=parameters71(tap,branches,mode,trace);stable_jump72(branches,step,budget,condition_limit,trace);
            for(auto& b:branches){b.x*=1-epsilon;b.w*=std::sqrt(1-epsilon);}
            auto& value=output[std::size_t(row)*m+mode];need(value<std::numeric_limits<std::uint32_t>::max(),"Signed PNR photon storage overflow");++value;++photons;
        }
        local[id][0]+=double(photons);local[id][1]+=1;local[id][2]=std::max(local[id][2],double(photons));local[id][3]=std::max(local[id][3],trace.peak);local[id][4]=std::max(local[id][4],double(trace.cancellation));local[id][5]+=trace.merges;local[id][6]+=trace.evaluations;local[id][7]=root_stats72.calls;local[id][8]=root_stats72.iterations;local[id][9]=root_stats72.fallbacks;local[id][10]=root_stats72.residual;++completed;
    }}catch(...){failures[id]=std::current_exception();failed.store(true);}};
    try{for(int id=0;id<workers;++id)pool.emplace_back(worker,id);}catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}
    for(auto& thread:pool)thread.join();for(auto& row:local)for(int j=0;j<11;++j){if(j==2||j==3||j==4||j==10)stats[j]=std::max(stats[j],row[j]);else stats[j]+=row[j];}
    for(auto& failure:failures)if(failure)std::rethrow_exception(failure);need(completed.load()==count,"Incomplete signed PNR batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
