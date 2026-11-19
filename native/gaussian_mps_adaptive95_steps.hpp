// One jump of an already selected detector. The stored Gaussian/MPS state
// is right-canonical on entry. Each extra physical slice is exactly zero
// before the jump and represents the single next Fock level after it.
struct Truncation95 {
    int max_dimension, max_bond;
    double target;
    double* trace;
};

static long double compress95(Mps93& state, int base_bond, Truncation95 step, double* stats) {
    long double log_retained=0;
    const long double cut_target=step.target*step.target/std::max(1,int(state.size())-1);
    for(int i=0;i+1<int(state.size());++i) {
        int left=state[i][0].rows(),right=state[i][0].cols(),d=state[i].size();
        Mat93 flat(d*left,right);
        for(int s=0;s<d;++s) flat.middleRows(s*left,left)=state[i][s];
        Mat93 vectors,vt;
        Vec93 singular;
        thin_svd93(flat,vectors,singular,vt);
        int keep=std::min<int>(base_bond,singular.size());
        long double mass=0,discard=0;
        for(int j=0;j<singular.size();++j) {
            long double weight=static_cast<long double>(singular[j])*singular[j];
            mass+=weight;
            if(j>=keep) discard+=weight;
        }
        need93(mass>0,"MPS SVD zero norm");
        const int cap=std::min<int>(step.max_bond,singular.size());
        while(keep<cap && discard>cut_target*mass) {
            discard-=static_cast<long double>(singular[keep])*singular[keep];
            ++keep;
        }
        while(keep>1&&singular[keep-1]<=1e-13*singular[0]) --keep;
        discard=0;
        for(int j=keep;j<singular.size();++j) discard+=static_cast<long double>(singular[j])*singular[j];
        const long double fraction=discard/mass;
        if(fraction>step.trace[5]) {step.trace[5]=double(fraction);step.trace[6]=i;}
        step.trace[7]+=keep>base_bond;
        step.trace[8]=std::max(step.trace[8],double(keep));
        log_retained+=std::log1p(-fraction);
        if(discard>0) ++stats[7];
        Mat93 u=vectors.leftCols(keep);
        Mat93 transfer=singular.head(keep).asDiagonal()*vt.topRows(keep);
        transfer/=std::sqrt(mass-discard);
        for(int s=0;s<d;++s) state[i][s]=u.middleRows(s*left,left);
        for(auto& a:state[i+1]) a=(transfer*a).eval();
        stats[8]=std::max(stats[8],double(keep));
    }
    auto norm=canonical93(state);
    double residual=canonical_residual93(state);
    stats[9]=std::max(stats[9],double(std::abs(norm-1)));
    stats[10]=std::max(stats[10],residual);
    need93(residual<1e-9&&std::abs(norm-1)<1e-8L,"MPS canonical residual failed");
    return log_retained;
}

static void finish_jump95(Mps93& prior,Mps93& state,const Gaussian93& g,
        const std::vector<Z93>& weights,const std::vector<long double>& leaks,
        long double full,long double outside,int port,int bond,double budget,double* stats,Truncation95 step) {
    auto trace=step.trace;
    trace[0]=port;trace[1]=stats[5];trace[2]=std::sqrt(outside/full);
    trace[6]=-1;trace[11]=-1;
    const long double allowance=full*step.target*step.target;
    std::vector<std::pair<long double,int>> eligible;
    long double largest=0;
    for(int i=0;i<int(prior.size());++i) {
        long double contribution=std::norm(weights[i])*leaks[i];
        if(contribution>largest) {largest=contribution;trace[11]=i;}
        if(outside>allowance && int(prior[i].size())<step.max_dimension && contribution>0)
            eligible.emplace_back(contribution,i);
    }
    std::sort(eligible.rbegin(),eligible.rend());
    for(auto [weight,i]:eligible) {
        if(outside<=allowance) break;
        auto& site=prior[i];
        site.emplace_back(Mat93::Zero(site[0].rows(),site[0].cols()));
        outside-=weight;
        ++trace[9];
    }
    if(trace[9]) {
        outside=0;
        // Expanded modes have no input amplitude in their new top slice.
        for(int i=0;i<int(prior.size());++i)
            if(int(prior[i].size())==int(state[i].size())) outside+=std::norm(weights[i])*leaks[i];
        state=jump_mpo93(prior,g,weights);
        long double reconstructed=canonical93(state)+outside;
        trace[13]=double(std::abs(reconstructed-full)/full);
        need93(trace[13]<1e-9,"Expanded jump changes detector hazard");
        full=reconstructed;
    }
    for(const auto& site:prior) trace[12]=std::max(trace[12],double(site.size()));
    trace[3]=std::sqrt(outside/full);
    long double log_bond=compress95(state,bond,step,stats);
    trace[4]=std::sqrt(-std::expm1(log_bond));
    double delta=std::sqrt(-std::expm1(std::log1p(-outside/full)+log_bond));
    trace[10]=delta;
    charge93(delta,budget,stats);
    prior=std::move(state);
}
