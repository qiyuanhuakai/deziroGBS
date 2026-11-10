// Reusable coherent-reference envelope for displaced reset detector selection.
// Exact-arithmetic identity: z = alpha*z_ref + delta. Cauchy--Schwarz and
// |a+b|^2 <= (1+t)|a|^2 + (1+1/t)|b|^2 yield a fixed-reference plus leverage
// proposal. Covariance occupation uses the existing diagonal envelope.
// This changes detector selection only, not the displaced Gaussian reset law.
#include "iteration69_displaced_v6.cpp"
using LC=std::complex<long double>;
struct Anchor {Eigen::VectorXcd z;std::vector<long double> weights,cdf;long double total=0.,norm=0.;};
static Anchor make_anchor(const Model& model,const Vector& w){
    Anchor a;a.z.resize(model.n);a.weights.resize(model.m);a.cdf.resize(model.m);
    for(int j=0;j<model.n;++j){a.z[j]={w[j]/std::sqrt(2.),w[j+model.n]/std::sqrt(2.)};a.norm+=std::norm(LC(a.z[j].real(),a.z[j].imag()));}
    const long double eps=std::numeric_limits<long double>::epsilon();
    for(int i=0;i<model.m;++i){LC sum=0.;long double magnitudes=0.;for(int j=0;j<model.n;++j){LC v(model.v(i,j).real(),model.v(i,j).imag()),z(a.z[j].real(),a.z[j].imag());sum+=v*z;magnitudes+=std::abs(v)*std::abs(z);}long double amplitude=std::abs(sum)+64.L*model.n*eps*magnitudes;a.weights[i]=amplitude>0?up(amplitude*amplitude):0.;long double next=a.total+a.weights[i];need(!(a.weights[i]>0.&&next<=a.total),"Positive coherent-anchor interval lost");a.total=next;a.cdf[i]=next;}
    return a;
}
static int choose_anchor(const Model& model,const Complex& h,const Vector& w,const Anchor& anchor,std::mt19937_64& rng,std::uint64_t& trials,long double& chosen_rate){
    const int n=model.n;const long double eps=std::numeric_limits<long double>::epsilon(),guard=64.L*n*n*eps;
    std::vector<long double> diagonal(n),mixture(n+2);long double total=0.;
    for(int j=0;j<n;++j){long double d=0.;for(int k=0;k<n;++k){d=up(d+std::abs((long double)h(j,k).real()));d=up(d+std::abs((long double)h(j,k).imag()));}diagonal[j]=up(d*(1+guard));long double weight=diagonal[j]*model.column_total[j],next=total+weight;need(!(weight>0.&&next<=total),"Positive anchor covariance interval lost");total=next;mixture[j]=total;}
    // z is rounded exactly as total_occupation; any alpha gives a valid split.
    Eigen::VectorXcd z(n);for(int j=0;j<n;++j)z[j]=std::complex<double>(w[j],w[j+n])/std::sqrt(2.);
    LC numerator=0.;for(int j=0;j<n;++j)numerator+=std::conj(LC(anchor.z[j].real(),anchor.z[j].imag()))*LC(z[j].real(),z[j].imag());
    LC alpha=anchor.norm>0?numerator/anchor.norm:LC(0.);long double residual_norm=0.;
    for(int j=0;j<n;++j){LC ref(anchor.z[j].real(),anchor.z[j].imag()),current(z[j].real(),z[j].imag()),residual=current-alpha*ref;long double rounding=64.L*eps*(std::abs(current)+std::abs(alpha)*std::abs(ref));long double length=std::abs(residual)+rounding;residual_norm=up(residual_norm+up(length*length));}
    long double aa=up(std::norm(alpha)*(1+guard)),reference_mass=aa*anchor.total,residual_mass=residual_norm*model.total,reference_coefficient=0.,leverage_coefficient=0.;
    if(reference_mass>0.&&residual_mass>0.){long double t=std::sqrt(residual_mass/reference_mass);reference_coefficient=up((1+t)*aa*(1+guard));leverage_coefficient=up((1+1/t)*residual_norm*(1+guard));}
    else if(reference_mass>0.)reference_coefficient=aa;else leverage_coefficient=residual_norm;
    // Inflate a tiny positive proposal component, never discard target mass.
    // This only loosens the rejection envelope; the same coefficients enter
    // the sampled mixture and its acceptance denominator.
    long double scale=total+reference_coefficient*anchor.total+leverage_coefficient*model.total;
    if(reference_coefficient>0.&&anchor.total>0.)reference_coefficient=std::max(reference_coefficient,up(1024.L*eps*scale/anchor.total));
    if(leverage_coefficient>0.&&model.total>0.)leverage_coefficient=std::max(leverage_coefficient,up(1024.L*eps*scale/model.total));
    for(int which=0;which<2;++which){long double weight=which==0?reference_coefficient*anchor.total:leverage_coefficient*model.total;long double next=total+weight;need(!(weight>0.&&next<=total),"Positive anchor mixture interval lost");total=next;mixture[n+which]=total;}
    // Evaluate covariance and coherent rates separately to avoid a rounded double outer-product violating the long-double envelope.
    for(;;){need(trials<std::numeric_limits<std::uint64_t>::max(),"Anchor proposal counter overflow");++trials;int component=categorical(mixture,total,uniform(rng)),i;
        if(component<n){const auto* first=model.column_cdf.data()+std::size_t(component)*model.m;auto found=std::upper_bound(first,first+model.m,(long double)uniform(rng)*model.column_total[component]);need(found!=first+model.m,"Anchor column endpoint lost");i=int(found-first);}
        else if(component==n)i=categorical(anchor.cdf,anchor.total,uniform(rng));else i=categorical(model.cdf,model.total,uniform(rng));
        long double ceiling=reference_coefficient*anchor.weights[i]+leverage_coefficient*model.ell[i];for(int j=0;j<n;++j)ceiling+=diagonal[j]*model.entry_weights[std::size_t(i)*n+j];LC amplitude=0.;for(int j=0;j<n;++j)amplitude+=LC(model.v(i,j).real(),model.v(i,j).imag())*LC(z[j].real(),z[j].imag());long double covariance_rate=rate(model,h,i),r=covariance_rate+std::norm(amplitude);if(!(ceiling>0.&&r<=ceiling)){std::ostringstream details;details<<std::setprecision(25)<<"Coherent-anchor envelope violated r="<<r<<" ceiling="<<ceiling<<" covariance="<<covariance_rate<<" coherent="<<std::norm(amplitude)<<" reference_coefficient="<<reference_coefficient<<" leverage_coefficient="<<leverage_coefficient;throw std::runtime_error(details.str());}if((long double)uniform(rng)*ceiling<r){chosen_rate=r;return i;}
    }
}
// Same 7 counters as ds_sample, then total anchor-construction seconds.
API int ds_sample_anchor(void* handle,int count,std::uint64_t seed,const double* mean,const double* variance,std::uint32_t* output,double* stats,char* message,int cap){try{PrecisionGuard rounding;
    need(handle&&count>=0&&mean&&variance&&output&&stats,"Invalid anchor sampler call");auto& model=*static_cast<Model*>(handle);Vector center(2*model.n),deviation(2*model.n);
    for(int j=0;j<2*model.n;++j){need(std::isfinite(mean[j])&&std::isfinite(variance[j])&&variance[j]>=0.,"Invalid anchor displacement distribution");center[j]=mean[j];deviation[j]=std::sqrt(variance[j]);}
    std::fill(stats,stats+8,0.);std::fill(output,output+std::size_t(count)*model.m,0);
    for(int sample=0;sample<count;++sample){std::mt19937_64 waits(mix(seed+2*sample)),detectors(mix(seed+2*sample+1)),noise(mix(seed+0x81c6f2d54a970e31ULL+sample));std::normal_distribution<double> normal;Real x=model.initial;Vector w=center;for(int j=0;j<w.size();++j)if(deviation[j]>0.)w[j]+=deviation[j]*normal(noise);
        auto start=Clock::now();Anchor anchor=make_anchor(model,w);stats[7]+=std::chrono::duration<double>(Clock::now()-start).count();std::uint64_t photons=0,trials=0;
        for(;;){start=Clock::now();auto flow=displaced_evolve(x,w,uniform(waits));stats[2]+=std::chrono::duration<double>(Clock::now()-start).count();if(flow.stopped)break;x=std::move(flow.x);w=std::move(flow.w);start=Clock::now();auto h=occupation(x);long double lambda=0.;int mode=choose_anchor(model,h,w,anchor,detectors,trials,lambda);stats[3]+=std::chrono::duration<double>(Clock::now()-start).count();start=Clock::now();displaced_jump(model,x,w,mode,lambda);stats[4]+=std::chrono::duration<double>(Clock::now()-start).count();auto& value=output[std::size_t(sample)*model.m+mode];need(value<std::numeric_limits<std::uint32_t>::max(),"Anchor photon storage overflow");++value;++photons;}
        stats[0]+=double(photons);stats[1]+=double(trials);stats[5]+=(photons==0);stats[6]=std::max(stats[6],double(photons));}
    return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
