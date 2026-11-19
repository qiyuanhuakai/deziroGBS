// Sequential Hermite polynomial in input creation coordinates, with a tangent
// for no-count time. This avoids enumerating all subsets of photon factors.
// Variables y=sqrt(tau) z keep the tau=0 endpoint finite and preserve parity.
using Z75=std::complex<LD73>;
struct Jet75 {Z75 value=0,prime=0;};
using Hermite75=std::unordered_map<std::string,Jet75>;
static Hermite75 act75(const Hermite75& p,Z75 alpha,Z75 alphad,const CVector74& lower,const CVector74& upper,const CVector74& lowerd,const CVector74& upperd){
    Hermite75 out;int n=lower.size();
    for(const auto& [key,h]:p){if(alpha!=Z75(0)||alphad!=Z75(0)){auto& a=out[key];a.value+=alpha*h.value;a.prime+=alphad*h.value+alpha*h.prime;}
        for(int j=0;j<n;++j){int degree=static_cast<unsigned char>(key[j]);
            if(upper[j]!=Z75(0)||upperd[j]!=Z75(0)){need(degree<250,"Hermite degree storage overflow");auto next=key;++next[j];auto& a=out[next];a.value+=upper[j]*h.value;a.prime+=upperd[j]*h.value+upper[j]*h.prime;}
            if(degree&&(lower[j]!=Z75(0)||lowerd[j]!=Z75(0))){auto next=key;--next[j];auto& a=out[next];a.value+=LD73(degree)*lower[j]*h.value;a.prime+=LD73(degree)*(lowerd[j]*h.value+lower[j]*h.prime);}}
    }
    need(out.size()<=200000,"Hermite polynomial workspace bound exceeded");return out;
}
static Hermite75 hermite75(const ClickModel70& c,const Vector& t,const LVec73& mean,const LVec73& meand,const std::vector<int>& modes,LD73 tau){
    int n=t.size();Hermite75 p;p[std::string(n,'\0')].value=1;
    for(int mode:modes){CVector74 low(n),up(n),lowd(n),upd(n);for(int j=0;j<n;++j){LD73 tj=t[j],den=1-tau*tau*tj*tj,f=1/std::sqrt(den),fd=-tau*tj*tj*f/den;auto z=c.model->v(mode,j);Z75 v(z.real(),z.imag());low[j]=v*f;up[j]=-v*tj*f;lowd[j]=v*fd;upd[j]=-v*tj*fd;}
        Z75 alpha(c.a.col(mode).cast<LD73>().dot(mean),c.b.col(mode).cast<LD73>().dot(mean)),ad(c.a.col(mode).cast<LD73>().dot(meand),c.b.col(mode).cast<LD73>().dot(meand));p=act75(p,alpha/std::sqrt(2.L),ad/std::sqrt(2.L),low,up,lowd,upd);}
    return p;
}
static std::pair<LD73,LD73> norm75(const Hermite75& p,LD73 tau){
    LD73 norm=0,prime=0;for(const auto& [key,h]:p){int degree=0;LD73 factorial=1;for(unsigned char k:key){degree+=k;for(int j=2;j<=k;++j)factorial*=j;}LD73 power=std::pow(tau,degree);norm+=factorial*power*std::norm(h.value);prime+=factorial*(power*2*std::real(std::conj(h.value)*h.prime)-(degree?degree*std::pow(tau,degree-1)*std::norm(h.value):0));}
    need(std::isfinite(norm)&&norm>=0&&std::isfinite(prime),"Hermite norm overflow");return {norm,prime};
}
static Z75 inner75(const Hermite75& a,const Hermite75& b){Z75 sum=0;for(const auto& [key,h]:a){auto found=b.find(key);if(found==b.end())continue;LD73 weight=1;for(unsigned char k:key)for(int j=2;j<=k;++j)weight*=j;sum+=weight*std::conj(h.value)*found->second.value;}return sum;}
static Hermite75 input_annihilate75(const Hermite75& p,const Vector& t,const Vector& mean,int index){
    int n=t.size();CVector74 low=CVector74::Zero(n),up=low,zero=low;LD73 tj=t[index],f=1/std::sqrt(1-tj*tj);low[index]=f;up[index]=-tj*f;return act75(p,Z75(mean[index],mean[index+n])/std::sqrt(2.L),0,low,up,zero,zero);
}
static void compress_hermite75(const ClickModel70& c,State73& state,const Vector& t,Stats73& stats){
    int n=t.size();auto p=hermite75(c,t,state.mean.cast<LD73>(),LVec73::Zero(2*n),state.factors,1);LD73 norm=norm75(p,1).first;need(norm>0,"Hermite compression normalization zero");std::vector<Hermite75> acted;LVec73 mean(2*n);LMat73 x=LMat73::Zero(2*n,2*n);
    for(int i=0;i<n;++i){acted.push_back(input_annihilate75(p,t,state.mean,i));auto z=inner75(p,acted.back())/norm;mean[i]=std::sqrt(2.L)*z.real();mean[i+n]=std::sqrt(2.L)*z.imag();}
    for(int i=0;i<n;++i)for(int j=0;j<n;++j){Z75 normal=inner75(acted[i],acted[j])/norm,anomalous=inner75(p,input_annihilate75(acted[j],t,state.mean,i))/norm;x(i,j)=normal.real()+anomalous.real()-mean[i]*mean[j];x(n+i,n+j)=normal.real()-anomalous.real()-mean[n+i]*mean[n+j];x(i,n+j)=anomalous.imag()+normal.imag()-mean[i]*mean[n+j];x(n+j,i)=x(i,n+j);}
    state.mean=mean.cast<double>();state.x=((x+x.transpose())/2).cast<double>();need(state.x.allFinite()&&state.mean.allFinite(),"Hermite compression overflow");state.factors.clear();++stats.compressions;
}
static bool hermite_flow75(const ClickModel70& c,State73& state,Vector& squeezing,double uniform,Stats73& stats){
    ++stats.flow_calls;int n=squeezing.size(),d=2*n;LD73 normalization=1;bool normalized=false;
    auto evaluate=[&](LD73 s){LD73 logp=0,slope=0;LVec73 tilted(d),tiltedd(d);
        for(int i=0;i<d;++i){LD73 t=squeezing[i%n],lambda=i<n?-t/(1+t):t/(1-t),a=state.mean[i],den=1+s*lambda,z=a/den,x=lambda/den;tilted[i]=z;tiltedd[i]=-x*z;logp-=.5L*(std::log1p(s*lambda)+s*a*a/den);slope-=.5L*(x+z*z);}
        auto norm=norm75(hermite75(c,squeezing,tilted,tiltedd,state.factors,1-s),1-s);LD73 gaussian=std::exp(logp),p=gaussian*norm.first/normalization,dp=gaussian*(slope*norm.first+norm.second)/normalization;need(std::isfinite(p)&&p>=0&&(!normalized||p<=1),"Hermite survival outside [0,1]; no clipping");return std::pair<LD73,LD73>(p,dp);
    };
    auto initial=evaluate(0);normalization=initial.first;need(normalization>0,"Hermite state normalization zero");normalized=true;if(uniform<=evaluate(1).first)return false;LD73 lo=0,hi=1,s=.5L,guess=std::log((LD73)uniform)/(initial.second/normalization);if(guess>0&&guess<1&&std::isfinite(guess))s=guess;bool converged=false;
    for(int i=0;i<32;++i){auto [p,dp]=evaluate(s);++stats.root_iterations;LD73 residual=p-uniform;if(std::abs(residual)<=64*std::numeric_limits<LD73>::epsilon()*(1+std::abs(p))){converged=true;break;}if(residual>0)lo=s;else hi=s;auto proposed=s-residual/dp;if(!(dp<0&&std::isfinite(proposed)&&proposed>lo&&proposed<hi))proposed=(lo+hi)/2;s=proposed;}
    if(!converged){++stats.root_fallbacks;for(int i=0;i<64;++i){s=(lo+hi)/2;++stats.root_iterations;if(evaluate(s).first>uniform)lo=s;else hi=s;}s=(lo+hi)/2;}double rounded=double(s);need(rounded>=0&&rounded<1,"Hermite root rounded to endpoint");stats.residual=std::max(stats.residual,double(std::abs(evaluate(rounded).first-uniform)));state.x.setZero();
    for(int i=0;i<n;++i){LD73 t=squeezing[i],remain=1-(LD73)rounded;state.mean[i]*=double(std::sqrt(remain)*(1+t)/(1+remain*t));state.mean[i+n]*=double(std::sqrt(remain)*(1-t)/(1-remain*t));squeezing[i]=double(remain*t);t=squeezing[i];state.x(i,i)=double(-t/(1+t));state.x(i+n,i+n)=double(t/(1-t));}return true;
}
static int hermite_detector75(const ClickModel70& c,const State73& state,const Vector& t,double uniform){std::vector<int> modes=state.factors;modes.push_back(0);std::vector<LD73> cumulative;LD73 total=0;
    for(int i=0;i<c.model->m;++i){modes.back()=i;LD73 rate=norm75(hermite75(c,t,state.mean.cast<LD73>(),LVec73::Zero(state.mean.size()),modes,1),1).first,next=total+rate;need(!(rate>0&&next<=total),"Hermite detector interval lost");total=next;cumulative.push_back(total);}return categorical(cumulative,total,uniform);}
