// Coherent total-photon Gaussian core + correlated classical displacement.
// Dense, explicitly capped validation prototype. No MPS/scalability claim.
// The refined channel is latent: trajectory probabilities are not binary logQ.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Real = long double;
using Complex = std::complex<Real>;
using Clock = std::chrono::steady_clock;
constexpr int MAX_MODES = 64, MAX_CUTOFF = 8;
constexpr std::uint64_t MAX_BASIS = 100000, MAX_STAGE_BASIS = 1000000;
constexpr std::uint64_t MAX_PAYLOAD = 256ULL * 1024 * 1024;
constexpr Real EPS = std::numeric_limits<Real>::epsilon();
struct NumericError : std::runtime_error { using std::runtime_error::runtime_error; };
struct EnumerationError : std::runtime_error { using std::runtime_error::runtime_error; };

double seconds(Clock::time_point started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}
struct Sum {
    Real s=0, c=0;
    void add(Real x) {
        if (!std::isfinite(x)) throw NumericError("nonfinite summand");
        const Real next=s+x;
        c += std::abs(s)>=std::abs(x) ? (s-next)+x : (x-next)+s;
        s=next;
    }
    Real value() const {
        const Real v=s+c;
        if (!std::isfinite(v)) throw NumericError("nonfinite compensated sum");
        return v;
    }
};
struct CSum {
    Sum re,im;
    void add(Complex z) { re.add(z.real()); im.add(z.imag()); }
    Complex value() const { return {re.value(),im.value()}; }
};
void finite(Real x,const char* message) {
    if (!std::isfinite(x)) throw NumericError(message);
}
void finite(Complex x,const char* message) {
    if (!std::isfinite(x.real()) || !std::isfinite(x.imag())) throw NumericError(message);
}
Real positive_product(Real a,Real b,const char* message) {
    const Real v=a*b;
    if (!std::isfinite(v) || (a!=0 && b!=0 && (v==0 || std::fpclassify(v)==FP_SUBNORMAL)))
        throw NumericError(message);
    return v;
}
Complex product(Complex a,Complex b) {
    const Complex v=a*b;
    finite(v,"nonfinite complex product");
    if (a!=Complex{} && b!=Complex{} && v==Complex{})
        throw NumericError("positive complex magnitude underflowed in multiplication");
    return v;
}
Real square(Complex v) {
    finite(v,"nonfinite amplitude");
    const Real result=std::norm(v);
    if (!std::isfinite(result) || (v!=Complex{} && (result==0 || std::fpclassify(result)==FP_SUBNORMAL)))
        throw NumericError("nonzero amplitude norm underflowed or overflowed");
    return result;
}
Real weight(const std::vector<Complex>& state) {
    Sum result;
    for (Complex x:state) result.add(square(x));
    return result.value();
}
Real positive_exp(Real x,const char* message) {
    const Real v=std::exp(x);
    if (!(v>0) || !std::isfinite(v) || std::fpclassify(v)==FP_SUBNORMAL)
        throw NumericError(message);
    return v;
}

struct Options {
    std::string action="sample",unitary,state,alpha,uniforms,normals;
    int modes=0,cutoff=0;
    Real squeezing=0,loss=1,guard=1e-10L;
    std::uint64_t samples=0,seed=0,max_basis=MAX_BASIS,max_stage_basis=MAX_STAGE_BASIS;
    std::uint64_t max_payload=MAX_PAYLOAD,max_nodes=1000000;
    bool raw=false,return_state=false,trace=false;
};
std::uint64_t integer(const std::string& text) {
    if (text.empty() || text[0]=='-') throw std::invalid_argument("expected nonnegative integer");
    size_t pos=0; const auto value=std::stoull(text,&pos);
    if (pos!=text.size()) throw std::invalid_argument("invalid integer argument");
    return value;
}
Real decimal(const std::string& text) {
    size_t pos=0; const Real value=std::stold(text,&pos);
    if (pos!=text.size() || !std::isfinite(value)) throw std::invalid_argument("expected finite decimal");
    return value;
}
Options options(int argc,char** argv) {
    Options o;
    for (int i=1;i<argc;++i) {
        const std::string name=argv[i];
        if (name=="--raw-samples") { o.raw=true; continue; }
        if (name=="--return-state") { o.return_state=true; continue; }
        if (name=="--trace") { o.trace=true; continue; }
        if (++i==argc) throw std::invalid_argument("missing CLI option value");
        const std::string value=argv[i];
        if (name=="--action") o.action=value;
        else if (name=="--unitary") o.unitary=value;
        else if (name=="--state") o.state=value;
        else if (name=="--alpha") o.alpha=value;
        else if (name=="--uniforms") o.uniforms=value;
        else if (name=="--normals") o.normals=value;
        else if (name=="--modes") { auto v=integer(value); if(v>MAX_MODES) throw std::invalid_argument("mode cap exceeded"); o.modes=int(v); }
        else if (name=="--cutoff") { auto v=integer(value); if(v>MAX_CUTOFF) throw std::invalid_argument("cutoff cap exceeded"); o.cutoff=int(v); }
        else if (name=="--samples") o.samples=integer(value);
        else if (name=="--seed") o.seed=integer(value);
        else if (name=="--max-basis") o.max_basis=integer(value);
        else if (name=="--max-stage-basis") o.max_stage_basis=integer(value);
        else if (name=="--max-payload-bytes") o.max_payload=integer(value);
        else if (name=="--max-enumeration-nodes") o.max_nodes=integer(value);
        else if (name=="--squeezing") o.squeezing=decimal(value);
        else if (name=="--loss") o.loss=decimal(value);
        else if (name=="--numeric-guard") o.guard=decimal(value);
        else throw std::invalid_argument("unknown CLI option: "+name);
    }
    if (o.modes<1 || o.modes>MAX_MODES || o.cutoff<0 || o.cutoff>MAX_CUTOFF)
        throw std::invalid_argument("invalid modes/cutoff");
    if (o.squeezing<0 || o.loss<0 || o.loss>1 || !(o.guard>0 && o.guard<=1e-6L))
        throw std::invalid_argument("invalid physical parameters or numeric guard");
    if (!o.max_basis || o.max_basis>MAX_BASIS || !o.max_stage_basis || o.max_stage_basis>MAX_STAGE_BASIS
        || !o.max_payload || o.max_payload>MAX_PAYLOAD || !o.max_nodes || o.max_nodes>1000000)
        throw std::invalid_argument("resource limit exceeds hard cap");
    if (o.action!="sample" && o.action!="state" && o.action!="channels"
        && o.action!="vacuum-law" && o.action!="refined-law")
        throw std::invalid_argument("unsupported action");
    if (o.action=="channels" && (o.state.empty() || o.alpha.empty()))
        throw std::invalid_argument("channels requires state and first-mode alpha");
    if (o.action!="channels" && o.unitary.empty())
        throw std::invalid_argument("Gaussian core action requires unitary");
    if ((o.action=="vacuum-law" || o.action=="refined-law") && o.alpha.empty())
        throw std::invalid_argument("fixed-law action requires alpha");
    if (!o.alpha.empty() && !o.normals.empty()) throw std::invalid_argument("fixed alpha and normal draws conflict");
    if (o.action!="sample" && (o.samples || !o.uniforms.empty() || !o.normals.empty()))
        throw std::invalid_argument("sample options require sample action");
    if (o.action=="vacuum-law" && o.modes>12) throw std::invalid_argument("vacuum full law supports at most 12 modes");
    if (o.action=="refined-law" && o.modes>4) throw std::invalid_argument("refined full law supports at most 4 modes");
    if (o.samples>1000000 || o.samples>o.max_payload/sizeof(std::uint64_t))
        throw std::invalid_argument("sample count/storage cap exceeded");
    if (o.trace && o.samples>100000ULL/std::uint64_t(o.modes))
        throw std::invalid_argument("trace entry cap exceeded");
    return o;
}
// CSV tokens for U/alpha/random inputs are deliberately parsed through double:
// the model starts from the supplied binary64 inputs, then widens exactly.
std::vector<Real> read_numbers(const std::string& path,bool binary64) {
    std::ifstream stream(path);
    if (!stream) throw std::invalid_argument("cannot open input file: "+path);
    std::vector<Real> values;
    std::string line,token;
    while (std::getline(stream,line)) {
        std::replace(line.begin(),line.end(),',',' ');
        std::istringstream row(line);
        while (row>>token) {
            size_t pos=0;
            Real x=binary64 ? Real(std::stod(token,&pos)) : std::stold(token,&pos);
            if (pos!=token.size() || !std::isfinite(x)) throw std::invalid_argument("invalid numeric input token");
            values.push_back(x);
        }
    }
    return values;
}
std::vector<Complex> read_complex(const std::string& path,size_t count) {
    auto raw=read_numbers(path,true);
    if (raw.size()!=2*count) throw std::invalid_argument("complex input shape mismatch");
    std::vector<Complex> result(count);
    for(size_t j=0;j<count;++j) result[j]={raw[2*j],raw[2*j+1]};
    return result;
}

std::uint64_t choose(int n,int k) {
    if(k<0 || n<k) return 0;
    k=std::min(k,n-k);
    std::uint64_t value=1;
    for(int i=1;i<=k;++i) {
        // n<=73,k<=9 in this prototype; the intermediate fits uint64.
        if(value>std::numeric_limits<std::uint64_t>::max()/std::uint64_t(n-k+i))
            throw std::invalid_argument("basis combinatorial count overflow");
        value=value*std::uint64_t(n-k+i)/std::uint64_t(i);
    }
    return value;
}
std::uint64_t dimension(int modes,int K) { return choose(modes+K,K); }
struct Resources {
    std::uint64_t full=0,even=0,stages=0,fibers=0,planned_payload=0;
};
Resources resources(const Options& o) {
    Resources r;
    r.full=dimension(o.modes,o.cutoff);
    for(int n=0;n<=o.cutoff;n+=2) r.even+=choose(o.modes+n-1,n);
    for(int modes=0;modes<=o.modes;++modes) r.stages+=dimension(modes,o.cutoff);
    for(int modes=1;modes<=o.modes;++modes) r.fibers+=dimension(modes-1,o.cutoff)*std::uint64_t(o.cutoff+1);
    // Payload of all basis codes/degrees/indices, U/B, core + two state
    // vectors, a full set of contracted channel vectors, and packed output.
    r.planned_payload=r.stages*(sizeof(std::uint64_t)+sizeof(std::uint8_t))
        +r.fibers*sizeof(std::int32_t)+2ULL*o.modes*o.modes*sizeof(Complex)
        +(3*r.full+std::uint64_t(o.cutoff+1)*dimension(o.modes-1,o.cutoff))*sizeof(Complex)
        +r.full*sizeof(Real)+o.samples*sizeof(std::uint64_t)
        +(o.uniforms.empty() ? 0:o.samples*2ULL*o.modes*sizeof(Real))
        +(o.normals.empty() ? 0:o.samples*std::uint64_t(o.modes)*sizeof(Real))
        +(o.trace ? o.samples*std::uint64_t(o.modes)*512:0);
    if(r.full>o.max_basis || r.stages>o.max_stage_basis || r.fibers>std::uint64_t(std::numeric_limits<std::int32_t>::max())
        || r.planned_payload>o.max_payload)
        throw std::invalid_argument("dense Fock basis/stage/index/payload resource cap exceeded");
    return r;
}
// Code stores <=8 sorted photon mode labels in 7-bit slots; label is mode+1.
// Unlike per-mode bit packing this covers 64 modes without exceeding uint64.
std::uint64_t encode(const std::vector<int>& photons) {
    std::uint64_t code=0;
    for(size_t j=0;j<photons.size();++j) code|=std::uint64_t(photons[j]+1)<<(7*j);
    return code;
}
std::vector<int> decode(std::uint64_t code) {
    std::vector<int> result;
    while(code) { result.push_back(int(code&127)-1); code>>=7; }
    return result;
}
std::uint64_t rank_code(std::uint64_t code,int modes,int K) {
    auto p=decode(code); const int N=int(p.size());
    if(N>K) throw std::logic_error("Fock rank exceeds cutoff");
    std::uint64_t rank=0;
    for(int n=0;n<N;++n) rank+=choose(modes+n-1,n);
    int minimum=0;
    for(int j=0;j<N;++j) {
        if(p[j]<minimum || p[j]>=modes) throw std::logic_error("invalid sorted photon code");
        const int left=N-j-1;
        rank+=choose(modes-minimum+left,left+1)-choose(modes-p[j]+left,left+1);
        minimum=p[j];
    }
    return rank;
}
struct Basis {
    int modes=0,K=0;
    std::vector<std::uint64_t> codes;
    std::vector<std::uint8_t> degree;
    std::vector<std::int32_t> fibers; // [remaining_index][n_first], -1 for padded zero
};
struct Bases {
    std::vector<Basis> level;
    explicit Bases(const Options& o) {
        level.resize(o.modes+1);
        for(int r=0;r<=o.modes;++r) {
            auto& b=level[r]; b.modes=r; b.K=o.cutoff;
            const auto size=dimension(r,o.cutoff);
            b.codes.reserve(size); b.degree.reserve(size);
            b.codes.push_back(0); b.degree.push_back(0);
            if(r) for(int N=1;N<=o.cutoff;++N) {
                std::vector<int> photons(N);
                std::function<void(int,int)> visit=[&](int pos,int lower) {
                    if(pos==N) { b.codes.push_back(encode(photons)); b.degree.push_back(std::uint8_t(N)); return; }
                    for(int j=lower;j<r;++j) { photons[pos]=j; visit(pos+1,j); }
                };
                visit(0,0);
            }
            if(b.codes.size()!=size) throw std::logic_error("basis count disagrees with combinatorial dimension");
            for(size_t j=0;j<b.codes.size();++j)
                if(rank_code(b.codes[j],r,o.cutoff)!=j) throw std::logic_error("basis ranking mismatch");
            if(r) {
                const auto& rem=level[r-1];
                b.fibers.assign(rem.codes.size()*size_t(o.cutoff+1),-1);
                for(size_t h=0;h<rem.codes.size();++h) {
                    auto rest=decode(rem.codes[h]);
                    for(int& p:rest) ++p;
                    for(int n=0;n<=o.cutoff-int(rem.degree[h]);++n) {
                        std::vector<int> p(n,0); p.insert(p.end(),rest.begin(),rest.end());
                        const auto at=rank_code(encode(p),r,o.cutoff);
                        if(at>=b.codes.size()) throw std::logic_error("fiber index out of range");
                        b.fibers[h*size_t(o.cutoff+1)+n]=std::int32_t(at);
                    }
                }
            }
        }
    }
};

struct Diagnostics {
    std::uint64_t coefficient_evaluations=0,measurements=0,contracted_amplitudes=0;
    std::uint64_t poisson_tail_terms=0,normal_variates=0,gaussian_uniforms=0,quantum_uniforms=0;
    std::uint64_t displacement_evaluations=0,enumeration_nodes=0,vacuum_subsets=0;
    std::uint64_t completed_samples=0,numeric_failures=0;
    Real maximum_state_norm_residual=0,maximum_channel_norm_residual=0,maximum_poisson_norm_residual=0;
    Real maximum_poisson_tail_remainder=0,coefficient_error_l2_indicator=0,retained_weight_residual=0;
};
struct Model {
    int m,K;
    std::vector<Complex> U,B,initial;
    Real r=0,eta=0,a=1,b=1,s=0,tanh_s=0,lambda=0,delta=0;
    Real retained_weight=1,analytic_retained_weight=1,tail=0,unitarity_residual=0;
    explicit Model(const Options& o,const Bases& bases,Diagnostics& diag,std::vector<Complex> unitary)
        :m(o.modes),K(o.cutoff),U(std::move(unitary)),r(o.squeezing),eta(o.loss) {
        for(int i=0;i<m;++i) for(int j=0;j<m;++j) {
            CSum dot;
            for(int l=0;l<m;++l) dot.add(product(std::conj(U[l*m+i]),U[l*m+j]));
            unitarity_residual=std::max(unitarity_residual,std::abs(dot.value()-Complex(i==j ? 1:0,0)));
        }
        if(unitarity_residual>1e-10L) throw std::invalid_argument("input matrix is not unitary within 1e-10");
        const Real one_minus_a=eta*(-std::expm1(-2*r));
        a=(eta==0 || r==0) ? 1:(1-eta)+eta*std::exp(-2*r);
        b=(eta==0 || r==0) ? 1:(1-eta)+eta*std::exp(2*r);
        if(!(a>0) || !std::isfinite(a) || !std::isfinite(b)) throw NumericError("Gaussian core parameters overflowed or underflowed");
        s=one_minus_a<0.5L ? -0.5L*std::log1p(-one_minus_a) : -0.5L*std::log(a);
        tanh_s=one_minus_a/(1+a);
        if(tanh_s<0 || !(tanh_s<1) || (r>0 && eta>0 && tanh_s==0))
            throw NumericError("nonzero finite core squeezing unresolved");
        lambda=positive_product(tanh_s,tanh_s,"core lambda underflowed");
        if(r!=0 && eta!=0 && eta!=1) {
            const Real sh=std::sinh(r);
            delta=positive_product(positive_product(4*eta*(1-eta),sh,"displacement variance underflowed"),sh,"displacement variance overflowed")/a;
            if(!(delta>0) || !std::isfinite(delta)) throw NumericError("nonzero displacement variance unresolved");
        }
        B.resize(size_t(m)*m);
        std::vector<Real> Berror(B.size());
        for(int j=0;j<m;++j) for(int l=0;l<m;++l) {
            CSum sum; Sum absolute;
            for(int h=0;h<m;++h) { Complex z=product(U[j*m+h],U[l*m+h]); sum.add(z); absolute.add(std::abs(z)); }
            B[j*m+l]=-tanh_s*sum.value();
            Berror[j*m+l]=32*EPS*tanh_s*(absolute.value()+std::abs(sum.value()));
        }
        const auto& basis=bases.level[m];
        initial.assign(basis.codes.size(),Complex{});
        std::vector<Real> errors(initial.size());
        const Real log_vac=Real(m)/4*std::log1p(-lambda);
        initial[0]=Complex(positive_exp(log_vac,"vacuum core amplitude underflowed"),0);
        errors[0]=16*EPS*(1+std::abs(log_vac))*std::abs(initial[0]);
        for(size_t at=1;at<initial.size();++at) {
            if(basis.degree[at]&1) continue; // exact even TOTAL parity only in initial core
            auto photons=decode(basis.codes[at]);
            const int j=photons.front();
            const int nj=int(std::count(photons.begin(),photons.end(),j));
            photons.erase(photons.begin());
            CSum sum; Sum error,absolute;
            for(size_t lpos=0;lpos<photons.size();) {
                const int l=photons[lpos];
                size_t end=lpos+1; while(end<photons.size() && photons[end]==l) ++end;
                const Real factor=std::sqrt(Real(end-lpos));
                auto predecessor=photons; predecessor.erase(predecessor.begin()+std::ptrdiff_t(lpos));
                const auto prev=rank_code(encode(predecessor),m,K);
                const Complex term=product(B[j*m+l],initial[prev])*factor;
                sum.add(term); absolute.add(std::abs(term));
                error.add(factor*(std::abs(B[j*m+l])*errors[prev]+Berror[j*m+l]*std::abs(initial[prev])));
                lpos=end;
            }
            const Real divisor=std::sqrt(Real(nj));
            initial[at]=sum.value()/divisor;
            finite(initial[at],"nonfinite Fock recurrence amplitude");
            errors[at]=(error.value()+32*EPS*(absolute.value()+std::abs(sum.value())))/divisor;
            ++diag.coefficient_evaluations;
        }
        retained_weight=weight(initial);
        if(!(retained_weight>0) || !std::isfinite(retained_weight)) throw NumericError("truncated core has unresolved norm");
        Sum e2; for(Real e:errors) e2.add(e*e);
        diag.coefficient_error_l2_indicator=std::sqrt(e2.value()/retained_weight);
        if(!std::isfinite(diag.coefficient_error_l2_indicator) || diag.coefficient_error_l2_indicator>o.guard)
            throw NumericError("Fock recurrence error indicator exceeds guard");
        // NB(m/2,1-lambda), counting pairs. Compute the small tail as a
        // positive series; subtraction is used only when tail >=1/2.
        const Real shape=Real(m)/2;
        Real term=positive_exp(shape*std::log1p(-lambda),"core retained mass underflowed");
        Sum cdf; cdf.add(term); const int J=K/2;
        for(int j=1;j<=J;++j) { term=positive_product(term,lambda*(shape+j-1)/j,"NB CDF term underflowed"); cdf.add(term); }
        analytic_retained_weight=cdf.value();
        if(lambda==0) tail=0;
        else if(analytic_retained_weight<=0.5L) tail=1-analytic_retained_weight;
        else {
            term=positive_product(term,lambda*(shape+J)/(J+1),"positive core tail underflowed");
            Sum sf; sf.add(term); int j=J+1;
            for(;;) {
                const Real ratio=lambda*(shape+j)/(j+1);
                const Real upper_ratio=std::max(lambda,ratio);
                if(upper_ratio<1 && term*upper_ratio/(1-upper_ratio)<=4*EPS*sf.value()) break;
                if(++j>100000) throw NumericError("core-tail iteration budget exceeded");
                term=positive_product(term,ratio,"positive core-tail term underflowed");
                sf.add(term);
            }
            tail=sf.value();
        }
        if(!std::isfinite(tail) || tail<0 || tail>=1 || (lambda>0 && !(tail>0)))
            throw NumericError("invalid positive core tail");
        diag.retained_weight_residual=std::abs(retained_weight-analytic_retained_weight);
        if(diag.retained_weight_residual>o.guard*analytic_retained_weight
            || std::abs(analytic_retained_weight+tail-1)>o.guard)
            throw NumericError("Fock norm and analytic retained weight disagree");
        const Real norm=std::sqrt(retained_weight);
        for(Complex& z:initial) z/=norm;
    }
};

struct POVM {
    std::vector<Complex> w;
    Real x=0,t=1,q=0,t_raw=1,q_raw=0,norm_residual=0,tail_remainder=0;
};
POVM local_povm(Complex alpha,int K,Real guard,Diagnostics& diag) {
    POVM p; p.w.assign(K+1,Complex{});
    p.x=square(alpha);
    if(alpha==Complex{}) { p.w[0]=1; return p; }
    if(!(p.x>0)) throw NumericError("nonzero alpha squared magnitude unresolved");
    const Real root_x=std::sqrt(p.x);
    const Complex phase=-alpha/root_x;
    std::vector<Real> logterms(K+1,0);
    for(int n=1;n<=K;++n) logterms[n]=logterms[n-1]+std::log(p.x)-std::log(Real(n));
    const Real maxlog=*std::max_element(logterms.begin(),logterms.end());
    Sum scale_norm; Complex phase_power=1;
    for(int n=0;n<=K;++n) {
        if(n) phase_power=product(phase_power,phase);
        const Real v=positive_exp((logterms[n]-maxlog)/2,"normalized coherent power underflowed");
        p.w[n]=phase_power*v; scale_norm.add(square(p.w[n]));
    }
    const Real normalizer=std::sqrt(scale_norm.value());
    for(Complex& v:p.w) v/=normalizer;
    // Enforce only the analytic phase of w_0, not a probability clipping rule.
    if(p.w[0].imag()!=0 || !(p.w[0].real()>0)) throw NumericError("invalid vacuum component of coherent basis");
    Real term=positive_exp(-p.x,"strictly positive Poisson CDF underflowed");
    Sum cdf; cdf.add(term);
    for(int n=1;n<=K;++n) { term=positive_product(term,p.x/n,"Poisson CDF term underflowed"); cdf.add(term); }
    p.t_raw=cdf.value();
    term=positive_product(term,p.x/Real(K+1),"strictly positive Poisson tail underflowed");
    Sum tail; tail.add(term); int n=K+1; ++diag.poisson_tail_terms;
    for(;;) {
        const Real ratio=p.x/Real(n+1);
        if(ratio<1) {
            const Real remainder=term*ratio/(1-ratio);
            if(remainder<=4*EPS*tail.value()) { p.tail_remainder=remainder; break; }
        }
        if(++n>100000) throw NumericError("Poisson tail iteration budget exceeded");
        term=positive_product(term,ratio,"Poisson tail term underflowed"); tail.add(term);
        ++diag.poisson_tail_terms;
    }
    p.q_raw=tail.value();
    if(!(p.t_raw>0 && p.q_raw>0) || !std::isfinite(p.t_raw) || !std::isfinite(p.q_raw))
        throw NumericError("finite nonzero alpha requires strictly positive t and q");
    const Real total=p.t_raw+p.q_raw;
    p.norm_residual=std::abs(total-1);
    diag.maximum_poisson_norm_residual=std::max(diag.maximum_poisson_norm_residual,p.norm_residual);
    diag.maximum_poisson_tail_remainder=std::max(diag.maximum_poisson_tail_remainder,p.tail_remainder);
    if(p.norm_residual>guard) throw NumericError("Poisson CDF and positive-tail norm guard failed");
    // Explicit normalization of two independently summed positive masses.
    // Keep both tails; never form q by subtracting rounded t from one.
    p.t=p.t_raw/total; p.q=p.q_raw/total;
    if(!(p.t>0 && p.q>0)) throw NumericError("positive POVM split lost during normalization");
    return p;
}
struct Channels {
    POVM povm;
    size_t remaining_size=0;
    std::vector<Complex> vectors; // channel-major, NOT yet normalized
    std::vector<Real> raw_weights,probabilities;
    Real total=0,input_norm=0;
};
Channels measure(const std::vector<Complex>& state,int modes,Complex alpha,
                 const Bases& bases,const Options& o,Diagnostics& diag) {
    if(modes<1 || state.size()!=bases.level[modes].codes.size())
        throw std::logic_error("measurement state shape mismatch");
    Channels c; c.povm=local_povm(alpha,o.cutoff,o.guard,diag);
    c.input_norm=weight(state);
    const Real residual=std::abs(c.input_norm-1);
    diag.maximum_state_norm_residual=std::max(diag.maximum_state_norm_residual,residual);
    if(residual>o.guard) throw NumericError("incoming pure trajectory is not normalized");
    c.remaining_size=bases.level[modes-1].codes.size();
    c.vectors.assign(size_t(o.cutoff+1)*c.remaining_size,Complex{});
    c.raw_weights.assign(o.cutoff+1,0); c.probabilities.assign(o.cutoff+1,0);
    std::vector<Sum> sums(o.cutoff+1);
    const auto& fiber=bases.level[modes].fibers;
    const Real denominator=1+c.povm.w[0].real(); // >=1: stable at alpha=0
    for(size_t h=0;h<c.remaining_size;++h) {
        CSum projection;
        for(int n=0;n<=o.cutoff;++n) {
            const auto at=fiber[h*size_t(o.cutoff+1)+n];
            if(at>=0) projection.add(product(std::conj(c.povm.w[n]),state[size_t(at)]));
        }
        const auto at0=fiber[h*size_t(o.cutoff+1)];
        if(at0<0) throw std::logic_error("missing n=0 fiber entry");
        const Complex g0=projection.value();
        const Complex common=(state[size_t(at0)]+g0)/denominator;
        for(int channel=0;channel<=o.cutoff;++channel) {
            Complex g=g0;
            if(channel) {
                g=product(c.povm.w[channel],common);
                const auto at=fiber[h*size_t(o.cutoff+1)+channel];
                if(at>=0) g-=state[size_t(at)];
            }
            finite(g,"nonfinite Householder contraction");
            c.vectors[size_t(channel)*c.remaining_size+h]=g;
            sums[channel].add(square(g));
            ++diag.contracted_amplitudes;
        }
        // All K+1 projective labels are kept even if channel+degree(rest)>K.
        // A projective channel is not a detected photon-number constraint.
    }
    Sum total;
    for(int channel=0;channel<=o.cutoff;++channel) {
        c.raw_weights[channel]=sums[channel].value(); total.add(c.raw_weights[channel]);
    }
    c.total=total.value();
    if(!(c.total>0)) throw NumericError("all projective channel norms vanished");
    const Real defect=std::abs(c.total-c.input_norm);
    diag.maximum_channel_norm_residual=std::max(diag.maximum_channel_norm_residual,defect);
    if(defect>o.guard) throw NumericError("Householder channel completeness guard failed");
    for(int j=0;j<=o.cutoff;++j) {
        c.probabilities[j]=c.raw_weights[j]/c.total;
        if(c.raw_weights[j]>0 && !(c.probabilities[j]>0))
            throw NumericError("positive channel mass lost during normalization");
    }
    ++diag.measurements;
    return c;
}
std::vector<Complex> normalized_channel(const Channels& c,int channel) {
    if(!(c.raw_weights[channel]>0)) throw std::logic_error("cannot normalize zero channel");
    const Real norm=std::sqrt(c.raw_weights[channel]);
    std::vector<Complex> result(c.remaining_size);
    for(size_t h=0;h<c.remaining_size;++h) result[h]=c.vectors[size_t(channel)*c.remaining_size+h]/norm;
    return result;
}

std::vector<Complex> generic_state(const Options& o,const Bases& bases) {
    auto values=read_numbers(o.state,false);
    const size_t columns=size_t(o.modes)+2;
    if(values.empty() || values.size()%columns) throw std::invalid_argument("generic state rows require m occupations and complex amplitude");
    std::vector<Complex> state(bases.level[o.modes].codes.size(),Complex{});
    std::vector<bool> seen(state.size(),false);
    for(size_t row=0;row<values.size();row+=columns) {
        std::vector<int> p;
        for(int j=0;j<o.modes;++j) {
            const Real n=values[row+j];
            if(n<0 || n>o.cutoff || n!=std::floor(n)) throw std::invalid_argument("invalid Fock occupation");
            for(int h=0;h<int(n);++h) p.push_back(j);
        }
        if(p.size()>size_t(o.cutoff)) throw std::invalid_argument("state exceeds total photon cutoff");
        const auto at=rank_code(encode(p),o.modes,o.cutoff);
        if(seen[at]) throw std::invalid_argument("duplicate Fock occupation");
        seen[at]=true;
        state[at]={values[row+o.modes],values[row+o.modes+1]};
    }
    const Real norm=weight(state);
    if(!(norm>0) || std::abs(norm-1)>o.guard) throw std::invalid_argument("generic input state must be normalized within numeric guard");
    return state;
}

struct Random {
    std::mt19937_64 engine;
    const std::vector<Real>& injected;
    const std::vector<Real>& normals;
    Diagnostics& diag;
    explicit Random(std::uint64_t seed,const std::vector<Real>& u,const std::vector<Real>& z,Diagnostics& d)
        :engine(seed),injected(u),normals(z),diag(d) {}
    Real quantum(size_t at) {
        ++diag.quantum_uniforms;
        if(!injected.empty()) return injected.at(at);
        return Real(engine()>>11)*0x1p-53L;
    }
    Real gaussian_uniform() {
        ++diag.gaussian_uniforms;
        // Exact half-grid shift in long double (at least 64 significand bits).
        return (Real(engine()>>11)+0.5L)*0x1p-53L;
    }
    std::vector<Real> normal_vector(int m,std::uint64_t sample) {
        std::vector<Real> z(m);
        if(!normals.empty()) {
            for(int j=0;j<m;++j) z[j]=normals.at(size_t(sample)*m+j);
        } else {
            const Real pi=std::acos(-1.0L);
            for(int j=0;j<m;j+=2) {
                const Real u=gaussian_uniform(),v=gaussian_uniform();
                const Real radius=std::sqrt(-2*std::log(u)),angle=2*pi*v;
                z[j]=radius*std::cos(angle);
                if(j+1<m) z[j+1]=radius*std::sin(angle);
            }
        }
        diag.normal_variates+=m;
        return z;
    }
};
std::vector<Complex> displace(const Model& model,Random& random,std::uint64_t sample,Diagnostics& diag) {
    std::vector<Complex> alpha(model.m,Complex{});
    ++diag.displacement_evaluations;
    if(model.delta==0) return alpha; // analytic eta/r boundary
    const auto z=random.normal_vector(model.m,sample);
    const Complex scale(0,std::sqrt(model.delta)/2);
    for(int j=0;j<model.m;++j) {
        CSum sum;
        for(int l=0;l<model.m;++l) sum.add(model.U[j*model.m+l]*z[l]);
        alpha[j]=product(scale,sum.value());
        finite(alpha[j],"nonfinite correlated Gaussian displacement");
    }
    return alpha;
}
int categorical(const std::vector<Real>& probabilities,Real u) {
    // Small-to-large cumulative order retains tiny positive channels better
    // than adding them after a near-one channel. Ties use the channel index.
    std::vector<int> order(probabilities.size());
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){ return probabilities[a]<probabilities[b]; });
    Sum total; for(Real p:probabilities) total.add(p);
    const Real threshold=u*total.value();
    Sum prefix; int last=-1;
    for(int j:order) if(probabilities[j]>0) {
        last=j; prefix.add(probabilities[j]);
        if(threshold<prefix.value()) return j;
    }
    if(last<0) throw std::logic_error("categorical distribution has no positive channel");
    if(u>=0 && u<1) return last;
    throw std::logic_error("uniform outside [0,1)");
}
int split_bit(const POVM& p,Real u) {
    if(p.q==0) return 0; // only alpha=0 has this analytic boundary
    const Real total=p.t+p.q;
    // Compare to the small side, never to a rounded 1-q.
    if(p.q<=p.t) return u<p.q/total ? 1:0;
    return u<p.t/total ? 0:1;
}
struct Trace {
    std::uint64_t sample=0,prefix=0;
    int mode=0,channel=0,bit=0;
    Complex alpha;
    Real t=0,q=0,u_channel=0,u_split=0;
    std::vector<Real> probabilities;
};
struct Failure {
    int mode=0;
    std::uint64_t prefix=0;
    Real mass=0;
    std::string error;
};
struct Enumeration {
    std::vector<Real> distribution,vacuum;
    Real failure_mass=0,absolute_indicator=0;
    std::vector<Failure> failures;
    bool complete=false;
};
Real branch_mass(Real parent,Real conditional) {
    if(conditional==0) return 0;
    const Real mass=parent*conditional;
    if(!(mass>0) || !std::isfinite(mass) || std::fpclassify(mass)==FP_SUBNORMAL)
        throw EnumerationError("positive enumeration branch lost mass; traversal aborted");
    return mass;
}
Enumeration refined_law(const std::vector<Complex>& initial,const std::vector<Complex>& alpha,
                        const Bases& bases,const Options& o,Diagnostics& diag) {
    Enumeration result;
    std::vector<Sum> leaves(size_t(1)<<o.modes); Sum failed;
    std::function<void(int,const std::vector<Complex>&,Real,std::uint64_t)> visit;
    visit=[&](int mode,const std::vector<Complex>& state,Real mass,std::uint64_t prefix) {
        if(++diag.enumeration_nodes>o.max_nodes) throw EnumerationError("refined enumeration node cap exceeded");
        if(mode==o.modes) { leaves[size_t(prefix)].add(mass); return; }
        Channels c;
        try { c=measure(state,o.modes-mode,alpha[mode],bases,o,diag); }
        catch(const NumericError& e) {
            failed.add(mass); result.failures.push_back({mode,prefix,mass,e.what()});
            ++diag.numeric_failures; return;
        }
        for(int channel=0;channel<=o.cutoff;++channel) {
            if(c.probabilities[channel]==0) continue;
            auto next=normalized_channel(c,channel);
            const Real channel_mass=branch_mass(mass,c.probabilities[channel]);
            if(channel==0) {
                visit(mode+1,next,branch_mass(channel_mass,c.povm.t),prefix);
                if(c.povm.q>0) visit(mode+1,next,branch_mass(channel_mass,c.povm.q),prefix|(std::uint64_t(1)<<mode));
            } else visit(mode+1,next,channel_mass,prefix|(std::uint64_t(1)<<mode));
        }
    };
    visit(0,initial,1,0);
    result.distribution.resize(leaves.size()); Sum total;
    for(size_t j=0;j<leaves.size();++j) { result.distribution[j]=leaves[j].value(); total.add(result.distribution[j]); }
    result.failure_mass=failed.value(); total.add(result.failure_mass);
    if(std::abs(total.value()-1)>o.guard) throw EnumerationError("refined complete law lost normalization");
    result.complete=true;
    return result;
}
Enumeration vacuum_law(const Model& model,const std::vector<Complex>& alpha,
                       const Bases& bases,const Options& o,Diagnostics& diag) {
    Enumeration result;
    const size_t count=size_t(1)<<o.modes;
    result.vacuum.assign(count,0); result.vacuum[0]=1;
    std::vector<POVM> local;
    for(Complex a:alpha) local.push_back(local_povm(a,o.cutoff,o.guard,diag));
    std::vector<Real> error(count,0);
    const auto& initial_basis=bases.level[o.modes];
    for(size_t zero=1;zero<count;++zero) {
        std::array<int,MAX_MODES> remaining_map{};
        int rem=0; for(int j=0;j<o.modes;++j) remaining_map[j]=(zero>>j)&1 ? -1:rem++;
        std::vector<CSum> projected(bases.level[rem].codes.size());
        for(size_t at=0;at<model.initial.size();++at) if(model.initial[at]!=Complex{}) {
            const auto photons=decode(initial_basis.codes[at]);
            std::array<int,MAX_MODES> occupations{}; std::vector<int> rest;
            for(int j:photons) {
                ++occupations[j];
                if(remaining_map[j]>=0) rest.push_back(remaining_map[j]);
            }
            Complex value=model.initial[at];
            for(int j=0;j<o.modes;++j) if((zero>>j)&1)
                value=product(value,std::conj(local[j].w[occupations[j]])*std::sqrt(local[j].t));
            projected[rank_code(encode(rest),rem,o.cutoff)].add(value);
        }
        Sum value; for(const auto& z:projected) value.add(square(z.value()));
        result.vacuum[zero]=value.value();
        if(result.vacuum[zero]<0 || result.vacuum[zero]>1+o.guard)
            throw NumericError("invalid finite-state vacuum marginal");
        error[zero]=64*EPS*(model.initial.size()+o.cutoff+1);
        ++diag.vacuum_subsets;
    }
    auto exact_zero=result.vacuum;
    for(int bit=0;bit<o.modes;++bit) for(size_t mask=0;mask<count;++mask) if(!((mask>>bit)&1)) {
        const size_t other=mask|(size_t(1)<<bit);
        error[mask]+=error[other]+8*EPS*(std::abs(exact_zero[mask])+std::abs(exact_zero[other]));
        exact_zero[mask]-=exact_zero[other];
    }
    result.distribution.resize(count); Sum total,indicator;
    for(size_t zero=0;zero<count;++zero) {
        const Real p=exact_zero[zero];
        if(!std::isfinite(p) || p<0) throw NumericError("negative/nonfinite Mobius probability; no clipping permitted");
        result.distribution[(count-1)^zero]=p; total.add(p); indicator.add(error[zero]);
    }
    result.absolute_indicator=indicator.value();
    if(std::abs(total.value()-1)>o.guard || result.absolute_indicator>o.guard)
        throw NumericError("vacuum/Mobius full-law numerical guard exceeded");
    result.complete=true;
    return result;
}

void json_quote(const std::string& text) {
    std::cout << '"';
    for(unsigned char c:text) {
        if(c=='"' || c=='\\') std::cout << '\\' << char(c);
        else if(c=='\n') std::cout << "\\n";
        else if(c=='\r') std::cout << "\\r";
        else if(c=='\t') std::cout << "\\t";
        else if(c<32) std::cout << '?';
        else std::cout << char(c);
    }
    std::cout << '"';
}
void real_json(Real value) {
    if(!std::isfinite(value)) { std::cout << "null"; return; }
    // Strings preserve positive sub-binary64 masses and long-double phases.
    std::ostringstream stream; stream << std::setprecision(std::numeric_limits<Real>::max_digits10) << value;
    json_quote(stream.str());
}
void real_array(const std::vector<Real>& values) {
    std::cout << '[';
    for(size_t j=0;j<values.size();++j) { if(j) std::cout << ','; real_json(values[j]); }
    std::cout << ']';
}
void complex_array(const std::vector<Complex>& values,bool imag) {
    std::cout << '[';
    for(size_t j=0;j<values.size();++j) { if(j) std::cout << ','; real_json(imag ? values[j].imag():values[j].real()); }
    std::cout << ']';
}
void occupations_json(const Basis& basis) {
    std::cout << '[';
    for(size_t at=0;at<basis.codes.size();++at) {
        if(at) std::cout << ',';
        std::array<int,MAX_MODES> n{}; for(int j:decode(basis.codes[at])) ++n[j];
        std::cout << '['; for(int j=0;j<basis.modes;++j) { if(j)std::cout << ',';std::cout<<n[j]; } std::cout << ']';
    }
    std::cout << ']';
}
void state_json(const Basis& b,const std::vector<Complex>& amplitudes) {
    std::cout << "{\"occupations\":"; occupations_json(b);
    std::cout << ",\"amplitudes_real\":"; complex_array(amplitudes,false);
    std::cout << ",\"amplitudes_imag\":"; complex_array(amplitudes,true);
    std::cout << '}';
}
void channels_json(const Channels& c,const Basis& remaining) {
    std::cout << "{\"projective_probabilities\":"; real_array(c.probabilities);
    std::cout << ",\"projective_raw_weights\":"; real_array(c.raw_weights);
    std::cout << ",\"projective_total\":"; real_json(c.total);
    std::cout << ",\"input_norm\":"; real_json(c.input_norm);
    std::cout << ",\"t\":"; real_json(c.povm.t);
    std::cout << ",\"q\":"; real_json(c.povm.q);
    std::cout << ",\"t_raw\":"; real_json(c.povm.t_raw);
    std::cout << ",\"q_raw\":"; real_json(c.povm.q_raw);
    std::cout << ",\"x\":"; real_json(c.povm.x);
    std::cout << ",\"poisson_norm_residual\":"; real_json(c.povm.norm_residual);
    std::cout << ",\"poisson_tail_remainder\":"; real_json(c.povm.tail_remainder);
    std::cout << ",\"w_real\":"; complex_array(c.povm.w,false);
    std::cout << ",\"w_imag\":"; complex_array(c.povm.w,true);
    std::cout << ",\"remaining_occupations\":"; occupations_json(remaining);
    std::cout << ",\"remaining_states\":[";
    for(size_t channel=0;channel<c.probabilities.size();++channel) {
        if(channel) std::cout << ',';
        if(c.raw_weights[channel]==0) { std::cout << "null"; continue; }
        const auto state=normalized_channel(c,int(channel));
        std::cout << "{\"amplitudes_real\":"; complex_array(state,false);
        std::cout << ",\"amplitudes_imag\":"; complex_array(state,true); std::cout << '}';
    }
    std::cout << "],\"flattened_probabilities\":";
    std::vector<Real> probabilities{c.probabilities[0]*c.povm.t,c.probabilities[0]*c.povm.q};
    probabilities.insert(probabilities.end(),c.probabilities.begin()+1,c.probabilities.end());
    real_array(probabilities);
    std::cout << ",\"flattened_labels\":\"(channel0,zero),(channel0,click),(channel1,click),...\"}";
}
void diagnostics_json(const Diagnostics& d) {
    std::cout << "{\"coefficient_evaluations\":" << d.coefficient_evaluations
        << ",\"measurements\":" << d.measurements << ",\"contracted_amplitudes\":" << d.contracted_amplitudes
        << ",\"poisson_tail_terms\":" << d.poisson_tail_terms << ",\"normal_variates\":" << d.normal_variates
        << ",\"gaussian_uniforms\":" << d.gaussian_uniforms << ",\"quantum_uniforms\":" << d.quantum_uniforms
        << ",\"displacement_evaluations\":" << d.displacement_evaluations
        << ",\"enumeration_nodes\":" << d.enumeration_nodes << ",\"vacuum_subsets\":" << d.vacuum_subsets
        << ",\"completed_samples\":" << d.completed_samples << ",\"numeric_failures\":" << d.numeric_failures;
    std::cout << ",\"maximum_state_norm_residual\":"; real_json(d.maximum_state_norm_residual);
    std::cout << ",\"maximum_channel_norm_residual\":"; real_json(d.maximum_channel_norm_residual);
    std::cout << ",\"maximum_poisson_norm_residual\":"; real_json(d.maximum_poisson_norm_residual);
    std::cout << ",\"maximum_poisson_tail_remainder\":"; real_json(d.maximum_poisson_tail_remainder);
    std::cout << ",\"coefficient_error_l2_indicator\":"; real_json(d.coefficient_error_l2_indicator);
    std::cout << ",\"retained_weight_residual\":"; real_json(d.retained_weight_residual);
    std::cout << '}';
}

int main(int argc,char** argv) {
    Options o;
    try { o=options(argc,argv); }
    catch(const std::exception& e) {
        std::cout << "{\"schema_version\":1,\"status\":\"error\",\"numerical_failure\":false,\"error\":";
        json_quote(e.what()); std::cout << "}\n"; return 1;
    }
    Diagnostics diag; Resources res;
    std::unique_ptr<Bases> bases; std::unique_ptr<Model> model;
    std::vector<Complex> U,alpha,input_state;
    std::vector<Real> uniforms,normals;
    std::vector<std::uint64_t> packed;
    std::vector<Trace> traces;
    std::unique_ptr<Channels> inspection;
    Enumeration law;
    std::string phase="input",error;
    bool numeric_failure=false;
    std::uint64_t failed_sample=0,partial_prefix=0;
    int failed_position=0;
    double input_seconds=0,setup_seconds=0,sampling_seconds=0,reference_seconds=0;
    auto started=Clock::now();
    try {
        if(std::numeric_limits<Real>::digits<64) throw std::runtime_error("requires at least 64 long-double significand bits");
        res=resources(o); // all dimensions checked before basis/state allocation
        if(o.action!="channels") U=read_complex(o.unitary,size_t(o.modes)*o.modes);
        if(!o.alpha.empty()) alpha=read_complex(o.alpha,o.action=="channels" ? 1:size_t(o.modes));
        if(!o.uniforms.empty()) {
            uniforms=read_numbers(o.uniforms,true);
            if(uniforms.size()!=o.samples*2ULL*o.modes) throw std::invalid_argument("uniforms must have shape samples x (2*modes)");
            for(Real u:uniforms) if(!(u>=0 && u<1)) throw std::invalid_argument("uniform outside [0,1)");
        }
        if(!o.normals.empty()) {
            normals=read_numbers(o.normals,true);
            if(normals.size()!=o.samples*std::uint64_t(o.modes)) throw std::invalid_argument("normal draws must have shape samples x modes");
        }
        input_seconds=seconds(started); phase="setup"; started=Clock::now();
        bases=std::make_unique<Bases>(o);
        if(o.action=="channels") input_state=generic_state(o,*bases);
        else model=std::make_unique<Model>(o,*bases,diag,std::move(U));
        setup_seconds=seconds(started);
        if(o.action=="sample") {
            phase="sampling"; started=Clock::now();
            packed.reserve(size_t(o.samples));
            if(o.trace) traces.reserve(size_t(o.samples)*o.modes);
            Random random(o.seed,uniforms,normals,diag);
            for(std::uint64_t sample=0;sample<o.samples;++sample) {
                failed_sample=sample; failed_position=0; partial_prefix=0;
                auto displacement=alpha.empty() ? displace(*model,random,sample,diag):alpha;
                auto state=model->initial; // coherent copy; no initial sector sampling
                for(int mode=0;mode<o.modes;++mode) {
                    failed_position=mode;
                    auto c=measure(state,o.modes-mode,displacement[mode],*bases,o,diag);
                    const size_t uat=(size_t(sample)*o.modes+mode)*2;
                    const Real u=random.quantum(uat),v=random.quantum(uat+1);
                    const int channel=categorical(c.probabilities,u);
                    const int bit=channel==0 ? split_bit(c.povm,v):1;
                    if(o.trace) traces.push_back({sample,partial_prefix,mode,channel,bit,displacement[mode],
                        c.povm.t,c.povm.q,u,v,c.probabilities});
                    state=normalized_channel(c,channel);
                    if(bit) partial_prefix|=std::uint64_t(1)<<mode;
                    failed_position=mode+1;
                }
                packed.push_back(partial_prefix); ++diag.completed_samples;
                partial_prefix=0; failed_position=0;
            }
            sampling_seconds=seconds(started);
        } else if(o.action=="channels") {
            phase="channels"; started=Clock::now();
            inspection=std::make_unique<Channels>(measure(input_state,o.modes,alpha[0],*bases,o,diag));
            reference_seconds=seconds(started);
        } else if(o.action=="refined-law" || o.action=="vacuum-law") {
            phase="enumeration"; started=Clock::now();
            law=o.action=="refined-law" ? refined_law(model->initial,alpha,*bases,o,diag)
                                        :vacuum_law(*model,alpha,*bases,o,diag);
            reference_seconds=seconds(started);
        }
        phase="complete";
    } catch(const NumericError& e) {
        error=e.what(); numeric_failure=true; ++diag.numeric_failures;
    } catch(const std::exception& e) { error=e.what(); }
    if(!error.empty()) {
        if(phase=="input") input_seconds=seconds(started);
        else if(phase=="setup") setup_seconds=seconds(started);
        else if(phase=="sampling") sampling_seconds=seconds(started);
        else reference_seconds=seconds(started);
    }
    std::cout << std::setprecision(17);
    std::cout << "{\"schema_version\":1,\"implementation\":\"coherent_dense_core_v1\",\"status\":";
    json_quote(error.empty() ? "ok":"error");
    std::cout << ",\"numerical_failure\":" << (numeric_failure ? "true":"false") << ",\"error\":";
    if(error.empty()) std::cout << "null"; else json_quote(error);
    std::cout << ",\"failed_phase\":"; if(error.empty()) std::cout << "null"; else json_quote(phase);
    std::cout << ",\"action\":"; json_quote(o.action);
    std::cout << ",\"modes\":" << o.modes << ",\"cutoff\":" << o.cutoff
        << ",\"seed\":" << o.seed << ",\"requested_samples\":" << o.samples
        << ",\"completed_samples\":" << packed.size()
        << ",\"failed_sample\":" << failed_sample << ",\"partial_prefix_mask\":" << partial_prefix
        << ",\"partial_prefix_length\":" << failed_position
        << ",\"long_double_digits\":" << std::numeric_limits<Real>::digits
        << ",\"long_double_bytes\":" << sizeof(Real) << ",\"complex_long_double_bytes\":" << sizeof(Complex);
    std::cout << ",\"numeric_guard\":"; real_json(o.guard);
    std::cout << ",\"numeric_policy\":\"compensated sums, recurrence/norm/completeness guards; no clipping; no rigorous numerical TV certificate\""
        << ",\"law_semantics\":\"coherent normalized total-N truncation, latent rank-one channels, correlated Gaussian alpha; no trajectory logQ\""
        << ",\"failure_record_semantics\":\"enumeration failure records are latent-path mass contributions; other latent paths with the same binary prefix may succeed\""
        << ",\"quantum_uniform_policy\":\"two top-53-bit uniforms per accepted mode; second unused for nonzero projective channel\""
        << ",\"gaussian_rng_policy\":\"Box-Muller with open half-shifted 53-bit uniforms; finite PRNG approximation\"";
    std::cout << ",\"resources\":{\"full_basis\":" << res.full << ",\"even_initial_basis\":" << res.even
        << ",\"all_stage_basis\":" << res.stages << ",\"fiber_index_entries\":" << res.fibers
        << ",\"planned_major_payload_bytes\":" << res.planned_payload
        << ",\"max_basis\":" << o.max_basis << ",\"max_stage_basis\":" << o.max_stage_basis
        << ",\"max_payload_bytes\":" << o.max_payload
        << ",\"memory_scope\":\"planned major numerical/index/sample containers; excludes allocator overhead, process RSS and JSON transport\"}";
    std::cout << ",\"timings_seconds\":{\"input\":" << input_seconds << ",\"setup\":" << setup_seconds
        << ",\"sampling\":" << sampling_seconds << ",\"cold_batch\":" << setup_seconds+sampling_seconds
        << ",\"reference\":" << reference_seconds << '}';
    std::cout << ",\"parameters\":";
    if(model) {
        std::cout << "{\"squeezing\":"; real_json(model->r); std::cout << ",\"loss\":"; real_json(model->eta);
        std::cout << ",\"a\":"; real_json(model->a); std::cout << ",\"b\":"; real_json(model->b);
        std::cout << ",\"core_squeezing\":"; real_json(model->s);
        std::cout << ",\"tanh_core_squeezing\":"; real_json(model->tanh_s);
        std::cout << ",\"lambda\":"; real_json(model->lambda);
        std::cout << ",\"displacement_variance\":"; real_json(model->delta);
        std::cout << ",\"unitarity_residual\":"; real_json(model->unitarity_residual); std::cout << '}';
    } else std::cout << "null";
    std::cout << ",\"retained_weight\":"; if(model)real_json(model->retained_weight);else std::cout<<"null";
    std::cout << ",\"analytic_retained_weight\":"; if(model)real_json(model->analytic_retained_weight);else std::cout<<"null";
    std::cout << ",\"tail_probability\":"; if(model)real_json(model->tail);else std::cout<<"null";
    std::cout << ",\"trace_distance_bound\":"; if(model)real_json(std::sqrt(model->tail));else std::cout<<"null";
    std::cout << ",\"tail_scope\":\"sqrt(tau) is the exact-arithmetic coherent projection budget; floating state, Gaussian RNG and measurement errors are additional\"";
    std::cout << ",\"state\":";
    if(model && (o.return_state || o.action=="state")) state_json(bases->level[o.modes],model->initial);
    else std::cout << "null";
    std::cout << ",\"channels\":";
    if(inspection) channels_json(*inspection,bases->level[o.modes-1]); else std::cout<<"null";
    std::cout << ",\"raw_samples\":";
    if(o.raw) {
        std::cout << '[';
        for(size_t sample=0;sample<packed.size();++sample) {
            if(sample) std::cout << ',';
            std::cout << '[';
            for(int j=0;j<o.modes;++j) { if(j)std::cout<<',';std::cout<<((packed[sample]>>j)&1); }
            std::cout << ']';
        }
        std::cout << ']';
    } else std::cout << "null";
    std::cout << ",\"completed_packed_words\":[";
    for(size_t j=0;j<packed.size();++j) { if(j)std::cout<<',';std::cout<<packed[j]; }
    std::cout << "],\"trace\":[";
    for(size_t j=0;j<traces.size();++j) {
        const auto& tr=traces[j]; if(j)std::cout<<',';
        std::cout << "{\"sample\":"<<tr.sample<<",\"mode\":"<<tr.mode<<",\"prefix_mask\":"<<tr.prefix
            <<",\"channel\":"<<tr.channel<<",\"bit\":"<<tr.bit<<",\"alpha_real\":";real_json(tr.alpha.real());
        std::cout << ",\"alpha_imag\":";real_json(tr.alpha.imag());std::cout<<",\"t\":";real_json(tr.t);
        std::cout << ",\"q\":";real_json(tr.q);std::cout<<",\"uniform_channel\":";real_json(tr.u_channel);
        std::cout << ",\"uniform_split\":";real_json(tr.u_split);std::cout<<",\"projective_probabilities\":";
        real_array(tr.probabilities);std::cout << '}';
    }
    std::cout << "],\"enumeration_complete\":" << (law.complete ? "true":"false") << ",\"distribution\":";
    if(law.complete)real_array(law.distribution);else std::cout<<"null";
    std::cout << ",\"vacuum_marginals\":";
    if(!law.vacuum.empty())real_array(law.vacuum);else std::cout<<"null";
    std::cout << ",\"failure_mass\":";real_json(law.failure_mass);
    std::cout << ",\"vacuum_absolute_indicator\":";real_json(law.absolute_indicator);
    std::cout << ",\"enumeration_failures\":[";
    for(size_t j=0;j<law.failures.size();++j) {
        const auto& f=law.failures[j];if(j)std::cout<<',';
        std::cout << "{\"mode\":"<<f.mode<<",\"prefix_mask\":"<<f.prefix<<",\"mass\":";
        real_json(f.mass);std::cout<<",\"error\":";json_quote(f.error);std::cout<<'}';
    }
    std::cout << "],\"diagnostics\":";diagnostics_json(diag);
    std::cout << "}\n";
    return error.empty() ? 0:(numeric_failure ? 2:1);
}
