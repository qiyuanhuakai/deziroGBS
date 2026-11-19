// Direct threshold trajectories in a span of normalized Gaussian projections.
// Compression projects the purification into selected branch vectors. Every
// discarded direction is charged; failures and partial records are explicit.
#include "gaussian_click_gram94_v1.cpp"
using Mask94=std::vector<std::uint64_t>;
struct Term94{Mask94 mask;long double coefficient;};
using State94=std::vector<Term94>;
struct Vacuum94{
 int m,words;const double* q;const double* mean;std::map<Mask94,long double> cache;
 Vacuum94(int modes,const double* covariance,const double* displacement):m(modes),words((m+63)/64),q(covariance),mean(displacement){}
 long double value(const Mask94& mask){
  auto found=cache.find(mask);if(found!=cache.end())return found->second;
  if(cache.size()>=262144)throw Failure93(3,"Gaussian vacuum cache cap");
  std::vector<int> ids;for(int i=0;i<m;++i)if((mask[i/64]>>(i%64))&1)ids.push_back(i);
  int k=ids.size();for(int i=0;i<k;++i)ids.push_back(ids[i]+m);long double result=1;
  if(k){Real94 sub(2*k,2*k);RealVec94 mu(2*k);for(int i=0;i<2*k;++i){mu[i]=mean[ids[i]];for(int j=0;j<2*k;++j)sub(i,j)=q[std::size_t(ids[i])*2*m+ids[j]];}
   Eigen::LLT<Real94> chol(sub);need93(chol.info()==Eigen::Success,"Branch vacuum Cholesky failure");Real94 L=chol.matrixL();RealVec94 white=L.template triangularView<Eigen::Lower>().solve(mu);long double logp=-.5L*white.squaredNorm();for(int i=0;i<2*k;++i)logp-=std::log(L(i,i));result=std::exp(logp);
  }
  need93(result>0&&result<=1+1e-12L&&std::isfinite(result),"Branch vacuum underflow or invalid probability");cache.emplace(mask,result);return result;
 }
 long double overlap(const Mask94& a,const Mask94& b){Mask94 joined(words);for(int w=0;w<words;++w)joined[w]=a[w]|b[w];return value(joined)/std::sqrt(value(a)*value(b));}
 Real94 gram(const State94& state){int n=state.size();Real94 G(n,n);for(int i=0;i<n;++i)for(int j=0;j<=i;++j)G(i,j)=G(j,i)=i==j?1.L:overlap(state[i].mask,state[j].mask);return G;}
};
static RealVec94 coefficients94(const State94& state){RealVec94 x(state.size());for(int i=0;i<int(state.size());++i)x[i]=state[i].coefficient;return x;}
static long double quadratic94(const RealVec94& x,const Real94& G){
 long double sum=0,correction=0,absolute=0;
 for(int i=0;i<x.size();++i)for(int j=0;j<=i;++j){long double term=(i==j?1.L:2.L)*x[i]*G(i,j)*x[j],next=sum+term;correction+=std::abs(sum)>=std::abs(term)?(sum-next)+term:(term-next)+sum;sum=next;absolute+=std::abs(term);}
 sum+=correction;need93(sum>=0&&std::isfinite(sum),"Negative/nonfinite branch norm; no clipping");
 if(sum>0)need93(absolute/sum<=1e12L,"Branch norm cancellation limit");return sum;
}
static State94 merged94(const State94& terms){std::map<Mask94,long double> combined;for(const auto& term:terms)combined[term.mask]+=term.coefficient;State94 result;for(const auto& [mask,a]:combined)if(a!=0)result.push_back({mask,a});return result;}
static State94 project_zero94(const State94& state,int mode,Vacuum94& model){State94 out;for(const auto& term:state){auto mask=term.mask;mask[mode/64]|=std::uint64_t(1)<<(mode%64);out.push_back({mask,term.coefficient*std::sqrt(model.value(mask)/model.value(term.mask))});}return merged94(out);}
static void normalize94(State94& state,long double mass){need93(mass>0,"Normalize zero conditional state");for(auto& term:state)term.coefficient/=std::sqrt(mass);}
static State94 compress_branch94(const State94& state,Vacuum94& model,int limit,double budget,double* stats){
 if(int(state.size())<=limit)return state;
 auto G=model.gram(state);auto a=coefficients94(state);long double original=quadratic94(a,G);need93(original>0,"Zero branch compression input");
 int n=state.size();Real94 C=Real94::Zero(n,limit);RealVec94 overlap=G*a,t=RealVec94::Zero(limit);std::vector<int> chosen;std::vector<bool> used(n,false);
 for(int k=0;k<limit;++k){int best=-1;long double best_gain=-1,best_diagonal=0;
  for(int i=0;i<n;++i)if(!used[i]){long double diagonal=1,residual=overlap[i];for(int j=0;j<k;++j){diagonal-=C(i,j)*C(i,j);residual-=C(i,j)*t[j];}
   // Unusable near-dependent candidate columns are omitted from the chosen
   // subspace. Their contribution remains in the explicitly measured residual.
   if(diagonal<=1e-14L)continue;long double gain=residual*residual/diagonal;if(gain>best_gain){best=i;best_gain=gain;best_diagonal=diagonal;}
  }
  if(best<0)break;used[best]=true;chosen.push_back(best);long double scale=std::sqrt(best_diagonal);
  for(int i=0;i<n;++i){long double value=G(i,best);for(int j=0;j<k;++j)value-=C(i,j)*C(best,j);C(i,k)=value/scale;}
  long double value=overlap[best];for(int j=0;j<k;++j)value-=C(best,j)*t[j];t[k]=value/scale;
 }
 int rank=chosen.size();need93(rank>0,"No independent Gaussian branch pivot");RealVec94 weights=RealVec94::Zero(rank);
 for(int j=rank-1;j>=0;--j){long double value=t[j];for(int k=j+1;k<rank;++k)value-=C(chosen[k],j)*weights[k];weights[j]=value/C(chosen[j],j);}
 State94 retained;RealVec94 residual=a,embedded=RealVec94::Zero(n);for(int j=0;j<rank;++j){retained.push_back({state[chosen[j]].mask,weights[j]});residual[chosen[j]]-=weights[j];embedded[chosen[j]]=weights[j];}
 long double kept=quadratic94(embedded,G),lost=quadratic94(residual,G),cross=std::abs(residual.dot(G*embedded));need93(kept>0&&lost<=original*(1+1e-10L)&&cross<1e-10L*original,"Gaussian branch projection orthogonality failed");
 stats[11]=std::max(stats[11],double(cross/original));long double delta=std::sqrt(lost/original);need93(delta>=0&&delta<=1,"Invalid Gaussian projection charge");++stats[6];stats[5]=std::max(stats[5],double(delta));
 if(stats[4]+delta>budget)throw Failure93(2,"Gaussian branch trace-distance budget exhausted");stats[4]+=double(delta);normalize94(retained,kept);return retained;
}
struct Split94{State94 zero,one;long double p0,p1;};
static Split94 split94(const State94& state,int mode,Vacuum94& model,double* stats){
 auto zero=project_zero94(state,mode,model);State94 one=state;for(const auto& term:zero)one.push_back({term.mask,-term.coefficient});one=merged94(one);
 long double mass=quadratic94(coefficients94(state),model.gram(state)),p0=zero.empty()?0:quadratic94(coefficients94(zero),model.gram(zero)),p1=one.empty()?0:quadratic94(coefficients94(one),model.gram(one)),total=p0+p1;
 need93(total>0&&std::abs(total-mass)<5e-11L*mass,"Conditional projector mass mismatch");stats[7]=std::max(stats[7],double(std::abs(total-mass)/mass));return {std::move(zero),std::move(one),p0/total,p1/total};
}
static State94 follow94(State94 state,int mode,Vacuum94& model,int limit,double budget,double* stats){
 auto mass=quadratic94(coefficients94(state),model.gram(state));normalize94(state,mass);stats[0]=mode+1;stats[3]=std::max(stats[3],double(state.size()));
 if(mode+1<model.m)state=compress_branch94(state,model,limit,budget,stats);for(const auto& term:state)stats[10]=std::max(stats[10],double(std::abs(term.coefficient)));return state;
}
static void trajectory94(Vacuum94& model,int limit,double budget,DomainRng84& rng,std::uint32_t* bits,double* stats){
 State94 state{{Mask94(model.words),1}};stats[3]=1;
 for(int mode=0;mode<model.m;++mode){long double uniform=uniform84(rng);auto branches=split94(state,mode,model,stats);bool click=uniform>=branches.p0;bits[mode]=click;state=follow94(click?std::move(branches.one):std::move(branches.zero),mode,model,limit,budget,stats);}
 stats[1]=1;
}
API93 int branch94_sample(int m,const double* q,const double* mean,int limit,double budget,int count,std::uint64_t experiment,unsigned sector,std::uint64_t offset,int threads,std::uint32_t* output,double* statistics,int* errors,char* error,int cap){try{
 Guard93 guard;need93(m>=1&&m<=2048&&q&&mean&&limit>=1&&limit<=64&&budget>=0&&budget<=1&&count>=0&&sector<=65535&&offset<=0x100000000ULL&&std::uint64_t(count)<=0x100000000ULL-offset&&threads>=1&&threads<=64&&output&&statistics&&errors,"Branch sampler arguments");int dim=2*m;
 for(int i=0;i<dim;++i){need93(std::isfinite(mean[i]),"Nonfinite branch mean");for(int j=0;j<dim;++j)need93(std::isfinite(q[std::size_t(i)*dim+j])&&std::abs(q[std::size_t(i)*dim+j]-q[std::size_t(j)*dim+i])<1e-11,"Invalid branch covariance");}
 std::fill(output,output+std::size_t(count)*m,0);std::fill(statistics,statistics+std::size_t(count)*16,0);std::fill(errors,errors+count,0);std::atomic<int> next{0};std::vector<std::thread> pool;std::vector<std::string> messages(count);
 auto worker=[&](){Guard93 guard;for(;;){int row=next.fetch_add(1);if(row>=count)break;double* stats=statistics+16*row;DomainRng84 rng(experiment,sector,Click84,offset+row);Vacuum94 model(m,q,mean);try{trajectory94(model,limit,budget,rng,output+std::size_t(row)*m,stats);}catch(const Failure93& e){errors[row]=e.code;messages[row]=e.what();}catch(const std::exception& e){errors[row]=4;messages[row]=e.what();}stats[2]=model.cache.size();stats[9]=double(rng.used);stats[15]=budget;}};
 try{for(int i=0;i<std::min(count,threads);++i)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[row]){message93(error,cap,"row="+std::to_string(row)+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){message93(error,cap,e.what());return 2;}}

// Diagnostic complete tree; the stochastic sampler never calls enumeration.
static void enumerate94(const State94& state,int mode,std::uint64_t prefix,long double mass,Vacuum94& model,int limit,double budget,std::array<double,16> trace,long double* law,long double* diagnostics){
 if(mode==model.m){law[prefix]+=mass;diagnostics[0]+=mass*trace[4];return;}
 try{auto branches=split94(state,mode,model,trace.data());for(int bit=0;bit<2;++bit){long double probability=bit?branches.p1:branches.p0;if(probability==0)continue;auto child_trace=trace;long double weight=mass*probability;
  try{auto child=follow94(bit?branches.one:branches.zero,mode,model,limit,budget,child_trace.data());enumerate94(child,mode+1,prefix|(std::uint64_t(bit)<<mode),weight,model,limit,budget,child_trace,law,diagnostics);}
  catch(const Failure93& e){law[std::uint64_t(1)<<model.m]+=weight;diagnostics[0]+=weight*child_trace[4];diagnostics[e.code==2?3:2]+=weight;}
 }}catch(const Failure93&){law[std::uint64_t(1)<<model.m]+=mass;diagnostics[0]+=mass*trace[4];diagnostics[2]+=mass;}
}
API93 int branch94_enumerate(int m,const double* q,const double* mean,int limit,double budget,double* output,double* stats,char* error,int cap){try{
 Guard93 guard;need93(m>=1&&m<=10&&q&&mean&&limit>=1&&limit<=64&&budget>=0&&budget<=1&&output&&stats,"Branch enumeration arguments");Vacuum94 model(m,q,mean);State94 initial{{Mask94(model.words),1}};std::vector<long double> law((std::size_t(1)<<m)+1,0);long double diagnostics[4]={};enumerate94(initial,0,0,1,model,limit,budget,{},law.data(),diagnostics);for(int i=0;i<int(law.size());++i)output[i]=double(law[i]);for(int i=0;i<4;++i)stats[i]=double(diagnostics[i]);stats[1]=double(law.back());stats[4]=model.cache.size();return 0;
 }catch(const std::exception& e){message93(error,cap,e.what());return 1;}}
API93 int branch94_compress_probe(int m,const double* q,const double* mean,int count,const std::uint64_t* masks,const double* coefficients,int limit,double* new_coefficients,std::uint64_t* new_masks,double* stats,char* error,int cap){try{
 Guard93 guard;need93(m>=1&&m<=10&&count>=1&&count<=128&&limit>=1&&limit<count,"Branch compression probe arguments");Vacuum94 model(m,q,mean);State94 state;for(int i=0;i<count;++i){Mask94 mask(model.words);for(int w=0;w<model.words;++w)mask[w]=masks[i*model.words+w];state.push_back({mask,coefficients[i]});}normalize94(state,quadratic94(coefficients94(state),model.gram(state)));std::fill(stats,stats+16,0);auto result=compress_branch94(state,model,limit,1.,stats);stats[12]=result.size();for(int i=0;i<int(result.size());++i){new_coefficients[i]=double(result[i].coefficient);for(int w=0;w<model.words;++w)new_masks[i*model.words+w]=result[i].mask[w];}return 0;
 }catch(const std::exception& e){message93(error,cap,e.what());return 1;}}
