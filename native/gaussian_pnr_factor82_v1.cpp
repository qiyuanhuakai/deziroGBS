// Factorized photon-pair sampler. Counts identify (i,j) with (j,i), so an
// oriented draw implements the symmetric physical pair table without a coin.
#include "gaussian_pnr_transport80_v3.cpp"
#include <omp.h>
using Row82=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
struct Compact82 {
 int size,bits,categories;std::vector<std::uint64_t> thresholds;std::vector<int> aliases;long double discrepancy;
 Compact82(const double* p,int n){Alias80 a(p,n);size=a.size;bits=a.bits;categories=n;discrepancy=a.tv();thresholds=std::move(a.thresholds);aliases=std::move(a.aliases);}
 int draw(std::mt19937_64& rng)const{auto word=rng();int col=word&(size-1);return (word>>bits)<thresholds[col]?col:aliases[col];}
 std::vector<long double> law()const{std::vector<long double> p(size);long double levels=std::ldexp(1.L,64-bits);for(int i=0;i<size;++i){long double q=thresholds[i]/levels/size;p[i]+=q;p[aliases[i]]+=1.L/size-q;}return p;}
 std::uint64_t bytes()const{return std::uint64_t(size)*(sizeof(std::uint64_t)+sizeof(int));}
};
struct FactorPlan82 {
 const ClickModel70* owner;int m,latent;std::unique_ptr<Compact82> labels;std::vector<std::unique_ptr<Compact82>> roots,proposals;std::vector<std::uint64_t> accept;double minimum_acceptance=1;
 std::pair<int,int> draw(int label,std::mt19937_64& rng,std::uint64_t& attempts)const{
  int i=roots[label]->draw(rng);for(int trial=0;trial<1000000;++trial){++attempts;int j=proposals[i]->draw(rng);if((rng()>>11)<accept[std::size_t(label)*m+j])return {i,j};}throw std::runtime_error("Factor pair rejection exceeded explicit one-million-attempt bound");
 }
};
API void* factor82_create(void* handle,int latent,const double* weights,const double* kernel,const double* left,const double* right,int threads,double* stats,char* message,int cap){try{
 PrecisionGuard precision;need(handle&&weights&&kernel&&left&&right&&stats&&latent>=1&&latent<=131072&&threads>=1&&threads<=64,"Invalid factor plan arguments");auto& c=*static_cast<ClickModel70*>(handle);int m=c.model->m,n=c.model->n;need(m<=1024&&std::uint64_t(latent)*m<=64000000,"Factor plan workspace bound");
 struct Guard{int old;Guard(int n):old(omp_get_max_threads()){omp_set_num_threads(n);}~Guard(){omp_set_num_threads(old);}} guard(threads);
 Eigen::Map<const Row82> K(kernel,m,m),A(left,latent,m),B(right,latent,m);need(K.allFinite()&&A.allFinite()&&B.allFinite()&&std::min({K.minCoeff(),A.minCoeff(),B.minCoeff()})>=0,"Nonfinite or negative factor");
 auto plan=std::make_unique<FactorPlan82>();plan->owner=&c;plan->m=m;plan->latent=latent;plan->labels=std::make_unique<Compact82>(weights,latent);std::fill(stats,stats+8,0);stats[4]=double(plan->labels->discrepancy);
 Row82 marg=(A.cwiseProduct(B*K.transpose())+B.cwiseProduct(A*K))*.5,weighted=B;for(int l=0;l<latent;++l)weighted.row(l)*=weights[l];Row82 average=K.cwiseProduct(A.transpose()*weighted);average=((average+average.transpose())*.5).eval();
 Row82 Q=(c.model->v*c.model->v.transpose()).cwiseAbs2()/n;stats[2]=(average-Q).cwiseAbs().maxCoeff();weighted=marg;for(int l=0;l<latent;++l)weighted.row(l)*=weights[l];Row82 cross=marg.transpose()*weighted;auto v=c.model->v;Eigen::VectorXd ell=v.cwiseAbs2().rowwise().sum();Row82 R=(ell*ell.transpose()+(v*v.transpose()).cwiseAbs2()+(v*v.adjoint()).cwiseAbs2())/(n*(n+2.));stats[3]=(cross-R).cwiseAbs().maxCoeff();
 need(std::max(stats[2],stats[3])<1e-8&&(marg.rowwise().sum().array()-1).abs().maxCoeff()<1e-8,"Factor plan does not match physical pair constraints");marg.resize(0,0);weighted.resize(0,0);
 Row82 proposal(m,m),alpha(latent,m);Eigen::VectorXd rowsum=K.rowwise().sum(),maximum=B.rowwise().maxCoeff();std::vector<double> p(m);plan->proposals.reserve(m);plan->roots.reserve(latent);plan->accept.resize(std::size_t(latent)*m);
 for(int i=0;i<m;++i){for(int j=0;j<m;++j)p[j]=rowsum[i]>0?K(i,j)/rowsum[i]:(j==0?1.:0.);plan->proposals.push_back(std::make_unique<Compact82>(p.data(),m));auto law=plan->proposals.back()->law();for(int j=0;j<m;++j)proposal(i,j)=double(law[j]);stats[5]=std::max(stats[5],double(plan->proposals.back()->discrepancy));}
 constexpr std::uint64_t levels=1ULL<<53;
 for(int l=0;l<latent;++l){need(maximum[l]>0,"Zero acceptance label unsupported");for(int j=0;j<m;++j){long double ratio=static_cast<long double>(B(l,j))/maximum[l];auto cutoff=std::uint64_t(std::ceil(ratio*levels));need(cutoff<=levels,"Acceptance integer overflow");plan->accept[std::size_t(l)*m+j]=cutoff;alpha(l,j)=double(cutoff)/levels;}}
 Row82 accepted=alpha*proposal.transpose();
 for(int l=0;l<latent;++l){double total=0;for(int i=0;i<m;++i){p[i]=A(l,i)*rowsum[i]*maximum[l]*accepted(l,i);if(p[i]>0)plan->minimum_acceptance=std::min(plan->minimum_acceptance,accepted(l,i));total+=p[i];}need(std::isfinite(total)&&total>0,"Invalid factor root mass");for(auto& value:p)value/=total;plan->roots.push_back(std::make_unique<Compact82>(p.data(),m));stats[6]=std::max(stats[6],double(plan->roots.back()->discrepancy));}
 need(plan->minimum_acceptance>=1e-4,"Factor acceptance below explicit work guarantee");stats[1]=plan->minimum_acceptance;stats[0]=plan->labels->bytes()+plan->accept.size()*sizeof(std::uint64_t);for(auto& p:plan->roots)stats[0]+=p->bytes();for(auto& p:plan->proposals)stats[0]+=p->bytes();stats[7]=1e6;return plan.release();
}catch(const std::exception& e){error(message,cap,e.what());return nullptr;}}
API void factor82_destroy(void* ptr){delete static_cast<FactorPlan82*>(ptr);}
// Actual ideal-iid integer law, limited to small audits, no quantum reference.
API int factor82_law(void* ptr,double* weights,double* pairs,char* message,int cap){try{
 PrecisionGuard guard;need(ptr&&weights&&pairs,"Invalid factor law pointers");auto& p=*static_cast<FactorPlan82*>(ptr);need(p.m<=16&&p.latent<=512,"Full factor law export audit bound");auto w=p.labels->law();for(int l=0;l<p.latent;++l){weights[l]=double(w[l]);auto root=p.roots[l]->law();for(int i=0;i<p.m;++i){auto q=p.proposals[i]->law();if(root[i]==0){for(int j=0;j<p.m;++j)pairs[(std::size_t(l)*p.m+i)*p.m+j]=0;continue;}long double total=0;for(int j=0;j<p.m;++j)total+=q[j]*p.accept[std::size_t(l)*p.m+j];need(total>0,"Zero conditional acceptance");for(int j=0;j<p.m;++j)pairs[(std::size_t(l)*p.m+i)*p.m+j]=double(root[i]*q[j]*p.accept[std::size_t(l)*p.m+j]/total);}}return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
// One label's integer table data, permitting exact Fraction reconstruction.
API int factor82_integer(void* ptr,int label,std::uint64_t* thresholds,int* aliases,std::uint64_t* accept,int* sizes,char* message,int cap){try{
 need(ptr&&thresholds&&aliases&&accept&&sizes,"Invalid factor integer export");auto& p=*static_cast<FactorPlan82*>(ptr);need(label>=0&&label<p.latent&&p.m<=16&&p.latent<=512,"Integer export audit bound");std::size_t index=0;std::vector<const Compact82*> tables{p.labels.get(),p.roots[label].get()};for(auto& q:p.proposals)tables.push_back(q.get());for(std::size_t t=0;t<tables.size();++t){auto& q=*tables[t];sizes[t]=q.size;for(int i=0;i<q.size;++i){thresholds[index]=q.thresholds[i];aliases[index++]=q.aliases[i];}}for(int j=0;j<p.m;++j)accept[j]=p.accept[std::size_t(label)*p.m+j];return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}

API int factor82_sample(void* handle,void* plan,double t,int count,std::uint64_t seed,int threads,int fixed_total,int exact_total_budget,std::uint32_t* output,std::uint64_t* totals,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(exact_total_budget>=0&&exact_total_budget<=498&&exact_total_budget%2==0,"Invalid exact-total budget");need(handle&&plan&&output&&totals&&stats&&count>=0&&threads>=1&&threads<=256&&std::isfinite(t)&&t>=0&&t<1&&(fixed_total==-1||(fixed_total>=0&&fixed_total%2==0)),"Invalid number-conditioned batch");need(t>0||fixed_total<=0,"Positive fixed total has zero vacuum probability");auto& c=*static_cast<ClickModel70*>(handle);int n=c.model->n,m=c.model->m;number_capacity77(n,exact_total_budget);std::fill(output,output+std::size_t(count)*m,0);std::fill(totals,totals+count,0);std::fill(stats,stats+12,0);if(!count)return 0;std::vector<LD73> first_cdf;LD73 first_total=0;for(int i=0;i<m;++i){LD73 value=0;for(int j=0;j<n;++j){auto z=c.model->v(i,j);value+=LD73(z.real())*z.real()+LD73(z.imag())*z.imag();}first_total+=value;first_cdf.push_back(first_total);}auto& cached=*static_cast<FactorPlan82*>(plan);need(cached.owner==&c,"Factor plan/model ownership mismatch");auto& latent_alias=*cached.labels;

 int workers=std::min(count,threads);std::vector<std::array<double,12>> local(workers);std::atomic<std::int64_t> next{0},completed{0};std::atomic<bool> failed{false};std::vector<std::exception_ptr> failures(workers);std::vector<std::thread> pool;
 auto worker=[&](int id){try{PrecisionGuard guard;auto& trace=local[id];for(;;){if(failed.load())break;auto row=next.fetch_add(1);if(row>=count)break;std::mt19937_64 number(mix(seed+2ULL*row)),detector(mix(seed+2ULL*row+1));std::uint64_t photons=0;
 if(fixed_total>=0)photons=fixed_total;else if(t>0){LD73 z=LD73(t)*t;std::gamma_distribution<LD73> gamma(LD73(n)/2,z/(1-z));LD73 intensity=gamma(number);need(std::isfinite(intensity)&&intensity>=0&&intensity<1e15L,"Total-count generator resource bound");std::poisson_distribution<std::uint64_t> poisson{double(intensity)};auto pairs=poisson(number);need(pairs<std::uint64_t(std::numeric_limits<int>::max()/2),"Total-count integer resource bound");photons=2*pairs;}
 totals[row]=photons;if(photons>std::uint64_t(exact_total_budget)){
 std::mt19937_64 latent_rng(mix(seed+0xb492b66fbe98f273ULL+row));int label=latent_alias.draw(latent_rng);
 for(std::uint64_t k=0;k<photons/2;++k){std::uint64_t attempts=0;auto ports_pair=cached.draw(label,detector,attempts);trace[10]+=attempts;trace[11]+=attempts-1;++output[std::size_t(row)*m+ports_pair.first];++output[std::size_t(row)*m+ports_pair.second];}
 trace[0]+=photons;trace[1]+=1;trace[2]=std::max(trace[2],double(photons));trace[3]+=1;trace[4]+=photons;trace[8]+=1;trace[9]+=photons;++completed;continue;}number_capacity77(n,photons);std::vector<int> modes;
 for(std::uint64_t k=0;k<photons;++k){if(k==0){int mode=categorical(first_cdf,first_total,uniform(detector));modes.push_back(mode);++output[std::size_t(row)*m+mode];trace[4]+=1;trace[5]=std::max(trace[5],1.);trace[6]=std::max(trace[6],1.);continue;}auto r=rates77(c,t,photons,modes);std::vector<LD73> cumulative;LD73 total=0;for(auto rate:r.rates){auto next=total+rate;need(!(rate>0&&next<=total),"Conditional categorical interval lost");total=next;cumulative.push_back(total);}int mode=categorical(cumulative,total,uniform(detector));modes.push_back(mode);++output[std::size_t(row)*m+mode];trace[4]+=1;trace[5]=std::max(trace[5],double(modes.size()));trace[6]=std::max(trace[6],double(r.terms));trace[7]=std::max(trace[7],double(r.rank));}
 trace[0]+=photons;trace[1]+=1;trace[2]=std::max(trace[2],double(photons));trace[3]+=1;++completed;
 }}catch(...){failures[id]=std::current_exception();failed.store(true);}};
 try{for(int i=0;i<workers;++i)pool.emplace_back(worker,i);}catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}for(auto& thread:pool)thread.join();for(auto& trace:local)for(int j=0;j<12;++j){if(j==2||(j>=5&&j<=7))stats[j]=std::max(stats[j],trace[j]);else stats[j]+=trace[j];}for(auto& e:failures)if(e)std::rethrow_exception(e);need(completed.load()==count,"Incomplete number-conditioned batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
