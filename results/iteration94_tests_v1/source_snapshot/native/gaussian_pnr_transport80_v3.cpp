#include "gaussian_pnr_number77_v2.cpp"
struct Alias80 {
 int bits=1,size=2,categories;std::uint64_t levels;std::vector<std::uint64_t> thresholds;std::vector<int> aliases;std::vector<LD73> original;
 Alias80(const double* probabilities,int count):categories(count){
  need(probabilities&&count>=1&&count<=262144,"Invalid pair alias dimensions");while(size<count){size*=2;++bits;}levels=1ULL<<(64-bits);thresholds.resize(size);aliases.resize(size);original.resize(size,0);LD73 total=0;
  for(int i=0;i<count;++i){need(std::isfinite(probabilities[i])&&probabilities[i]>=0,"Invalid alias mass");original[i]=probabilities[i];total+=original[i];}need(std::abs(total-1)<1e-8L,"Invalid alias total");
  std::vector<LD73> scaled(size);std::vector<int> small,large;
  for(int i=0;i<size;++i){original[i]/=total;scaled[i]=original[i]*size;(scaled[i]<1?small:large).push_back(i);}
  while(!small.empty()&&!large.empty()){int a=small.back();small.pop_back();int b=large.back();large.pop_back();need(scaled[a]>=0&&scaled[a]<=1,"Invalid alias fraction");thresholds[a]=std::uint64_t(std::ceil(scaled[a]*LD73(levels)));aliases[a]=b;scaled[b]+=scaled[a]-1;(scaled[b]<1?small:large).push_back(b);}
  for(int i:small){thresholds[i]=levels;aliases[i]=i;}for(int i:large){thresholds[i]=levels;aliases[i]=i;}
  auto actual=law();for(int i=count;i<size;++i)need(actual[i]==0,"Alias padding acquired probability");
 }
 std::vector<LD73> law()const{std::vector<LD73> out(size,0);for(int i=0;i<size;++i){LD73 probability=LD73(thresholds[i])/levels;out[i]+=probability/size;out[aliases[i]]+=(1-probability)/size;}return out;}
 int draw(std::mt19937_64& generator)const{std::uint64_t word=generator();int col=word&(size-1);return (word>>bits)<thresholds[col]?col:aliases[col];}
 LD73 tv()const{auto actual=law();LD73 sum=0;for(int i=0;i<size;++i)sum+=std::abs(actual[i]-original[i]);return sum/2;}
};
API int transport80_alias_law(int m,int latent,const double* weights,const double* masses,double* actual_weights,double* actual_masses,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(m>=1&&m<=384&&latent>=1&&latent<=8192&&weights&&masses&&actual_weights&&actual_masses&&stats,"Invalid transport alias export");int categories=m*(m+1)/2;need(std::size_t(latent)*categories<=16000000,"Transport alias resource bound");Alias80 direction(weights,latent);auto law=direction.law();for(int l=0;l<latent;++l)actual_weights[l]=double(law[l]);stats[0]=double(direction.tv());stats[1]=0;
 for(int l=0;l<latent;++l){Alias80 pair(masses+std::size_t(l)*categories,categories);auto p=pair.law();for(int k=0;k<categories;++k)actual_masses[std::size_t(l)*categories+k]=double(p[k]);stats[1]=std::max(stats[1],double(pair.tv()));}return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}

API int transport80a_sample(void* handle,double t,int count,std::uint64_t seed,int threads,int fixed_total,int exact_total_budget,int latent,const double* latent_weights,const double* pair_masses,std::uint32_t* output,std::uint64_t* totals,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(exact_total_budget>=0&&exact_total_budget<=498&&exact_total_budget%2==0,"Invalid exact-total budget");need(handle&&output&&totals&&stats&&count>=0&&threads>=1&&threads<=256&&std::isfinite(t)&&t>=0&&t<1&&(fixed_total==-1||(fixed_total>=0&&fixed_total%2==0)),"Invalid number-conditioned batch");need(t>0||fixed_total<=0,"Positive fixed total has zero vacuum probability");auto& c=*static_cast<ClickModel70*>(handle);int n=c.model->n,m=c.model->m;number_capacity77(n,exact_total_budget);std::fill(output,output+std::size_t(count)*m,0);std::fill(totals,totals+count,0);std::fill(stats,stats+10,0);if(!count)return 0;std::vector<LD73> first_cdf;LD73 first_total=0;for(int i=0;i<m;++i){LD73 value=0;for(int j=0;j<n;++j){auto z=c.model->v(i,j);value+=LD73(z.real())*z.real()+LD73(z.imag())*z.imag();}first_total+=value;first_cdf.push_back(first_total);}need(latent>=1&&latent<=8192&&m<=384&&latent_weights&&pair_masses,"Invalid transport table dimensions");
 int categories=m*(m+1)/2;need(std::size_t(latent)*categories<=16000000,"Transport table resource bound");
 std::vector<std::pair<int,int>> ports;for(int i=0;i<m;++i)for(int j=i;j<m;++j)ports.emplace_back(i,j);
 Alias80 latent_alias(latent_weights,latent);std::vector<Alias80> pair_aliases;pair_aliases.reserve(latent);for(int l=0;l<latent;++l)pair_aliases.emplace_back(pair_masses+std::size_t(l)*categories,categories);
 int workers=std::min(count,threads);std::vector<std::array<double,10>> local(workers);std::atomic<std::int64_t> next{0},completed{0};std::atomic<bool> failed{false};std::vector<std::exception_ptr> failures(workers);std::vector<std::thread> pool;
 auto worker=[&](int id){try{PrecisionGuard guard;auto& trace=local[id];for(;;){if(failed.load())break;auto row=next.fetch_add(1);if(row>=count)break;std::mt19937_64 number(mix(seed+2ULL*row)),detector(mix(seed+2ULL*row+1));std::uint64_t photons=0;
 if(fixed_total>=0)photons=fixed_total;else if(t>0){LD73 z=LD73(t)*t;std::gamma_distribution<LD73> gamma(LD73(n)/2,z/(1-z));LD73 intensity=gamma(number);need(std::isfinite(intensity)&&intensity>=0&&intensity<1e15L,"Total-count generator resource bound");std::poisson_distribution<std::uint64_t> poisson{double(intensity)};auto pairs=poisson(number);need(pairs<std::uint64_t(std::numeric_limits<int>::max()/2),"Total-count integer resource bound");photons=2*pairs;}
 totals[row]=photons;if(photons>std::uint64_t(exact_total_budget)){
 std::mt19937_64 latent_rng(mix(seed+0xb492b66fbe98f273ULL+row));int label=latent_alias.draw(latent_rng);
 for(std::uint64_t k=0;k<photons/2;++k){int choice=pair_aliases[label].draw(detector);auto ports_pair=ports[choice];++output[std::size_t(row)*m+ports_pair.first];++output[std::size_t(row)*m+ports_pair.second];}
 trace[0]+=photons;trace[1]+=1;trace[2]=std::max(trace[2],double(photons));trace[3]+=1;trace[4]+=photons;trace[8]+=1;trace[9]+=photons;++completed;continue;}number_capacity77(n,photons);std::vector<int> modes;
 for(std::uint64_t k=0;k<photons;++k){if(k==0){int mode=categorical(first_cdf,first_total,uniform(detector));modes.push_back(mode);++output[std::size_t(row)*m+mode];trace[4]+=1;trace[5]=std::max(trace[5],1.);trace[6]=std::max(trace[6],1.);continue;}auto r=rates77(c,t,photons,modes);std::vector<LD73> cumulative;LD73 total=0;for(auto rate:r.rates){auto next=total+rate;need(!(rate>0&&next<=total),"Conditional categorical interval lost");total=next;cumulative.push_back(total);}int mode=categorical(cumulative,total,uniform(detector));modes.push_back(mode);++output[std::size_t(row)*m+mode];trace[4]+=1;trace[5]=std::max(trace[5],double(modes.size()));trace[6]=std::max(trace[6],double(r.terms));trace[7]=std::max(trace[7],double(r.rank));}
 trace[0]+=photons;trace[1]+=1;trace[2]=std::max(trace[2],double(photons));trace[3]+=1;++completed;
 }}catch(...){failures[id]=std::current_exception();failed.store(true);}};
 try{for(int i=0;i<workers;++i)pool.emplace_back(worker,i);}catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}for(auto& thread:pool)thread.join();for(auto& trace:local)for(int j=0;j<10;++j){if(j==2||(j>=5&&j<=7))stats[j]=std::max(stats[j],trace[j]);else stats[j]+=trace[j];}for(auto& e:failures)if(e)std::rethrow_exception(e);need(completed.load()==count,"Incomplete number-conditioned batch");return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
