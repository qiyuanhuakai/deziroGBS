// RNG correction: experiment/sector/global-row/purpose fields do not overlap.
// A given pure Gaussian core plus a FULL correlated classical displacement.
// Draw displacement before coherent photon-jump conditioning. Never sample
// core and noise PNR independently and add them. K+1-factor Gaussian closure
// is still approximate; the core decomposition is not its error certificate.
#include "gaussian_pnr_resplit75_v1.cpp"
#include <climits>
#include "gaussian_rng_domain84_v1.hpp"
static bool resplit_rng84(ClickModel70& c,State73& state,Vector& t,DomainRng84& rng,ResplitStats75& stats){
    ++stats.attempts;need(state.factors.empty(),"Resplit requires compressed Gaussian");auto split=decompose75(state.x);if(!split.valid)return false;
    int n=c.model->n,m=c.model->m;std::normal_distribution<double> normal;Vector z(2*n);for(int j=0;j<2*n;++j)z[j]=normal(rng);Vector nextmean=split.rotation.transpose()*(state.mean+split.noise_factor*z);
    Real nexta=split.rotation.transpose()*c.a,nextb=split.rotation.transpose()*c.b;Complex nextv(m,n);
    for(int i=0;i<m;++i)for(int j=0;j<n;++j)nextv(i,j)={nexta(j,i),nextb(j,i)};
    need((nextv.adjoint()*nextv-Complex::Identity(n,n)).norm()<1e-8,"Resplit passive isometry residual");
    c.model->v=nextv;for(int i=0;i<m;++i)for(int j=0;j<n;++j){auto v=nextv(i,j);c.a(j,i)=v.real();c.a(j+n,i)=-v.imag();c.b(j,i)=v.imag();c.b(j+n,i)=v.real();}
    t=split.t;state.mean=nextmean;state.x.setZero();for(int j=0;j<n;++j){state.x(j,j)=-t[j]/(1+t[j]);state.x(n+j,n+j)=t[j]/(1-t[j]);}
    ++stats.successes;stats.residual=std::max(stats.residual,split.residual);return true;
}

API int pnr84d_correlated_sample(void* handle,const double* squeezing,const double* noise_factor,int count,std::uint64_t seed,unsigned sector,std::uint64_t offset,int threads,int retained,int resplit,const double* mean,const double* normals,std::uint32_t* output,double* statistics,int* errors,char* message,int cap){try{
 PrecisionGuard precision;need(handle&&squeezing&&noise_factor&&count>=0&&count<=4096&&sector<=65535&&offset<=0x100000000ULL&&std::uint64_t(count)<=0x100000000ULL-offset&&threads>=1&&threads<=64&&retained>=0&&retained<=7&&(resplit==0||resplit==1)&&mean&&output&&statistics&&errors,"Invalid correlated PNR arguments");auto& original=*static_cast<ClickModel70*>(handle);int n=original.model->n,d=2*n,m=original.model->m;Vector initial_t(n),center(d);Real L(d,d),xp=Real::Zero(d,d);
 for(int j=0;j<n;++j){auto t=squeezing[j];need(std::isfinite(t)&&std::abs(t)<1,"Invalid correlated PNR squeezing");initial_t[j]=t;xp(j,j)=-t/(1+t);xp(n+j,n+j)=t/(1-t);}
 for(int j=0;j<d;++j){center[j]=mean[j];for(int k=0;k<d;++k)L(j,k)=noise_factor[j*d+k];}need(center.allFinite()&&L.allFinite(),"Invalid correlated displacement");need((original.model->initial-xp).cwiseAbs().maxCoeff()<1e-10,"Supplied model differs from pure diagonal core");
 std::fill(output,output+std::size_t(count)*m,0);std::fill(statistics,statistics+16*count,0);std::fill(errors,errors+2*count,0);std::atomic<int> next{0};std::vector<std::string> messages(count);std::vector<std::thread> pool;
 auto worker=[&](){PrecisionGuard guard;for(;;){int row=next.fetch_add(1);if(row>=count)break;Stats73 trace;ResplitStats75 splits;int stage=0;std::uint64_t photons=0;DomainRng84 splitting(seed,sector,Resplitting84,offset+row),waits(seed,sector,Waiting84,offset+row),detectors(seed,sector,Detector84,offset+row),noise(seed,sector,Displacement84,offset+row);try{
  ClickModel70 c;c.model=std::make_unique<Model>();c.model->n=n;c.model->m=m;c.model->v=original.model->v;c.a=original.a;c.b=original.b;
  std::normal_distribution<double> normal;Vector z(d);for(int j=0;j<d;++j){z[j]=normals?normals[std::size_t(row)*d+j]:normal(noise);need(std::isfinite(z[j]),"Invalid supplied correlated normal");}
  State73 state{xp,center+L*z,{}};Vector current_t=initial_t;bool pure=true;
  for(;;){stage=1;bool jump=pure?pure_flow74(c,state,current_t,uniform84(waits),trace):flow73(c,state,uniform84(waits),trace);if(!jump)break;stage=2;int mode=pure?pure_detector74(c,state,current_t,uniform84(detectors)):detector73(c,state,uniform84(detectors));state.factors.push_back(mode);trace.maximum_factors=std::max(trace.maximum_factors,double(state.factors.size()));if(state.factors.size()>std::size_t(retained)){stage=3;compress73(c,state,trace);pure=resplit?resplit_rng84(c,state,current_t,splitting,splits):false;}auto& value=output[std::size_t(row)*m+mode];need(value<std::numeric_limits<std::uint32_t>::max(),"Correlated PNR storage overflow");++value;++photons;}
  trace.photons=double(photons);trace.completed=1;trace.maximum_photons=double(photons);
 }catch(const std::exception& e){errors[2*row]=stage+1;errors[2*row+1]=int(std::min<std::uint64_t>(photons,INT_MAX));messages[row]=e.what();trace.photons=double(photons);trace.maximum_photons=double(photons);}
 auto* out=statistics+16*row;out[0]=trace.photons;out[1]=trace.completed;out[2]=trace.maximum_photons;out[3]=trace.compressions;out[4]=trace.maximum_factors;out[5]=trace.flow_calls;out[6]=trace.root_iterations;out[7]=trace.root_fallbacks;out[8]=trace.residual;out[9]=splits.attempts;out[10]=splits.successes;out[11]=splits.residual;out[12]=double(waits.used);out[13]=double(detectors.used);out[14]=double(noise.used);out[15]=double(splitting.used);}};
 try{for(int j=0;j<std::min(count,threads);++j)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[2*row]){error(message,cap,"row="+std::to_string(row)+" completed_photons="+std::to_string(errors[2*row+1])+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
