// Exact conditional quadratic detector moments, routed through optical Gram sums.
// The Gaussian/polynomial evolution and moment closure remain frozen84.
#include "gaussian_pnr_correlated84_v2.cpp"
struct DetectorTrace85 {double calls=0,fallbacks=0;};
struct GramNode85 {int first,last,left=-1,right=-1;std::vector<LD73> leftgram;};
struct GramTree85 {
 int d,m;Real a,b;std::vector<GramNode85> nodes;std::vector<LD73> totalgram;
 GramTree85(const ClickModel70& c):d(c.a.rows()),m(c.a.cols()),a(c.a),b(c.b){nodes.reserve(2*m-1);totalgram=build(0,m).second;}
 std::pair<int,std::vector<LD73>> build(int first,int last){
  int id=int(nodes.size());nodes.push_back({first,last});std::vector<LD73> gram(std::size_t(d)*(d+1)/2);
  if(last-first==1){std::size_t at=0;for(int i=0;i<d;++i)for(int j=0;j<=i;++j)gram[at++]=(LD73(a(i,first))*a(j,first)+LD73(b(i,first))*b(j,first))/2;return {id,std::move(gram)};}
  int middle=first+(last-first)/2;auto left=build(first,middle),right=build(middle,last);nodes[id].left=left.first;nodes[id].right=right.first;nodes[id].leftgram=std::move(left.second);for(std::size_t i=0;i<gram.size();++i)gram[i]=nodes[id].leftgram[i]+right.second[i];return {id,std::move(gram)};
 }
};
static LMat73 raw_second85(const ClickModel70& c,const State73& state){
 Real forms=forms73(c,state.factors),products=state.x*forms;auto moment=moments73(state.x,state.mean,forms,true);need(std::isfinite(moment.value)&&moment.value>0,"Tree85 polynomial normalization invalid");
 LMat73 px=products.cast<LD73>();LVec73 mean=state.mean.cast<LD73>(),delta=px*(moment.gradient/moment.value);LMat73 result=state.x.cast<LD73>()+mean*mean.transpose()+mean*delta.transpose()+delta*mean.transpose()+px*(moment.hessian/moment.value)*px.transpose();result=((result+result.transpose())/2).eval();need(result.allFinite(),"Tree85 second moment invalid");return result;
}
static std::vector<LD73> pack_second85(const LMat73& second){std::vector<LD73> out;out.reserve(second.rows()*(second.rows()+1)/2);for(int i=0;i<second.rows();++i)for(int j=0;j<=i;++j)out.push_back(second(i,j)*(i==j?1:2));return out;}
static std::pair<LD73,LD73> contraction85(const std::vector<LD73>& gram,const std::vector<LD73>& second){LD73 value=0,absolute=0;for(std::size_t j=0;j<gram.size();++j){LD73 term=gram[j]*second[j];value+=term;absolute+=std::abs(term);}return {value,absolute};}
static int detector85(const ClickModel70& c,const State73& state,double uniform,const GramTree85& tree,DetectorTrace85& trace){
 ++trace.calls;auto fallback=[&](){++trace.fallbacks;return detector73(c,state,uniform);};
 auto raw=raw_second85(c,state);int n=raw.rows()/2;using ComplexMoment85=Eigen::Matrix<std::complex<LD73>,Eigen::Dynamic,Eigen::Dynamic>;ComplexMoment85 occupation(n,n);
 for(int i=0;i<n;++i)for(int j=0;j<n;++j)occupation(i,j)={ (raw(i,j)+raw(i+n,j+n))/2,(raw(i,j+n)-raw(i+n,j))/2};
 // A tree must not conceal negative unvisited detector rates. Require a
 // numerical positive-definite one-particle moment; otherwise use every old rate.
 Eigen::LLT<ComplexMoment85> positive(occupation);if(positive.info()!=Eigen::Success)return fallback();
 auto second=pack_second85(raw);auto [total,scale]=contraction85(tree.totalgram,second);
 LD73 tolerance=4096*std::numeric_limits<double>::epsilon()*std::max(scale,std::abs(total));if(!(total>tolerance&&std::isfinite(total)))return fallback();
 LD73 residual=LD73(uniform)*total;int id=0;LD73 mass=total;
 while(tree.nodes[id].last-tree.nodes[id].first>1){auto& node=tree.nodes[id];auto [left,absleft]=contraction85(node.leftgram,second);LD73 guard=4096*std::numeric_limits<double>::epsilon()*(scale+absleft+std::abs(mass));
  if(!std::isfinite(left)||left<0||left>mass||std::abs(residual-left)<=guard||left<=guard||mass-left<=guard)return fallback();
  if(residual<left){id=node.left;mass=left;}else{residual-=left;mass-=left;id=node.right;}
 }
 return tree.nodes[id].first;
}
API void* pnr85_tree_create(void* handle,char* message,int cap){try{PrecisionGuard guard;need(handle,"Tree85 null model");return new GramTree85(*static_cast<ClickModel70*>(handle));}catch(const std::exception& e){error(message,cap,e.what());return nullptr;}}
API void pnr85_tree_destroy(void* handle){delete static_cast<GramTree85*>(handle);}
API int pnr85_tree_info(void* handle,double* out){if(!handle||!out)return 1;auto& tree=*static_cast<GramTree85*>(handle);std::size_t entries=tree.totalgram.size();for(auto& n:tree.nodes)entries+=n.leftgram.size();out[0]=double(tree.nodes.size());out[1]=double(entries*sizeof(LD73));out[2]=tree.d;out[3]=tree.m;return 0;}
API int pnr85_detector_probe(void* handle,void* tree_handle,const double* x,const double* mean,int length,const int* modes,int count,const double* draws,double* rates_old,double* rates_new,int* selected,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(handle&&tree_handle&&x&&mean&&modes&&length>=0&&length<=7&&count>=0&&draws&&rates_old&&rates_new&&selected&&stats,"Tree85 probe arguments");auto& c=*static_cast<ClickModel70*>(handle);auto& tree=*static_cast<GramTree85*>(tree_handle);int d=c.a.rows(),m=c.a.cols();need(tree.d==d&&tree.m==m&&(tree.a-c.a).norm()==0&&(tree.b-c.b).norm()==0,"Tree85 model mismatch");State73 state{Real(d,d),Vector(d),{}};
 for(int i=0;i<d;++i){state.mean[i]=mean[i];for(int j=0;j<d;++j)state.x(i,j)=x[i*d+j];}for(int i=0;i<length;++i){need(modes[i]>=0&&modes[i]<m,"Tree85 probe mode");state.factors.push_back(modes[i]);}
 auto second=raw_second85(c,state);std::vector<int> factors=state.factors;factors.push_back(0);LD73 oldsum=0,newsum=0;
 for(int mode=0;mode<m;++mode){factors.back()=mode;LD73 old=moments73(state.x,state.mean,forms73(c,factors),false).value;LVec73 a=c.a.col(mode).cast<LD73>(),b=c.b.col(mode).cast<LD73>();LD73 next=(a.dot(second*a)+b.dot(second*b))/2;need(old>=0&&next>=0,"Tree85 probe negative rate");rates_old[mode]=double(old);rates_new[mode]=double(next);oldsum+=old;newsum+=next;}
 for(int i=0;i<m;++i){rates_old[i]/=double(oldsum);rates_new[i]/=double(newsum);}DetectorTrace85 trace;
 for(int i=0;i<count;++i){need(std::isfinite(draws[i])&&draws[i]>=0&&draws[i]<1,"Tree85 probe draw");selected[2*i]=detector73(c,state,draws[i]);selected[2*i+1]=detector85(c,state,draws[i],tree,trace);}stats[0]=trace.calls;stats[1]=trace.fallbacks;return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}

API int pnr85_correlated_sample(void* handle,void* tree_handle,const double* squeezing,const double* noise_factor,int count,std::uint64_t seed,unsigned sector,std::uint64_t offset,int threads,int retained,int resplit,const double* mean,const double* normals,std::uint32_t* output,double* statistics,int* errors,char* message,int cap){try{
 PrecisionGuard precision;need(handle&&squeezing&&noise_factor&&count>=0&&count<=4096&&sector<=65535&&offset<=0x100000000ULL&&std::uint64_t(count)<=0x100000000ULL-offset&&threads>=1&&threads<=64&&retained>=0&&retained<=7&&resplit==0&&tree_handle&&mean&&output&&statistics&&errors,"Invalid correlated PNR arguments");auto& original=*static_cast<ClickModel70*>(handle);auto& tree=*static_cast<GramTree85*>(tree_handle);need(tree.d==original.a.rows()&&tree.m==original.a.cols()&&(tree.a-original.a).norm()==0&&(tree.b-original.b).norm()==0,"Tree85 model mismatch");int n=original.model->n,d=2*n,m=original.model->m;Vector initial_t(n),center(d);Real L(d,d),xp=Real::Zero(d,d);
 for(int j=0;j<n;++j){auto t=squeezing[j];need(std::isfinite(t)&&std::abs(t)<1,"Invalid correlated PNR squeezing");initial_t[j]=t;xp(j,j)=-t/(1+t);xp(n+j,n+j)=t/(1-t);}
 for(int j=0;j<d;++j){center[j]=mean[j];for(int k=0;k<d;++k)L(j,k)=noise_factor[j*d+k];}need(center.allFinite()&&L.allFinite(),"Invalid correlated displacement");need((original.model->initial-xp).cwiseAbs().maxCoeff()<1e-10,"Supplied model differs from pure diagonal core");
 std::fill(output,output+std::size_t(count)*m,0);std::fill(statistics,statistics+18*count,0);std::fill(errors,errors+2*count,0);std::atomic<int> next{0};std::vector<std::string> messages(count);std::vector<std::thread> pool;
 auto worker=[&](){PrecisionGuard guard;for(;;){int row=next.fetch_add(1);if(row>=count)break;Stats73 trace;DetectorTrace85 detector_trace;ResplitStats75 splits;int stage=0;std::uint64_t photons=0;DomainRng84 splitting(seed,sector,Resplitting84,offset+row),waits(seed,sector,Waiting84,offset+row),detectors(seed,sector,Detector84,offset+row),noise(seed,sector,Displacement84,offset+row);try{
  ClickModel70 c;c.model=std::make_unique<Model>();c.model->n=n;c.model->m=m;c.model->v=original.model->v;c.a=original.a;c.b=original.b;
  std::normal_distribution<double> normal;Vector z(d);for(int j=0;j<d;++j){z[j]=normals?normals[std::size_t(row)*d+j]:normal(noise);need(std::isfinite(z[j]),"Invalid supplied correlated normal");}
  State73 state{xp,center+L*z,{}};Vector current_t=initial_t;bool pure=true;
  for(;;){stage=1;bool jump=pure?pure_flow74(c,state,current_t,uniform84(waits),trace):flow73(c,state,uniform84(waits),trace);if(!jump)break;stage=2;int mode=pure?pure_detector74(c,state,current_t,uniform84(detectors)):detector85(c,state,uniform84(detectors),tree,detector_trace);state.factors.push_back(mode);trace.maximum_factors=std::max(trace.maximum_factors,double(state.factors.size()));if(state.factors.size()>std::size_t(retained)){stage=3;compress73(c,state,trace);pure=resplit?resplit_rng84(c,state,current_t,splitting,splits):false;}auto& value=output[std::size_t(row)*m+mode];need(value<std::numeric_limits<std::uint32_t>::max(),"Correlated PNR storage overflow");++value;++photons;}
  trace.photons=double(photons);trace.completed=1;trace.maximum_photons=double(photons);
 }catch(const std::exception& e){errors[2*row]=stage+1;errors[2*row+1]=int(std::min<std::uint64_t>(photons,INT_MAX));messages[row]=e.what();trace.photons=double(photons);trace.maximum_photons=double(photons);}
 auto* out=statistics+18*row;out[0]=trace.photons;out[1]=trace.completed;out[2]=trace.maximum_photons;out[3]=trace.compressions;out[4]=trace.maximum_factors;out[5]=trace.flow_calls;out[6]=trace.root_iterations;out[7]=trace.root_fallbacks;out[8]=trace.residual;out[9]=splits.attempts;out[10]=splits.successes;out[11]=splits.residual;out[12]=double(waits.used);out[13]=double(detectors.used);out[14]=double(noise.used);out[15]=double(splitting.used);out[16]=detector_trace.calls;out[17]=detector_trace.fallbacks;}};
 try{for(int j=0;j<std::min(count,threads);++j)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[2*row]){error(message,cap,"row="+std::to_string(row)+" completed_photons="+std::to_string(errors[2*row+1])+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
