// Store only upper levels; regenerate leaf-block Gram sums in the EXACT frozen85
// binary summation order. No alternate quadratic-rate normalization is used.
#include "gaussian_pnr_tree85_v1.cpp"
#include <xmmintrin.h>
struct PnrRoundGuard87 {PrecisionGuard x87;unsigned saved;PnrRoundGuard87():saved(_mm_getcsr()){_mm_setcsr(saved&~0x6000u);}~PnrRoundGuard87(){unsigned after=_mm_getcsr();_mm_setcsr((saved&~0x3fu)|((saved|after)&0x3fu));}};
struct BlockTrace87 {double calls=0,fallbacks=0,ondemand=0,leaves=0;};
struct BlockTree87 {
 int d,m,block;Real a,b;std::vector<GramNode85> nodes;std::vector<LD73> totalgram;
 BlockTree87(const ClickModel70& c,int block_size):d(c.a.rows()),m(c.a.cols()),block(block_size),a(c.a),b(c.b){need(block>=1&&block<=1024,"Block87 leaf size");nodes.reserve(std::min<std::size_t>(2*std::size_t(m)-1,4*((std::size_t(m)+block-1)/block)));totalgram=build(0,m).second;}
 std::vector<LD73> sum(int first,int last,BlockTrace87* trace=nullptr)const{
  std::vector<LD73> gram(std::size_t(d)*(d+1)/2);
  if(last-first==1){if(trace)++trace->leaves;std::size_t at=0;for(int i=0;i<d;++i)for(int j=0;j<=i;++j)gram[at++]=(LD73(a(i,first))*a(j,first)+LD73(b(i,first))*b(j,first))/2;return gram;}
  int middle=first+(last-first)/2;auto left=sum(first,middle,trace),right=sum(middle,last,trace);for(std::size_t i=0;i<gram.size();++i)gram[i]=left[i]+right[i];return gram;
 }
 std::pair<int,std::vector<LD73>> build(int first,int last){
  int id=int(nodes.size());nodes.push_back({first,last});if(last-first<=block)return {id,sum(first,last)};
  int middle=first+(last-first)/2;auto left=build(first,middle),right=build(middle,last);nodes[id].left=left.first;nodes[id].right=right.first;nodes[id].leftgram=std::move(left.second);std::vector<LD73> gram(std::size_t(d)*(d+1)/2);for(std::size_t i=0;i<gram.size();++i)gram[i]=nodes[id].leftgram[i]+right.second[i];return {id,std::move(gram)};
 }
};
static int detector87(const ClickModel70& c,const State73& state,double uniform,const BlockTree87& tree,BlockTrace87& trace){
 ++trace.calls;auto fallback=[&](){++trace.fallbacks;return detector73(c,state,uniform);};auto raw=raw_second85(c,state);int n=raw.rows()/2;using CM=Eigen::Matrix<std::complex<LD73>,Eigen::Dynamic,Eigen::Dynamic>;CM occupation(n,n);
 for(int i=0;i<n;++i)for(int j=0;j<n;++j)occupation(i,j)={(raw(i,j)+raw(i+n,j+n))/2,(raw(i,j+n)-raw(i+n,j))/2};Eigen::LLT<CM> positive(occupation);if(positive.info()!=Eigen::Success)return fallback();
 auto second=pack_second85(raw);auto [total,scale]=contraction85(tree.totalgram,second);LD73 tolerance=4096*std::numeric_limits<double>::epsilon()*std::max(scale,std::abs(total));if(!(total>tolerance&&std::isfinite(total)))return fallback();LD73 residual=LD73(uniform)*total,mass=total;int id=0,first=0,last=tree.m;
 while(last-first>1){int middle=first+(last-first)/2;std::vector<LD73> temporary;const std::vector<LD73>* gram=nullptr;bool stored=id>=0&&tree.nodes[id].left>=0;
  if(stored)gram=&tree.nodes[id].leftgram;else{++trace.ondemand;temporary=tree.sum(first,middle,&trace);gram=&temporary;}
  auto [left,absleft]=contraction85(*gram,second);LD73 guard=4096*std::numeric_limits<double>::epsilon()*(scale+absleft+std::abs(mass));if(!std::isfinite(left)||left<0||left>mass||std::abs(residual-left)<=guard||left<=guard||mass-left<=guard)return fallback();
  if(residual<left){if(stored)id=tree.nodes[id].left;else id=-1;last=middle;mass=left;}else{if(stored)id=tree.nodes[id].right;else id=-1;first=middle;residual-=left;mass-=left;}
 }return first;
}
API void* pnr87_tree_create(void* handle,int block,char* message,int cap){try{PnrRoundGuard87 guard;need(handle,"Block87 null model");return new BlockTree87(*static_cast<ClickModel70*>(handle),block);}catch(const std::exception& e){error(message,cap,e.what());return nullptr;}}
API void pnr87_tree_destroy(void* handle){delete static_cast<BlockTree87*>(handle);}
API int pnr87_tree_info(void* handle,double* out){if(!handle||!out)return 1;auto& t=*static_cast<BlockTree87*>(handle);std::size_t entries=t.totalgram.size();for(auto& n:t.nodes)entries+=n.leftgram.size();out[0]=double(t.nodes.size());out[1]=double(entries*sizeof(LD73));out[2]=t.d;out[3]=t.m;out[4]=t.block;out[5]=double((t.a.size()+t.b.size())*sizeof(double));out[6]=double(t.nodes.capacity()*sizeof(GramNode85));return 0;}
API int pnr87_gram_audit(void* handle,void* tree_handle,double* stats,char* message,int cap){try{PnrRoundGuard87 guard;need(handle&&tree_handle&&stats,"Block87 Gram audit arguments");auto& c=*static_cast<ClickModel70*>(handle);auto& t=*static_cast<BlockTree87*>(tree_handle);need(t.m==c.a.cols()&&t.d==c.a.rows()&&(t.a-c.a).norm()==0&&(t.b-c.b).norm()==0,"Block87 model mismatch");GramTree85 old(c);double different=0,maximum=0;std::size_t compared=0;
 auto compare=[&](const std::vector<LD73>& a,const std::vector<LD73>& b){need(a.size()==b.size(),"Block87 Gram dimensions");for(std::size_t i=0;i<a.size();++i){different+=a[i]!=b[i];maximum=std::max(maximum,double(std::abs(a[i]-b[i])));++compared;}};compare(t.totalgram,old.totalgram);
 for(auto& node:old.nodes)if(node.left>=0){int middle=node.first+(node.last-node.first)/2;compare(t.sum(node.first,middle),node.leftgram);}stats[0]=double(compared);stats[1]=different;stats[2]=maximum;return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int pnr87_detector_probe(void* handle,void* tree_handle,const double* x,const double* mean,int length,const int* modes,int count,const double* draws,int* selected,double* stats,char* message,int cap){try{
 PnrRoundGuard87 guard;need(handle&&tree_handle&&x&&mean&&modes&&length>=0&&length<=7&&count>=0&&draws&&selected&&stats,"Block87 probe arguments");auto& c=*static_cast<ClickModel70*>(handle);auto& tree=*static_cast<BlockTree87*>(tree_handle);int d=c.a.rows(),m=c.a.cols();need(tree.d==d&&tree.m==m&&(tree.a-c.a).norm()==0&&(tree.b-c.b).norm()==0,"Block87 model mismatch");State73 state{Real(d,d),Vector(d),{}};
 for(int i=0;i<d;++i){state.mean[i]=mean[i];for(int j=0;j<d;++j)state.x(i,j)=x[i*d+j];}for(int i=0;i<length;++i){need(modes[i]>=0&&modes[i]<m,"Block87 probe mode");state.factors.push_back(modes[i]);}BlockTrace87 trace;
 for(int i=0;i<count;++i){need(std::isfinite(draws[i])&&draws[i]>=0&&draws[i]<1,"Block87 draw");selected[2*i]=detector73(c,state,draws[i]);selected[2*i+1]=detector87(c,state,draws[i],tree,trace);}stats[0]=trace.calls;stats[1]=trace.fallbacks;stats[2]=trace.ondemand;stats[3]=trace.leaves;return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}

API int pnr87_correlated_sample(void* handle,void* tree_handle,const double* squeezing,const double* noise_factor,int count,std::uint64_t seed,unsigned sector,std::uint64_t offset,int threads,int retained,int resplit,const double* mean,const double* normals,std::uint32_t* output,double* statistics,int* errors,char* message,int cap){try{
 PnrRoundGuard87 precision;need(handle&&squeezing&&noise_factor&&count>=0&&count<=4096&&sector<=65535&&offset<=0x100000000ULL&&std::uint64_t(count)<=0x100000000ULL-offset&&threads>=1&&threads<=64&&retained>=0&&retained<=7&&resplit==0&&tree_handle&&mean&&output&&statistics&&errors,"Invalid correlated PNR arguments");auto& original=*static_cast<ClickModel70*>(handle);auto& tree=*static_cast<BlockTree87*>(tree_handle);need(tree.d==original.a.rows()&&tree.m==original.a.cols()&&(tree.a-original.a).norm()==0&&(tree.b-original.b).norm()==0,"Tree85 model mismatch");int n=original.model->n,d=2*n,m=original.model->m;Vector initial_t(n),center(d);Real L(d,d),xp=Real::Zero(d,d);
 for(int j=0;j<n;++j){auto t=squeezing[j];need(std::isfinite(t)&&std::abs(t)<1,"Invalid correlated PNR squeezing");initial_t[j]=t;xp(j,j)=-t/(1+t);xp(n+j,n+j)=t/(1-t);}
 for(int j=0;j<d;++j){center[j]=mean[j];for(int k=0;k<d;++k)L(j,k)=noise_factor[j*d+k];}need(center.allFinite()&&L.allFinite(),"Invalid correlated displacement");need((original.model->initial-xp).cwiseAbs().maxCoeff()<1e-10,"Supplied model differs from pure diagonal core");
 std::fill(output,output+std::size_t(count)*m,0);std::fill(statistics,statistics+20*count,0);std::fill(errors,errors+2*count,0);std::atomic<int> next{0};std::vector<std::string> messages(count);std::vector<std::thread> pool;
 auto worker=[&](){PnrRoundGuard87 guard;for(;;){int row=next.fetch_add(1);if(row>=count)break;Stats73 trace;BlockTrace87 detector_trace;ResplitStats75 splits;int stage=0;std::uint64_t photons=0;DomainRng84 splitting(seed,sector,Resplitting84,offset+row),waits(seed,sector,Waiting84,offset+row),detectors(seed,sector,Detector84,offset+row),noise(seed,sector,Displacement84,offset+row);try{
  ClickModel70 c;c.model=std::make_unique<Model>();c.model->n=n;c.model->m=m;c.model->v=original.model->v;c.a=original.a;c.b=original.b;
  std::normal_distribution<double> normal;Vector z(d);for(int j=0;j<d;++j){z[j]=normals?normals[std::size_t(row)*d+j]:normal(noise);need(std::isfinite(z[j]),"Invalid supplied correlated normal");}
  State73 state{xp,center+L*z,{}};Vector current_t=initial_t;bool pure=true;
  for(;;){stage=1;bool jump=pure?pure_flow74(c,state,current_t,uniform84(waits),trace):flow73(c,state,uniform84(waits),trace);if(!jump)break;stage=2;int mode=pure?pure_detector74(c,state,current_t,uniform84(detectors)):detector87(c,state,uniform84(detectors),tree,detector_trace);state.factors.push_back(mode);trace.maximum_factors=std::max(trace.maximum_factors,double(state.factors.size()));if(state.factors.size()>std::size_t(retained)){stage=3;compress73(c,state,trace);pure=resplit?resplit_rng84(c,state,current_t,splitting,splits):false;}auto& value=output[std::size_t(row)*m+mode];need(value<std::numeric_limits<std::uint32_t>::max(),"Correlated PNR storage overflow");++value;++photons;}
  trace.photons=double(photons);trace.completed=1;trace.maximum_photons=double(photons);
 }catch(const std::exception& e){errors[2*row]=stage+1;errors[2*row+1]=int(std::min<std::uint64_t>(photons,INT_MAX));messages[row]=e.what();trace.photons=double(photons);trace.maximum_photons=double(photons);}
 auto* out=statistics+20*row;out[0]=trace.photons;out[1]=trace.completed;out[2]=trace.maximum_photons;out[3]=trace.compressions;out[4]=trace.maximum_factors;out[5]=trace.flow_calls;out[6]=trace.root_iterations;out[7]=trace.root_fallbacks;out[8]=trace.residual;out[9]=splits.attempts;out[10]=splits.successes;out[11]=splits.residual;out[12]=double(waits.used);out[13]=double(detectors.used);out[14]=double(noise.used);out[15]=double(splitting.used);out[16]=detector_trace.calls;out[17]=detector_trace.fallbacks;out[18]=detector_trace.ondemand;out[19]=detector_trace.leaves;}};
 try{for(int j=0;j<std::min(count,threads);++j)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[2*row]){error(message,cap,"row="+std::to_string(row)+" completed_photons="+std::to_string(errors[2*row+1])+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
