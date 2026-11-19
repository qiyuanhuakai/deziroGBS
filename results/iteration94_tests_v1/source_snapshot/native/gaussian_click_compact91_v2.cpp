// Match the full conditional second moment at each closure, then retain the
// latest click factor by solving its inverse Gaussian moment map. Failed fits
// choose the explicitly defined all-Gaussian closure, before future sampling.
// This is an approximation with measured residuals, not a full-law certificate.
#include "gaussian_click_guarded87_v1.cpp"

// Conservative interior test of V+i Omega/2. Rejection selects the predefined
// full-moment Gaussian closure; this test never adds noise or clips probability.
using ComplexMatrix91=Eigen::Matrix<std::complex<long double>,Eigen::Dynamic,Eigen::Dynamic>;
static bool physical91(const Mat82& x,long double* diagnostics=nullptr){
 int d=x.rows();need(d>=2&&d%2==0&&x.cols()==d&&x.allFinite(),"Physical91 covariance shape");int n=d/2;ComplexMatrix91 q=x.cast<std::complex<long double>>();q.diagonal().array()+=.5L;for(int i=0;i<n;++i){q(i,n+i)+=std::complex<long double>(0,.5L);q(n+i,i)-=std::complex<long double>(0,.5L);}
 long double scale=q.cwiseAbs().rowwise().sum().maxCoeff();long double shift=1024*d*std::numeric_limits<long double>::epsilon()*std::max(1.L,scale);ComplexMatrix91 shifted=q;shifted.diagonal().array()-=shift;Eigen::LLT<ComplexMatrix91> chol(shifted);if(diagnostics){diagnostics[0]=shift;diagnostics[1]=0;diagnostics[2]=0;}
 if(chol.info()!=Eigen::Success)return false;ComplexMatrix91 L=chol.matrixL();ComplexMatrix91 error=shifted-L*L.adjoint();long double residual=error.cwiseAbs().rowwise().sum().maxCoeff();if(diagnostics){diagnostics[1]=residual;diagnostics[2]=L.diagonal().cwiseAbs().minCoeff();}return std::isfinite(residual)&&residual<shift/4;
}

// Suffix QR cache. A_j^* A_j equals the Gram matrix of optical rows j..m-1.
// No singular-value cutoff or eigenvalue projection is used. Rank-deficient
// compressed vacuum directions may still trigger the conservative interior test.
// Packed upper triangles and rank-one Givens insertion: O(m n^2) QR work.
// The independent Gram residual is evaluated only at 17 fixed checkpoints.
// Full independent suffix reconstruction is exercised by the audit API below.
struct Optical91 {
 int n,m;decltype(Model::v) original;std::vector<std::complex<long double>> packed;
 std::vector<std::size_t> offsets;double residual=0;std::size_t bytes=0;
 Optical91(const ClickModel70& c):n(c.model->n),m(c.model->m),original(c.model->v),offsets(m+1){
  for(int j=0;j<m;++j){int k=std::min(n,m-j);offsets[j+1]=offsets[j]+std::size_t(k)*n-std::size_t(k)*(k-1)/2;}
  packed.resize(offsets[m]);bytes=packed.size()*sizeof(packed[0]);
  ComplexMatrix91 R=ComplexMatrix91::Zero(n,n),gram=R;
  for(int j=m-1;j>=0;--j){Eigen::Matrix<std::complex<long double>,1,Eigen::Dynamic> row=c.model->v.row(j).cast<std::complex<long double>>();
   gram+=row.adjoint()*row;int k=std::min(n,m-j);
   for(int a=0;a<k;++a){Eigen::JacobiRotation<std::complex<long double>> rot;std::complex<long double> diagonal;rot.makeGivens(R(a,a),row[a],&diagonal);
    for(int b=a+1;b<n;++b){auto top=R(a,b),bottom=row[b];R(a,b)=std::conj(rot.c())*top-std::conj(rot.s())*bottom;row[b]=rot.s()*top+rot.c()*bottom;}R(a,a)=diagonal;row[a]=0;
   }
   auto at=offsets[j];for(int a=0;a<k;++a)for(int b=a;b<n;++b)packed[at++]=R(a,b);
   if(j==0||j==m-1||j%std::max(1,m/16)==0)residual=std::max(residual,double((R.adjoint()*R-gram).norm()/(1+gram.norm())));
  }
 }
 ComplexMatrix91 map(int j)const{need(j>=0&&j<m,"Optical91 suffix index");int k=std::min(n,m-j);ComplexMatrix91 a=ComplexMatrix91::Zero(k,n);auto at=offsets[j];for(int r=0;r<k;++r)for(int c=r;c<n;++c)a(r,c)=packed[at++];return a;}
};
static Mat82 real_map91(const ComplexMatrix91& a){int k=a.rows(),n=a.cols();Mat82 r(2*k,2*n);r.topLeftCorner(k,n)=a.real();r.topRightCorner(k,n)=-a.imag();r.bottomLeftCorner(k,n)=a.imag();r.bottomRightCorner(k,n)=a.real();return r;}
static bool optical_physical91(const Mat82& x,const ComplexMatrix91& a,long double* diagnostic=nullptr){Mat82 r=real_map91(a),effective=r*x*r.transpose();effective=((effective+effective.transpose())/2).eval();return physical91(effective,diagnostic);}
API int tc91_optical_create(void* handle,void** output,double* stats,char* message,int cap){try{SharedRoundGuard86 guard;need(handle&&output&&stats,"Optical91 create arguments");auto p=std::make_unique<Optical91>(*static_cast<ClickModel70*>(handle));stats[0]=double(p->bytes);stats[1]=p->residual;stats[2]=p->m;*output=p.release();return 0;}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API void tc91_optical_destroy(void* handle){delete static_cast<Optical91*>(handle);}
API int tc91_optical_probe(int n,int rows,const double* covariance,const double* optical,double* stats,char* message,int cap){try{SharedRoundGuard86 guard;need(n>=1&&rows>=1&&covariance&&optical&&stats,"Optical91 probe arguments");Mat82 x(2*n,2*n);ComplexMatrix91 a(rows,n);for(int i=0;i<2*n;++i)for(int j=0;j<2*n;++j)x(i,j)=covariance[i*2*n+j];for(int i=0;i<rows;++i)for(int j=0;j<n;++j)a(i,j)={optical[2*(i*n+j)],optical[2*(i*n+j)+1]};Eigen::HouseholderQR<ComplexMatrix91> qr(a);int k=std::min(n,rows);ComplexMatrix91 b=ComplexMatrix91::Zero(k,n);for(int i=0;i<k;++i)for(int j=i;j<n;++j)b(i,j)=qr.matrixQR()(i,j);long double diag[3];stats[0]=optical_physical91(x,b,diag);for(int i=0;i<3;++i)stats[i+1]=double(diag[i]);stats[4]=double((a.adjoint()*a-b.adjoint()*b).norm());stats[5]=physical91(x);return 0;}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
struct Fit91 {Mat82 base;long double residual=0;int reason=0;};
static Fit91 inverse_click91(const Mat82& target,const Mat82& forms,const ComplexMatrix91* optical=nullptr){
 Fit91 result;Mat82 product=target*forms,B=forms.transpose()*product;B=((B+B.transpose())/2).eval();Eigen::SelfAdjointEigenSolver<Mat82> eig(B);
 if(eig.info()!=Eigen::Success||!(eig.eigenvalues().minCoeff()>0)){result.reason=1;return result;}
 Q82 b0=eig.eigenvalues()[0],b1=eig.eigenvalues()[1],sum=b0+b1+b0*b1,lo=0,hi=sum/(sqrtq(1+sum)+1);
 if(!(hi>0&&finite82(hi))){result.reason=2;return result;}
 auto solve=[](Q82 b,Q82 h){Q82 c=1-b;return 2*b/(sqrtq(c*c+4*(1+1/h)*b)+c);};
 for(int j=0;j<128;++j){Q82 h=(lo+hi)/2;if(h==lo||h==hi)break;Q82 a0=solve(b0,h),a1=solve(b1,h),s=a0+a1+a0*a1,next=s/(sqrtq(1+s)+1);if(next>h)lo=h;else hi=h;}
 Q82 h=(lo+hi)/2,a0=solve(b0,h),a1=solve(b1,h);if(!(h>0&&finite82(h)&&finite82(a0)&&finite82(a1))){result.reason=2;return result;}
 Vec82 coefficients(2);Q82 aa[2]={a0,a1};for(int j=0;j<2;++j){Q82 d=1+aa[j]/(h*(1+aa[j]));coefficients[j]=static_cast<long double>(1/(h*d*d*(1+aa[j])));}
 Mat82 C=eig.eigenvectors()*coefficients.asDiagonal()*eig.eigenvectors().transpose();result.base=target-product*C*product.transpose();result.base=((result.base+result.base.transpose())/2).eval();
 if(!result.base.allFinite()){result.reason=2;return result;}Mat82 husimi=result.base;husimi.diagonal().array()+=1;Eigen::LLT<Mat82> domain(husimi);if(domain.info()!=Eigen::Success){result.reason=3;return result;}
 Mat82 p=result.base*forms,G=forms.transpose()*p;G=((G+G.transpose())/2).eval();G.diagonal().array()+=1;Q82 g00=G(0,0),g11=G(1,1),g01=G(0,1),det=g00*g11-g01*g01;
 if(!(g00>0&&g11>0&&det>1&&finite82(det))){result.reason=3;return result;}Q82 r=1/expm1q(logq(det)/2);Mat82 correction(2,2);correction(0,0)=static_cast<long double>(r*g11/det);correction(0,1)=correction(1,0)=static_cast<long double>(-r*g01/det);correction(1,1)=static_cast<long double>(r*g00/det);Mat82 recovered=result.base+p*correction*p.transpose();result.residual=(recovered-target).norm()/(1+target.norm());if(!(std::isfinite(result.residual)&&result.residual<=1e-11L)){result.reason=4;return result;}if(!(optical?optical_physical91(result.base,*optical):physical91(result.base))){result.reason=5;return result;}return result;
}
struct ProjectionTrace91 {GuardTrace87 decision;double attempts=0,successes=0,rejected=0,max_residual=0;std::array<double,5> reasons{};};
static int project_capacity91(int budget){need(budget>=1&&budget<=128&&(budget&(budget-1))==0,"Project91 power-of-two budget");return 31-__builtin_clz(unsigned(budget));}
static void projected_update91(SharedState86& state,SharedStep86& step,int mode,bool bit,int budget,ProjectionTrace91& trace,const Optical91& optical){
 auto& base=trace.decision.base;if(!bit){shared_update86(state,step,mode,false,budget,base);return;}need(step.p1>Weight83(0),"Project91 impossible click");state.clicks.push_back(mode);base.peak=std::max(base.peak,double(state.clicks.size()));
 if((1U<<state.clicks.size())>unsigned(budget)){
  Mat82 R=step.wide?correction86(*step.wide):correction86(*step.narrow);Mat82 target=state.x-step.products*R*step.products.transpose();target=((target+target.transpose())/2).eval();need(target.allFinite(),"Project91 full moment overflow");++base.merges;
  bool retained=false;if(budget>=2){++trace.attempts;auto suffix=optical.map(mode);auto fit=inverse_click91(target,step.forms.rightCols(2),&suffix);trace.max_residual=std::max(trace.max_residual,double(fit.residual));if(!fit.reason){state.x=std::move(fit.base);state.clicks.assign(1,mode);++trace.successes;retained=true;}else{++trace.rejected;++trace.reasons[fit.reason-1];}}
  if(!retained){state.x=std::move(target);state.clicks.clear();}
 }
 state.x=((state.x+state.x.transpose())/2).eval();need(state.x.allFinite(),"Project91 covariance overflow");
}
API int tc91_project_prefix(void* handle,void* optical_handle,int length,const std::uint8_t* bits,int budget,double* probability,double* stats,char* message,int cap){try{
 SharedRoundGuard86 guard;project_capacity91(budget);need(handle&&bits&&probability&&stats,"Project91 prefix arguments");auto& c=*static_cast<ClickModel70*>(handle);need(optical_handle,"Missing Optical91 cache");auto& optical=*static_cast<Optical91*>(optical_handle);need(optical.n==c.model->n&&optical.m==c.model->m&&(optical.original-c.model->v).norm()==0,"Optical91 model mismatch");need(length>=0&&length<=std::min(16,c.model->m),"Project91 prefix length");SharedState86 state{c.model->initial.cast<long double>(),{}};ProjectionTrace91 trace;Weight83 joint(1);std::fill(stats,stats+10,0);
 for(int mode=0;mode<length;++mode){need(bits[mode]<=1,"Project91 bit");auto step=shared_step86(c,state,mode,trace.decision.base);joint=joint*(bits[mode]?step.p1:step.p0);if(!(joint>Weight83(0)))break;if(mode+1<length)projected_update91(state,step,mode,bits[mode],budget,trace,optical);stats[5]=mode+1;}
 auto& s=trace.decision.base;*probability=double83(joint);stats[0]=s.evaluations;stats[1]=s.merges;stats[2]=s.peak;stats[3]=s.condition;stats[4]=s.wide;stats[6]=trace.attempts;stats[7]=trace.successes;stats[8]=trace.rejected;stats[9]=trace.max_residual;return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int tc91_project_sample(void* handle,void* optical_handle,int count,int threads,int budget,int use_guard,const double* draws,std::uint8_t* bits,double* midpoint,double* bounds,std::uint8_t* kind,int* errors,double* stats,char* message,int cap){try{
 SharedRoundGuard86 guard;project_capacity91(budget);need(handle&&count>=0&&count<=4096&&threads>=1&&threads<=64&&(use_guard==0||use_guard==1)&&draws&&bits&&midpoint&&bounds&&kind&&errors&&stats,"Project91 sample arguments");auto& c=*static_cast<ClickModel70*>(handle);need(optical_handle,"Missing Optical91 cache");auto& optical=*static_cast<Optical91*>(optical_handle);need(optical.n==c.model->n&&optical.m==c.model->m&&(optical.original-c.model->v).norm()==0,"Optical91 model mismatch");int m=c.model->m;std::size_t entries=std::size_t(count)*m;std::fill(bits,bits+entries,0);std::fill(midpoint,midpoint+entries,std::numeric_limits<double>::quiet_NaN());std::fill(bounds,bounds+2*entries,std::numeric_limits<double>::quiet_NaN());std::fill(kind,kind+entries,255);std::fill(errors,errors+2*count,0);std::fill(stats,stats+21*count,0);std::atomic<int> next{0};std::vector<std::string> messages(count);std::vector<std::thread> pool;
 auto worker=[&](){SharedRoundGuard86 precision;for(;;){int row=next.fetch_add(1);if(row>=count)break;int mode=0,stage=1;ProjectionTrace91 trace;try{SharedState86 state{c.model->initial.cast<long double>(),{}};for(;mode<m;++mode){stage=1;auto at=std::size_t(row)*m+mode;double u=draws[at];need(std::isfinite(u)&&u>=0&&u<1,"Project91 explicit uniform");GuardStep87 step;
   if(use_guard)step=guarded_step87(c,state,mode,u,budget,trace.decision);
   else{step.step=shared_step86(c,state,mode,trace.decision.base);step.bit=Weight83(Q82(u))<step.step.p1;step.midpoint=double83(step.step.p1);step.lower=std::nextafter(step.midpoint,-std::numeric_limits<double>::infinity());step.upper=std::nextafter(step.midpoint,std::numeric_limits<double>::infinity());step.kind=step.step.wide?2:0;}
   bits[at]=step.bit;midpoint[at]=step.midpoint;bounds[2*at]=step.lower;bounds[2*at+1]=step.upper;kind[at]=step.kind;if(mode+1<m){stage=2;projected_update91(state,step.step,mode,step.bit,budget,trace,optical);}
  }}catch(const std::exception& e){errors[2*row]=stage;errors[2*row+1]=mode;messages[row]=e.what();}auto& d=trace.decision;auto* s=stats+21*row;s[0]=d.base.evaluations;s[1]=d.base.merges;s[2]=d.base.peak;s[3]=d.base.condition;s[4]=d.base.wide;s[5]=mode;s[6]=d.balls;s[7]=d.certified;s[8]=d.ambiguous;s[9]=d.closure_wide;s[10]=d.ball_failure;s[11]=d.max_width;s[12]=trace.attempts;s[13]=trace.successes;s[14]=trace.rejected;s[15]=trace.max_residual;for(int j=0;j<5;++j)s[16+j]=trace.reasons[j];}};
 try{for(int i=0;i<std::min(count,threads);++i)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[2*row]){error(message,cap,"row="+std::to_string(row)+" mode="+std::to_string(errors[2*row+1])+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int tc91_project_probe(int dimension,const double* target,const double* forms,double* output,double* stats,char* message,int cap){try{
 SharedRoundGuard86 guard;need(dimension>=2&&dimension<=512&&target&&forms&&output&&stats,"Project91 probe arguments");Mat82 T(dimension,dimension),F(dimension,2);for(int i=0;i<dimension;++i){for(int j=0;j<dimension;++j)T(i,j)=target[i*dimension+j];for(int j=0;j<2;++j)F(i,j)=forms[2*i+j];}need(T.allFinite()&&F.allFinite()&&T==T.transpose(),"Project91 finite symmetric probe");auto fit=inverse_click91(T,F);stats[0]=fit.reason;stats[1]=double(fit.residual);if(!fit.reason)for(int i=0;i<dimension;++i)for(int j=0;j<dimension;++j)output[i*dimension+j]=double(fit.base(i,j));return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}

API int tc91_physical_probe(int dimension,const double* covariance,double* output,char* message,int cap){try{
 SharedRoundGuard86 guard;need(dimension>=2&&dimension<=512&&covariance&&output,"Physical91 probe arguments");Mat82 x(dimension,dimension);for(int i=0;i<dimension;++i)for(int j=0;j<dimension;++j)x(i,j)=covariance[i*dimension+j];need(x==x.transpose(),"Physical91 symmetric input");long double details[3];bool accepted=physical91(x,details);output[0]=accepted;for(int j=0;j<3;++j)output[1+j]=double(details[j]);return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}

// Diagnostic export: exact stored values converted to binary64, no hidden cutoff.
API int tc91_optical_export(void* handle,int mode,double* output,char* message,int cap){try{SharedRoundGuard86 guard;need(handle&&output,"Optical91 export arguments");auto a=static_cast<Optical91*>(handle)->map(mode);for(int i=0;i<a.rows();++i)for(int j=0;j<a.cols();++j){auto z=a(i,j);auto p=2*(i*a.cols()+j);output[p]=double(z.real());output[p+1]=double(z.imag());}return 0;}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
