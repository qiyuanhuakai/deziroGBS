// Active creation-coordinate Hermite representation. The tangent span is
// included at each evaluation; Q is held fixed for that local derivative.
struct Projected76 { CMatrix74 q; Hermite75 polynomial; LD73 residual=0; };
static void capacity76(int rank,int degree){LD73 estimate=1;for(int j=1;j<=degree+2;++j)estimate*=LD73(rank+j)/j;need(estimate*(rank+3)<=2000000,"Projected prefix polynomial resource estimate exceeds limit");}
static Projected76 project76(const ClickModel70& c,const Vector& t,const LVec73& mean,const LVec73& meand,const std::vector<int>& modes,LD73 tau,bool tangent){
    int n=t.size(),k=modes.size();CMatrix74 low(k,n),up(k,n),lowd(k,n),upd(k,n);CVector74 alpha(k),alphad(k);
    for(int a=0;a<k;++a){int mode=modes[a];for(int j=0;j<n;++j){LD73 tj=t[j],den=1-tau*tau*tj*tj,f=1/std::sqrt(den),fd=tangent?-tau*tj*tj*f/den:0;auto z=c.model->v(mode,j);Z75 v(z.real(),z.imag());low(a,j)=v*f;up(a,j)=-v*tj*f;lowd(a,j)=v*fd;upd(a,j)=-v*tj*fd;}
        alpha[a]=Z75(c.a.col(mode).cast<LD73>().dot(mean),c.b.col(mode).cast<LD73>().dot(mean))/std::sqrt(2.L);alphad[a]=tangent?Z75(c.a.col(mode).cast<LD73>().dot(meand),c.b.col(mode).cast<LD73>().dot(meand))/std::sqrt(2.L):Z75(0);}
    Projected76 out;CMatrix74 columns(n,tangent?2*k:k);if(k){columns.leftCols(k)=up.adjoint();if(tangent)columns.rightCols(k)=upd.adjoint();}
    for(int j=0;j<columns.cols();++j){LD73 norm=columns.col(j).norm();if(norm>0)columns.col(j)/=norm;}
    int rank=0;if(columns.cols()){Eigen::ColPivHouseholderQR<CMatrix74> qr(columns);qr.setThreshold(128*std::numeric_limits<LD73>::epsilon());rank=qr.rank();out.q=qr.householderQ()*CMatrix74::Identity(n,rank);out.residual=(columns-out.q*(out.q.adjoint()*columns)).norm();need(out.residual<=1e-14L*std::max<LD73>(1,columns.norm()),"Projected Hermite tangent span residual");need((out.q.adjoint()*out.q-CMatrix74::Identity(rank,rank)).norm()<1e-14L*std::max(1,rank),"Projected Hermite basis orthogonality");}else out.q=CMatrix74(n,0);
    capacity76(rank,k);out.polynomial[std::string(rank,'\0')].value=1;
    for(int a=0;a<k;++a){CVector74 l=(low.row(a)*out.q.conjugate()).transpose(),u=(up.row(a)*out.q).transpose(),ld=(lowd.row(a)*out.q.conjugate()).transpose(),ud=(upd.row(a)*out.q).transpose();out.polynomial=act75(out.polynomial,alpha[a],alphad[a],l,u,ld,ud);}
    return out;
}
static Hermite75 annihilate_active76(const Hermite75& p,int rank,int index){CVector74 low=CVector74::Zero(rank),zero=low;low[index]=1;return act75(p,0,0,low,zero,zero,zero);}
static void moments76(const ClickModel70& c,const State73& state,const Vector& t,Real& covariance,Vector& center){
    int n=t.size();auto projected=project76(c,t,state.mean.cast<LD73>(),LVec73::Zero(2*n),state.factors,1,false);auto& p=projected.polynomial;int r=projected.q.cols();LD73 norm=norm75(p,1).first;need(norm>0,"Projected compression normalization zero");std::vector<Hermite75> acted;LVec73 mu(2*r);LMat73 x=LMat73::Zero(2*r,2*r);
    for(int i=0;i<r;++i){acted.push_back(annihilate_active76(p,r,i));auto z=inner75(p,acted.back())/norm;mu[i]=std::sqrt(2.L)*z.real();mu[i+r]=std::sqrt(2.L)*z.imag();}
    for(int i=0;i<r;++i)for(int j=0;j<r;++j){auto normal=inner75(acted[i],acted[j])/norm,anomalous=inner75(p,annihilate_active76(acted[j],r,i))/norm;x(i,j)=normal.real()+anomalous.real()-mu[i]*mu[j];x(r+i,r+j)=normal.real()-anomalous.real()-mu[r+i]*mu[r+j];x(i,r+j)=anomalous.imag()+normal.imag()-mu[i]*mu[r+j];x(r+j,i)=x(i,r+j);}
    // Physical b = conjugate(Q) b_active + orthogonal vacuum. Follow by
    // diagonal pure squeezing and the original phase-space displacement.
    CMatrix74 w=projected.q.conjugate();LMat73 map(2*n,2*r);map.topLeftCorner(n,r)=w.real();map.topRightCorner(n,r)=-w.imag();map.bottomLeftCorner(n,r)=w.imag();map.bottomRightCorner(n,r)=w.real();LMat73 full=LMat73::Zero(2*n,2*n);
    for(int i=0;i<n;++i){LD73 tj=t[i],s=std::sqrt((1-tj)/(1+tj));map.row(i)*=s;map.row(i+n)/=s;full(i,i)=-tj/(1+tj);full(i+n,i+n)=tj/(1-tj);}
    full.noalias()+=map*x*map.transpose();center=(state.mean.cast<LD73>()+map*mu).cast<double>();covariance=((full+full.transpose())/2).cast<double>();need(covariance.allFinite()&&center.allFinite(),"Projected compression overflow");
}
static void compress_projected76(const ClickModel70& c,State73& state,const Vector& t,Stats73& stats){Real x;Vector mean;moments76(c,state,t,x,mean);state.x=std::move(x);state.mean=std::move(mean);state.factors.clear();++stats.compressions;}
static std::vector<LD73> rates76(const ClickModel70& c,const State73& state,const Vector& t){
    int n=t.size();auto projected=project76(c,t,state.mean.cast<LD73>(),LVec73::Zero(2*n),state.factors,1,false);auto& p=projected.polynomial;auto& q=projected.q;LD73 norm=norm75(p,1).first;std::vector<LD73> rates;CVector74 zero=CVector74::Zero(q.cols());
    for(int mode=0;mode<c.model->m;++mode){CMatrix74 low(1,n),up(1,n);for(int j=0;j<n;++j){LD73 tj=t[j],f=1/std::sqrt(1-tj*tj);auto z=c.model->v(mode,j);Z75 v(z.real(),z.imag());low(0,j)=v*f;up(0,j)=-v*tj*f;}CVector74 l=(low*q.conjugate()).transpose(),u=(up*q).transpose();Z75 alpha(c.a.col(mode).cast<LD73>().dot(state.mean.cast<LD73>()),c.b.col(mode).cast<LD73>().dot(state.mean.cast<LD73>()));auto acted=act75(p,alpha/std::sqrt(2.L),0,l,u,zero,zero);CMatrix74 residual=up-(up*q)*q.adjoint();LD73 rate=norm75(acted,1).first+residual.squaredNorm()*norm;need(std::isfinite(rate)&&rate>=0,"Projected detector rate invalid");rates.push_back(rate);}
    return rates;
}
static int detector76(const ClickModel70& c,const State73& state,const Vector& t,double uniform){auto rates=rates76(c,state,t);std::vector<LD73> cumulative;LD73 total=0;for(auto rate:rates){LD73 next=total+rate;need(!(rate>0&&next<=total),"Projected detector interval lost");total=next;cumulative.push_back(total);}return categorical(cumulative,total,uniform);}
