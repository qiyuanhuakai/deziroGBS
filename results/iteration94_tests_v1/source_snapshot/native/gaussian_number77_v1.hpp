// Equal positive squeezing, zero displacement: fixed-number positive coefficients.
struct Weights77 {std::vector<LD73> values;LD73 logscale=0;};
static int degree77(const std::string& key){int d=0;for(unsigned char k:key)d+=k;return d;}
static LD73 factorial77(const std::string& key){LD73 f=1;for(unsigned char k:key)for(int j=2;j<=k;++j)f*=j;return f;}
static Weights77 weights77(int n,int k,int remaining,LD73 t,int degree){
    Weights77 out;out.values.assign(degree+1,0);std::vector<LD73> logs(degree+1,-std::numeric_limits<LD73>::infinity());LD73 largest=-std::numeric_limits<LD73>::infinity(),shape=LD73(n)/2+k;
    for(int d=0;d<=degree;++d)if(remaining>=d&&(remaining-d)%2==0){int pairs=(remaining-d)/2;if(t==0&&pairs)continue;LD73 value=(pairs?2*pairs*std::log(t):0)+std::lgamma(shape+pairs)-std::lgamma(shape)-std::lgamma(LD73(pairs)+1);logs[d]=value;largest=std::max(largest,value);}
    if(!std::isfinite(largest))return out;out.logscale=largest;for(int d=0;d<=degree;++d)if(std::isfinite(logs[d]))out.values[d]=std::exp(logs[d]-largest);return out;
}
static LD73 weighted_norm77(const Hermite75& p,const Weights77& w,int shift=0){LD73 total=0;for(auto& [key,h]:p){int d=degree77(key)+shift;if(d<int(w.values.size()))total+=factorial77(key)*std::norm(h.value)*w.values[d];}need(std::isfinite(total)&&total>=0,"Number-conditioned norm overflow");return total;}
static CMatrix74 weighted_factor77(const Hermite75& p,int rank,const Weights77& w){
    std::vector<Hermite75> basis;for(int i=0;i<rank;++i)basis.push_back(annihilate_active76(p,rank,i));for(int i=0;i<rank;++i){CVector74 up=CVector74::Zero(rank),zero=up;up[i]=1;basis.push_back(act75(p,0,0,zero,up,zero,zero));}
    if(!rank)return CMatrix74(0,0);std::unordered_map<std::string,int> index;for(auto& a:basis)for(auto& [key,h]:a){int d=degree77(key);if(d<int(w.values.size())&&w.values[d]>0&&index.find(key)==index.end()){int row=index.size();index.emplace(key,row);}}
    int rows=index.size(),cols=basis.size();if(!rows)return CMatrix74::Zero(0,cols);need(std::size_t(rows)*cols<=4000000,"Number-conditioned Gram resource bound");CMatrix74 matrix=CMatrix74::Zero(rows,cols);
    for(int j=0;j<cols;++j)for(auto& [key,h]:basis[j]){auto at=index.find(key);if(at!=index.end())matrix(at->second,j)=std::sqrt(factorial77(key)*w.values[degree77(key)])*h.value;}
    Eigen::HouseholderQR<CMatrix74> qr(matrix);CMatrix74 upper=qr.matrixQR().topRows(std::min(rows,cols)).template triangularView<Eigen::Upper>();need(upper.allFinite(),"Number-conditioned QR overflow");return upper;
}
struct Rates77 {std::vector<LD73> rates;LD73 logscale=0;int terms=0,rank=0;};
static Rates77 rates77(const ClickModel70& c,double t,int total,const std::vector<int>& modes){
    int n=c.model->n,m=c.model->m,k=modes.size();need(total>k,"No remaining photon for detector choice");Vector squeezing=Vector::Constant(n,t);auto projected=project76(c,squeezing,LVec73::Zero(2*n),LVec73::Zero(2*n),modes,0,false);auto& p=projected.polynomial;auto& q=projected.q;int rank=q.cols();auto weights=weights77(n,k+1,total-k-1,t,k+1);auto factor=weighted_factor77(p,rank,weights);LD73 outside=weighted_norm77(p,weights,1);CMatrix74 optics=c.model->v.cast<Complex74>(),lower=optics*q.conjugate(),upper=(-LD73(t)*optics)*q,coefficients(2*rank,m);coefficients.topRows(rank)=lower.transpose();coefficients.bottomRows(rank)=upper.transpose();CMatrix74 values=factor*coefficients;CMatrix74 residual=-LD73(t)*optics-upper*q.adjoint();Rates77 out;out.logscale=weights.logscale;out.terms=p.size();out.rank=rank;
    for(int mode=0;mode<m;++mode){LD73 rate=values.col(mode).squaredNorm()+residual.row(mode).squaredNorm()*outside;need(std::isfinite(rate)&&rate>=0,"Invalid conditional detector rate");out.rates.push_back(rate);}return out;
}
static LD73 coefficient77(const ClickModel70& c,double t,int total,const std::vector<int>& modes){int n=c.model->n,k=modes.size();if(total<k)return 0;Vector squeezing=Vector::Constant(n,t);auto projected=project76(c,squeezing,LVec73::Zero(2*n),LVec73::Zero(2*n),modes,0,false);auto weights=weights77(n,k,total-k,t,k);LD73 mass=weighted_norm77(projected.polynomial,weights);return mass*std::exp(weights.logscale+LD73(n)/2*std::log1p(-LD73(t)*t));}
