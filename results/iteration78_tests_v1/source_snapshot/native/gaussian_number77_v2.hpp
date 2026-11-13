
static void number_capacity77(int n,int total){
    need(total<=498,"Number-conditioned degree storage resource bound; no resampling");
    for(int step=0;step<=total;++step){int rank=n<=8?n:std::min(n,step),degree=std::min(step,total-step);LD73 size=1;for(int j=1;j<=degree;++j)size*=LD73(rank+j)/j;need(size<=200000,"Number-conditioned polynomial resource estimate exceeds limit; no resampling");}
}
static Hermite75 act77(const Hermite75& p,const CVector74& lower,const CVector74& upper,int maximum){
    Hermite75 out;int rank=lower.size();for(auto& [key,h]:p){int sum=0;for(unsigned char a:key)sum+=a;
        for(int j=0;j<rank;++j){int degree=static_cast<unsigned char>(key[j]);if(sum<maximum&&upper[j]!=Z75(0)){need(degree<250,"Conditional degree storage overflow");auto next=key;++next[j];out[next].value+=upper[j]*h.value;}if(degree&&sum-1<=maximum&&lower[j]!=Z75(0)){auto next=key;--next[j];out[next].value+=LD73(degree)*lower[j]*h.value;}}
        need(out.size()<=200000,"Conditional polynomial workspace exceeded");
    }return out;
}
static Projected76 project77(const ClickModel70& c,double t,int total,const std::vector<int>& modes){
    int n=c.model->n,k=modes.size(),rank=n<=8?n:std::min(n,k);Projected76 out;CMatrix74 selected(k,n);for(int i=0;i<k;++i)selected.row(i)=c.model->v.row(modes[i]).cast<Complex74>();CMatrix74 lower,upper;
    if(n<=8){out.q=CMatrix74::Identity(n,n);lower=selected;upper=-LD73(t)*selected;}
    else if(rank){CMatrix74 columns=selected.adjoint();Eigen::HouseholderQR<CMatrix74> qr(columns);out.q=qr.householderQ()*CMatrix74::Identity(n,rank);CMatrix74 triangular=qr.matrixQR().topRows(rank).template triangularView<Eigen::Upper>();lower=selected*out.q.conjugate();upper=-LD73(t)*triangular.adjoint();out.residual=(columns-out.q*triangular).norm();need(out.residual<1e-14L*std::max<LD73>(1,columns.norm()),"Conditional QR reconstruction residual");need((out.q.adjoint()*out.q-CMatrix74::Identity(rank,rank)).norm()<1e-14L*rank,"Conditional QR orthogonality");}
    else{out.q=CMatrix74(n,0);lower=CMatrix74(k,0);upper=CMatrix74(k,0);}
    out.polynomial[std::string(rank,'\0')].value=1;
    for(int i=0;i<k;++i){int active=n<=8?n:std::min(rank,i+1),degree=std::min(i+1,total-i-1);LD73 size=1;for(int j=1;j<=degree;++j)size*=LD73(active+j)/j;need(size<=200000,"Conditional intermediate polynomial resource bound");CVector74 l=lower.row(i).transpose(),u=upper.row(i).transpose();out.polynomial=act77(out.polynomial,l,u,total-i-1);}
    return out;
}
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
    std::vector<Hermite75> basis;for(int i=0;i<rank;++i){CVector74 low=CVector74::Zero(rank),zero=low;low[i]=1;basis.push_back(act77(p,low,zero,int(w.values.size())-1));}for(int i=0;i<rank;++i){CVector74 up=CVector74::Zero(rank),zero=up;up[i]=1;basis.push_back(act77(p,zero,up,int(w.values.size())-1));}
    if(!rank)return CMatrix74(0,0);std::unordered_map<std::string,int> index;for(auto& a:basis)for(auto& [key,h]:a){int d=degree77(key);if(d<int(w.values.size())&&w.values[d]>0&&index.find(key)==index.end()){int row=index.size();index.emplace(key,row);}}
    int rows=index.size(),cols=basis.size();if(!rows)return CMatrix74::Zero(0,cols);need(std::size_t(rows)*cols<=4000000,"Number-conditioned Gram resource bound");CMatrix74 matrix=CMatrix74::Zero(rows,cols);
    for(int j=0;j<cols;++j)for(auto& [key,h]:basis[j]){auto at=index.find(key);if(at!=index.end())matrix(at->second,j)=std::sqrt(factorial77(key)*w.values[degree77(key)])*h.value;}
    Eigen::HouseholderQR<CMatrix74> qr(matrix);CMatrix74 upper=qr.matrixQR().topRows(std::min(rows,cols)).template triangularView<Eigen::Upper>();need(upper.allFinite(),"Number-conditioned QR overflow");return upper;
}
struct Rates77 {std::vector<LD73> rates;LD73 logscale=0;int terms=0,rank=0;};
static Rates77 rates77(const ClickModel70& c,double t,int total,const std::vector<int>& modes){
    int n=c.model->n,m=c.model->m,k=modes.size();need(total>k,"No remaining photon for detector choice");auto projected=project77(c,t,total,modes);auto& p=projected.polynomial;auto& q=projected.q;int rank=q.cols();auto weights=weights77(n,k+1,total-k-1,t,std::min(k+1,total-k-1));auto factor=weighted_factor77(p,rank,weights);LD73 outside=weighted_norm77(p,weights,1);CMatrix74 optics=c.model->v.cast<Complex74>(),lower=optics*q.conjugate(),upper=(-LD73(t)*optics)*q,coefficients(2*rank,m);coefficients.topRows(rank)=lower.transpose();coefficients.bottomRows(rank)=upper.transpose();CMatrix74 values=factor*coefficients;CMatrix74 residual=-LD73(t)*optics-upper*q.adjoint();Rates77 out;out.logscale=weights.logscale;out.terms=p.size();out.rank=rank;
    for(int mode=0;mode<m;++mode){LD73 rate=values.col(mode).squaredNorm()+residual.row(mode).squaredNorm()*outside;need(std::isfinite(rate)&&rate>=0,"Invalid conditional detector rate");out.rates.push_back(rate);}return out;
}
static LD73 coefficient77(const ClickModel70& c,double t,int total,const std::vector<int>& modes){int n=c.model->n,k=modes.size();if(total<k)return 0;auto projected=project77(c,t,total,modes);auto weights=weights77(n,k,total-k,t,std::min(k,total-k));LD73 mass=weighted_norm77(projected.polynomial,weights);return mass*std::exp(weights.logscale+LD73(n)/2*std::log1p(-LD73(t)*t));}
