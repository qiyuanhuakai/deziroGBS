"""Remaining-number pruning is exact: a degree above R cannot return below R
when each annihilation lowers degree by at most one and R decreases by one.
Use the QR triangular structure, never a numerical coefficient threshold.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'native/gaussian_number77_v1.hpp').read_text(encoding='utf-8')
prefix='''
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
    out.polynomial[std::string(rank,'\\0')].value=1;
    for(int i=0;i<k;++i){int active=n<=8?n:std::min(rank,i+1),degree=std::min(i+1,total-i-1);LD73 size=1;for(int j=1;j<=degree;++j)size*=LD73(active+j)/j;need(size<=200000,"Conditional intermediate polynomial resource bound");CVector74 l=lower.row(i).transpose(),u=upper.row(i).transpose();out.polynomial=act77(out.polynomial,l,u,total-i-1);}
    return out;
}
'''
s=prefix+s
s=s.replace('for(int i=0;i<rank;++i)basis.push_back(annihilate_active76(p,rank,i));','for(int i=0;i<rank;++i){CVector74 low=CVector74::Zero(rank),zero=low;low[i]=1;basis.push_back(act77(p,low,zero,int(w.values.size())-1));}')
s=s.replace('basis.push_back(act75(p,0,0,zero,up,zero,zero));','basis.push_back(act77(p,zero,up,int(w.values.size())-1));')
s=s.replace('Vector squeezing=Vector::Constant(n,t);auto projected=project76(c,squeezing,LVec73::Zero(2*n),LVec73::Zero(2*n),modes,0,false);','auto projected=project77(c,t,total,modes);')
s=s.replace('total-k-1,t,k+1','total-k-1,t,std::min(k+1,total-k-1)').replace('total-k,t,k)','total-k,t,std::min(k,total-k))')
with (ROOT/'native/gaussian_number77_v2.hpp').open('xb') as f:f.write(s.encode())
s=(ROOT/'native/gaussian_pnr_number77_v1.cpp').read_text(encoding='utf-8').replace('gaussian_number77_v1.hpp','gaussian_number77_v2.hpp').replace('need(photons<=249,"Number-conditioned degree storage resource bound; no resampling");capacity76(std::min<std::uint64_t>(n,photons),photons);','number_capacity77(n,photons);')
# Share the exact first-port row-norm CDF within the batch. The rounding-up
# proposal envelope in Model::cdf is intentionally not reused as a probability.
s=s.replace('int workers=std::min(count,threads);','std::vector<LD73> first_cdf;LD73 first_total=0;for(int i=0;i<m;++i){LD73 value=0;for(int j=0;j<n;++j){auto z=c.model->v(i,j);value+=LD73(z.real())*z.real()+LD73(z.imag())*z.imag();}first_total+=value;first_cdf.push_back(first_total);}int workers=std::min(count,threads);')
old='for(std::uint64_t k=0;k<photons;++k){auto r=rates77(c,t,photons,modes);'
new='for(std::uint64_t k=0;k<photons;++k){if(k==0){int mode=categorical(first_cdf,first_total,uniform(detector));modes.push_back(mode);++output[std::size_t(row)*m+mode];trace[4]+=1;trace[5]=std::max(trace[5],1.);trace[6]=std::max(trace[6],1.);continue;}auto r=rates77(c,t,photons,modes);'
assert old in s;s=s.replace(old,new)
with (ROOT/'native/gaussian_pnr_number77_v2.cpp').open('xb') as f:f.write(s.encode())
s=(ROOT/'experiments/iteration77_number_pilot_v1.py').read_text(encoding='utf-8').replace('gaussian_pnr_number77_v1.cpp','gaussian_pnr_number77_v2.cpp').replace('gaussian_number77_v1.hpp','gaussian_number77_v2.hpp').replace('Independent Fock coefficient checks for total-number-conditioned pure sampling.','Remaining-number pruning: independent Fock coefficient checks for pure sampling.')
with (ROOT/'experiments/iteration77_number_pilot_v2.py').open('xb') as f:f.write(s.encode())
