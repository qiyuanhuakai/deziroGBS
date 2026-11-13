"""Freeze a second core using one positive QR norm factor for all detectors.
The first executed projected kernel and all its evidence remain unchanged.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'native/gaussian_projected76_v1.hpp').read_text(encoding='utf-8')
factor='''
static CMatrix74 gram_factor76(const Hermite75& p,int rank){
    std::vector<Hermite75> basis;basis.push_back(p);for(int i=0;i<rank;++i)basis.push_back(annihilate_active76(p,rank,i));
    for(int i=0;i<rank;++i){CVector74 up=CVector74::Zero(rank),zero=up;up[i]=1;basis.push_back(act75(p,0,0,zero,up,zero,zero));}
    std::unordered_map<std::string,int> index;for(auto& a:basis)for(auto& [key,h]:a)if(index.find(key)==index.end()){int row=index.size();index.emplace(key,row);}
    int rows=index.size(),cols=basis.size();need(rows>0&&std::size_t(rows)*cols<=4000000,"Projected Gram workspace bound");CMatrix74 matrix=CMatrix74::Zero(rows,cols);
    for(int j=0;j<cols;++j)for(auto& [key,h]:basis[j]){LD73 weight=1;for(unsigned char k:key)for(int a=2;a<=k;++a)weight*=a;matrix(index.at(key),j)=std::sqrt(weight)*h.value;}
    Eigen::HouseholderQR<CMatrix74> qr(matrix);CMatrix74 upper=qr.matrixQR().topRows(std::min(rows,cols)).template triangularView<Eigen::Upper>();need(upper.allFinite(),"Projected Gram QR overflow");return upper;
}
'''
at=s.index('static void moments76');s=s[:at]+factor+s[at:]
a=s.index('std::vector<Hermite75> acted;',s.index('static void moments76'));b=s.index('    // Physical b',a)
s=s[:a]+'''CMatrix74 factor=gram_factor76(p,r),gram=factor.adjoint()*factor;LVec73 mu(2*r);LMat73 x=LMat73::Zero(2*r,2*r);
    for(int i=0;i<r;++i){auto z=gram(0,1+i)/norm;mu[i]=std::sqrt(2.L)*z.real();mu[i+r]=std::sqrt(2.L)*z.imag();}
    for(int i=0;i<r;++i)for(int j=0;j<r;++j){auto normal=gram(1+i,1+j)/norm,anomalous=gram(1+r+i,1+j)/norm;x(i,j)=normal.real()+anomalous.real()-mu[i]*mu[j];x(r+i,r+j)=normal.real()-anomalous.real()-mu[r+i]*mu[r+j];x(i,r+j)=anomalous.imag()+normal.imag()-mu[i]*mu[r+j];x(r+j,i)=x(i,r+j);}
'''+s[b:]
s=s.replace('std::vector<LD73> rates;CVector74 zero=CVector74::Zero(q.cols());','std::vector<LD73> rates;int rank=q.cols();CMatrix74 factor=gram_factor76(p,rank);')
a=s.index('auto acted=act75(p,alpha/std::sqrt(2.L)',s.index('static std::vector<LD73> rates76'));b=s.index('need(std::isfinite(rate)',a)
s=s[:a]+'''CVector74 coefficients(1+2*rank);coefficients[0]=alpha/std::sqrt(2.L);coefficients.segment(1,rank)=l;coefficients.segment(1+rank,rank)=u;CMatrix74 residual=up-(up*q)*q.adjoint();LD73 rate=(factor*coefficients).squaredNorm()+residual.squaredNorm()*norm;'''+s[b:]
with (ROOT/'native/gaussian_projected76_v2.hpp').open('xb') as f:f.write(s.encode())
s=(ROOT/'native/gaussian_pnr_projected76_v1.cpp').read_text(encoding='utf-8').replace('gaussian_projected76_v1.hpp','gaussian_projected76_v2.hpp')
with (ROOT/'native/gaussian_pnr_projected76_v2.cpp').open('xb') as f:f.write(s.encode())
s=(ROOT/'experiments/iteration76_projected_pilot_v1.py').read_text(encoding='utf-8').replace('gaussian_pnr_projected76_v1.cpp','gaussian_pnr_projected76_v2.cpp').replace('gaussian_projected76_v1.hpp','gaussian_projected76_v2.hpp').replace('Active-span moments, tangents, Gaussian compression and detector-rate checks.','Second core: positive QR norm shared by all detectors and compression moments.')
with (ROOT/'experiments/iteration76_projected_pilot_v2.py').open('xb') as f:f.write(s.encode())
