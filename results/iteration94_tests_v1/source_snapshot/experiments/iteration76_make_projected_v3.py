"""Use the full thin QR span, with no numerical-rank truncation.
Equal absolute squeezing makes every upper tangent a common scalar multiple,
so its columns need not be repeated. This is an exact identity before rounding.
Version2 remains an unexecuted QR-Gram draft; version1 remains the tested baseline.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'native/gaussian_projected76_v2.hpp').read_text(encoding='utf-8')
a=s.index('Projected76 out;CMatrix74 columns');b=s.index('    capacity76(rank,k);',a)
s=s[:a]+'''Projected76 out;bool equal_abs=true;for(int j=1;j<n;++j)equal_abs=equal_abs&&(std::abs(t[j])==std::abs(t[0]));bool extra=tangent&&!equal_abs&&tau!=0;
    CMatrix74 columns(n,extra?2*k:k);if(k){columns.leftCols(k)=up.adjoint();if(extra)columns.rightCols(k)=upd.adjoint();}
    for(int j=0;j<columns.cols();++j){LD73 norm=columns.col(j).norm();if(norm>0)columns.col(j)/=norm;}
    int rank=std::min<int>(n,columns.cols());if(rank){Eigen::HouseholderQR<CMatrix74> qr(columns);out.q=qr.householderQ()*CMatrix74::Identity(n,rank);out.residual=(columns-out.q*(out.q.adjoint()*columns)).norm();need(out.residual<=1e-14L*std::max<LD73>(1,columns.norm()),"Projected Hermite tangent span residual");need((out.q.adjoint()*out.q-CMatrix74::Identity(rank,rank)).norm()<1e-14L*std::max(1,rank),"Projected Hermite basis orthogonality");}else out.q=CMatrix74(n,0);
'''+s[b:]
with (ROOT/'native/gaussian_projected76_v3.hpp').open('xb') as f:f.write(s.encode())
s=(ROOT/'native/gaussian_pnr_projected76_v2.cpp').read_text(encoding='utf-8').replace('gaussian_projected76_v2.hpp','gaussian_projected76_v3.hpp')
with (ROOT/'native/gaussian_pnr_projected76_v3.cpp').open('xb') as f:f.write(s.encode())
s=(ROOT/'experiments/iteration76_projected_pilot_v2.py').read_text(encoding='utf-8').replace('gaussian_pnr_projected76_v2.cpp','gaussian_pnr_projected76_v3.cpp').replace('gaussian_projected76_v2.hpp','gaussian_projected76_v3.hpp').replace('Second core: positive QR norm shared by all detectors and compression moments.','Full thin span without numerical-rank truncation; positive QR norm shared by detectors.')
with (ROOT/'experiments/iteration76_projected_pilot_v3.py').open('xb') as f:f.write(s.encode())
