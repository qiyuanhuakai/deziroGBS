"""Generate a new frozen deep-retention core without editing earlier sources."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'native/gaussian_pnr_polynomial73_v1.cpp').read_text(encoding='utf-8')
a=s.index('struct Wick73 {');b=s.index('static Real forms73',a)
s=s[:a]+'#include "gaussian_complex_moments75_v1.hpp"\n'+s[b:]
s=s.replace('retained<=3','retained<=7').replace('length<=4','length<=8')
t=(ROOT/'native/gaussian_pnr_structured74_v1.cpp').read_text(encoding='utf-8')
t=t.replace('#include "gaussian_pnr_polynomial73_v1.cpp"','').replace('retained<=3','retained<=7')
probe='''
API int deep75_moments(int length,const double* mu,const double* covariance,double* result,double* gradient,double* hessian,char* message,int cap){try{
    PrecisionGuard guard;need(length>=0&&length<=16&&length%2==0&&mu&&covariance&&result&&gradient&&hessian,"Invalid deep moment probe");LVec73 mean(length);LMat73 cov(length,length);
    for(int i=0;i<length;++i){mean[i]=mu[i];for(int j=0;j<length;++j)cov(i,j)=covariance[i*length+j];}
    auto r=polynomial73(mean,cov,true);*result=double(r.value);for(int i=0;i<length;++i){gradient[i]=double(r.gradient[i]);for(int j=0;j<length;++j)hessian[i*length+j]=double(r.hessian(i,j));}return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
'''
with (ROOT/'native/gaussian_pnr_deep75_v1.cpp').open('xb') as f:f.write(('// Iteration75: shared complex moments, retained factors 0..7.\n'+s+'\n'+t+probe).encode())
