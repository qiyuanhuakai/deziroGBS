"""Generate exact pure prefix followed by bounded Gaussian closure, without output cutoff."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'native/gaussian_pnr_structured74_v1.cpp').read_text(encoding='utf-8');s=s[s.index('API int pnrs74_sample'):].replace('pnrs74_sample','pnrp75_sample').replace('retained<=3','retained<=31').replace('pure_flow74','hermite_flow75').replace('pure_detector74','hermite_detector75')
s=s.replace('if(state.factors.size()>std::size_t(retained)){compress73(c,state,trace);pure=false;}','if(state.factors.size()>std::size_t(pure?retained:3)){if(pure)compress_hermite75(c,state,current_t,trace);else compress73(c,state,trace);pure=false;}')
s=s.replace('Vector center(d),deviation(d),initial_t(d/2);','LD73 capacity=1;for(int j=1;j<=retained+3;++j)capacity*=LD73(d/2+j)/j;need(capacity*(d/2+3)<=2000000,"Prefix polynomial resource estimate exceeds limit");Vector center(d),deviation(d),initial_t(d/2);')
probe='''
API int prefix75_probe(void* handle,const double* squeezing,const double* mean,const double* meanprime,int length,const int* modes,double tau,double* mass,double* derivative,double* covariance,double* nextmean,char* message,int cap){try{
    PrecisionGuard guard;need(handle&&squeezing&&mean&&meanprime&&modes&&mass&&derivative&&covariance&&nextmean&&length>=0&&length<=32&&tau>=0&&tau<=1,"Invalid prefix probe");auto& c=*static_cast<ClickModel70*>(handle);int n=c.model->n;Vector t(n);LVec73 mu(2*n),md(2*n);std::vector<int> ids;for(int i=0;i<n;++i)t[i]=squeezing[i];for(int i=0;i<2*n;++i){mu[i]=mean[i];md[i]=meanprime[i];}for(int j=0;j<length;++j){need(modes[j]>=0&&modes[j]<c.model->m,"Prefix mode index");ids.push_back(modes[j]);}auto p=hermite75(c,t,mu,md,ids,tau);auto r=norm75(p,tau);*mass=double(r.first);*derivative=double(r.second);
    if(tau==1&&r.first>0){State73 state{Real::Zero(2*n,2*n),mu.cast<double>(),ids};Stats73 stats;compress_hermite75(c,state,t,stats);for(int i=0;i<2*n;++i){nextmean[i]=state.mean[i];for(int j=0;j<2*n;++j)covariance[i*2*n+j]=state.x(i,j);}}return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
'''
with (ROOT/'native/gaussian_pnr_prefix75_v1.cpp').open('xb') as f:f.write(('#include "gaussian_pnr_deep75_v1.cpp"\n#include "gaussian_hermite75_v1.hpp"\n'+s+probe).encode())
s=(ROOT/'src/gbslab/native_structured_pnr.py').read_text(encoding='utf-8');s=s[s.index('    def sample('):].replace('pnrs74_sample','pnrp75_sample')
header='''"""Exact pure prefix up to first compression; later closure remains approximate.
Retention is not an output photon cutoff. Sparse polynomial workspace is bounded.
"""
import ctypes as C
import time
import numpy as np
from .native_structured_pnr import NativeStructuredPNRSampler
from .native_polynomial_pnr import DP,UP,STATS
class NativePrefixPNRSampler(NativeStructuredPNRSampler):
    def __init__(self,*args,retained=15,**kwargs):
        if isinstance(retained,bool) or not isinstance(retained,int) or not 0<=retained<=31:raise ValueError('retained must be an integer in [0,31]')
        super().__init__(*args,retained=0,**kwargs);self.retained=retained
        self.lib.pnrp75_sample.argtypes=[C.c_void_p,DP,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
        self.lib.pnrp75_sample.restype=C.c_int
'''
with (ROOT/'src/gbslab/native_prefix_pnr.py').open('xb') as f:f.write((header+s).encode())
