"""Create experimental repeated pure-plus-noise decomposition after compression."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'native/gaussian_pnr_structured74_v1.cpp').read_text(encoding='utf-8');s=s[s.index('API int pnrs74_sample'):]
s=s.replace('pnrs74_sample','pnrr75_sample').replace('retained<=3','retained<=7')
s=s.replace('auto& c=*static_cast<ClickModel70*>(handle);int d=2*c.model->n,m=c.model->m;','auto& original=*static_cast<ClickModel70*>(handle);int d=2*original.model->n,m=original.model->m;')
s=s.replace('stats+9','stats+12').replace('std::vector<Stats73> local(workers);','std::vector<Stats73> local(workers);std::vector<ResplitStats75> splits(workers);')
s=s.replace('if(row>=count)break;','if(row>=count)break;ClickModel70 c;c.model=std::make_unique<Model>();c.model->n=original.model->n;c.model->m=original.model->m;c.model->v=original.model->v;c.a=original.a;c.b=original.b;std::mt19937_64 splitting(mix(seed+0x5a74e910cd8236b1ULL+row));')
s=s.replace('compress73(c,state,trace);pure=false;','compress73(c,state,trace);pure=resplit75(c,state,current_t,splitting,splits[id]);')
s=s.replace('for(auto& e:failures)if(e)', 'for(auto& r:splits){stats[9]+=r.attempts;stats[10]+=r.successes;stats[11]=std::max(stats[11],r.residual);}for(auto& e:failures)if(e)')
probe='''
API int split75_probe(int n,const double* input,double* rotation,double* t,double* noise,double* residual,char* message,int cap){try{
    PrecisionGuard guard;need(n>0&&n<=128&&input&&rotation&&t&&noise&&residual,"Invalid split probe");int d=2*n;Real x(d,d);for(int i=0;i<d;++i)for(int j=0;j<d;++j)x(i,j)=input[i*d+j];auto r=decompose75(x);if(!r.valid)return 2;
    for(int i=0;i<d;++i)for(int j=0;j<d;++j){rotation[i*d+j]=r.rotation(i,j);noise[i*d+j]=r.noise_factor(i,j);}for(int j=0;j<n;++j)t[j]=r.t[j];*residual=r.residual;return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
'''
with (ROOT/'native/gaussian_pnr_resplit75_v1.cpp').open('xb') as f:f.write(('#include "gaussian_pnr_deep75_v1.cpp"\n#include "gaussian_resplit75_v1.hpp"\n'+s+probe).encode())
s=(ROOT/'src/gbslab/native_structured_pnr.py').read_text(encoding='utf-8');a=s.index('    def sample(');s=s[a:].replace('pnrs74_sample','pnrr75_sample').replace('np.zeros(9)','np.zeros(12)').replace('zip(STATS,','zip(STATS+("resplit_attempts","resplit_successes","maximum_resplit_residual"),')
header='''"""Experimental Gaussian closure with repeated pure/noise decomposition; no global accuracy claim."""
import ctypes as C
import time
import numpy as np
from .native_deep_pnr import NativeDeepPNRSampler
from .native_polynomial_pnr import DP,UP,STATS
class NativeResplitPNRSampler(NativeDeepPNRSampler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lib.pnrr75_sample.argtypes=[C.c_void_p,DP,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
        self.lib.pnrr75_sample.restype=C.c_int
'''
with (ROOT/'src/gbslab/native_resplit_pnr.py').open('xb') as f:f.write((header+s).encode())
