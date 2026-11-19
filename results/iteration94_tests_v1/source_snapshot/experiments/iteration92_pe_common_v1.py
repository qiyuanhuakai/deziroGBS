"""Pinned upstream photonic-engines v0.2.2, unmodified CPU v2 sampler."""
import sys
from iteration92_common_v2 import *
UPSTREAM=ROOT.parent/'external_sources/photonic_engines_iteration92'
sys.path.insert(0,str(UPSTREAM))
import torch
torch.set_num_threads(1)
from scipy.linalg import null_space
from photonic_engines.states import GaussianInput
from photonic_engines.gaussian import sample as pe_sample

def setup_pe(u,r):
    u=np.asarray(u,complex);w=np.column_stack([1j*(u[:,0]+u[:,1]),u[:,0]-u[:,1]])/np.sqrt(2);U=np.column_stack([w,null_space(w.conj().T)]) if len(u)>2 else w;state=GaussianInput(np.repeat(np.arctanh(r),2),(0,1),len(u));B,_,_=state.bargmann(U);expected=r*(np.outer(u[:,0],u[:,1])+np.outer(u[:,1],u[:,0]));assert np.max(abs(B-expected))<2e-14;return state,U

def run_pe(state,U,count,seed,cap=128):
    start=time.perf_counter();counts,stats=pe_sample(state,U,count,seed=seed,device='cpu',return_stats=True,impl='v2',window=4,max_photons=cap);return np.asarray(counts),stats,dict(seconds=time.perf_counter()-start,censored=int(sum(stats.censored)),max_norm_relative_error=float(max(stats.norm_rel_err,default=0)),max_round_off=float(max(stats.round_off,default=0)))
