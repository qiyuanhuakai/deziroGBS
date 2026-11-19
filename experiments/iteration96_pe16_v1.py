"""Matched PNR counts for the connected sixteen-mode interferometer."""
import argparse
import ctypes as ct
import subprocess
from pathlib import Path
from time import perf_counter
from iteration92_pe_common_v1 import UPSTREAM,GaussianInput,pe_sample
from iteration96_group_benchmark_v1 import BUILD,Native,fock
from iteration96_group_common_v1 import ROOT,np,H,save,manifest,I,U,D,T,B,P,bind
from iteration92_quality_v1 import confidence
p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
hfile=ROOT/'results/iteration96_herald_v1/herald.dll';src=ROOT/'results/iteration96_herald_v1/mixed.npz'
ref=ROOT/'results/iteration96_optical_v1/mixed_reference.npz'
files=[Path(__file__),BUILD,hfile,src,ref,ROOT/'experiments/iteration92_quality_v1.py',ROOT/'experiments/iteration92_pe_common_v1.py']
save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},seed=96060000,records=16384,threads=1,
    upstream=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
    quality='pilot 4096, holdout 12288, four one-sided bounds per two methods, delta=.01/4 each for PNR TV',settings=dict(window=4,cap=64)))
native=Native(BUILD);lib=ct.CDLL(str(hfile));prep=bind(lib,'herald96_prepare',[I,D,D,T],P);free=bind(lib,'herald96_free',[P],None)
draw=bind(lib,'herald96_draw',[P,I,U,U,I,T,B]);z=np.load(src);v=z['unitary'];probs=z['source_p'];d=len(probs)
err=np.ones(1,np.uint32);h=prep(d,v.view(float),probs,err);assert h and not err[0]
u=np.zeros((16,16),complex)
for j in range(d):
    u[:d,2*j:2*j+2]=v[:,j,None]/np.sqrt(2);u[d+j,2*j:2*j+2]=[1/np.sqrt(2),-1/np.sqrt(2)]
t=np.tile([.24,-.24],d);Upe=u*np.where(t>0,1j,1)
state=GaussianInput(np.arctanh(abs(t)),tuple(range(16)),16)
target=np.load(ref);barg,mean,_=state.bargmann(Upe);assert abs(barg-target['barg']).max()<1e-14 and not np.any(mean)
occ,pf=fock(native,target['barg'],target['q'],8)
shifts=(4*np.arange(16)).astype(np.uint64);keys=np.sum(occ.astype(np.uint64)<<shifts,axis=1,dtype=np.uint64)
order=np.argsort(keys);keys=keys[order];pf=pf[order]
tail=float(1-pf.sum());law=np.r_[pf,tail,0.,0.]
def sample(name,n,seed):
    if name=='pe':
        x,st=pe_sample(state,Upe,n,seed=seed,device='cpu',return_stats=True,impl='v2',window=4,max_photons=64)
        return np.asarray(x),np.asarray(st.censored,bool)
    x=np.empty((n,16),np.uint32);flags=np.empty(n,np.uint8);assert draw(h,n,seed,0,1,x,flags)==0
    return x,flags!=0

rows=[];times={'pe':[],'herald':[]}
try:
    for rep in range(-1,5):
        for j,name in enumerate((['pe','herald'] if rep%2==0 else ['herald','pe'])):
            start=perf_counter();x,bad=sample(name,64 if rep<0 else 2048,96060000+(rep+1)*2+j)
            tsec=perf_counter()-start
            if rep>=0:times[name].append(tsec*1e6/2048)
    for i,name in enumerate(['herald','pe']):
        batches=[sample(name,1024,96060100+32*i+r) for r in range(16)]
        x=np.concatenate([r[0] for r in batches]);bad=np.concatenate([r[1] for r in batches])
        assert np.all(x>=0);n=len(x);summ=x.sum(1);small=summ<=8
        raw=np.sum(x.astype(np.uint64)<<shifts,axis=1,dtype=np.uint64)
        pos=np.minimum(len(keys)-1,np.searchsorted(keys,raw))
        label=np.where(small,np.where(keys[pos]==raw,pos,len(keys)+1),len(keys));label[bad]=len(keys)+2
        stats,_=confidence(label,law,n,4096,.0025,tail)
        rows.append(dict(method=name,records=n,failures=int(bad.sum()),us=times[name],median=float(np.median(times[name])),
            tv=stats,signal_idler_imbalance=int(np.count_nonzero(x[:,:8].sum(1)!=x[:,8:].sum(1))),tail=tail))
        np.savez_compressed(out/f'{name}.npz',counts=x,failed=bad);print(rows[-1],flush=True)
finally:free(h)
save(out,'summary.json',rows);manifest(out)
