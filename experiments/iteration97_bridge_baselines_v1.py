"""Same-input Frugal and photonic-engines baselines for two weakly coupled optical cells."""
import argparse,ctypes as ct,subprocess
from pathlib import Path
from time import perf_counter
from iteration92_pe_common_v1 import GaussianInput,pe_sample,UPSTREAM
from iteration92_quality_v1 import confidence
from iteration96_group_benchmark_v1 import fock
from iteration97_conditionals_v1 import ROOT,Kernel,I,D,B,bind,np,H,save,manifest
from iteration96_group_common_v1 import read
p=argparse.ArgumentParser();p.add_argument('output',type=Path)
p.add_argument('--input',type=Path,default=ROOT/'results/iteration97_bridges_v1/input_1.npz')
p.add_argument('--seed',type=int,default=97120000)
args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
data=ROOT/'results';src=args.input.resolve();dll=data/'iteration96_frugal_build_v1/frugal.dll'
files=[Path(__file__),src,dll,data/'iteration96_tree_build_v1/tree.dll',ROOT/'experiments/iteration92_quality_v1.py',ROOT/'experiments/iteration92_pe_common_v1.py']
save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},seed=args.seed,
    batch=2048,repetitions=5,pe_samples=16384,pe_revision=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
    scope='all 16 detected ports, single CPU worker, bunched occupations retained; no conditioning on observed successes'))
z=np.load(src);kernel=Kernel();lib=ct.CDLL(str(dll))
tree=bind(lib,'fc96_full16',[I,D,D,D,D,D,ct.c_char_p,I]);sample=bind(lib,'fc96_sample16',[I,I,ct.c_uint,D,B,D,ct.c_char_p,I])
msg=ct.create_string_buffer(2048);cov=2*z['q'][0]-np.eye(32);rows=[]
for k in (2,4):
    th=np.empty(65535);prev=th.copy();prop=th.copy();stats=np.zeros(3)
    assert tree(k,cov,th,prev,prop,stats,msg,len(msg))==0 and stats[2]==64
    law=np.ones(1)
    for j in range(16):
        zero=np.nan_to_num(th[len(law)-1:2*len(law)-1],nan=1).clip(0,1);law=np.r_[law*zero,law*(1-zero)]
    tv=float(abs(law-z['click']).sum()/2)
    times=[];walls=[]
    for j in range(6):
        t=perf_counter();x=np.empty((2048,16),np.uint8);st=np.empty(7)
        assert sample(k,2048,args.seed+100*k+j,cov,x,st,msg,len(msg))==0 and st[5]==64
        wall=(perf_counter()-t)*1e6/len(x)
        if j:walls.append(wall);times.append((st[3]+st[4])*1e6/len(x))
    rows.append(dict(method=f'Frugal K{k}',click_tv=tv,us=times,median=float(np.median(times)),whole_call_us=walls))
    np.savez_compressed(out/f'k{k}.npz',threshold=th,law=law,sample=x)
v=z['v'];d=len(v);u=np.zeros((16,16),complex)
for j in range(d):
    u[:d,2*j:2*j+2]=v[:,j,None]/np.sqrt(2);u[d+j,2*j:2*j+2]=[1/np.sqrt(2),-1/np.sqrt(2)]
t=np.tile([.24,-.24],d);state=GaussianInput(np.arctanh(abs(t)),tuple(range(16)),16);U=u*np.where(t>0,1j,1)
b,mu,_=state.bargmann(U);assert abs(b-z['barg']).max()<1e-13 and not mu.any()
def pe(n,seed):
    x,st=pe_sample(state,U,n,seed=seed,device='cpu',return_stats=True,impl='v2',window=4,max_photons=64)
    return np.asarray(x),np.asarray(st.censored,bool)
pe(64,args.seed+3000);times=[]
for j in range(5):
    start=perf_counter();x,bad=pe(2048,args.seed+3001+j);times.append((perf_counter()-start)*1e6/2048);assert not bad.any()
batches=[pe(1024,args.seed+4000+j) for j in range(16)]
x=np.concatenate([r[0] for r in batches]);bad=np.concatenate([r[1] for r in batches])
occ,pf=fock(kernel.prior,z['barg'],z['q'],8);shifts=(4*np.arange(16)).astype(np.uint64)
keys=np.sum(occ.astype(np.uint64)<<shifts,axis=1,dtype=np.uint64);order=np.argsort(keys);keys=keys[order];pf=pf[order]
tail=float(1-pf.sum());target=np.r_[pf,tail,0.,0.]
raw=np.sum(x.astype(np.uint64)<<shifts,axis=1,dtype=np.uint64);pos=np.minimum(len(keys)-1,np.searchsorted(keys,raw))
labels=np.where(x.sum(1)<=8,np.where(keys[pos]==raw,pos,len(keys)+1),len(keys));labels[bad]=len(keys)+2
quality,_=confidence(labels,target,len(x),4096,.005,tail)
rows.append(dict(method='photonic-engines CPU v2',us=times,median=float(np.median(times)),records=len(x),failures=int(bad.sum()),pnr=quality,tail=tail))
np.savez_compressed(out/'pe.npz',records=x,failed=bad);save(out,'summary.json',rows);manifest(out)
