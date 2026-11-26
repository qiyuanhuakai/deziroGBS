"""Optical marginal correction that preserves both photon and click totals per record."""
import argparse,ctypes as ct,json,math,os,shutil,subprocess
from pathlib import Path
from time import perf_counter
from threadpoolctl import threadpool_limits
from iteration100_projection_v1 import Projector
from iteration100_fast_v1 import ROOT,np,Fast,Tree,I,W,T,U,B,SHORT,bind,H,save,manifest,alias_table
from iteration99_marks_v1 import integer_mass

def count_messages(proj,groups,limit):
    model=proj.model
    # The last column is the moment E[2^B], used for an upper-tail Chernoff bound.
    cond={}
    for i in range(len(model.caps)):
        cap=int(model.caps[i])
        if i<model.leaves:
            if i not in groups:continue
            pop=np.array([j.bit_count() for j in range(1<<model.width)])
            q=proj.mask[i];a=np.zeros((len(q),limit+1))
            for k in range(min(model.width+1,limit)):a[:,k]=q[:,pop==k].sum(1)
            a[:,-1]=q@(2.**pop)
        else:
            l,r=model.children[i]
            if l not in cond and r not in cond:continue
            if l not in cond:a=proj.conditional[i][1]@cond.pop(r)
            elif r not in cond:a=proj.conditional[i][0]@cond.pop(l)
            else:
                left,right=cond.pop(l),cond.pop(r);a=np.zeros((cap+1,limit+1))
                for n,row in enumerate(proj.conditional[i][0]):
                    ix=np.arange(max(0,n-len(right)+1),min(n+1,len(left)))
                    joint=(left[ix,:limit]*row[ix,None]).T@right[n-ix,:limit]
                    for k in range(limit):a[n,k]=np.trace(joint[:k+1,:k+1][:,::-1])
                    a[n,-1]=(left[ix,-1]*row[ix])@right[n-ix,-1]
        cond[i]=a
    root=len(model.caps)-1;return proj.totals[root]@cond[root]

class Calibrated:
    def __init__(self,fast,dll,arrays):
        self.fast=fast;self.arrays=arrays;self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        self.lib=ct.CDLL(str(dll))
        self.fn=bind(self.lib,'relabel100',[I,I,W,I,W,T,T,I,U,U,I,SHORT,B])
    def correct(self,x,seed,off,workers):
        flags=np.empty(len(x),np.uint8);a=self.arrays
        assert self.fn(x.shape[1],8,a['keep'],(len(a['cut'])-1).bit_length(),a['cut'],a['alias'],a['words'],
                       len(x),seed,off,workers,x,flags)==0
        return flags
    def sample(self,n,route,seed,off,workers):
        x,f=self.fast.sample(n,route,seed,off,workers);flags=self.correct(x,seed,off,workers)
        return x,f,flags

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);out=ap.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';base=data/'iteration100_spatial_v1';fs=data/'iteration100_fast_v1';pan=data/'iteration100_projection_v1'
    src=ROOT/'native/spatial_relabel100_v1.cpp';files=[Path(__file__),src,base/'prepared.npz',fs/'shortcuts.npz',fs/'fast.dll',pan/'panel_0.npz']
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},seed=100050000,
         calibration='first branch of each of the first sixteen physical splitters, exact optical probabilities; swaps within splitter, then region, then outside',
         tests='both complete count observables invariant; one million fresh records per route for old frozen tests'))
    dll=out/'relabel.dll';cmd=[shutil.which('g++'),'-std=c++20','-O3','-shared','-fopenmp',str(src),'-o',str(dll)]
    run=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr));run.check_returncode()
    with np.load(base/'prepared.npz') as f:arrays=dict(f)
    with np.load(fs/'shortcuts.npz') as f:fast=Fast(fs/'fast.dll',arrays,dict(f))
    z=np.load(pan/'panel_0.npz');old=integer_mass(z['law']);new=integer_mass(z['truth'])
    so,sn=sum(old),sum(new);ex=[max(0,v*so-u*sn) for u,v in zip(old,new)]
    keep=np.array([min(2**53,(v*so*2**53)//(u*sn)) if u else 2**53 for u,v in zip(old,new)],np.uint64)
    bits,cut,alias,words,alias_tv,real=alias_table({i:k for i,k in enumerate(ex) if k>0})
    correction=dict(keep=keep,cut=cut,alias=alias,words=words)
    np.savez_compressed(out/'prepared.npz',**correction)
    start=perf_counter();proj=Projector(fast)
    global_counts=count_messages(proj,set(range(144)),16);regional=count_messages(proj,set(range(16)),16)
    probs=np.bincount([int(x).bit_count() for x in words],weights=real,minlength=17)
    def floor_risk(c):return float(np.r_[0,c[:16].cumsum()]@probs)
    raw=floor_risk(global_counts);spill=floor_risk(regional)+regional[-1]*2.**-113
    transfer=json.loads((fs/'summary.json').read_text())['full_tv_transfer_bound']
    # Conservative forward error for nonnegative contractions. Each operation
    # has at most max(cap,mask_count)+1 summands at every node along a term.
    ops=32*len(fast.caps)*(int(fast.caps.max())+258)
    rel=ops*2.**-53/(1-ops*2.**-53)
    slack=3*rel+3*proj.overflow+2*transfer+alias_tv+2**-52
    budget_s=perf_counter()-start
    model=Calibrated(fast,dll,correction);rows=[]
    prior=data/'iteration99_release_v1';ports=[d['ports'] for d in json.loads((prior/'validation.json').read_text())['heldout']]
    masks=[np.load(prior/f'heldout_{i}.npz')['event'] for i in range(3)]
    for route in (0,1):
        seed=100050000+route;flags=np.zeros(8,np.int64);hits=np.zeros(3,np.int64);base_hits=hits.copy();times=[]
        for rep in range(256):
            off=rep*4096;x,e=fast.sample(4096,route,seed,off,16);assert not e.any()
            before=x.sum(1);bc=(x>0).sum(1)
            def score(a):return np.array([event[(a[:,g]>0)@(1<<np.arange(len(g)))].sum() for g,event in zip(ports,masks)])
            base_hits+=score(x);f=model.correct(x,seed,off,16);hits+=score(x)
            assert np.array_equal(before,x.sum(1)) and np.array_equal(bc,(x>0).sum(1))
            flags+=np.bincount(f,minlength=8)
            if rep<4:
                other,ef,ff=model.sample(4096,1-route,seed,off,1);assert np.array_equal(x>0,other>0) and np.array_equal(f,ff) and not ef.any()
        for workers in (1,16):
            ts=[]
            for rep in range(7):
                t=perf_counter();model.sample(8192,route,seed,2**22+(workers*7+rep)*8192,workers);ts.append((perf_counter()-t)*1e6/8192)
            times.append(dict(workers=workers,us=ts,median=float(np.median(ts))))
        rows.append(dict(route=route,records=int(flags.sum()),flag_counts=flags.tolist(),frozen_hits=hits.tolist(),base_hits=base_hits.tolist(),timing=times))
        print(route,times[-1]['median'],flags.tolist(),flush=True)
    save(out,'summary.json',dict(rows=rows,source_tv=float(sum(ex)/(so*sn)),global_below_16=float(global_counts[:16].sum()),
         infeasible_upper_before_rounding=raw,regional_spill_upper_before_rounding=spill,conservative_numerical_allowance=slack,
         selected_full_click_tv_upper=raw+slack,outside_region_change_upper=spill+slack,
         whole_record_change_upper=float(sum(ex)/(so*sn))+slack,bound_computation_s=budget_s))
    manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
