"""Integer conditional sampling for the zero-phase two-cell ring."""
import argparse
import ctypes as ct
import math
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
from fractions import Fraction as Rat
from iteration99_loss_v1 import ROOT,P,CAP,SIGNAL,IDLERS,LIMIT,np,LossRing,draw,bind,I,W,U,T,B,H,save,manifest
from iteration96_group_common_v1 import J
from iteration98_cycle_v1 import fixture
from iteration92_quality_v1 import confidence

def coefficient_table(lib,local):
    capacity=sum(math.comb(n+3,3)**2 for n in range(CAP+1))
    keys=np.empty(capacity,np.uint32);re=np.empty(capacity,np.int32);im=np.empty_like(re)
    count=bind(lib,'cycle98_coefficients',[I,I,T,J,J])(CAP,capacity,keys,re,im)
    assert count>0
    lookup=dict(zip(map(int,local[3]),map(int,local[4])))
    coeff=np.zeros((int(local[4].max())+1,CAP+1,2),np.int32)
    for code,a,b in zip(keys[:count],re[:count],im[:count]):
        coeff[lookup[int(code)],(int(code)>>12)&15]=[a,b]
    return coeff

class IntegerRing(LossRing):
    def __init__(self,lib,exact,p=P,signal=SIGNAL,idlers=IDLERS):
        super().__init__(lib,p,signal,idlers)
        self.coeff=coefficient_table(lib,self.table)
        self.close=bind(exact,'exact99_close2',[I,I,I,W,T,T,T,J,I,U,U,I,B,B])
    def virtual(self,table,route,n,seed,offset,workers):
        x,flags=draw(self.ring,table,self.edges,2,route,0.,0,n,seed,offset,workers)
        bits,cut,alias,keys,groups,_,_=table
        rc=self.close(route,bits,CAP,cut,alias,keys,groups,self.coeff,n,seed,offset,workers,flags,x)
        if rc:raise RuntimeError(('integer closure',rc))
        return x,flags
    def sample(self,cells,route,n,seed,offset,workers,phase=0.):
        if cells!=2 or phase!=0.:return super().sample(cells,route,n,seed,offset,workers,phase)
        x,flags=self.virtual(self.table,route,n,seed,offset,workers)
        rc=self.measure(8,route,CAP,LIMIT,self.cuts,n,seed,offset,workers,x)
        if rc:raise RuntimeError(('idler posterior',rc))
        return x,flags
    def bound(self,cells,route,phase=0.):
        if cells!=2 or phase!=0.:return super().bound(cells,route)
        tail=1-sum(math.comb(t+7,7)*self.q**t*(1-self.q)**8 for t in range(13))
        return float(1-self.survival**2+tail)+2*(self.table[-1]+self.edges[2][route]+4*(self.quant[route]+route*self.tail))

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';dll=data/'iteration99_loss_v1/loss.dll';src=ROOT/'native/exact_ring99_v1.cpp'
    source=data/'iteration99_baselines_v1'
    inputs=[Path(__file__),src,dll,source/'input.npz',source/'fock.npz',ROOT/'experiments/iteration99_loss_v1.py']
    save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in inputs},max_total=12,seed=99050000,
         assumptions='balanced two-cell ring, zero closing phase, exact integer sums and uniform integer sampling in the rare conditional branch',
         bound='uncut NB(8) upper tail above 12 plus cell truncation and rational CDF/alias rounding'))
    exact=out/'integer.dll';cmd=[shutil.which('g++'),'-shared','-fopenmp','-std=c++20','-O3',str(src),'-o',str(exact)]
    run=subprocess.run(cmd,text=True,capture_output=True);save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr));run.check_returncode()
    runtime=os.add_dll_directory(str(Path(cmd[0]).parent));lib=ct.CDLL(str(dll))
    start=perf_counter();ring=IntegerRing(lib,ct.CDLL(str(exact)));prep=perf_counter()-start
    cap=12;max_uint=2*(cap+1)**4*64**cap*math.factorial(cap)
    assert max_uint<2**128 and (cap+1)*8**cap<2**63
    rows=[]
    for route in (0,1):
        table=ring.table
        local=np.array([65537,65536+4096],np.uint32)
        ids=np.array([table[4][np.flatnonzero(table[3]==k)[0]] for k in local],np.uint32)
        fixed=(1,np.array([2**63]*2,np.uint64),np.array([0,1],np.uint32),local,ids,table[5],0.)
        x,flags=ring.virtual(fixed,route,131072,99051000+route,0,16)
        controls=__import__('json').loads((data/'iteration98_cycle_v1/controls.json').read_text())
        truth=np.array(next(row['truth'] for row in controls if row['phase']==0 and row['route']==route))
        hist=np.bincount(x[:,[0,3,4,7]]@np.array([1,4,16,64]),minlength=256)
        assert hist[truth<1e-14].sum()==0 and abs(hist/hist.sum()-truth).sum()/2<.012
        seed=99050000+route
        x,_=ring.sample(2,route,131072,seed,0,16)
        y,_=ring.sample(2,route,480,seed,32,1)
        assert np.array_equal(x[32:512],y)
        times=[]
        for j in range(7):
            t=perf_counter();ring.sample(2,route,2048,seed,2**20+2048*j,1);times.append((perf_counter()-t)*1e6/2048)
        if route:
            ref=np.load(source/'fock.npz');keys=ref['keys'];p=np.r_[ref['completed'],0.]
            raw=(x.astype(np.uint64)<<(4*np.arange(16,dtype=np.uint64))).sum(1)
            idx=np.minimum(len(keys)-1,np.searchsorted(keys,raw))
            labels=np.where((keys[idx]==raw)&(x.sum(1)<=8),idx,len(keys));tail=float(ref['omitted'])
        else:
            p=np.load(source/'input.npz')['click'];labels=x@(1<<np.arange(16));tail=0.
        stats,h=confidence(labels[:16384],p,16384,4096,.0025,tail);stats['lower']=max(0.,stats['lower']-tail)
        np.savez_compressed(out/f'quality_{route}.npz',records=x,forced=hist,**h)
        rows.append(dict(route=route,full_tv_upper=ring.bound(2,route),sample_tv=stats,median=float(np.median(times)),
            warm_us=times,prep_s=prep,conditional_records=int(hist.sum()),forbidden=int(hist[truth<1e-14].sum()),
            conditional_empirical_tv=float(abs(hist/hist.sum()-truth).sum()/2)))
    save(out,'summary.json',dict(rows=rows,integer_sum_upper=str(max_uint),maximum_accumulator_bits=max_uint.bit_length(),
        fallback_probability=float(1-sum(math.comb(t+7,7)*ring.q**t*(1-ring.q)**8 for t in range(13)))))
    manifest(out)
if __name__=='__main__':main()
