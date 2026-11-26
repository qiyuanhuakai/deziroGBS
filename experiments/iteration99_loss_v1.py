"""Uniform signal loss commutes through the ring; observe each true idler conditional on the surviving input."""
import argparse
import ctypes as ct
import math
import os
import shutil
import subprocess
from fractions import Fraction as Rat
from pathlib import Path
from time import perf_counter
from iteration98_cycle_v1 import ROOT,P,CAP,np,bind,I,F,W,T,D,U,B,H,save,manifest,tables,couplers,draw

SIGNAL=Rat(3,5)
IDLERS=tuple(map(Rat,['2/5','11/20','7/10','17/20']))
LIMIT=32
SCALE=1<<53

def idler_tables(p,signal,idlers,cap,limit):
    a=p*(1-signal)
    cuts=np.empty((4,cap+1,limit+1),np.uint64)
    tail=Rat(0)
    quant=[Rat(0),Rat(0)]
    for j,eta in enumerate(idlers):
        b=a*eta/(1-a+a*eta)
        for m in range(cap+1):
            binom=[math.comb(m,k)*eta**k*(1-eta)**(m-k) for k in range(m+1)]
            nb=[math.comb(m+k,k)*(1-b)**(m+1)*b**k for k in range(limit+1)]
            law=[sum(binom[h]*nb[k-h] for h in range(min(k,m)+1)) for k in range(limit+1)]
            tail=max(tail,1-sum(law))
            law[-1]+=1-sum(law)
            assert law[0]==(1-eta)**m*((1-a)/(1-a+a*eta))**(m+1)
            cdf=Rat(0);previous=0;error=Rat(0)
            for k,prob in enumerate(law):
                cdf+=prob;cut=int(cdf*SCALE)
                cuts[j,m,k]=cut;error+=abs(Rat(cut-previous,SCALE)-prob);previous=cut
            quant[0]=max(quant[0],abs(law[0]-Rat(int(cuts[j,m,0]),SCALE)))
            quant[1]=max(quant[1],error/2)
    return cuts,float(tail),list(map(float,quant))

class LossRing:
    def __init__(self,lib,p=P,signal=SIGNAL,idlers=IDLERS):
        if not (0<=p<1 and 0<=signal<=1 and len(idlers)==4 and all(0<=x<=1 for x in idlers)):
            raise ValueError('invalid squeezing or efficiencies')
        self.q=signal*p/(1-p+signal*p)
        local,self.survival=tables(lib,CAP,self.q)
        self.table=local[1]
        self.edges=couplers(CAP,Rat(1,2))
        self.cuts,self.tail,self.quant=idler_tables(p,signal,idlers,CAP,LIMIT)
        self.ring=bind(lib,'cycle98_draw',[I,I,I,I,I,F,W,T,T,T,D,D,W,I,U,U,I,B,B])
        self.measure=bind(lib,'loss99_idlers',[I,I,I,I,W,I,U,U,I,B])
    def sample(self,cells,route,n,seed,offset,workers,phase=0.):
        x,flags=draw(self.ring,self.table,self.edges,cells,route,phase,1,n,seed,offset,workers)
        rc=self.measure(cells*4,route,CAP,LIMIT,self.cuts,n,seed,offset,workers,x)
        if rc:raise RuntimeError(('idler measurement',rc))
        return x,flags
    def bound(self,cells,route):
        q=self.q
        return float(1-self.survival**cells+2*q**cells-q**(2*cells))+cells*(
            self.table[-1]+self.edges[2][route]+4*(self.quant[route]+route*self.tail))

def parity(cells):
    factors=[(1-P)/(1-P*(1-2*SIGNAL)*(1-2*eta)) for eta in IDLERS]
    return float((1-math.prod(factors)**cells)/2)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    src=[ROOT/'native/gaussian_loss99_v1.cpp',ROOT/'native/fourier_cell98_v1.cpp']
    inputs=[*src,ROOT/'native/gaussian_cycle98_v1.cpp',ROOT/'native/gaussian_rng_domain84_v1.hpp',
            Path(__file__),ROOT/'experiments/iteration98_cycle_v1.py']
    save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in inputs},seed=99001000,
        cells=[2,16,144],cap=CAP,idler_limit=LIMIT,p=str(P),signal=str(SIGNAL),idlers=list(map(str,IDLERS)),
        comparison='7 warmed batches; independent 131072 shots per route at 16 and 1152 modes; five parity/CDF checks use tolerance .01'))
    dll=out/'loss.dll'
    cmd=[shutil.which('g++'),'-shared','-fopenmp','-std=c++20','-O3','-ffp-contract=off',*map(str,src),'-o',str(dll)]
    run=subprocess.run(cmd,text=True,capture_output=True)
    save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr));run.check_returncode()
    runtime=os.add_dll_directory(str(Path(cmd[0]).parent));lib=ct.CDLL(str(dll))
    start=perf_counter();sampler=LossRing(lib);prep=perf_counter()-start
    rows=[]
    for j,cells in enumerate([2,16,144]):
        for route in (0,1):
            dom=99001000+100*j+route
            sample=lambda n,off,w:sampler.sample(cells,route,n,dom,off,w)
            x,flags=sample(512,0,1);y,replay=sample(480,32,4)
            assert np.array_equal(x[32:],y) and np.array_equal(flags[32:],replay)
            timings=[]
            for workers in (1,16):
                batch=2048 if cells==2 else 8192
                times=[]
                for k in range(7):
                    t=perf_counter();x,_=sample(batch,4096+(workers*7+k)*batch,workers)
                    times.append((perf_counter()-t)*1e6/batch)
                timings.append(dict(workers=workers,batch=batch,us=times,median=float(np.median(times))))
            row=dict(ports=8*cells,route=route,prep_s=prep,full_tv_upper=sampler.bound(cells,route),
                     signal_q=str(sampler.q),cell_tail=float(1-sampler.survival),posterior_tail=sampler.tail,timing=timings)
            if cells in (2,144):
                total=0;hist=np.zeros(2049,np.int64);n=131072
                batches=[]
                for k in range(n//2048):
                    x,_=sample(2048,2**20+2048*k,16)
                    hist+=np.bincount(x.sum(1).astype(int),minlength=len(hist))
                    if cells==2:batches.append(x)
                if cells==2:np.savez_compressed(out/f'records_{route}.npz',x=np.concatenate(batches))
                if route:
                    odd=float(hist[1::2].sum()/n);truth=parity(cells)
                    assert abs(odd-truth)<.01
                    row.update(odd=odd,odd_truth=truth)
                else:assert x.max()<=1
                np.savez_compressed(out/f'hist_{cells}_{route}.npz',counts=hist)
                row['records']=n
            rows.append(row);print(row['ports'],route,row['full_tv_upper'],timings[-1]['median'],flush=True)
    save(out,'summary.json',rows);manifest(out)
if __name__=='__main__':main()
