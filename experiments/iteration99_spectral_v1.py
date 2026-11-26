"""Preserve the optical photon-total generating function and port means on the measured Jiuzhang input."""
import argparse
import ctypes as ct
import json
import math
import os
import shutil
from pathlib import Path
from time import perf_counter
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from iteration97_forest_v1 import alias_table
from iteration96_group_common_v1 import ROOT,I,D,B,W,U,T,np,bind,H,save,manifest
from iteration96_group_benchmark_v1 import HIST,ORDER

COUNT_CAP=384
SHORT=np.ctypeslib.ndpointer(dtype=np.uint16,flags='C_CONTIGUOUS')

def fit(z):
    v=z['v'][np.argsort(z['order'])]
    n=v.shape[1]
    assert abs(v.conj().T@v-np.eye(n)).max()<1e-11
    pos=[];neg=[];mark=[];diagnostics=[]
    for x in z['x']:
        lam,vec=np.linalg.eigh(x)
        lo=np.flatnonzero(lam<0)
        hi=np.flatnonzero(lam>0)
        assert len(hi)==len(lo)==n
        # Conjugate squeezed quadratures in an ideal independent squeezed source.
        axes=vec[:n]+1j*vec[n:]
        overlap=abs((axes[:,hi].conj().T@axes[:,lo]).imag)
        score=np.where((1+2*lam[hi,None])*(1+2*lam[lo])>=1-1e-13,overlap,-1e6)
        i,j=linear_sum_assignment(-score)
        assert min(score[i,j])>=0
        hi=hi[i];lo=lo[j]
        profiles=abs(v@axes)**2
        target=profiles@lam/2
        means=(lam[hi]+lam[lo])/2
        assert target.min()>0 and means.min()>0
        flux=(profiles[:,hi].T+profiles[:,lo].T)*means[:,None]/2
        for k in range(1000):
            flux*=target/flux.sum(0)
            flux*=(means/flux.sum(1))[:,None]
            if abs(flux.sum(0)-target).max()<3e-15:break
        assert abs(flux.sum(0)-target).max()<1e-12
        pos.extend(lam[hi]);neg.extend(lam[lo]);mark.extend(flux/means[:,None])
        diagnostics.append(dict(mean_error=float(abs(flux.sum(0)-target).max()),steps=k+1,lowest_overlap=float(score[i,j].min())))
    return np.array(pos),np.array(neg),np.ascontiguousarray(mark),diagnostics

def count_tables(pos,neg):
    cap=COUNT_CAP
    laws=[];cuts=[];tail=0.
    for a,b in zip(pos,neg):
        u,v=a/(1+a),b/(1+b)
        assert u>=abs(v) and 0<=u<1
        a1,b1=np.longdouble(u),np.longdouble(v)
        law=np.zeros(cap+1,np.longdouble)
        law[0]=1/np.sqrt(np.longdouble(1+a)*(1+b));law[1]=(a1+b1)*law[0]/2
        for n in range(1,cap):
            law[n+1]=((n+.5)*(a1+b1)*law[n]-n*a1*b1*law[n-1])/(n+1)
        # Bound P(N>cap) with the positive PGF at R inside its convergence disk.
        radius=1+(1/u-1)*.95
        tail+=math.exp(-(cap+1)*math.log(radius))/math.sqrt((1+a*(1-radius))*(1+b*(1-radius)))
        assert law.min()>=0 and abs(law.sum()-1)<3e-14
        c=np.minimum(2**53,np.rint(law.cumsum()*2**53)).astype(np.uint64);c[-1]=2**53
        assert np.all(c[1:]>=c[:-1])
        cuts.append(c);laws.append(np.diff(c,prepend=np.uint64(0)).astype(float)/2**53)
    return np.array(cuts),laws,tail

class Spectral:
    def __init__(self,lib,z):
        self.pos,self.neg,self.mark,self.diag=fit(z)
        self.counts,self.laws,self.tail=count_tables(self.pos,self.neg)
        cuts=[];aliases=[];actual=[];self.qt=0.
        for row in self.mark:
            # Quantize first to positive integer weights then use a 64-bit alias draw.
            weights=np.rint(row*2**56).astype(np.int64)
            self.bits,cut,alias,keys,tv,real=alias_table(dict(enumerate(map(int,weights))))
            assert np.array_equal(keys[:len(row)],np.arange(len(row))) and real[len(row):].sum()==0
            self.qt=max(self.qt,float(abs(real[:len(row)]-row).sum()/2))
            actual.append(real[:len(row)]);cuts.append(cut);aliases.append(alias)
        self.actual=np.array(actual);self.cut=np.array(cuts);self.alias=np.array(aliases)
        self.draw=bind(lib,'loss99_sources',[I,I,I,I,I,W,W,T,I,U,U,I,SHORT])
        self.law=bind(lib,'loss99_law',[I,I,D,D,D,D])
    def sample(self,n,route,seed,offset,workers):
        m=self.mark.shape[1];x=np.empty((n,m),np.uint16)
        assert self.draw(m,len(self.pos),COUNT_CAP,self.bits,route,self.counts,self.cut,self.alias,n,
                         seed,offset,workers,x)==0
        return x
    def small(self,ports):
        law=np.empty(1<<len(ports))
        assert self.law(len(ports),len(self.pos),np.ascontiguousarray(self.actual[:,ports]),self.pos,self.neg,law)==0
        assert abs(law.sum()-1)<2e-12 and law.min()>-1e-12
        return law

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';dll=data/'iteration99_loss_v1/loss.dll'
    quad=data/'iteration96_quad_v1/actual16.npz';phot=data/'iteration84_photon_cdf_v1/reference.npz'
    files=[Path(__file__),ORDER,HIST,quad,phot,dll]
    save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},seed=99030000,records=131072,
        matched='no fitting to hardware samples or reference histograms; preserve photon-total PGF and single-port photon means',
        calibration='symplectic overlap matching within each spectrum, positive optical spatial profiles balanced to optical means',
        confidence='simulated and observed CDFs: 2 click routes, one optical photon check and hardware; each DKW failure probability .0025 conditional iid'))
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    lib=ct.CDLL(str(dll))
    start=perf_counter();model=Spectral(lib,np.load(ORDER));prep=perf_counter()-start
    target=np.load(phot)['photon_pmf'];law=np.array([1.])
    for pr in model.laws:law=np.convolve(law,pr)[:len(target)]
    total_tv=float((abs(law-target).sum()+abs(law.sum()-target.sum()))/2)
    assert total_tv<1e-11
    truth=np.load(quad)['reference'];small=model.small(np.arange(16)*8)
    full16=float(abs(small-truth).sum()/2)
    hw=np.load(HIST)['total_click_histogram'];hardware=hw.cumsum()/hw.sum()
    rows=[]
    for route in (0,1):
        dom=99030000+route
        x=model.sample(512,route,dom,0,1)
        assert np.array_equal(x[32:],model.sample(480,route,dom,32,4))
        h=np.zeros(1153,np.int64);nh=np.zeros(2049,np.int64);bits=np.zeros(65536,np.int64)
        for k in range(64):
            x=model.sample(2048,route,dom,2**20+k*2048,16)
            f=x>0
            h+=np.bincount(f.sum(1),minlength=len(h))
            bits+=np.bincount(f[:,np.arange(16)*8]@(1<<np.arange(16)),minlength=65536)
            if route:nh+=np.bincount(x.sum(1).astype(int),minlength=len(nh))
            else:assert x.max()==1
        timings=[]
        for workers in [1,16]:
            times=[]
            for k in range(7):
                start=perf_counter();model.sample(8192,route,dom,(workers*7+k+1)*8192,workers)
                times.append((perf_counter()-start)*1e6/8192)
            timings.append(dict(workers=workers,batch=8192,us=times,median=float(np.median(times))))
        row=dict(route=route,records=int(h.sum()),ks_hardware=float(abs(h.cumsum()/h.sum()-hardware).max()),
                 timing=timings,small_empirical_tv_to_model=float(abs(bits/bits.sum()-small).sum()/2))
        if route:
            observed=nh.cumsum()/nh.sum()
            row.update(odd=float(nh[1::2].sum()/nh.sum()),ks_photon=float(abs(observed[:len(target)]-target.cumsum()).max()))
            assert row['ks_photon']<.01
        np.savez_compressed(out/f'hist_{route}.npz',click=h,photon=nh,subset=bits)
        rows.append(row);print(row['ks_hardware'],timings[-1]['median'],flush=True)
    summary=dict(physical_ports=1152,sources=len(model.pos),prep_s=prep,rows=rows,total_tv_to_reference=total_tv,
                 cutoff_union_chernoff=model.tail,odd_truth=float(target[1::2].sum()),mean=float(np.arange(len(target))@target),
                 complete_reduced_click_tv=full16,marking_quant_tv_per_photon=model.qt,flux_fit=model.diag,hardware_records=int(hw.sum()),
                 hardware_and_sampling_ks_halfwidth=math.sqrt(math.log(2/.0025)/(2*hw.sum()))+math.sqrt(math.log(2/.0025)/(2*131072)))
    np.savez_compressed(out/'laws.npz',measured_total=law,reference_total=target,click16=small,reference16=truth,
                        pos=model.pos,neg=model.neg,mark=model.actual,count_cdf=model.counts)
    save(out,'summary.json',summary);manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
