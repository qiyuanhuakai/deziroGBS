"""Correct an enumerated click marginal while preserving the total photon count of each record."""
import argparse
import ctypes as ct
import json
import math
import os
import shutil
import subprocess
from decimal import Decimal as Dec,localcontext
from fractions import Fraction as Rat
from pathlib import Path
from time import perf_counter
from threadpoolctl import threadpool_limits
from iteration99_spectral_v1 import ROOT,I,D,B,W,U,T,SHORT,np,bind,H,save,manifest,ORDER,HIST,Spectral,COUNT_CAP,alias_table
from iteration96_group_common_v1 import J

def integer_mass(row):
    vals=[float(max(0,x)).as_integer_ratio() for x in row]
    den=max(b for a,b in vals)
    return [a*(den//b) for a,b in vals]

def total_check(model):
    # Compute the low-count probability for the emitted integer tables exactly.
    low=[1]+[0]*15
    for cuts in model.counts:
        prob=[int(x) for x in np.diff(cuts[:16],prepend=np.uint64(0))]
        low=[sum(low[j]*prob[i-j] for j in range(i+1)) for i in range(16)]
    small=Rat(sum(low),2**(53*len(model.pos)))
    with localcontext() as ctx:
        ctx.prec=80
        tv=Dec(0)
        for row,plus,minus in zip(model.counts,model.pos,model.neg):
            a,b=map(Dec.from_float,[float(plus),float(minus)])
            u=a/(1+a);v=b/(1+b)
            p=[1/((1+a)*(1+b)).sqrt()]
            p.append((u+v)*p[0]/2)
            for n in range(1,COUNT_CAP):
                p.append(((2*n+1)*(u+v)*p[n]/2-n*u*v*p[n-1])/(n+1))
            finite=[Dec(int(x))/2**53 for x in np.diff(row,prepend=np.uint64(0))]
            tv+=(sum(abs(r-t) for r,t in zip(finite,p))+max(Dec(0),1-sum(p)))/2
    return float(small),float(tv)

def physical(z,ports):
    v=z['v'][np.argsort(z['order'])]
    a=np.column_stack((v.real,-v.imag));b=np.column_stack((v.imag,v.real))
    vacuum=np.ones(len(v));sub=np.r_[a[ports],b[ports]]
    q=[]
    for x in z['x']:
        u=a@x;w=b@x
        vacuum/=np.sqrt((1+(a*u).sum(1))*(1+(b*w).sum(1))-((u*b).sum(1))**2)
        q.append(np.eye(2*len(ports))+sub@x@sub.T)
    return 1-vacuum,np.array(q)

class Marked:
    def __init__(self,model,lib,oracle,z,ports):
        self.source=model;self.group=np.array(ports,np.int32)
        width=len(ports);m=model.mark.shape[1]
        self.draw=bind(lib,'mark99_draw',[I,I,I,I,I,W,W,T,J,I,W,W,I,W,T,T,I,U,U,I,SHORT,B])
        law=bind(lib,'mark99_law',[I,I,D,D,D,D])
        reduced=np.empty(1<<width)
        assert law(width,len(model.pos),np.ascontiguousarray(model.actual[:,ports]),model.pos,model.neg,reduced)==0
        single_target,q=physical(z,ports)
        optical=np.empty_like(reduced);diag=np.empty(2)
        assert oracle(width,len(q),q,np.zeros((len(q),2*width)),optical,diag)==0
        assert min(reduced.min(),optical.min())>=-1e-20
        old=integer_mass(reduced);true=integer_mass(optical)
        oldsum=sum(old);truesum=sum(true)
        excess=[max(0,t*oldsum-o*truesum) for o,t in zip(old,true)]
        self.keep=np.array([min(2**53,(t*oldsum*2**53)//(o*truesum)) if o else 2**53 for o,t in zip(old,true)],np.uint64)
        self.dbits,self.dcut,self.dalias,self.words,alias_tv,_=alias_table({i:w for i,w in enumerate(excess) if w})
        self.tv=float(Rat(sum(excess),oldsum*truesum))
        single_old=-np.expm1(-.5*(np.log1p(model.pos[:,None]*model.actual)+np.log1p(model.neg[:,None]*model.actual)).sum(0))
        assert np.all(single_target<=single_old) and single_target.min()>0
        self.single=np.floor(2**53*single_target/single_old).astype(np.uint64)
        self.optical=optical;self.before=reduced;self.singles=single_target
        self.small,ctv=total_check(model)
        # Rounded mark masses have relative error <= 2^-53. Include binary64
        # output rounding/normalization and a binary128 arithmetic allowance.
        mean=sum(float(np.dot(np.arange(len(x)),x)) for x in model.laws)
        slack=ctv+(mean+4)*2**-52+alias_tv+2**-53+1e-17
        self.bound=self.small+slack
        self.details=dict(full_word_tv_before=self.tv,global_count_below_width=self.small,count_table_tv_sum=ctv,
            mean_photons=mean,deficit_alias_tv=alias_tv,rounding_allowance=slack,
            optical_clicks_mean=float(single_target.sum()),base_clicks_mean=float(single_old.sum()),
            min_port_click_reduction=float((single_old-single_target).min()),max_port_click_reduction=float((single_old-single_target).max()),
            precise_vs_long_double=float(abs(model.small(ports)-reduced).sum()/2),measured_group_tv_upper=self.bound)
    def sample(self,n,route,seed,offset,workers):
        s=self.source;m=s.mark.shape[1];x=np.empty((n,m),np.uint16);flags=np.empty(n,np.uint8)
        rc=self.draw(m,len(s.pos),COUNT_CAP,s.bits,route,s.counts,s.cut,s.alias,self.group,len(self.group),self.keep,self.single,
                     self.dbits,self.dcut,self.dalias,self.words,n,seed,offset,workers,x,flags)
        if rc or (flags&16).any():raise RuntimeError('mark draw failed')
        return x,flags

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';base=data/'iteration99_loss_v1/loss.dll';qd=data/'iteration96_quad_v1/oracle.dll'
    src=ROOT/'native/spectral_marks99_v1.cpp';truth=data/'iteration96_quad_v1/actual16.npz'
    files=[Path(__file__),src,base,qd,ORDER,HIST,truth,ROOT/'experiments/iteration99_spectral_v1.py']
    save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},group=(8*np.arange(16)).tolist(),
        seed=99070000,records=131072,fit='spectral and optical parameters only; no hardware sample enters calibration',
        checks='same-seed binary replay, per-record PNR total against original, all 65536 optical probabilities; DKW .0025 per CDF'))
    dll=out/'marks.dll';cmd=[shutil.which('g++'),'-std=c++20','-O3','-shared','-fopenmp','-ffp-contract=off',str(src),'-lquadmath','-o',str(dll)]
    run=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr));run.check_returncode()
    runtime=os.add_dll_directory(str(Path(cmd[0]).parent))
    oracle=bind(ct.CDLL(str(qd)),'oracle96',[I,I,D,D,D,D])
    z=np.load(ORDER);t=perf_counter();source=Spectral(ct.CDLL(str(base)),z)
    model=Marked(source,ct.CDLL(str(dll)),oracle,z,np.arange(16)*8);prep=perf_counter()-t
    reference=np.load(truth)['reference'];oracle_diff=float(abs(reference-model.optical).sum()/2)
    hw=np.load(HIST)['total_click_histogram'];rows=[]
    for route in (0,1):
        seed=99070000+route
        x,flags=model.sample(512,route,seed,0,1);y,f=model.sample(480,route,seed,32,16)
        assert np.array_equal(x[32:],y) and np.array_equal(flags[32:],f)
        hist=np.zeros(1153,np.int64);bits=np.zeros(65536,np.int64);flagcounts=np.zeros(32,np.int64)
        nh=np.zeros(2049,np.int64);portcounts=np.zeros(1152,np.int64)
        for j in range(64):
            off=2**20+2048*j
            x,flags=model.sample(2048,route,seed,off,16);click=x>0
            flagcounts+=np.bincount(flags,minlength=32)
            portcounts+=click.sum(0)
            hist+=np.bincount(click.sum(1),minlength=len(hist))
            bits+=np.bincount(click[:,model.group]@(1<<np.arange(16)),minlength=len(bits))
            if route:
                totals=x.sum(1)
                assert np.array_equal(totals,source.sample(2048,1,seed,off,16).sum(1))
                direct,df=model.sample(2048,0,seed,off,16)
                assert np.array_equal(direct,click) and np.array_equal(df,flags)
                nh+=np.bincount(totals.astype(int),minlength=len(nh))
        times=[]
        for workers in (1,16):
            tms=[]
            for j in range(7):
                t=perf_counter();model.sample(8192,route,seed,(workers*7+j+1)*8192,workers);tms.append((perf_counter()-t)*1e6/8192)
            times.append(dict(workers=workers,batch=8192,us=tms,median=float(np.median(tms))))
        ks=float(abs(hist.cumsum()/hist.sum()-hw.cumsum()/hw.sum()).max())
        row=dict(route=route,records=int(hist.sum()),hardware_ks=ks,click_mean=float(hist@np.arange(len(hist))/hist.sum()),
            timings=times,port_mean_max_sample_deviation=float(abs(portcounts/hist.sum()-model.singles).max()),
            edited=int(flagcounts[np.arange(32)&1==1].sum()),subset_infeasible=int(flagcounts[np.arange(32)&2!=0].sum()),
            outside_capacity=int(flagcounts[np.arange(32)&4!=0].sum()),empty_fix=int(flagcounts[np.arange(32)&8!=0].sum()))
        if route:
            target=np.load(data/'iteration84_photon_cdf_v1/reference.npz')['photon_cdf']
            row.update(total_ks=float(abs(nh.cumsum()[:len(target)]/nh.sum()-target).max()),odd=float(nh[1::2].sum()/nh.sum()))
        rows.append(row)
        np.savez_compressed(out/f'hist_{route}.npz',click=hist,subset=bits,pnr=nh,flags=flagcounts,ports=portcounts)
        print(route,ks,times[-1]['median'],flush=True)
    assert oracle_diff<1e-12
    np.savez_compressed(out/'marginals.npz',target=model.optical,base=model.before,ports=model.group,keep=model.keep,single=model.single)
    save(out,'summary.json',dict(rows=rows,prep_s=prep,oracle_tv_vs_archived=oracle_diff,model=model.details,
        joint_16_tv_upper_vs_archived=model.bound+oracle_diff,hardware_n=int(hw.sum()),cdf_halfwidth=math.sqrt(math.log(800)/(2*hw.sum()))+math.sqrt(math.log(800)/(2*131072))))
    manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
