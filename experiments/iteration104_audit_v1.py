"""Independent packed-sampler hardware panels, matched-width timing, and optical moment witness checks."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits

from iteration100_projection_v1 import physical
from iteration100_spatial_v1 import HIST,ORDER,I,D,bind,save,manifest
from iteration104_sample_v1 import FIT,PILOT,DATA
from iteration104_packed_v1 import Packed,OUT
from iteration104_optical_v1 import OPT


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    runner=Packed(OUT/'gpu.dll')
    config=dict(np.load(FIT/'tables.npz'))
    spec=dict(np.load(ORDER))
    hw=np.load(HIST)
    ref=[]
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    for ids in hw['ports']:
        truth=np.empty(256)
        assert oracle(8,2,physical(spec,ids),np.zeros((2,16)),truth,np.empty(2))==0
        ref.append(truth)
    ref=np.array(ref)
    oldhw=hw['histograms'][:,1]
    hardware=oldhw/oldhw.sum(1)[:,None]
    keys=np.uint16(1)<<np.arange(8,dtype=np.uint16)
    rows=[]
    speed=[]
    for i,label in enumerate(('raw','optical','hardware')):
        runner.set_table(config.get(label,np.ones((1,1))))
        rate=float(config.get(label+'_rate',1.))
        for pnr in (False,True):
            chosen=[289,861] if pnr else list(range(288,296))+list(range(856,864))
            times=[]
            for rep in range(10):
                tic=perf_counter()
                x,c,_=runner.draw(2048,pnr,104050000+100*i+2*rep+pnr,rate=rate)
                observed=x[:,chosen].copy()
                if rep>=2:times.append((perf_counter()-tic)*1e6/2048)
            speed.append(dict(label=label,pnr=pnr,ports=chosen,output_us=float(np.median(times)),times=times))
            num=524288
            count=np.zeros((3,256),np.int64)
            off=0
            for k in range(0,num,8192):
                x,c,off=runner.draw(8192,pnr,104051000+2*i+pnr,off,rate)
                for j,ids in enumerate(hw['ports']):
                    code=(x[:,ids]>0)@keys
                    count[j]+=np.bincount(code,minlength=256)
            assert (count.sum(1)==num).all()
            sim=count/num
            np.savez_compressed(out/f'panels_{label}_{int(pnr)}.npz',sim=count,hardware=oldhw,optical=ref)
            rows.append(dict(label=label,pnr=pnr,records=num,hw_tv=(abs(sim-hardware).sum(1)/2).tolist(),optical_tv=(abs(sim-ref).sum(1)/2).tolist()))
    # Reconstruct the quadratic upper bound without the fitting code, working
    # directly with uncentered unconditional moment masses at each photon number.
    z=np.load(OPT/'moments.npz')
    witness=np.load(FIT/'witness.npz')
    h=witness['h'];nn=witness['selected'];co=witness['coeff']
    p=z['prob'];f=z['click_mass'];s=z['factorial_mass']+f
    upper=1-p[nn].sum()
    repaired=0.
    for n,(a,b,c) in zip(nn,co):
        mu=f[n]/p[n];var=s[n]/p[n]-mu**2
        assert var>.02
        b1=b/np.sqrt(var)-2*c*mu/var;b2=c/var;b0=a-b*mu/np.sqrt(var)+c*mu**2/var
        support=np.arange(n+1)
        residual=max(0.,float(np.max(h[:n+1]-(b0+b1*support+b2*support**2))))
        upper+=p[n]*(b0+residual)+f[n]*b1+s[n]*b2
        repaired=max(repaired,residual)
    assert abs(upper-witness['upper'])<1e-10 and (np.diff(h)>=0).all() and h.min()>=0 and h.max()<=1
    total=hw['total_click_histogram']
    cdf=total.cumsum()/total.sum()
    # At KS distance delta the minimum achievable mean is sum (1-F_hw-delta)_+.
    # Compare the independent mean-only lower bound with the two-moment witness.
    def lower_mean(delta):return np.maximum(0,1-cdf-delta).sum()
    lo,hi=0.,1.
    for _ in range(60):
        mid=(lo+hi)/2
        if lower_mean(mid)>f.sum():lo=mid
        else:hi=mid
    save(out,'summary.json',dict(rows=rows,observed_speed=speed,hardware_panel_records=int(oldhw.sum(1)[0]),hardware_optical_tv=(abs(ref-hardware).sum(1)/2).tolist(),numerical_witness_upper=float(upper),quadratic_repair_max=float(repaired),mean_only_ks_lower_empirical=float(lo),bit_order='physical ports in archive order; least significant bit first',hardware_panels=hw['ports'].tolist()))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
