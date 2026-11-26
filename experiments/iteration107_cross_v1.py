"""19-bit Gaussian marginal plus a coupling bound for the three other touched tiles."""
import argparse
import ctypes as ct
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,manifest,save,bind,I,D
from iteration106_field_v1 import factors
from iteration106_sample_v1 import DATA
from iteration107_cliques_v1 import optical_x
from iteration107_fast_v1 import ONE,OUT as FAST,gaussian_click,oracle


def run(out):
    out.mkdir(exist_ok=False,parents=True)
    origin=(ROOT/'native/click_oracle96_v1.cpp').read_text()
    expanded=origin.replace('[36]','[38]').replace('m>18','m>19')
    source=expanded.replace('using Q=__float128;','using Q=long double; static_assert(sizeof(Q)>8);').replace('sqrtq(','std::sqrt(').replace('expq(','std::exp(')
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'oracle.cpp';p.write_text(source)
        subprocess.run([shutil.which('g++'),'-O3','-ffp-contract=off','-std=c++20','-shared',str(p),'-o',str(out/'extended.dll')],check=True)
    q=oracle(True);long=bind(ct.CDLL(str(out/'extended.dll')),'oracle96',[I,I,D,D,D,D])
    with np.load(DATA/'iteration107_rank_v1/input4096.npz') as z:t,r=z['transfer'],z['squeezing']
    pars=dict(np.load(FAST/'case4096.npz'));main=pars['primary_ids']
    ids=np.array([0,1,2,3,4,1367,2731,4095]);remote=ids[5:];assert np.array_equal(main,np.arange(16))
    def base(use,ref):
        f=factors(t[use],r)
        return gaussian_click(ref,np.stack([2*v.T@v for v in f]),pars)
    sub=base(np.r_[main,remote],long).reshape(8,-1)
    raw=np.array([np.bincount(np.arange(65536)%32,weights=row,minlength=32) for row in sub]).ravel()
    ctrl=base(ids,q);mismatch=float(abs(raw-ctrl).sum()/2)
    main_check=float(abs(sub.sum(0)-base(main,q)).sum()/2)
    assert max(mismatch,main_check)<1e-8
    move=pars['click_move'][:65536].astype(float)
    cdf=pars['click_cdf'][:65536];projection=np.arange(65536)%32
    excess=np.bincount(projection,weights=np.diff(cdf,prepend=np.uint64(0))/2**52,minlength=32)
    partial=np.stack([np.bincount(projection,weights=(1-move)*row,minlength=32)+(row@move)*excess for row in sub]).ravel()
    true=gaussian_click(q,optical_x(t[ids],r),ONE)
    impact=0.;bounds=[]
    for tile in pars['click_plan'][1:]:
        dim,start,size=tile[-3:];ports=tile[:dim];kept=[j for j,k in enumerate(ports) if k in remote]
        if not kept:continue
        obs=sum(((np.arange(size)>>j)&1)<<i for i,j in enumerate(kept));ref=base(ports,q)
        sl=slice(start,start+size);remove=ref*pars['click_move'][sl]
        replacement=np.diff(pars['click_cdf'][sl],prepend=np.uint64(0))/2**52
        loss=np.bincount(obs,weights=remove,minlength=2**len(kept))
        excess=np.bincount(obs,weights=replacement,minlength=len(loss))
        prob=float(loss@(1-excess));impact+=prob
        bounds.append(dict(ports=ports.tolist(),query=ports[kept].tolist(),change_probability=prob))
    center=float(abs(partial-true).sum()/2)
    save(out,'summary.json',dict(ids=ids.tolist(),raw_tv=float(abs(ctrl-true).sum()/2),partial_tv=center,remote= bounds,
            lower=max(0,center-impact),upper=center+impact,oracle_raw_eight_check=mismatch,oracle_raw_sixteen_check=main_check,
            negative_mass_19=float(-sub[sub<0].sum()),scope='Numerical Gaussian-field law under the deployed table probabilities. TV interval uses a union bound for changed remote observed bits.'))
    np.savez_compressed(out/'projection.npz',base=ctrl,primary_only=partial,target=true)
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
