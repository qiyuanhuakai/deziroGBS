"""Full-record error under a dense passive phase coupler spanning all otherwise independent cells."""
import argparse
import ctypes as ct
import os
import shutil
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest,realify
from iteration100_baselines_v1 import frugal
from iteration106_field_v1 import DATA
from iteration107_fast_v1 import oracle,gaussian_click,ONE
from iteration109_sample_v1 import tv


def bound(x,v,cells,theta,size):
    e,u=np.linalg.eigh(x);row=realify(v.conj()[None,:])@u
    z=np.exp(-2j*np.pi*np.arange(size)/size)
    den=1+(1-z[:,None])*e
    full=np.exp(-cells*.5*np.log(den).sum(1))
    tilted=np.einsum('ij,zj,kj->zik',row,e/den,row,optimize=True)
    determinant=np.linalg.det(np.eye(2)+(z*(1-np.exp(1j*theta)))[:,None,None]*tilted)
    phase=np.unwrap(np.angle(determinant))
    b=full*np.exp(-.5*(np.log(abs(determinant))+1j*phase))
    p=np.fft.ifft(full).real;over=np.fft.ifft(b)
    coarse=float(np.sqrt(np.maximum(0.,p**2-abs(over)**2)).sum())
    unconditional=float(np.sqrt(max(0.,1-abs(b[0])**2)))
    return unconditional,min(coarse,unconditional),float(np.maximum(abs(over)-p,0).sum())


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    data=DATA/'iteration110_collective_v1';x=np.load(data/'optical.npz')['core_x'];d=len(x)//2
    v=np.array([1+.8j,.4-.9j,-.7+.5j,.8+.2j]);v/=np.linalg.norm(v)
    small=[];wide=[];arrays={}
    o=oracle(True);base=gaussian_click(o,x[None],ONE)
    p=base
    for j in range(3):p=np.kron(base,p)
    identity=np.eye(16);vv=np.tile(v,4)/2
    block=np.block([[np.kron(np.eye(4),x[a*d:(a+1)*d,b*d:(b+1)*d]) for b in (0,1)] for a in (0,1)])
    equal=gaussian_click(o,block[None],ONE);assert tv(equal,p)<3e-10
    rt=os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
    lib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
    for i,theta in enumerate((.003,.02,.1,.4,1.,np.pi)):
        w=identity+np.expm1(1j*theta)*np.outer(vv,vv.conj());f=realify(w);xx=f@block@f.T
        truth=gaussian_click(o,xx[None],ONE);gap=tv(p,truth)
        full,sector,res=bound(x,v,4,theta,512);assert gap<=sector+1e-9
        # Independent full determinant on 16 physical modes at a complex fugacity.
        z=-.37+.27j;uu=realify(vv.conj()[None,:]);pbus=uu.T@uu
        D=np.linalg.det(np.eye(32)+(np.eye(32)-z*(np.eye(32)+np.expm1(1j*theta)*pbus))@block)
        ev,V=np.linalg.eigh(x);L=realify(v.conj()[None,:])@V;den=1+(1-z)*ev
        check=np.prod(den)**4*np.linalg.det(np.eye(2)-z*np.expm1(1j*theta)*(L*(ev/den))@L.T)
        assert abs(D/check-1)<1e-12
        row=dict(theta=float(theta),tv=gap,full_state_bound=full,total_sector_bound=sector,numerical_residual=res)
        if i in (2,4):
            for k in (2,4):
                fr,tm=frugal(lib,(np.eye(32)+xx)[None],k,110360000+100*i+10*k)
                row[f'frugal{k}']=dict(model_tv=tv(fr,truth),us=float(np.median(tm)))
        small.append(row);arrays[f'truth{i}']=truth
        # All 131,072 detectors participate in the passive coupler. Conditioning
        # on N modulo 65,536 is sufficient for an upper bound at any occupancy.
        whole,cond,bad=bound(x,v,32768,theta,65536)
        _,check2,_=bound(x,v,32768,theta,32768)
        wide.append(dict(theta=float(theta),full_state_bound=whole,total_sector_bound=cond,
                         half_grid_difference=abs(check2-cond),numerical_residual=bad))
    np.savez_compressed(out/'laws.npz',base=p,bus=v,**arrays)
    save(out,'summary.json',dict(small=small,wide=wide,wide_ports=131072,wide_cells=32768,
                                 mean_photons=float(x.trace()*32768/2)));manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(ap.parse_args().output.resolve())
