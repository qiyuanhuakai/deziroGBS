"""Optical third-cumulant calibration of a complete 1152-output click sampler."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import ctypes as C
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from scipy.optimize import least_squares
from iteration92_common_v2 import ROOT,H,save,manifest
from iteration95_count_moment_floor_v1 import moment_floor

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
src=ROOT/'native/gaussian_click_cumulant95_v1.cpp'
prev=ROOT/'results/iteration95_ensemble_v2'
layout=ROOT/'results/iteration83_loss_pilot_v1/input.npz'
hwdir=ROOT/'results/iteration81_jiuzhang_accuracy_v3'
small=ROOT/'results/iteration93_reference_v2'
paths=[ROOT/f'results/iteration94_actual_quality_v1/input_{i}.npz' for i in (0,1)]
files=[Path(__file__),Path(__file__).with_name('iteration95_count_moment_floor_v1.py'),src,
       src.with_name('gaussian_click_window95_v1.cpp'),src.with_name('gaussian_click_window95_v2.cpp'),
       src.with_name('gaussian_rng_domain84_v1.hpp'),layout,*paths,hwdir/'histograms.npz',
       prev/'quality_4_0.npz',prev/'quality_4_2.npz',*[hwdir/f'input_{j}.npz' for j in range(3)],
       *[small/f'{s}_{j}.npz' for j in (0,2) for s in ('input','law')]]
save(out,'plan.json',dict(hashes={q.relative_to(ROOT).as_posix():H(q) for q in files},
    scope='zero displacement; exact sum over all distinct triples, two independent spectra OR combined',
    calibration='nine fixed Gauss-Hermite weights, slopes a+b*z+c*(z*z-1); fit optical moments only; accept if max standardized residual <1e-8',
    comparison='physical order, same 4+4 windows, 2-moment vs 3-moment nine-node tables, count CDF and three selected eight-bit TVs',
    execution='8192 quality records per model with one/eight worker replay. Seven rotated timing measurements, 32768 records, 1/16 workers',
    rng='quality 0x95b00000+degree; timing 0x95b10000+256*(rep+1)+32*degree+threads',
    floor='LP over every count law matching one, two, or three optical moments; DKW shared hardware radius, conditional iid',
    controls='small exact click laws, independently enumerated three eight-port physical reference laws, product-state vacuum and balanced thermal controls'))
compiler=Path(shutil.which('g++'))
cmd=[str(compiler),'-shared','-O3','-std=c++20','-DEIGEN_DONT_PARALLELIZE','-fno-fast-math','-ffp-contract=off',
     '-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-o',str(out/'third.dll')]
b=subprocess.run(cmd,capture_output=True,text=True)
save(out,'build.json',dict(command=cmd,code=b.returncode,stderr=b.stderr));b.check_returncode()
runtime=os.add_dll_directory(str(compiler.parent));lib=C.CDLL(str(out/'third.dll'))
I,P,U=C.c_int,C.c_void_p,C.c_uint64
D,J,B=[np.ctypeslib.ndpointer(dtype=t,flags='C_CONTIGUOUS') for t in (np.float64,np.int32,np.uint8)]
def bind(name,args,rest=I):
    f=getattr(lib,name);f.argtypes=args;f.restype=rest;return f
optical=bind('window95_moments',[I,I,D,D,D,D,D,D])
third=bind('count95_cumulants',[I,I,D,I,D,D])
prepare=bind('window95_prepare',[I,I,I,I,D,D,D],P)
tilt=bind('window95_tilt',[P,I,C.c_double],P)
counts=bind('window95_counts',[P,I,D]);marginal=bind('ensemble95_marginal',[P,I,J,D])
ensemble=bind('ensemble95_prepare',[I,C.POINTER(P),D],P)
draw=bind('ensemble95_sample',[P,I,U,U,I,B])
alias=bind('ensemble95_alias_error',[P,I,P],C.c_double)
release=bind('window95_destroy',[P],None);free=bind('ensemble95_destroy',[P],None)

def q_for(paths):
    nsector=len(paths);m=len(np.load(paths[0])['u'])
    q=np.empty((nsector,2*m,2*m));mu=np.empty((nsector,2*m))
    for s,path in enumerate(paths):
        z=np.load(path);u=np.ascontiguousarray(z['u'])
        assert optical(m,len(z['t']),u.view(float),z['t'],z['alpha'].view(float),z['noise'],q[s],mu[s])==0
    assert not mu.any()
    return q,mu
def cumulants(q,workers):
    m=q.shape[1]//2
    target=np.zeros(3);rows=np.zeros((m,3))
    assert third(m,len(q),q,workers,target,rows)==0
    assert abs(target[1]-rows[:,0].sum())<1e-9 and abs(target[2]-rows[:,1:].sum())<1e-8
    return target,rows
def moments(prob,values):
    mean=prob@values
    return np.array([mean,prob@(values-mean)**2,prob@(values-mean)**3])
def check_law(q,law):
    x=np.array([j.bit_count() for j in range(len(law))])
    observed,_=cumulants(q,1)
    expected=moments(law,x)
    error=float(np.max(abs(observed-expected)))
    assert error<1e-10
    return error
controls={}
for j in (0,2):
    q,_=q_for([small/f'input_{j}.npz'])
    controls[f'small{j}']=check_law(q,np.load(small/f'law_{j}.npz')['click'])
for scale in (1.,1.3):
    q=np.array([scale*np.eye(16)])
    k=np.array([j.bit_count() for j in range(256)])
    inverse=1/scale
    controls[f'product_{scale}']=check_law(q,(1-inverse)**k*inverse**(8-k))
start=time.perf_counter()
q,mu=q_for(paths);m=mu.shape[1]//2
order=np.argsort(np.load(layout)['order']);idx=np.r_[order,order+m]
q=np.ascontiguousarray(q[:,idx][:,:,idx]);mu=np.ascontiguousarray(mu[:,idx])
optical_seconds=time.perf_counter()-start
hw=np.load(hwdir/'histograms.npz');ports=hw['ports']
for i,sub in enumerate(ports):
    ix=np.r_[sub,sub+m]
    controls[f'physical{i}']=check_law(np.ascontiguousarray(q[:,ix][:,:,ix]),
        np.asarray(np.load(hwdir/f'input_{i}.npz')['two_common_sectors'],float))
save(out,'controls.json',controls)
start=time.perf_counter();truth,parts=cumulants(q,16);third_seconds=time.perf_counter()-start
np.savez_compressed(out/'cumulants.npz',truth=truth,contributions=parts)
hist=hw['total_click_histogram'];empirical=hist/hist.sum();x=np.arange(m+1)
hardware=moments(empirical,x);radius=math.sqrt(math.log(200)/(2*hist.sum()))
save(out,'moments.json',dict(optical=truth.tolist(),hardware=hardware.tolist(),seconds=third_seconds,
    triples=math.comb(m,3),hardware_count_radius=radius))
print('moments',truth,third_seconds,flush=True)
floors=moment_floor(hist,truth)
for row,raw in floors:
    row['population_lower']=max(0.,row['dual_lower']-radius)
    np.savez_compressed(out/('floor_%d.npz'%row['degree']),**raw)
save(out,'floors.json',[r for r,_ in floors])
start=time.perf_counter();diag=np.zeros(4)
base=prepare(m,4,4,2,q,mu,diag);assert base
base_count=np.zeros(m+1);assert counts(base,4,base_count)==0
window_seconds=time.perf_counter()-start
orig=np.load(prev/'quality_4_2.npz');base_ref=np.load(prev/'quality_4_0.npz')['count']
assert abs(base_count-base_ref).sum()<1e-10
nodes,weights=np.polynomial.hermite.hermgauss(9);nodes*=math.sqrt(2);weights/=weights.sum()
assert np.max(abs(weights-orig['weights']))<1e-14
mask=base_count>0;logp=np.log(base_count[mask]);score=(x[mask]-truth[0])/np.sqrt(truth[1])
features=np.stack([score,score**2,score**3],axis=1)
target=np.array([0.,1.,truth[2]/truth[1]**1.5])
design=np.stack([np.ones(9),nodes,nodes**2-1],axis=1)
initial=np.linalg.lstsq(design,orig['slopes'],rcond=None)[0]
def model(parameters):
    mass=logp[None,:]+(design@parameters)[:,None]*x[mask]
    mass=np.exp(mass-mass.max(1,keepdims=True))
    return weights@(mass/mass.sum(1,keepdims=True))
def residual(parameters):return model(parameters)@features-target
start=time.perf_counter()
fit=least_squares(residual,initial,bounds=([-1.,0.,-.08],[1.,.8,.08]),xtol=1e-14,gtol=1e-14,ftol=1e-14)
fit_seconds=time.perf_counter()-start
save(out,'fit.json',dict(parameters=fit.x.tolist(),residual=residual(fit.x).tolist(),seconds=fit_seconds))
assert np.max(abs(residual(fit.x)))<1e-8
models={};summary=[]
for degree,slopes in [(2,orig['slopes']),(3,design@fit.x)]:
    start=time.perf_counter()
    components=[tilt(base,4,s) for s in slopes];assert all(components)
    handle=ensemble(9,(P*9)(*components),weights);assert handle;models[degree]=handle
    build_seconds=time.perf_counter()-start
    bound=max(alias(handle,i,c) for i,c in enumerate(components))+9*2**-52
    assert 0<=bound<1e-10
    count=np.zeros(m+1);local=np.zeros((3,256))
    for c,w in zip(components,weights):
        temp=np.empty(m+1);assert counts(c,4,temp)==0;count+=w*temp
        for g,subset in enumerate(ports):
            v=np.empty(256);assert marginal(c,4,np.ascontiguousarray(subset,dtype=np.int32),v)==0
            local[g]+=w*v
        release(c)
    if degree==2:
        assert max(abs(count-orig['count']))<1e-11 and np.max(abs(local-orig['marginal']))<1e-11
    if degree==3:assert max(abs(count[mask]-model(fit.x)))<1e-10
    ks=float(max(abs(count.cumsum()-empirical.cumsum())))
    bits=np.empty((8192,m),np.uint8);check=np.empty_like(bits)
    exp=0x95b00000+degree
    assert draw(handle,len(bits),exp,0,1,bits)==0 and draw(handle,len(bits),exp,0,8,check)==0
    assert np.array_equal(bits,check)
    sampled=np.bincount(bits.sum(1).astype(int),minlength=m+1).cumsum()/len(bits)
    assert max(abs(sampled-count.cumsum()))<.04
    row=dict(degree=degree,ks=ks,interval=[max(0.,ks-radius-bound),min(1.,ks+radius+bound)],
             moments=moments(count,x).tolist(),local_tv=(abs(local-orig['ideal']).sum(1)/2).tolist(),
             construction_seconds=build_seconds,alias_bound=bound)
    summary.append(row)
    np.savez_compressed(out/f'quality_{degree}.npz',count=count,marginal=local,slopes=slopes,weights=weights,
        packed_bits=np.packbits(bits,axis=1,bitorder='little'))
timings=[]
schedule=list(itertools.product((2,3),(1,16)))
for rep in range(-1,7):
    for degree,threads in schedule[rep%4:]+schedule[:rep%4]:
        n=128 if rep<0 else 32768
        start=time.perf_counter();bits=np.empty((n,m),np.uint8)
        rc=draw(models[degree],n,0x95b10000+256*(rep+1)+32*degree+threads,0,threads,bits)
        elapsed=time.perf_counter()-start;assert rc==0
        if rep>=0:timings.append(dict(degree=degree,threads=threads,n=n,rep=rep,seconds=elapsed))
for h in models.values():free(h)
release(base)
save(out,'summary.json',dict(rows=summary,timings=timings,optical_seconds=optical_seconds,
    window_seconds=window_seconds,triple_seconds=third_seconds,fit_seconds=fit_seconds))
manifest(out)
