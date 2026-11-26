"""Independent optical Fock, held-out port and prepared sampler checks for round 99."""
import argparse
import ctypes as ct
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
from scipy.stats import beta
from threadpoolctl import threadpool_limits
from iteration99_marks_v1 import ROOT,I,J,B,T,D,np,bind,H,save,manifest,Spectral,Marked,physical,ORDER
from iteration99_integer_v1 import IntegerRing,CAP,P,Rat
from iteration99_sample_v1 import PreparedActual
from iteration96_group_benchmark_v1 import fock
from iteration97_conditionals_v1 import Kernel

def heldout_groups(z,calibrated):
    v=z['v'][np.argsort(z['order'])]
    a=np.column_stack((v.real,-v.imag));b=np.column_stack((v.imag,v.real))
    cov=np.zeros((len(v),len(v)));mean=np.zeros(len(v))
    for x in z['x']:
        ax=a@x;bx=b@x
        mean+=((ax*a).sum(1)+(bx*b).sum(1))/2
        cov+=((ax@a.T)**2+(ax@b.T)**2+(bx@a.T)**2+(bx@b.T)**2)/2
    np.fill_diagonal(cov,0)
    available=np.ones(len(v),bool);available[calibrated]=False
    bright=np.where(available)[0][np.argsort(-mean[available])[:16]]
    available[bright]=False
    pair=np.unravel_index(np.where(available[:,None]&available,cov,-1).argmax(),cov.shape)
    cluster=list(pair);available[cluster]=False
    while len(cluster)<16:
        i=int(np.where(available,cov[cluster].sum(0),-1).argmax());cluster.append(i);available[i]=False
    consecutive=np.flatnonzero(available)[:8]
    return [bright,np.array(cluster),consecutive]

def main():
    arg=argparse.ArgumentParser();arg.add_argument('output',type=Path)
    out=arg.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results';verified=0;source_hashes=0
    for stage in ['loss','baselines','integer','spectral','marks']:
        folder=data/f'iteration99_{stage}_v1'
        for name,digest in json.loads((folder/'manifest.json').read_text()).items():
            assert H(folder/name)==digest;verified+=1
        for name,digest in json.loads((folder/'plan.json').read_text())['hashes'].items():
            assert H(ROOT/name)==digest;source_hashes+=1
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    lib=ct.CDLL(str(data/'iteration99_loss_v1/loss.dll'))
    integer=ct.CDLL(str(data/'iteration99_integer_v1/integer.dll'))
    oracle=bind(ct.CDLL(str(data/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    src=ROOT/'native/verify_fock99_v1.cpp';dll=out/'check.dll'
    cmd=[shutil.which('g++'),'-std=c++20','-O3','-shared','-fopenmp','-ffp-contract=off',str(src),'-o',str(dll)]
    run=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr));run.check_returncode()
    check=bind(ct.CDLL(str(dll)),'verify99_fock',[I,T,J,J,I,B,ct.c_double,D])
    bound=sum(math.comb(n+3,3)**2 for n in range(5))
    keys=np.empty(bound,np.uint32);re=np.empty(bound,np.int32);im=np.empty_like(re)
    size=bind(lib,'cycle98_coefficients',[I,I,T,J,J])(4,bound,keys,re,im);assert size>0
    archived=np.load(data/'iteration98_optics_v1/input.npz')
    occ,truth=fock(Kernel().prior,archived['barg'],archived['q'],8);computed=np.empty_like(truth)
    assert check(size,keys,re,im,len(occ),occ,float(P),computed)==0
    maxerr=float(abs(truth-computed).max());l1=float(abs(truth-computed).sum());assert l1<2e-13
    for j,(s,i) in enumerate([(0,1),(1,0),(1,1)]):
        ring=IntegerRing(lib,integer,P,Rat(s),[Rat(i)]*4)
        x,_=ring.sample(2,1,8192,99082000+j,0,8)
        if not s:assert not x[:,:8].any() and x[:,8:].any()
        if not i:assert not x[:,8:].any() and x[:,:8].any()
        if s*i:assert np.array_equal(x[:,:8].sum(1),x[:,8:].sum(1))
    z=np.load(ORDER);s=Spectral(lib,z);group=np.arange(16)*8
    model=Marked(s,ct.CDLL(str(data/'iteration99_marks_v1/marks.dll')),oracle,z,group)
    cache=out/'prepared.npz'
    np.savez(cache,counts=s.counts,cut=s.cut,alias=s.alias,group=model.group,keep=model.keep,single=model.single,
             dcut=model.dcut,dalias=model.dalias,words=model.words)
    groups=heldout_groups(z,group)
    save(out,'plan.json',dict(seed=99081000,pilot=131072,validation=1048576,alpha=.01,
         groups=[g.tolist() for g in groups],selection='brightest sixteen, disjoint optical covariance cluster, first eight remaining physical ports',
         hashes={f.relative_to(ROOT).as_posix():H(f) for f in [Path(__file__),src,ROOT/'experiments/iteration99_sample_v1.py',
         ROOT/'native/exact_ring99_v1.cpp',data/'iteration98_optics_v1/input.npz',ORDER]}))
    ref=[]
    for g in groups:
        _,q=physical(z,g);v=np.empty(1<<len(g))
        assert oracle(len(g),2,q,np.zeros((2,2*len(g))),v,np.empty(2))==0
        assert abs(v.sum()-1)<1e-12 and v.min()>-1e-14;ref.append(v)
    pilot=[np.zeros(len(p),np.int64) for p in ref];valid=[p.copy() for p in pilot];load_times=[]
    for rep in range(5):
        t=perf_counter();runner=PreparedActual(cache,verify=False);runner.sample(2048,0,99083000+rep,0,1)
        load_times.append((perf_counter()-t)*1e6/2048)
    for route in (0,1):
        expect=model.sample(256,route,99083020+route,0,4)
        actual=runner.sample(256,route,99083020+route,0,2)
        assert all(np.array_equal(a,b) for a,b in zip(actual,expect))
    for target,reps,off in [(pilot,32,0),(valid,256,2**20)]:
        for rep in range(reps):
            x,_=runner.sample(4096,0,99081000,off+4096*rep,16)
            for h,g in zip(target,groups):
                labels=x[:,g]@(1<<np.arange(len(g)));h+=np.bincount(labels,minlength=len(h))
    rows=[]
    for j,(g,p,h,v) in enumerate(zip(groups,ref,pilot,valid)):
        event=h/h.sum()>p
        mass=float(p[event].sum());k=int(v[event].sum());n=int(v.sum())
        lo=float(beta.ppf(.01/len(groups),k,n-k+1)) if k else 0.
        rows.append(dict(ports=g.tolist(),pilot=int(h.sum()),validation=n,validation_hits=k,optical_event=mass,
            sample_event=k/n,observed_gap=k/n-mass,simultaneous_tv_lower=max(0,lo-mass)))
        np.savez_compressed(out/f'heldout_{j}.npz',truth=p,pilot=h,validation=v,event=event)
    save(out,'validation.json',dict(passed=True,verified_artifacts=verified,verified_source_hashes=source_hashes,
        fock_patterns=len(occ),fock_max_error=maxerr,fock_l1_error=l1,heldout=rows,
        full_1152_click_and_pnr_tv_lower=max(row['simultaneous_tv_lower'] for row in rows),
        cached_read_and_first_batch_us=load_times,cache_bytes=cache.stat().st_size,cache_sha256=H(cache),
        confidence='one-sided exact binomial bounds, joint 99% for three pilot-selected validation events, conditional iid; PNR follows by thresholding'))
    manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
