"""Check native ring records against a Gaussian click oracle and Fock probabilities."""
import argparse
import ctypes as ct
import json
import math
import os
import shutil
from collections import defaultdict
from decimal import Decimal,localcontext
from pathlib import Path
from time import perf_counter
from fractions import Fraction
from iteration98_cycle_v1 import ROOT,CAP,P,np,draw,tables,couplers,alias_table,bind,I,F,W,T,D,U,B,H,save,manifest
from iteration96_group_benchmark_v1 import fock
from iteration97_conditionals_v1 import Kernel
from iteration92_quality_v1 import confidence

def six_photon(fn,local,edges,route):
    # Two unresolved idler occupations with identical click mask per cell.
    # Postselect vacuum at both internal signal ports. Each cell heralds three photons.
    idlers=[(2,1),(1,2)]
    weights={}
    payload_group=dict(zip(map(int,local[3]),map(int,local[4])))
    phase=.674
    fourier=np.exp(2j*np.pi*np.outer(np.arange(4),np.arange(4))/4)/2
    pair=np.array([[1,1],[-1,1]])/np.sqrt(2)
    v=np.zeros((4,4),complex)
    v[:2,:2]=fourier[[0,3],:2]
    v[2:,2:]=fourier[[0,3],:2]
    initial=v.copy()
    v[3]*=np.exp(1j*phase)
    for a,b in [(1,2),(3,0)]:v[[a,b]]=pair@v[[a,b]]
    def optics(unitary,occ):
        terms={(0,)*len(unitary):1.+0j}
        for col,total in enumerate(occ):
            for _ in range(total):
                more=defaultdict(complex)
                for counts,z in terms.items():
                    for j,u in enumerate(unitary[:,col]):
                        more[counts[:j]+(counts[j]+1,)+counts[j+1:]]+=z*u
                terms=more
        return {key:abs(z)**2*math.prod(map(math.factorial,key))/math.prod(map(math.factorial,occ)) for key,z in terms.items()}
    for m in idlers:
        idle=(m[0]+16*m[1]) if route else 17
        for (left,right),prob in optics(initial[:2,:2],m).items():
            key=left+(right<<12)+(idle<<16)
            weights[key]=weights.get(key,0)+int(round(prob*64))
    bits,cut,alias,words,_,_=alias_table(weights)
    ids=np.array([payload_group[int(key)] for key in words],np.uint32)
    fixed=(bits,cut,alias,words,ids,local[5],0)
    target=defaultdict(float)
    for m in idlers:
        for n in idlers:
            for counts,p in optics(v,m+n).items():
                label=counts+m+n if route else tuple(int(t>0) for t in counts)
                target[label]+=p
    norm=sum(target.values())
    x,flags=draw(fn,fixed,edges,2,route,phase,1,131072,98006000+route,0,16)
    data=x[:,[0,3,4,7,8,9,12,13] if route else [0,3,4,7]]
    labels,freq=np.unique(data,axis=0,return_counts=True)
    empirical={tuple(k):h/len(data) for k,h in zip(labels,freq)}
    tv=sum(abs(empirical.get(k,0.)-target.get(k,0.)/norm) for k in target.keys()|empirical.keys())/2
    assert tv<(.036 if route else .011)
    return dict(records=len(x),empirical_tv=tv,ambiguous_records=int(flags.sum()),phase=phase)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    out=p.parse_args().output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'results'
    verified={}
    for folder in ['iteration98_cycle_v1','iteration98_optics_v1','iteration98_baselines_v1']:
        src=data/folder
        for name,digest in json.loads((src/'manifest.json').read_text()).items():
            assert H(src/name)==digest
            verified[folder+'/'+name]=digest
    with np.load(data/'iteration98_optics_v1/input.npz') as z:
        click=z['click'];barg=z['barg'];q=z['q']
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    lib=ct.CDLL(str(data/'iteration98_cycle_v1/cycle.dll'))
    fn=bind(lib,'cycle98_draw',[I,I,I,I,I,F,W,T,T,T,D,D,W,I,U,U,I,B,B])
    local,survival=tables(lib,CAP,P)
    edges=couplers(CAP,Fraction(1,2))
    occ,prob=fock(Kernel().prior,barg,q,8)
    shifts=4*np.arange(16,dtype=np.uint64)
    keys=np.sum(occ.astype(np.uint64)<<shifts,axis=1)
    order=keys.argsort()
    keys=keys[order];prob=prob[order]
    tail=float(1-prob.sum())
    pnr=np.r_[prob,tail,0.]
    rows=[]
    for route in (0,1):
        control=six_photon(fn,local[route],edges,route)
        N=131072
        x,_=draw(fn,local[route],edges,2,route,0.,1,N,98007000+route,0,16)
        if route:
            raw=np.sum(x.astype(np.uint64)<<shifts,axis=1)
            pos=np.minimum(len(keys)-1,np.searchsorted(keys,raw))
            labels=np.where(x.sum(1)<=8,np.where(keys[pos]==raw,pos,len(keys)+1),len(keys))
            target=pnr
        else:
            labels=x@(1<<np.arange(16));target=click
        stats,hist=confidence(labels,target,N,32768,.0025,tail if route else 0.)
        np.savez_compressed(out/f'hist_{route}.npz',**hist)
        times=[]
        for k in range(5):
            start=perf_counter()
            draw(fn,local[route],edges,2,route,0.,1,2048,98008000+route,2048*k,1)
            times.append((perf_counter()-start)*1e6/2048)
        rows.append(dict(route=route,records=N,observed=stats,one_worker_2048_us=times,six_photon=control))
    # All lossless ring outputs have even total photon number. The lossy two-spectrum
    # optical reference gives an independently checkable lower bound on the joint TV.
    ref=data/'iteration84_photon_cdf_v1/reference.json'
    with localcontext() as ctx:
        ctx.prec=85
        r=json.loads(ref.read_text())
        odd=sum(map(Decimal,r['probabilities'][1::2]))
        eig=np.array(r['eigenvalues'])
        parity=-np.expm1(-.5*np.log1p(2*eig).sum())/2
        assert abs(float(odd)-parity)<1e-14
    save(out,'validation.json',dict(passed=True,verified_files=len(verified),rows=rows,cap=CAP,p=str(P),
         parity_reference_sha256=H(ref),optical_odd_mass_through_1024=str(odd),odd_mass_from_determinant=parity,
         photon_tail_upper=r['tail_upper'],fock_records_through_eight=len(prob),fock_tail=tail))
    manifest(out)
if __name__=='__main__':main()
