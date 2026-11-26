"""Benchmark coherent terminal detection on a ring of four-mode Fourier cells."""
import argparse
import ctypes as ct
import math
import os
import shutil
import subprocess
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from time import perf_counter

from iteration97_forest_v1 import alias_table, edge_tables, transition
from iteration96_group_common_v1 import ROOT, np, bind, I, F, D, T, J, W, U, B, H, save, manifest

P = Fraction(36,625)
CAP = 7

def tables(lib,cap,p):
    nmax=sum(math.comb(i+3,3)**2 for i in range(cap+1))
    keys=np.empty(nmax,np.uint32)
    re=np.empty(nmax,np.int32)
    im=np.empty_like(re)
    count=bind(lib,'cycle98_coefficients',[I,I,T,J,J])(cap,nmax,keys,re,im)
    assert count>0
    fac=math.factorial
    weights=[defaultdict(int),defaultdict(int)]
    contexts=[{},{}]
    entries=[defaultdict(list),defaultdict(list)]
    for word,real,imag in zip(keys[:count],re[:count],im[:count]):
        key=int(word)
        n=[key>>(4*i)&15 for i in range(4)]
        m=[key>>(16+4*i)&15 for i in range(4)]
        total=sum(m)
        sf=math.prod(map(fac,n))
        mf=math.prod(map(fac,m))
        coeff=complex(int(real),int(imag))
        weight=(int(real)**2+int(imag)**2)*sf*(fac(cap)//mf)*p.numerator**total*(4*p.denominator)**(cap-total)
        amp=coeff*math.sqrt(sf/(mf*4**total))
        marks=sum(int(v>0)<<(4*i) for i,v in enumerate(m))
        for route,reported in enumerate([marks,key>>16]):
            payload=(key&65535)+(reported<<16)
            context=(n[1],n[2],n[0]+n[3],reported)
            weights[route][payload]+=weight
            contexts[route][payload]=context
            entries[route][context].append((key>>16,n[3],amp))
    result=[]
    for route in (0,1):
        bits,cut,alias,payload,qtv,_=alias_table(weights[route])
        grams=np.zeros((len(entries[route]),cap+1,cap+1),complex)
        ids={}
        for gid,(context,items) in enumerate(entries[route].items()):
            ids[context]=gid
            idlers=sorted({m for m,_,_ in items})
            amp=np.zeros((cap+1,len(idlers)),complex)
            for m,right,value in items:
                amp[right,idlers.index(m)]=value
            block=amp@amp.conj().T
            grams[gid]=block/np.diag(block).real.max()
        groups=np.array([ids[contexts[route][int(key)]] for key in payload],np.uint32)
        result.append((bits,cut,alias,payload,groups,grams.view(float),qtv))
    survival=sum(math.comb(n+3,3)*p**n*(1-p)**4 for n in range(cap+1))
    for w in weights:
        assert Fraction(sum(w.values()),(4*p.denominator)**cap*fac(cap))*(1-p)**4==survival
    return result,survival

def couplers(cap,r):
    # Real Fock amplitudes of [[sqrt(1-r),sqrt(r)],[-sqrt(r),sqrt(1-r)]].
    values=np.zeros((cap+1,cap+1,2*cap+1))
    for n in range(cap+1):
        for m in range(cap+1):
            for k in range(n+m+1):
                lo=max(0,k-m)
                hi=min(n,k)
                sign=sum((-1)**(n-j)*math.comb(n,j)*math.comb(m,k-j)*(1-r)**(j-lo)*r**(hi-j) for j in range(lo,hi+1))
                values[n,m,k]=math.copysign(math.sqrt(transition(n,m,k,r)),sign)
    cuts,errors=edge_tables(cap,r)
    return values,cuts,errors

def draw(fn,table,edges,cells,route,phase,coherent,n,dom,offset,workers):
    bits,cut,alias,words,groups,gram,_=table
    amp,cuts,_=edges
    x=np.empty((n,8*cells),np.uint8)
    ambiguous=np.empty(n,np.uint8)
    rc=fn(cells,route,bits,len(amp)-1,coherent,phase,cut,alias,words,groups,gram,amp,cuts[route],n,dom,offset,workers,x,ambiguous)
    if rc: raise RuntimeError(('cycle98_draw',rc,dom,offset))
    return x,ambiguous

def fixture(fn,table,edges,route,out):
    # One herald in Fourier input 0 and both interior ports vacuum in each cell.
    bits,cut,alias,words,groups,grams,qtv=table
    local=np.array([65537,65536+4096],np.uint32)
    group=np.array([groups[np.flatnonzero(words==k)[0]] for k in local],np.uint32)
    assert group[0]==group[1]
    fixed=(1,np.array([2**63]*2,np.uint64),np.array([0,1],np.uint32),local,group,grams,0.)
    rows=[]
    size=131072
    for j,phase in enumerate([0.,math.pi/2,math.pi]):
        # Independent optical reference: two single-photon columns, one in each cell.
        v=np.zeros((4,2),complex)
        v[[0,1],0]=1/math.sqrt(2)
        v[[2,3],1]=1/math.sqrt(2)
        v[3]*=np.exp(1j*phase)
        bs=np.array([[1,1],[-1,1]])/math.sqrt(2)
        for left,right in [(1,2),(3,0)]:v[[left,right]]=bs@v[[left,right]]
        def optical_law(v):
            law=np.zeros(256)
            for a in range(4):
                for b in range(a,4):
                    occ=np.bincount([a,b],minlength=4)
                    coef=v[a,0]*v[b,1]+v[b,0]*v[a,1]
                    key=int((occ if route else (occ>0))@np.array([1,4,16,64]))
                    law[key]+=abs(coef)**2/(1+(a==b))
            return law
        target=optical_law(v)
        for coherent in (0,1):
            x,_=draw(fn,fixed,edges,2,route,phase,coherent,size,98002000+100*route+2*j+coherent,0,16)
            labels=x[:,[0,3,4,7]]@np.array([1,4,16,64])
            hist=np.bincount(labels,minlength=256)/size
            tv=float(abs(hist-target).sum()/2)
            assert abs(tv-(0. if coherent else .25*abs(math.cos(phase))))<.012
            rows.append(dict(route=route,phase=phase,coherent=coherent,records=size,tv_to_optics=tv,
                             forbidden_observed=float(hist[target<1e-14].sum()),hist=hist.tolist(),truth=target.tolist()))
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    out=parser.parse_args().output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sources=[ROOT/'native/gaussian_cycle98_v1.cpp',ROOT/'native/fourier_cell98_v1.cpp']
    files=[*sources,Path(__file__),ROOT/'native/gaussian_rng_domain84_v1.hpp',ROOT/'experiments/iteration97_forest_v1.py']
    save(out,'plan.json',dict(sources={p.relative_to(ROOT).as_posix():H(p) for p in files},p=str(P),cap=CAP,
         cases=[2,16,144],reflectivity='1/2',phase=0,repetitions=7,records_per_batch=8192,workers=[1,16],
         fixture='two single heralds, interior outputs vacuum; phases 0, pi/2, pi; TV tolerance .012 for 131072 draws'))
    dll=out/'cycle.dll'
    cmd=[shutil.which('g++'),'-std=c++20','-O3','-shared','-fopenmp','-ffp-contract=off',*map(str,sources),'-o',str(dll)]
    result=subprocess.run(cmd,text=True,capture_output=True)
    save(out,'build.json',dict(command=cmd,code=result.returncode,stderr=result.stderr))
    result.check_returncode()
    runtime=os.add_dll_directory(str(Path(cmd[0]).parent))
    lib=ct.CDLL(str(dll))
    fn=bind(lib,'cycle98_draw',[I,I,I,I,I,F,W,T,T,T,D,D,W,I,U,U,I,B,B])
    t=perf_counter()
    local,survival=tables(lib,CAP,P)
    edge=couplers(CAP,Fraction(1,2))
    prep=perf_counter()-t
    controls=[]
    rows=[]
    for route in (0,1):
        controls+=fixture(fn,local[route],edge,route,out)
        for j,cells in enumerate([2,16,144]):
            timings=[]
            dom=98001000+100*route+j
            def sample(n,offset,workers):
                return draw(fn,local[route],edge,cells,route,0.,1,n,dom,offset,workers)
            x,flags=sample(512,0,1)
            replay=sample(480,32,4)
            assert np.array_equal(x[32:],replay[0]) and np.array_equal(flags[32:],replay[1])
            total_ambiguous=0
            for workers in (1,16):
                times=[]
                for k in range(7):
                    start=perf_counter()
                    x,flags=sample(8192,65536+(workers*7+k)*8192,workers)
                    times.append((perf_counter()-start)*1e6/8192)
                    total_ambiguous+=int(flags.sum())
                    if route:assert np.array_equal(x[:,:4*cells].sum(1),x[:,4*cells:].sum(1))
                    else:assert x.max()<=1
                timings.append(dict(workers=workers,us=times,median=float(np.median(times))))
            ambiguous=2*P**cells-P**(2*cells)
            tail=1-survival**cells
            bound=float(tail+ambiguous)+cells*(local[route][-1]+edge[2][route])
            rows.append(dict(cells=cells,ports=8*cells,route=['direct_click','PNR'][route],prep_s=prep,timing=timings,
                coherent_truncation_tv=float(tail),ambiguous_mass=float(ambiguous),ambiguous_records=total_ambiguous,
                records=14*8192,native_conservative_tv=bound,alias_quant_tv=local[route][-1],edge_quant_tv=edge[2][route]))
            print(cells,route,bound,timings[-1]['median'],flush=True)
    save(out,'summary.json',rows)
    save(out,'controls.json',controls)
    manifest(out)
if __name__=='__main__':main()
