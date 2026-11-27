"""Rejection-free refinement of coarse joint words for a nonuniform four-cell circuit."""
import argparse
import ctypes as ct
import json
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from iteration111_sampling_v1 import DATA,ROOT,alias
from iteration112_cases_v1 import optics,local_tables,click_table


def fourier(x,disp,radix,keep=None):
    d=len(x)//2
    grid=np.indices((radix,)*d).reshape(d,-1).T
    z=np.tile(np.exp((-2j*np.pi/radix)*grid),(1,2))
    kernel=np.eye(2*d)-np.linalg.inv(np.eye(2*d)+x)
    base=np.empty(len(grid),complex)
    i,j=np.triu_indices(2*d)
    forms=np.empty((len(grid),len(i)),complex)
    for off in range(0,len(grid),2048):
        sl=slice(off,off+2048);v=z[sl];t=1-v
        e=np.linalg.eigvals(v[:,:,None]*kernel)
        base[sl]=np.exp(-.5*np.log1p(-e).sum(1))/np.sqrt(np.linalg.det(np.eye(2*d)+x))
        matrix=np.linalg.solve(np.eye(2*d)+t[:,:,None]*x,t[:,:,None]*np.eye(2*d))
        forms[sl]=matrix[:,i,j]*(1+(i!=j))
    if keep is None:keep=np.arange(len(grid))
    out=np.empty((len(keep),len(disp)),complex)
    order=np.arange(len(grid)).reshape((radix,)*d).ravel(order='F')[keep]
    for off in range(0,len(disp),12):
        sl=slice(off,off+12)
        v=disp[sl]
        transformed=base[:,None]*np.exp(-(forms@(v[:,i]*v[:,j]).T))
        arr=np.fft.ifftn(transformed.reshape((radix,)*d+(-1,)),axes=tuple(range(d))).reshape(len(grid),-1)
        a=arr[order].copy();a[0]+=1-a.sum(0)
        out[:,sl]=a
    return out


def combinations(pos,c):
    k=len(c)
    dist=cdist(pos,-pos)
    mirror=np.argmin(dist,axis=1)
    assert dist[np.arange(k),mirror].max()<1e-10
    gram=np.exp(-.5*cdist(pos,pos,'sqeuclidean'))
    weight=np.outer(c,c)*gram
    pairs=[];w=[]
    for i in range(k):
        for j in range(i+1):
            orbit={(i,j),(j,i),(mirror[i],mirror[j]),(mirror[j],mirror[i])}
            if (i,j)!=max(orbit):continue
            pairs.append((i,j));w.append(sum(weight[a,b] for a,b in orbit))
    assert abs(sum(w)-1)<1e-10
    return np.array(pairs).T,np.array(w)


class Native:
    def __init__(self,out):
        self.rt=os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
        f=out/'coarse.dll'
        subprocess.run(['g++','-std=c++17','-O3','-fopenmp','-shared','-fPIC',str(ROOT/'native/coarse_readout112_v1.cpp'),'-o',str(f)],check=True)
        self.lib=ct.CDLL(str(f))
        self.fn=self.lib.coarse112
        self.fn.argtypes=[ct.c_int]*5+[ct.c_uint64,ct.c_uint32]+[ct.c_void_p]*10


def prepare(lib,local,xs,x,pos,axes,c,radix,h):
    pair,w=combinations(pos,c)
    counts=[];symbols=[];cats=[];prefix=[];widths=[];tail=0.;clipped=0.;offset=0
    target_totals=[];target_pairs=[]
    m=len(x)//2
    for g,(q,disp) in enumerate(local_tables(local,xs,pos,axes)):
        d=len(q)//2
        idx=np.r_[offset:offset+d,m+offset:m+offset+d]
        true=x[np.ix_(idx,idx)];offset+=d
        if radix==2:
            marg=click_table(true,np.zeros((1,1,2*d),complex))[:,0,0].real
            code=np.argsort(-marg)
            a=click_table(q,disp)[code][:,pair[0],pair[1]]
        else:
            marg=fourier(true,np.zeros((1,2*d),complex),radix)[:,0].real
            sort=np.argsort(-marg);end=np.searchsorted(np.cumsum(marg[sort],dtype=np.longdouble),1-5e-7)+1
            code=sort[:end]
            assert code[0]==0
            a=fourier(q,disp[pair[0],pair[1]],radix,code)
            tail+=max(0.,float(1-np.sum(marg[code],dtype=np.longdouble)))
            for mode in range(d):
                eig=np.linalg.eigvalsh(true[np.ix_([mode,d+mode],[mode,d+mode])])
                k=eig/(1+eig);p=[1/np.sqrt(np.prod(1+eig))]
                for n in range(1,radix):p.append(sum(np.sum(k**j)*p[n-j] for j in range(1,n+1))/(2*n))
                clipped+=max(0.,1-sum(p))
        t=np.zeros((h,len(w)),complex)
        limit=min(h-1,len(code))
        t[:limit]=a[:limit];t[-1]=a[limit:].sum(0)
        pr=np.zeros_like(a)
        if len(code)>=h:pr[h-1:]=a[h-1:].cumsum(0)
        cats.append(t);counts.append(a);prefix.append(pr);symbols.append(code);widths.append(d)
        digits=np.stack([code//radix**i%radix for i in range(d)])
        sums=digits.sum(0)
        total=np.zeros((128,len(w)),complex);np.add.at(total,sums,a);target_totals.append(np.fft.fft(total,axis=0))
        if g<2:
            ta=np.zeros((11,len(w)),complex);np.add.at(ta,np.minimum(10,digits[0]),a);target_pairs.append(ta)
    a,b,c0,d=cats
    left=(b[:,None]*a[None]).reshape(h*h,-1)
    right=(d[:,None]*c0[None]).reshape(h*h,-1)
    prob=((right*w)@left.T).real.ravel()
    assert prob.min()>-1e-11 and abs(prob.sum()-1)<1e-10
    prob=np.maximum(prob,0);prob/=prob.sum()
    th,aliases=alias(prob)
    arrays=[np.asarray(widths,np.int32),np.cumsum([0]+list(map(len,counts)),dtype=np.int32),aliases,th,w,
            np.stack(cats),np.vstack(counts),np.vstack(prefix),np.concatenate(symbols).astype(np.int32)]
    arrays=[np.ascontiguousarray(v) for v in arrays]
    def draw(n,seed,off=0,threads=1):
        out=np.empty((n,sum(widths)),np.uint8)
        rc=lib.fn(n,h,len(w),radix,threads,seed,off,*[v.ctypes.data for v in arrays],out.ctypes.data)
        if rc:raise ArithmeticError(rc)
        return out
    ptotal=np.fft.ifft((np.prod(target_totals,axis=0)*w).sum(1)).real
    ppair=((target_pairs[1]*w)@target_pairs[0].T).real.ravel()
    return draw,dict(pairs=len(w),local_words=list(map(len,counts)),coarse_words=h**4,wrapped=clipped,
                     omitted=tail),ptotal,ppair


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    lib=Native(out);fits=np.load(DATA/'iteration112_fit_v1/fits.npz')
    refs=np.load(DATA/'iteration112_cases_v1/laws.npz')
    fit=json.loads((DATA/'iteration112_fit_v1/fits.json').read_text())
    rows=[];arrays={}
    for k,(buses,budget,radix) in enumerate(((2,129,2),(2,65,12),(4,129,2),(4,65,12))):
        local,xs,s,axes,x,base=optics(buses)
        key=f'b{buses}n{budget}';start=perf_counter()
        model,info,total,pair=prepare(lib,local,xs,x,fits[key+'_pos'],axes,fits[key+'_c'],radix,16 if radix==2 else 10)
        prep=perf_counter()-start
        ns=65536 if radix==2 else 8192
        seed=112400000+1000*k
        times={}
        for nt in (1,8):
            tm=[];batch=2048 if radix==2 else 128
            for rep in range(12):
                start=perf_counter();a=model(batch,seed+20*nt+rep,threads=nt)
                if rep>=2:tm.append((perf_counter()-start)*1e6/batch)
            times[str(nt)]=tm
        a=model(ns,seed,threads=8)
        assert np.array_equal(a[3:13],model(10,seed,3,1))
        h=np.bincount((a>0)@(1<<np.arange(13)),minlength=8192)
        hp=np.bincount(np.minimum(a[:,0],10).astype(int)+11*np.minimum(a[:,3],10).astype(int),minlength=121)
        ht=np.bincount(a.sum(1).astype(int),minlength=128)
        spec=np.linalg.eigvalsh(x);z=np.exp(-2j*np.pi*np.arange(65)/128)
        true_total=np.fft.irfft(np.exp(-.5*np.log1p((1-z[:,None])*spec).sum(1)),n=128)
        row=next(r for r in fit if r['buses']==buses and r['budget']==budget)
        info.update(buses=buses,budget=budget,radix=radix,shots=ns,timing=times,setup_s=prep,state_bound=row['state_bound'],
                    full_tv_budget=min(1.,row['state_bound']+info['wrapped']+info['omitted']),
                    empirical_click_tv=float(abs(h/ns-refs[f'truth{buses}']).sum()/2),click_model_tv=row['click_tv'] if radix==2 else None)
        if radix>2:
            info.update(pair_model_tv=float(abs(pair-refs[f'pair{buses}']).sum()/2),pair_empirical_tv=float(abs(hp/ns-refs[f'pair{buses}']).sum()/2),
                        total_model_ks=float(abs((total-true_total).cumsum()).max()),total_empirical_ks=float(abs((ht/ns-true_total).cumsum()).max()))
        rows.append(info)
        print(buses,radix,prep,np.median(times['1']),info.get('pair_model_tv'),flush=True)
        arrays.update({f'{k}_'+s:v for s,v in [('first',a[:128]),('hist',h),('pair',hp),('total',ht),('model_pair',pair),('model_total',total)]})
        np.savez_compressed(out/'records.npz',**arrays)
        (out/'coarse.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=3):run(ap.parse_args().output.resolve())
