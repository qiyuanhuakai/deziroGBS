"""Dense complex interferometers at fixed source rank, with loss and both output readouts."""
import argparse
import ctypes as ct
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import mpmath as mp
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from iteration112_records_v1 import fock
from iteration114_positive_v1 import positive
from iteration112_cases_v1 import gaussian_click, oracle, ONE, pair_reference
from iteration111_memory_v1 import realify

ROOT=Path(__file__).resolve().parents[1]
P=ct.c_void_p
I=ct.c_int
U=ct.c_uint64
J=ct.c_uint32
BLOCK=32


def libopen(path):
    if os.name=='nt':
        libopen.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    lib=ct.CDLL(str(path.resolve()))
    lib.pnr114.argtypes=[I]*7+[U,J]+[P]*9
    lib.click114.argtypes=[I]*6+[U,J]+[P]*8
    lib.law114.argtypes=[I]*3+[P]*5
    return lib


def optical(m,r):
    rng=np.random.default_rng(114000000+8*m+r)
    raw=rng.standard_normal((m,r))+1j*rng.standard_normal((m,r))
    q=np.linalg.qr(raw,mode='reduced')[0]
    q*=np.sqrt(rng.uniform(.48,.97,m))[:,None]
    squeeze=np.array([.52,.40,.30,.23][:r])
    return np.ascontiguousarray(q),squeeze


def numlaw(s,cap):
    with mp.workdps(65):
        a=[mp.mpf(1)]+[mp.mpf(0)]*(cap//2)
        for v in s:
            t=mp.tanh(v)**2
            b=[1/mp.cosh(v)]
            for k in range(1,len(a)):b.append(b[-1]*t*(2*k-1)/(2*k))
            a=[mp.fsum(a[i]*b[k-i] for i in range(k+1)) for k in range(len(a))]
        tail=1-mp.fsum(a)
        q=np.zeros(cap+1)
        q[::2]=list(map(float,a))
    c=np.cumsum(q/q.sum());c[-1]=1.
    return c,float(tail)


def blockgram(t):
    blocks=(len(t)+BLOCK-1)//BLOCK
    blocks_t=np.pad(t,((0,blocks*BLOCK-len(t)),(0,0))).reshape(blocks,BLOCK,t.shape[1])
    return np.cumsum(blocks_t.conj().transpose(0,2,1)@blocks_t,axis=0)


def model(t,s,cap,budget,dictionary='positive'):
    start=perf_counter()
    vis=blockgram(t)
    vis_seconds=perf_counter()-start
    loss=np.eye(len(s))-vis[-1]
    eig,u=eigh(loss)
    assert min(eig)>-1e-10
    env=np.sqrt(eig.clip(0))[:,None]*u.conj().T
    full=np.vstack([t,env])
    grams=blockgram(full)
    assert np.max(abs(grams[-1]-np.eye(len(s))))<2e-10
    cdf,tail=numlaw(s,cap)
    graph=fock(s,cap)
    number=[cdf,*graph,full,grams]
    pnrtime=perf_counter()-start
    start=perf_counter()
    if dictionary=='signed':
        from iteration112_sparse_v1 import axis,project
        pos,c,_,info=project(s,[axis(v) for v in s],budget)
    elif dictionary=='hermite':
        import itertools
        order=round(budget**(1/len(s)))
        if order**len(s)!=budget:raise ValueError('Use a complete tensor quadrature')
        xi,wi=np.polynomial.hermite.hermgauss(order)
        pos=np.array(list(itertools.product(*[xi*np.sqrt(2*np.tanh(v)) for v in s])))
        c=np.array([math.prod(row) for row in itertools.product(wi/np.sqrt(np.pi),repeat=len(s))])
        c*=np.exp((pos**2).sum(1)/2)/np.sqrt(np.cosh(s)).prod()
        gram=np.exp(-.5*((pos[:,None]-pos[None])**2).sum(2))
        c/=np.sqrt(c@gram@c)
        g=np.exp(-.5*(pos**2)@(1-np.tanh(s)))/np.sqrt(np.cosh(s)).prod()
        info=dict(branches=budget,budget=budget,state_bound=math.sqrt(max(0.,1-(c@g)**2)),
                  acceptance=float(1/c.sum()**2))
    else:
        pos,c,info=positive(s,budget)
    norm=(pos*pos).sum(1)
    l1=abs(c).sum()
    diag=l1*abs(c)*np.exp(-np.einsum('ai,ij,aj->a',pos,vis[-1],pos).real)
    # Coefficient indexed by ket a and bra b; include loss via <env_b|env_a>.
    envcross=(pos@loss@pos.T).T
    base=np.outer(c,c)*np.exp(envcross-.5*(norm[:,None]+norm))
    lo=np.tril_indices(len(c))
    base=base[lo]*np.where(lo[0]==lo[1],1.,2.)
    # Each branch skips vacuum runs using cumulative exponential hazards.
    hazard=np.empty((len(pos),len(vis)))
    for i,p in enumerate(pos):
        hazard[i]=np.einsum('i,nij,j->n',p,vis,p).real
    assert hazard.min()>-1e-12 and np.diff(hazard,axis=1).min(initial=0)>-1e-10
    proposal=np.cumsum(abs(c)/l1);proposal[-1]=1.
    click=[t,pos,proposal,hazard,diag,base]
    # A separate caller preparing only clicks also needs the visible-prefix Grams.
    ctime=perf_counter()-start+vis_seconds
    return dict(pnr=[np.ascontiguousarray(a) for a in number],click=[np.ascontiguousarray(a) for a in click],
                tail=tail,info=info,pnr_setup_s=pnrtime,click_setup_s=ctime,cap=cap)


def draw(lib,model,pnr,n,threads,seed,offset,buf):
    args=model['pnr' if pnr else 'click']
    t,pos=model['click'][:2];m,r=t.shape
    ptr=[a.ctypes.data for a in args]
    attempts=np.zeros(1,np.uint64)
    if pnr:
        rc=lib.pnr114(n,m,m+r,r,model['cap'],BLOCK,threads,seed,offset,*ptr,buf.ctypes.data)
    else:
        rc=lib.click114(n,m,r,len(pos),BLOCK,threads,seed,offset,*ptr,buf.ctypes.data,attempts.ctypes.data)
    assert rc==0,(rc,pnr,n,m)
    return int(attempts[0])


def cov(t,s):
    a=realify(t)
    return (a*np.expm1(np.r_[-2*s,2*s])/2)@a.T


def physical_total(t,s,size=128):
    # Nonzero eigenvalues of X are recovered from a 2r by 2r symmetric matrix.
    g=realify(t.conj().T@t)
    val,u=eigh(g)
    root=(u*np.sqrt(val.clip(0)))@u.T
    e=np.linalg.eigvalsh((root*(np.expm1(np.r_[-2*s,2*s])/2))@root)
    z=np.exp(-2j*np.pi*np.arange(size//2+1)/size)
    return np.fft.irfft(np.exp(-.5*np.log1p((1-z[:,None])*e).sum(1)),n=size)


def run_case(lib,out,m,r,budget,shots,batch,domain,dictionary='positive'):
    start=perf_counter();t,s=optical(m,r);optical_s=perf_counter()-start
    spec=model(t,s,24 if r==2 else 16,budget,dictionary)
    modes=[0,3];x=cov(t[modes],s)
    pair=pair_reference((np.eye(4)+x)[None],8)
    expected_total=physical_total(t,s)
    arrays=dict(pair=pair,total=expected_total,s=s,rows=t[:min(13,m)])
    result=dict(m=m,r=r,budget=budget,cap=spec['cap'],shots=shots,batch=batch,seed_domain=domain,
                transfer_sha256=__import__('hashlib').sha256(t.tobytes()).hexdigest(),
                input_construction_s=optical_s,pnr_setup_s=spec['pnr_setup_s'],click_setup_s=spec['click_setup_s'],
                pnr_full_tv_upper=spec['tail'],click_projection=spec['info'],
                persistent_pnr_bytes=sum(a.nbytes for a in spec['pnr']),
                persistent_click_bytes=sum(a.nbytes for a in spec['click']),measurements=[])
    if m==13:
        full=gaussian_click(oracle(True),cov(t,s)[None],ONE)
        arrays['optical_click']=full
        law=np.empty_like(full)
        transfer,pos,_,_,diag,base=spec['click']
        assert lib.law114(m,r,len(pos),*[a.ctypes.data for a in (transfer,pos,diag,base,law)])==0
        assert min(law)>-1e-14 and abs(law.sum()-1)<2e-12
        result['direct_model_tv']=float(abs(law-full).sum()/2)
        assert result['direct_model_tv']<spec['info']['state_bound']+1e-10
        arrays['projected_click']=law
    # Binary partition with half the outputs on each side, independent of the optical input.
    half=m//2
    miss_left=np.exp(-.5*np.linalg.slogdet(np.eye(2*r)+
        realify(t[:half].conj().T@t[:half])*np.expm1(np.r_[-2*s,2*s])/2)[1])
    result['left_half_vacuum']=float(miss_left)
    for pnr in (False,True):
        shape=(batch,m);buf=np.empty(shape,np.uint8)
        for threads in (1,8):
            times=[];total_hist=np.zeros(128,int);pair_hist=np.zeros(81,int)
            click_hist=np.zeros(8192,int) if m==13 else None
            left_zero=0;attempts=0
            seed=114000000+domain*100+10*int(pnr)+int(threads==8)
            draw(lib,spec,pnr,batch,threads,seed,2**31,buf)
            for off in range(0,shots,batch):
                n=min(batch,shots-off);v=buf[:n]
                begin=perf_counter();attempts+=draw(lib,spec,pnr,n,threads,seed,off,v)
                times.append(1e6*(perf_counter()-begin)/n)
                total_hist+=np.bincount(v.sum(1).astype(int),minlength=128)
                pair_hist+=np.bincount(np.minimum(v[:,0],8).astype(int)+9*np.minimum(v[:,3],8).astype(int),minlength=81)
                left_zero+=int((v[:,:half].max(1)==0).sum())
                if click_hist is not None:click_hist+=np.bincount((v>0)@(1<<np.arange(m)),minlength=8192)
                if off==0 and threads==8:arrays[f'sample_{int(pnr)}']=v[:min(12,n)].copy()
            key=f'{int(pnr)}_{threads}'
            arrays['total_'+key]=total_hist;arrays['pair_'+key]=pair_hist
            row=dict(pnr=pnr,threads=threads,us=float(np.median(times)),times=times,
                     left_half_zero=left_zero,left_half_gap=abs(left_zero/shots-miss_left))
            if click_hist is not None:
                arrays['click_'+key]=click_hist
                row['empirical_click_tv']=float(abs(click_hist/shots-full).sum()/2)
            if pnr:
                row['empirical_pair_tv']=float(abs(pair_hist/shots-pair).sum()/2)
                row['total_ks']=float(abs(np.cumsum(total_hist/shots-expected_total)).max())
            else:row.update(attempts=attempts,acceptance=shots/attempts)
            result['measurements'].append(row)
    stem=f'm{m}_r{r}'
    np.savez_compressed(out/(stem+'.npz'),**arrays)
    (out/(stem+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(m,r,result.get('direct_model_tv'),spec['info'],spec['tail'],[(v['pnr'],round(v['us'],2)) for v in result['measurements']],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    ap.add_argument('--library',type=Path);ap.add_argument('--case',choices=['all','small','wide','precise'],default='all')
    ap.add_argument('--dictionary',choices=['positive','signed','hermite'],default='positive')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    dll=args.library or out/('rank.dll' if os.name=='nt' else 'rank.so')
    if not args.library:
        subprocess.run(['g++','-std=c++17','-fopenmp','-O3','-shared','-fPIC',
                        str(ROOT/'native/source_rank114_v1.cpp'),'-o',str(dll)],check=True)
    lib=libopen(dll)
    configs=[(13,2,25,32768,512),(13,4,49,16384,256),(1152,4,49,16384,256),
             (131072,4,49,4096,64),(1048576,2,25,1024,16)]
    if args.case=='precise':
        configs=[(13,2,81,4096,128),(13,4,129,4096,128),(131072,4,129,1024,32),(1048576,2,81,1024,16)]
    with threadpool_limits(limits=2):
        for i,args_ in enumerate(configs):
            if args.case=='precise':
                run_case(lib,out,*args_,20+i,'hermite' if args_[1]==2 else 'positive')
            elif args.case=='all' or (args.case=='small')==(args_[0]==13):
                run_case(lib,out,*args_,i,args.dictionary)
