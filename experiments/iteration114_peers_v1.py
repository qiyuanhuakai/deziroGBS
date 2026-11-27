"""Matched 13-port Frugal and photonic-engines timings and equal-shot validation."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import null_space,eigh
from threadpoolctl import threadpool_limits
from iteration109_peers_v1 import load_frugal
from iteration92_pe_common_v1 import GaussianInput,pe_sample
from iteration114_rank_v1 import ROOT,P,I,optical,cov,model,libopen,draw


def run(out,dll):
    out.mkdir(exist_ok=True,parents=True);lib=libopen(dll)
    handles,old,_=load_frugal()
    data=ROOT/'results'
    fc=ct.CDLL(str(data/'iteration112_frugal_build_v1/cpu.dll'))
    fg=ct.CDLL(str(data/'iteration112_frugal_build_v1/frugal_fp32.dll'))
    fc.fc112_full13.argtypes=[I]+[P]*6+[I]
    fc.fc112_sample13.argtypes=[I,I,ct.c_uint32]+[P]*4+[I]
    fg.fg72_sample.argtypes=[I,I,I,ct.c_uint32,I]+[P]*4+[I]
    rows=[];arrays={};count=8192;msg=ct.create_string_buffer(4096)
    for r in (2,4):
        z=np.load(data/f'iteration114_positive_v1/m13_r{r}.npz')
        target=z['optical_click'];pair=z['pair'];total=z['total']
        t,s=optical(13,r)
        x=cov(t,s);q=np.ascontiguousarray(2*x+np.eye(26))
        eig,u=eigh(np.eye(r)-t.conj().T@t)
        tall=np.vstack([t,np.sqrt(eig)[:,None]*u.conj().T])
        unitary=np.column_stack([tall,null_space(tall.conj().T)])
        gs=GaussianInput(s,tuple(range(r)),len(unitary))
        b,_,_=gs.bargmann(unitary)
        assert abs(b+(tall*np.tanh(s))@tall.T).max()<1e-12
        obj=model(t,s,24 if r==2 else 16,25 if r==2 else 49)
        def score(name,a,times,law=None,pnr=False):
            words=np.bincount((a>0)@(1<<np.arange(13)),minlength=8192)
            key=f'r{r}_{name}';arrays[key+'_click']=words
            row=dict(r=r,name=name,shots=len(a),us=float(np.median(times)),times=times,
                click_empirical_tv=float(abs(words/len(a)-target).sum()/2))
            if law is not None:
                arrays[key+'_model']=law
                row['click_model_tv']=float(abs(law-target).sum()/2)
            if pnr:
                p=np.bincount(np.minimum(a[:,0],8).astype(int)+9*np.minimum(a[:,3],8).astype(int),minlength=81)
                tot=np.bincount(a.sum(1).astype(int),minlength=128)
                arrays[key+'_pair']=p;arrays[key+'_total']=tot
                row.update(pair_empirical_tv=float(abs(p/len(a)-pair).sum()/2),
                           total_ks=float(abs((tot/len(a)-total).cumsum()).max()))
            rows.append(row)
            print(r,name,round(row['us'],3),flush=True)
        for pn in (0,1):
            a=np.empty((count,13),np.uint8);times=[]
            for off in range(0,count,256):
                start=perf_counter();draw(lib,obj,pn,256,8,114600000+r*10+pn,off,a[off:off+256])
                if off:times.append((perf_counter()-start)*1e6/256)
            score('source_pnr' if pn else 'source_click',a,times,None if pn else z['projected_click'],pnr=pn)
        for k in (2,4):
            prev=np.empty(8191);prop=prev.copy();th=prev.copy();st=np.zeros(3)
            assert fc.fc112_full13(k,*[v.ctypes.data for v in (q,th,prev,prop,st)],msg,len(msg))==0
            law=np.ones(1)
            for depth in range(13):
                cond=np.nan_to_num(th[len(law)-1:2*len(law)-1],nan=1).clip(0,1)
                law=np.r_[law*cond,law*(1-cond)]
            for gpu in (False,True):
                a=np.empty((count,13),np.uint8);times=[]
                for off in range(0,count,512):
                    seed=114650000+r*10000+k*1000+100*gpu+off//512
                    st=np.zeros(6 if gpu else 7);v=a[off:off+512]
                    args=[y.ctypes.data for y in (q,v,st)]
                    start=perf_counter()
                    if gpu:rc=fg.fg72_sample(13,k,512,seed,16,*args,msg,len(msg))
                    else:rc=fc.fc112_sample13(k,512,seed,*args,msg,len(msg))
                    if off:times.append((perf_counter()-start)*1e6/512)
                    assert rc==0,msg.value
                score(f'frugal{k}_'+('gpu' if gpu else 'cpu'),a,times,None if gpu else law)
        for device in ('cpu','cuda'):
            a=np.empty((count,13),np.uint8);times=[]
            for off in range(0,count,128):
                seed=114720000+r*10000+1000*(device=='cuda')+off//128
                start=perf_counter()
                vals,stat=pe_sample(gs,unitary,128,seed=seed,device=device,impl='v2',window=4,max_photons=64,return_stats=True)
                a[off:off+128]=vals[:,:13]
                if off:times.append((perf_counter()-start)*1e6/128)
                assert not np.any(stat.censored) and vals.max()<64
            score('photonic_engines_'+device,a,times,pnr=True)
        np.savez_compressed(out/'peers.npz',**arrays)
        (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('output',type=Path);a.add_argument('--library',type=Path,required=True)
    args=a.parse_args()
    with threadpool_limits(limits=1):run(args.output,args.library)
