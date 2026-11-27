"""Matched shot counts, native record offsets and full-width CUDA Frugal comparisons."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration109_peers_v1 import load_frugal
from iteration111_sampling_v1 import (DATA, Native, Readout, branches, prepare, dyads, click_table,
                                       realify, pnr_table, marginal_tail)


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    optical=np.load(DATA/'iteration110_collective_v1/optical.npz')
    ref=np.load(DATA/'iteration110_bus_v1/laws.npz')
    targets=np.load(DATA/'iteration111_peers_v1/reference.npz')
    x=optical['core_x']
    bus=ref['bus']
    wide=np.tile(bus,4)/2
    f=realify(np.eye(16)+np.expm1(1j)*np.outer(wide,wide.conj()))
    base=np.block([[np.kron(np.eye(4),x[4*i:4*i+4,4*j:4*j+4]) for j in (0,1)] for i in (0,1)])
    cov=np.ascontiguousarray(2*f@base@f.T+np.eye(32))
    handles,lib,fr=load_frugal()
    gpu=[];arrays={}
    for k in (2,4):
        def draw(n,seed):
            a=np.empty((n,16),np.uint8)
            stats=np.zeros(6);msg=ct.create_string_buffer(4096)
            rc=fr(16,k,n,seed,16,cov,a,stats,msg,len(msg))
            if rc or stats[-1]!=32:raise RuntimeError(msg.value)
            return a
        hist=np.zeros(65536,np.int64);us=[]
        for rep in range(32):
            start=perf_counter()
            a=draw(2048,111700000+40*k+rep)
            if rep:us.append((perf_counter()-start)*1e6/2048)
            hist+=np.bincount(a@(1<<np.arange(16)),minlength=65536)
        gpu.append(dict(k=k,precision='fp32',us=float(np.median(us)),times=us,shots=int(hist.sum()),
                        empirical_tv=float(abs(hist/hist.sum()-ref['truth4']).sum()/2)))
        arrays[f'frugal{k}']=hist
    native=Native(out)
    s,axes,obs=prepare(optical,bus,1.)
    rows=[]
    for index,(cells,m,radix) in enumerate(((4,6,2),(32768,8,2),(4,6,16))):
        start=perf_counter()
        z,c,g,bound=branches(s,axes,m)
        d=dyads(z,obs,cells)
        tab=click_table(x,d) if radix==2 else pnr_table(x,d,radix)
        model=Readout(native,tab,c,g,cells,radix)
        setup=perf_counter()-start
        size=2048 if radix==2 and cells==4 else 128 if cells==4 else 16
        ns=65536 if size==2048 else 4096 if cells==4 else 128
        seed=111710000+index*1000
        h=np.zeros(65536,int);hp=np.zeros(121,int);tot=np.zeros(256,int)
        row=dict(cells=cells,m=m,radix=radix,batch=size,shots=ns,setup_s=setup,
                 whole_tv_budget=bound+(marginal_tail(x,bus,cells,radix) if radix>2 else 0),timing={})
        for nt in (1,8):
            times=[]
            for rep in range(10):
                start=perf_counter()
                model.sample(size,seed+100+nt*10+rep,nt)
                if rep:times.append((perf_counter()-start)*1e6/size)
            row['timing'][str(nt)]=times
        for off in range(0,ns,size):
            a,_=model.sample(size,seed,8,off)
            if off==0:
                part,_=model.sample(2,seed,1,2)
                assert np.array_equal(a[2:4],part)
            if cells==4:
                h+=np.bincount((a>0)@(1<<np.arange(16)),minlength=65536)
                hp+=np.bincount(a[:,[0,4]].clip(0,10)@np.array([1,11]),minlength=121)
                tot+=np.bincount(a.sum(1).astype(int),minlength=256)
            else:
                if not off:arrays['wide_click_sample']=np.packbits(a[:8],axis=1,bitorder='little')
        if cells==4:
            row['empirical_threshold_tv']=float(abs(h/ns-ref['truth4']).sum()/2)
            arrays[f'hist{index}']=h
        if radix>2:
            counts=np.array([np.arange(radix**4)//radix**i%radix for i in range(4)])
            one=np.array([tab[np.minimum(counts[0],10)==n].sum(0) for n in range(11)])
            weights=np.outer(c,c)*g
            pair=((one.reshape(11,-1)*weights.ravel())@one.reshape(11,-1).T).real.ravel()
            local=np.array([tab[counts.sum(0)==n].sum(0) for n in range(4*(radix-1)+1)])
            total=np.fft.ifft((np.fft.fft(local,n=256,axis=0)**4*weights).sum(axis=(1,2))).real
            row.update(pair_model_tv=float(abs(pair-targets['pair']).sum()/2),
                       total_model_ks=float(abs((total-targets['total']).cumsum()).max()),
                       pair_empirical_tv=float(abs(hp/ns-targets['pair']).sum()/2),
                       total_empirical_ks=float(abs((tot/ns-targets['total']).cumsum()).max()))
            arrays['pnr_pair'],arrays['pnr_total']=hp,tot
        rows.append(row)
        print(cells,m,radix,{t:np.median(v) for t,v in row['timing'].items()},flush=True)
    np.savez_compressed(out/'matched.npz',**arrays)
    (out/'matched.json').write_text(json.dumps(dict(frugal_gpu=gpu,native=rows),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(ap.parse_args().output.resolve())
