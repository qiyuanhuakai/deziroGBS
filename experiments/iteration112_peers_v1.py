"""Run the 13-mode Frugal CPU/GPU instantiations and physical 26-mode photonic-engines dilation."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import null_space
from threadpoolctl import threadpool_limits
from iteration109_peers_v1 import load_frugal
from iteration92_pe_common_v1 import GaussianInput,pe_sample
from iteration112_cases_v1 import optics
from iteration111_sampling_v1 import DATA


def run(out,buses):
    out.mkdir(parents=True,exist_ok=True)
    handles,old,_=load_frugal()
    build=DATA/'iteration112_frugal_build_v1'
    cpu=ct.CDLL(str(build/'cpu.dll'));gpu=ct.CDLL(str(build/'frugal_fp32.dll'))
    P=ct.c_void_p;I=ct.c_int;U=ct.c_uint
    cpu.fc112_full13.argtypes=[I]+[P]*6+[I]
    cpu.fc112_sample13.argtypes=[I,I,U]+[P]*4+[I]
    gpu.fg72_sample.argtypes=[I,I,I,U,I]+[P]*4+[I]
    _,_,_,_,x,_=optics(buses);cov=np.ascontiguousarray(2*x+np.eye(26))
    t,r=optics(buses,return_transfer=True)
    q=np.column_stack((t,null_space(t.conj().T)))
    gs=GaussianInput(r,tuple(range(13)),26)
    b,_,_=gs.bargmann(q)
    assert np.max(abs(b+(t*np.tanh(r))@t.T))<1e-12
    from iteration111_memory_v1 import realify
    f=realify(t[:13]);np.testing.assert_allclose((f*np.expm1(np.r_[-2*r,2*r])/2)@f.T,x,atol=2e-12)
    ref=np.load(DATA/'iteration112_cases_v1/laws.npz')
    truth=ref[f'truth{buses}'];pair=ref[f'pair{buses}']
    z=np.exp(-2j*np.pi*np.arange(65)/128)
    total=np.fft.irfft(np.exp(-.5*np.log1p((1-z[:,None])*np.linalg.eigvalsh(x)).sum(1)),n=128)
    arrays=dict(truth=truth,total=total,pair=pair);rows=[]
    msg=ct.create_string_buffer(4096)
    for k in (2,4):
        size=8191
        th=np.empty(size);prev=np.empty(size);prop=np.empty(size);st=np.zeros(3)
        rc=cpu.fc112_full13(k,*[a.ctypes.data for a in (cov,th,prev,prop,st)],msg,len(msg))
        assert rc==0 and st[-1]==13*st[0] and st[1]==13,(rc,st,msg.value)
        law=np.ones(1)
        for depth in range(13):
            cond=np.nan_to_num(th[len(law)-1:2*len(law)-1],nan=1).clip(0,1)
            law=np.r_[law*cond,law*(1-cond)]
        arrays[f'frugal{k}']=law
        for backend in ('cpu','gpu'):
            hist=np.zeros(8192,int);times=[]
            for rep in range(32):
                n=2048;seed=112510000+10000*buses+100*k+40*(backend=='gpu')+rep
                a=np.empty((n,13),np.uint8);st=np.zeros(7 if backend=='cpu' else 6)
                args=[v.ctypes.data for v in (cov,a,st)]
                start=perf_counter()
                if backend=='cpu':rc=cpu.fc112_sample13(k,n,seed,*args,msg,len(msg))
                else:rc=gpu.fg72_sample(13,k,n,seed,16,*args,msg,len(msg))
                if rep>=2:times.append((perf_counter()-start)*1e6/n)
                assert rc==0 and st[5]==(64 if backend=="cpu" else 32),(msg.value,st)
                hist+=np.bincount(a@(1<<np.arange(13)),minlength=8192)
            arrays[f'hist{k}{backend}']=hist
            rows.append(dict(backend=backend,k=k,shots=int(hist.sum()),us=float(np.median(times)),times=times,
                model_tv=float(abs(law-truth).sum()/2) if backend=='cpu' else None,
                empirical_tv=float(abs(hist/hist.sum()-truth).sum()/2)))
    result=dict(buses=buses,frugal=rows,pe=[])
    def save():
        np.savez_compressed(out/'peers.npz',**arrays)
        (out/'peers.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    save()
    for backend in ('cpu','cuda'):
        ns=8192;hh=np.zeros(128,int);hp=np.zeros(121,int);times=[]
        for off in range(0,ns,128):
            start=perf_counter()
            a,stat=pe_sample(gs,q,128,device=backend,seed=112560000+10000*buses+500*(backend=='cuda')+off//128,
                        impl='v2',window=4,max_photons=80,return_stats=True)
            a=a[:,:13].copy()
            if off:times.append((perf_counter()-start)*1e6/128)
            assert not np.any(stat.censored)
            hh+=np.bincount(a.sum(1).astype(int),minlength=128)
            hp+=np.bincount(np.minimum(a[:,0],10)+11*np.minimum(a[:,3],10),minlength=121)
        arrays[backend+'_pair']=hp;arrays[backend+'_total']=hh
        result['pe'].append(dict(backend=backend,shots=ns,us=float(np.median(times)),times=times,censored=0,
                            pair_empirical_tv=float(abs(hp/ns-pair).sum()/2),total_empirical_ks=float(abs((hh/ns-total).cumsum()).max())))
        save();print(buses,backend,np.median(times),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--buses',type=int,required=True)
    args=ap.parse_args()
    with threadpool_limits(limits=1):run(args.output.resolve(),args.buses)
