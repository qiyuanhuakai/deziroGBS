"""Independent Gaussian targets, Frugal model laws and photonic-engines records on the 1-radian input."""
import argparse
import ctypes as ct
import json
import os
import shutil
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import null_space
from threadpoolctl import threadpool_limits
from iteration100_baselines_v1 import frugal, pair_reference
from iteration92_pe_common_v1 import GaussianInput, pe_sample
from iteration111_sampling_v1 import DATA, realify, prepare, branches, dyads, pnr_table


def run(out, pe_shots):
    out.mkdir(exist_ok=True,parents=True)
    optical = np.load(DATA/'iteration110_collective_v1/optical.npz')
    ref = np.load(DATA/'iteration110_bus_v1/laws.npz')
    native = np.load(DATA/'iteration111_sampling_v1/records.npz')
    bus = np.tile(ref['bus'],4)/2
    w = np.eye(16)+np.expm1(1j)*np.outer(bus,bus.conj())
    optical_u,r,eta = [optical[k] for k in ('unitary','squeezing','efficiency')]
    top = w@np.kron(np.eye(4), np.sqrt(eta)[:,None]*optical_u)
    loss = np.kron(np.eye(4), np.sqrt(1-eta)[:,None]*optical_u)
    t = np.r_[top,loss]
    u = np.column_stack([t,null_space(t.conj().T)])
    r = np.tile(r,4)
    gs = GaussianInput(r,tuple(range(16)),32)
    b,_,_ = gs.bargmann(u)
    assert np.max(abs(b+(t*np.tanh(r))@t.T))<1e-12
    f = realify(top)
    x = (f*(np.expm1(np.r_[-2*r,2*r])/2))@f.T
    sel = [0,4,16,20]
    q = np.eye(32)+x
    pair = pair_reference(q[np.ix_(sel,sel)][None],10)
    eig = np.linalg.eigvalsh(x)
    z = np.exp(-2j*np.pi*np.arange(129)/256)
    total = np.fft.irfft(np.exp(-.5*np.log1p((1-z[:,None])*eig).sum(1)),n=256)
    def denormalized(h): return h/h.sum()
    def tv(p,q):return float(abs(p-q).sum()/2)
    def ks(p,q):return float(abs((p-q).cumsum()).max())
    arrays = dict(pair=pair,total=total)
    result = dict(pe=[],frugal=[],pnr=[])
    path = out/'peers.json'
    def write():
        path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        np.savez_compressed(out/'reference.npz',**arrays)
    runtime = os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
    lib = ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
    for k in (2,4):
        law,times = frugal(lib,q[None],k,111200000+20*k)
        result['frugal'].append(dict(k=k,model_tv=tv(law,ref['truth4']),us=float(np.median(times)),times=times))
        arrays[f'frugal{k}'] = law
    write()
    for backend in ('cpu','cuda'):
        shots = pe_shots
        h,hh = np.zeros(121,int),np.zeros(256,int)
        times = []
        for off in range(0,shots,128):
            start = perf_counter()
            a,st = pe_sample(gs,u,128,seed=111210000+1000*(backend=='cuda')+off//128,
                             device=backend,impl='v2',window=4,max_photons=80,return_stats=True)
            a = a[:,:16].copy()
            if off:times.append((perf_counter()-start)*1e6/len(a))
            assert not np.any(st.censored)
            h += np.bincount(a[:,[0,4]].clip(0,10)@np.array([1,11]),minlength=121)
            hh += np.bincount(a.sum(1).astype(int),minlength=256)
        result['pe'].append(dict(backend=backend,shots=shots,us=float(np.median(times)),times=times,
              pair_empirical_tv=tv(denormalized(h),pair),total_empirical_ks=ks(denormalized(hh),total),censored=0))
        arrays[backend+'_pair'],arrays[backend+'_total'] = h,hh
        write()
        print(backend,result['pe'][-1]['us'],flush=True)
    s,axes,observed = prepare(optical,ref['bus'],1.)
    for idx,m,radix in ((4,3,12),(5,4,12),(6,4,16)):
        zeta,c,g,bound = branches(s,axes,m)
        tab = pnr_table(optical['core_x'],dyads(zeta,observed,4),radix)
        codes = np.arange(radix**4)
        digits = np.array([codes//radix**j%radix for j in range(4)])
        counts = digits.sum(0)
        one = np.array([tab[np.minimum(digits[0],10)==n].sum(0) for n in range(11)])
        weight = np.outer(c,c)*g
        model_pair = ((one.reshape(11,-1)*weight.ravel())@one.reshape(11,-1).T).real.ravel()
        local = np.array([tab[counts==n].sum(0) for n in range(max(counts)+1)])
        model_total = np.fft.ifft((np.fft.fft(local,n=256,axis=0)**4*weight).sum(axis=(1,2))).real
        hh = native[str(idx)+'_pair'].reshape(radix,radix)
        h = np.zeros((11,11),int)
        np.add.at(h,tuple(np.minimum(np.indices(hh.shape),10)),hh)
        result['pnr'].append(dict(nodes=m,radix=radix,pair_model_tv=tv(model_pair,pair),total_model_ks=ks(model_total,total),
              pair_empirical_tv=tv(denormalized(h.ravel()),pair),total_empirical_ks=ks(denormalized(native[str(idx)+'_total']),total)))
        arrays[str(idx)+'_pair_model'],arrays[str(idx)+'_total_model'] = model_pair,model_total
        write()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    ap.add_argument('--pe-shots',type=int,default=4096)
    args=ap.parse_args()
    with threadpool_limits(limits=1):run(args.output,args.pe_shots)
