"""Native record sampling for the shared phase on four-mode lossy cells."""
import argparse
import ctypes as ct
import json
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits
from iteration111_memory_v1 import prepare, branches, dyads, click_table, complete_clicks, realify

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results'
P = ct.c_void_p
I = ct.c_int


def pnr_table(x, disp, radix, symbols=None):
    """Finite detector register n mod radix. DFT evaluates the exact residue POVM."""
    d = len(x)//2
    assert d == 4
    shape = (radix,)*d
    coords = np.indices(shape).reshape(d, -1).T
    k = disp.shape[0]
    flat = disp.reshape(k*k, -1)
    _, eig, basis = np.linalg.svd(flat, full_matrices=False)
    basis = basis[eig>1e-11]
    coeff = flat@basis.conj().T
    kernel = np.eye(2*d)-np.linalg.inv(np.eye(2*d)+x)
    assert max(abs(np.linalg.eigvalsh(kernel))) < 1
    base = np.empty(len(coords), complex)
    a, b = np.triu_indices(len(basis))
    quadratic = np.empty((len(coords), len(a)), complex)
    for start in range(0, len(coords), 2048):
        sl = slice(start, start+2048)
        z = np.tile(np.exp((-2j*np.pi/radix)*coords[sl]), (1,2))
        # Every eigenvalue of ZK lies inside the unit disk; these individual logs
        # select the vacuum-connected square root even when det winds around zero.
        ev = np.linalg.eigvals(z[:, :, None]*kernel)
        base[sl] = np.exp(-.5*np.log1p(-ev).sum(1))/np.sqrt(np.linalg.det(np.eye(2*d)+x))
        t = 1-z
        matrix = np.eye(2*d)+t[:, :, None]*x
        solved = np.linalg.solve(matrix, t[:, :, None]*basis.T)
        projected = basis@solved
        quadratic[sl] = projected[:, a, b]*(1+(a!=b))
    if symbols is None: symbols = np.arange(len(coords))
    table = np.empty((len(symbols), k,k), complex)
    symmetric = np.max(abs(disp[::-1,::-1]+disp)) < 2e-13
    for i in range(k):
        for j in range(i+1):
            if symmetric and (i,j)>(k-1-j,k-1-i):continue
            v = coeff[i*k+j]
            series = base*np.exp(-(quadratic@(v[a]*v[b])))
            # Fortran order packs port 0 as the least significant digit.
            law = np.fft.ifftn(series.reshape(shape)).ravel(order='F')
            kept = law[symbols].copy()
            kept[0] += 1-kept.sum()
            table[:, i,j] = kept
            table[:, j,i] = kept.conj()
            if symmetric:
                table[:, k-1-i,k-1-j] = kept
                table[:, k-1-j,k-1-i] = kept.conj()
    assert abs(table.sum(0)-1).max() < 4e-11
    return table


def marginal_tail(x, bus, cells, radix):
    w = np.eye(4)+np.expm1(1j)*np.outer(bus,bus.conj())
    f = realify(w)
    local = x+(f@x@f.T-x)/cells
    tails = []
    for i in range(4):
        ev = np.linalg.eigvalsh(local[np.ix_([i,i+4],[i,i+4])])
        with mp.workdps(60):
            e = list(map(mp.mpf, ev.tolist()))
            q = [v/(1+v) for v in e]
            p = [1/mp.sqrt((1+e[0])*(1+e[1]))]
            for n in range(1, radix):
                p.append(sum((q[0]**j+q[1]**j)*p[n-j] for j in range(1,n+1))/(2*n))
            tails.append(float(1-sum(p)))
    return min(1., cells*sum(tails))


class Native:
    def __init__(self, directory):
        self.runtime = os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
        path = directory/('sampler.dll' if os.name=='nt' else 'sampler.so')
        subprocess.run(['g++','-std=c++17','-O3','-fopenmp','-shared','-fPIC',
                        str(ROOT/'native/shared_measure111_v1.cpp'),'-o',str(path)], check=True)
        self.lib = ct.CDLL(str(path.resolve()))
        self.lib.memory111.argtypes = [I]*6+[ct.c_uint64,ct.c_uint32]+[P]*9
        self.lib.cached111.argtypes = [I,I,ct.c_uint64,ct.c_uint32,P,P,P]
        self.lib.cached111.restype = None


class Readout:
    def __init__(self, native, tab, c, gram, cells, radix, symbols=None):
        self.native, self.cells, self.radix = native, cells, radix
        self.size, self.k, _ = tab.shape
        if symbols is None: symbols = np.arange(self.size,dtype=np.int32)
        diag = tab.diagonal(axis1=1,axis2=2).real
        assert diag.min() >= -1e-14 and np.max(abs(diag.sum(0)-1))<2e-11
        cond = np.maximum(diag.T, 0).cumsum(1)
        cond /= cond[:, -1:]
        total = np.abs(c).sum()
        branch = np.abs(c).cumsum()/total
        branch[-1] = 1.
        # Every likelihood uses the same positive symbol-wise scale. This cancels
        # from the ratio and avoids underflow even at 32,768 occupied cells.
        base = np.maximum(diag.mean(1), 1e-280)
        normalized = tab/base[:,None,None]
        with np.errstate(divide='ignore'):
            logs = np.log(normalized) if cells>64 else np.zeros(1,complex)
        self.arrays = [np.ascontiguousarray(v) for v in
                       (branch,cond,total*abs(c),np.outer(c,c)*gram,normalized,logs,np.asarray(symbols,np.int32))]

    def sample(self,n,seed,threads=1,offset=0):
        out = np.empty((n,self.cells*4),np.uint8)
        attempts = np.empty(n,np.int32)
        rc = self.native.lib.memory111(n,self.cells,self.size,self.k,self.radix,threads,
              seed,offset,*[v.ctypes.data for v in self.arrays],out.ctypes.data,attempts.ctypes.data)
        if rc: raise ArithmeticError(f'rejection envelope or input failed ({rc})')
        return out,attempts


def alias(p):
    mass = p*len(p)
    small = list(np.flatnonzero(mass<1))
    large = list(np.flatnonzero(mass>=1))
    prob = np.ones(len(p))
    other = np.arange(len(p),dtype=np.uint32)
    while small and large:
        i, j = small.pop(),large.pop()
        prob[i],other[i] = mass[i],j
        mass[j] -= 1-mass[i]
        (small if mass[j]<1 else large).append(j)
    implied = prob/len(p)+np.bincount(other,weights=(1-prob)/len(p),minlength=len(p))
    assert np.abs(implied-p).sum()<1e-12
    return prob,other


def run(out):
    out.mkdir(exist_ok=True,parents=True)
    native = Native(out)
    optical = np.load(DATA/'iteration110_collective_v1/optical.npz')
    ref = np.load(DATA/'iteration110_bus_v1/laws.npz')
    x = optical['core_x']
    s, axes, observed = prepare(optical,ref['bus'],1.)
    summaries = []
    samples = {}
    for index, (cells, m, radix, shots) in enumerate((
            (4,3,2,65536), (4,6,2,65536), (32768,3,2,128), (32768,6,2,128),
            (4,3,12,65536), (4,4,12,65536), (4,4,16,65536))):
        start = perf_counter()
        z,c,g,bound = branches(s,axes,m)
        disp = dyads(z,observed,cells)
        tab = click_table(x,disp) if radix==2 else pnr_table(x,disp,radix)
        model = Readout(native,tab,c,g,cells,radix)
        prep = perf_counter()-start
        tail = 0 if radix==2 else marginal_tail(x,ref['bus'],cells,radix)
        seed = 111000000+index*100
        a, attempts = model.sample(shots,seed,8)
        times = {}
        batch = 2048 if cells==4 else 16
        for nt in (1,8):
            values = []
            for rep in range(6):
                start = perf_counter()
                model.sample(batch,seed+10*nt+rep,nt)
                if rep: values.append((perf_counter()-start)*1e6/batch)
            times[str(nt)] = values
        row = dict(cells=cells,radix=radix,nodes=m,shots=shots,setup_s=prep,acceptance=shots/int(attempts.sum()),
                   predicted_acceptance=float(1/sum(abs(c))**2),us_by_threads=times,
                   state_bound=bound,physical_overflow_bound=tail,combined_bound=min(1,bound+tail))
        if cells==4:
            word = (a>0)@(1 << np.arange(16))
            hist = np.bincount(word,minlength=65536)
            key = f'{index}'
            samples[key+'_hist'] = hist
            samples[key+'_first'] = a[:512]
            row['threshold_empirical_tv'] = float(abs(hist/shots-ref['truth4']).sum()/2)
            if radix>2:
                samples[key+'_pair'] = np.bincount(a[:,0].astype(int)+radix*a[:,4].astype(int),minlength=radix**2)
                samples[key+'_total'] = np.bincount(a.sum(1).astype(int),minlength=256)
            if radix==2:
                start = perf_counter()
                p = complete_clicks(tab,np.outer(c,c)*g)
                prob,other = alias(p)
                row['cache_extra_setup_s'] = perf_counter()-start
                row['model_tv'] = float(abs(p-ref['truth4']).sum()/2)
                cache = lambda o,nt,sd:native.lib.cached111(len(o),nt,sd,0,other.ctypes.data,prob.ctypes.data,o.ctypes.data)
                o = np.empty((shots,16),np.uint8)
                cache(o,8,seed+90)
                hc = np.bincount(o@(1 << np.arange(16)),minlength=65536)
                samples[key+'_cached'] = hc
                row['cached_threshold_empirical_tv'] = float(abs(hc/shots-ref['truth4']).sum()/2)
                row['cached_us_by_threads'] = {}
                for nt in (1,8):
                    t = []
                    for rep in range(6):
                        start = perf_counter()
                        cache(o[:2048],nt,seed+91+rep)
                        if rep:t.append((perf_counter()-start)*1e6/2048)
                    row['cached_us_by_threads'][str(nt)] = t
        else:
            samples[f'{index}_first'] = np.packbits(a[:8],axis=1,bitorder='little')
        print(index, 'setup',round(prep,3),'accept',round(row['acceptance'],4),
              'us', {k:round(np.median(v),3) for k,v in times.items()}, flush=True)
        summaries.append(row)
        (out/'sampling.json').write_text(json.dumps(summaries,indent=2)+'\n',encoding='utf8')
        np.savez_compressed(out/'records.npz',**samples)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2):
        run(ap.parse_args().output.resolve())
