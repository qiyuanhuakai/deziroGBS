"""Conditional total and support table, with optical PNR coefficients on a downward-closed set."""
import argparse
import ctypes as ct
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration112_cases_v1 import optics, gaussian_click, oracle, ONE

ROOT = Path(__file__).resolve().parents[1]
P = ct.c_void_p
I = ct.c_int


def library(path):
    if os.name == 'nt':
        library.runtime = os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    lib = ct.CDLL(str(path.resolve()))
    lib.sector113.argtypes = [I,I,I,P,ct.c_double,P]
    lib.counts113.argtypes = [I]*5+[P,ct.c_double,P,P,P]
    lib.draw113.argtypes = [I,I,P,P,I,ct.c_uint64,P]
    lib.draw113.restype = None
    return lib


def sectors(lib, x, cap):
    m = len(x)//2
    q = np.eye(2*m)+x
    vacuum = np.exp(-np.linalg.slogdet(q)[1]/2)
    k = np.asfortranarray(np.eye(2*m)-np.linalg.inv(q))
    table = np.empty((1<<m,cap+2))
    assert lib.sector113(m,cap,8,k.ctypes.data,vacuum,table.ctypes.data)==0
    return k,vacuum,table


def compositions(n,d):
    if d == 1:
        yield (n,)
        return
    for i in range(1,n-d+2):
        for row in compositions(n-i,d-1):
            yield (i,)+row


def words(m, common, extended, width):
    blocks = [np.zeros(1,np.uint64)]
    for n in range(1,extended+1):
        batch = []
        for d in range(1,min(n,m if n<=common else width)+1):
            shifts = np.array(list(itertools.combinations(range(m),d)),np.uint64)*4
            for comp in compositions(n,d):
                batch.append((np.array(comp,np.uint64)<<shifts).sum(1))
        blocks.append(np.concatenate(batch))
    return np.concatenate(blocks), np.cumsum([0]+[len(v) for v in blocks],dtype=np.int32)


def tv(a,b):
    return float(np.abs(a-b).sum()/2)


def cdf_table(a):
    cumulative = np.cumsum(a,dtype=np.longdouble)/a.sum(dtype=np.longdouble)
    cdf = np.asarray(cumulative,float)
    np.minimum(cdf,1,out=cdf)
    cdf[-1]=1.
    actual = np.diff(cdf,prepend=0.)
    return cdf, tv(actual,a)


def benchmark(lib, law, rows, route):
    prepare=perf_counter()
    cdf, rounding = cdf_table(law)
    cdf_seconds=perf_counter()-prepare
    shots=8192
    draws=np.empty((shots,rows.shape[1]),np.uint8)
    times=[]
    for rep in range(17):
        # The disjoint counter intervals contain 2^32 outputs per (route,repeat).
        domain=1130000+100*route+rep
        seed=(0x9e3779b97f4a7c15*(domain<<32)) % (1<<64)
        start=perf_counter()
        lib.draw113(rows.shape[1],len(law),cdf.ctypes.data,rows.ctypes.data,shots,seed,draws.ctypes.data)
        if rep: times.append((perf_counter()-start)*1e6/shots)
    return dict(us=float(np.median(times)),times=times,cdf_tv=rounding,cdf_setup_s=cdf_seconds,
                uniform_grid_tv_bound=len(cdf)*2**-53,shots=shots), draws


def run(out, lib, common=8, extended=14, width=3):
    out.mkdir(parents=True,exist_ok=True)
    _,_,_,_,x,_=optics(4)
    m=len(x)//2
    cap=48
    start=perf_counter()
    k,vacuum,table=sectors(lib,x,cap)
    masks=np.arange(1<<m)
    bits=(masks[:,None]>>np.arange(m))&1
    weights=1<<np.arange(m)
    support_size=bits.sum(1)
    click=table[:,-1].clip(0)
    click/=click.sum()
    click_rows=bits.astype(np.uint8)
    direct_setup=perf_counter()-start
    truth=np.load(ROOT/'results/iteration112_four_peers_v1/peers.npz')
    click_agreement=tv(click,truth['truth'])
    assert click_agreement<1e-10
    direct,ds=benchmark(lib,click,click_rows,2*common)
    start=perf_counter()
    keys,offsets=words(m,common,extended,width)
    values=np.empty(len(keys))
    assert lib.counts113(m,common,extended,width,8,k.ctypes.data,vacuum,
                         keys.ctypes.data,offsets.ctypes.data,values.ctypes.data)==0
    rows=((keys[:,None]>>(4*np.arange(m,dtype=np.uint64)))&15).astype(np.uint8)
    cells=(rows>0)@weights
    total=rows.sum(1).astype(int)
    resolved=np.bincount(cells*(cap+1)+total,weights=values,minlength=(1<<m)*(cap+1)).reshape(-1,cap+1)
    expected=table[:,:cap+1]
    energies=np.arange(cap+1)
    feasible=energies>=support_size[:,None]
    impossible=float(abs(expected[~feasible]).sum())
    retained=(energies<=common)|((support_size[:,None]<=width)&(energies<=extended))
    retained_error=float(abs((resolved-expected)[retained]).sum())
    assert min(values)>-1e-15 and retained_error<1e-9 and impossible<1e-9
    assert tv(expected.sum(0),truth['total'][:cap+1])<1e-10
    # For n=s or s=1, the whole count vector follows uniquely from (n,S).
    unique=(energies==support_size[:,None])|(support_size[:,None]==1)
    residual=np.where(~retained & feasible,expected.clip(0),0)
    unresolved=float(residual[~unique].sum())
    masks2,n2=np.where(residual>0)
    select=masks2!=0
    masks2,n2=masks2[select],n2[select]
    extra=bits[masks2].astype(np.uint8)
    # Deterministic completion for unresolved sectors. All excess is put at
    # the occupied port with the largest optical intensity.
    intensity=np.diag(x)[:m]+np.diag(x)[m:]
    j=np.argmax(np.where(extra,intensity,-np.inf),axis=1)
    extra[np.arange(len(extra)),j]+=(n2-support_size[masks2]).astype(np.uint8)
    law=np.r_[values.clip(0),residual[masks2,n2]]
    records=np.ascontiguousarray(np.vstack([rows,extra]))
    law/=law.sum()
    eigen=np.linalg.eigvalsh(x)
    s=np.linspace(1+1e-4,1+.999/eigen.max(),1024)
    tail=float(np.exp(np.min(-.5*np.log1p((1-s[:,None])*eigen).sum(1)-(cap+1)*np.log(s))))
    pnr_setup=direct_setup+perf_counter()-start
    pnr,ps=benchmark(lib,law,records,2*common+1)
    observed=records.sum(1).astype(int)
    pair=np.minimum(records[:,0],10).astype(int)+11*np.minimum(records[:,3],10).astype(int)
    simpair=np.minimum(ps[:,0],10).astype(int)+11*np.minimum(ps[:,3],10).astype(int)
    pair_law=np.bincount(pair,weights=law,minlength=121)
    click_pnr=np.bincount((records>0)@weights,weights=law,minlength=1<<m)
    # Independent multivariate coefficient checks. z_i = t*w_i reduces each
    # total-degree slice to a one-variable analytic determinant on sqrt(W)Ksqrt(W).
    rng=np.random.default_rng(113991)
    max_test=0.
    for j in range(16):
        w=rng.uniform(.15,1.5,m)
        scale=np.sqrt(np.tile(w,2))
        ev=np.linalg.eigvalsh(scale[:,None]*k*scale)
        ref=[vacuum]
        traces=[0]+[sum(ev**i) for i in range(1,common+1)]
        for n in range(1,common+1):
            ref.append(sum(traces[q]*ref[n-q] for q in range(1,n+1))/(2*n))
        use=total<=common
        coef=np.bincount(total[use],weights=values[use]*np.exp(rows[use]@np.log(w)),minlength=common+1)
        max_test=max(max_test,float(abs(coef-ref).max()))
    assert max_test<1e-10
    summary=dict(modes=m,active_phases=4,common=common,extended=extended,support_width=width,
        cap=cap,native_coefficients=len(keys),entries=len(law),direct_setup_s=direct_setup+direct['cdf_setup_s'],pnr_setup_s=pnr_setup+pnr['cdf_setup_s'],
        tail_bound=tail,unresolved_mass=unresolved,numerical_checks=dict(click_tv=click_agreement,
        retained_sector_l1=retained_error,impossible_mass=impossible,min_coefficient=float(values.min()),
        weighted_polynomial_max_abs=max_test),direct=direct,pnr=pnr,
        pnr_truncation_tv_bound=unresolved+tail,pnr_pair_model_tv=tv(pair_law,truth['pair']),
        pnr_pair_empirical_tv=tv(np.bincount(simpair,minlength=121)/len(ps),truth['pair']),
        pnr_click_model_tv=tv(click_pnr,truth['truth']),
        max_total_cdf_gap=float(abs(np.cumsum(np.bincount(observed,weights=law,minlength=128)-truth['total'])).max()))
    np.savez_compressed(out/'laws.npz',sector=table,keys=keys,values=values,offsets=offsets,
                        pair=pair_law,click=click_pnr,direct_samples=ds,pnr_samples=ps)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,default=float),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('output',type=Path)
    a.add_argument('--library',type=Path)
    a.add_argument('--common',type=int,choices=[8,10],default=8)
    args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    dll=args.library
    if dll is None:
        dll=args.output/('sector.dll' if os.name=='nt' else 'sector.so')
        eigen=os.environ.get('EIGEN_INCLUDE_DIR',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'))
        subprocess.run(['g++','-std=c++20','-O3','-fopenmp','-shared','-fPIC','-DEIGEN_DONT_PARALLELIZE',
                        '-I',eigen,str(ROOT/'native/sector_support113_v1.cpp'),'-o',str(dll)],check=True)
    with threadpool_limits(limits=2):
        run(args.output,library(dll),common=args.common)
