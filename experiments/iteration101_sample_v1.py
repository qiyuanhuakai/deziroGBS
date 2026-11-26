"""Emit complete physical-order click or photon records and evaluate fixed optical events."""
import argparse
import ctypes as ct
import json
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import beta
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import ROOT, HIST, ORDER, bind, I, U, D, J, W, SHORT, save, manifest, H
from iteration101_copula_v1 import BASE, DATA
from iteration100_baselines_v1 import frugal
from iteration100_projection_v1 import physical

FIT = DATA/'iteration101_fit_v1'
RUN = DATA/'iteration101_sample_v1'
N = 262144
BLOCK = 2048


class Sampler:
    def __init__(self, dll, ids=None):
        self.runtime = os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        self.draw = bind(ct.CDLL(str(dll)), 'spatial101_draw',
            [I,I,I,I,I,I,U,U,D,D,I,D,D,ct.c_double,ct.c_double,J,W,I,J,W,D,D,SHORT,SHORT])
        with np.load(BASE/'prepared.npz') as archive:
            group_count = len(archive['ports'])
            self.ids = np.arange(group_count) if ids is None else np.asarray(ids)
            self.caps = archive['caps'][self.ids]
            self.starts = archive['leafstarts'][self.ids]
            self.root = archive['root']
            self.masks = archive['masks']
            self.prior = archive['prior']
            self.weights = archive['ports'][self.ids]
        with np.load(FIT/'fit.npz') as z:
            self.lower = np.ascontiguousarray(np.linalg.cholesky(z['corr'][np.ix_(self.ids,self.ids)]))
            self.cuts = z['cuts'][self.ids]
            self.direction = z['direction'][self.ids]
            self.score = z['score'][self.ids]
            self.mean, self.sigma = float(z['mean']), float(z['sigma'])
        self.full = ids is None

    def sample(self, n, pnr, locked, workers, seed, offset=0):
        if locked and not self.full:
            raise ValueError('The optical total CDF covers the full device.')
        result = np.empty((n,len(self.ids)*8),np.uint16)
        target = np.empty(n,np.uint16)
        rc = self.draw(len(self.ids),8,n,int(pnr),int(locked),workers,seed,offset,self.lower,self.cuts,
                       self.cuts.shape[1],self.direction,self.score,self.mean,self.sigma,self.caps,
                       self.root,len(self.root),self.starts,self.masks,self.prior,self.weights,result,target)
        if rc:
            raise RuntimeError(f'Native sampling failed with code {rc}')
        return result, target


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    src = ROOT/'native/spatial_copula101_v1.cpp'
    dll = out/'copula.dll'
    build = [shutil.which('g++'),'-std=c++20','-O3','-shared','-fopenmp','-ffp-contract=off',str(src),'-o',str(dll)]
    subprocess.run(build,check=True)
    sampler = Sampler(dll)
    fit = json.loads((FIT/'summary.json').read_text())
    masks, count_events, optical, count_optical = [], [], [], []
    for j in range(len(fit['rows'])):
        with np.load(FIT/f'pair_{j}.npz') as z:
            mask = z['bits']>z['truth']
            mask_n = z['counts']>z['reference']
            masks.append(mask)
            count_events.append(mask_n)
            optical.append(z['truth'][mask].sum())
            count_optical.append(z['reference'][mask_n].sum())
    original = np.load(HIST)['total_click_histogram']
    optical_total = np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
    hardware_cdf = original.cumsum()/original.sum()
    optical_cdf = np.pad(optical_total,(0,len(original)-len(optical_total))).cumsum()
    powers = (1<<np.arange(8)).astype(np.uint16)
    # Test shared latent draw equivalence for both measurement routes and thread/chunk independence.
    for lock in (False,True):
        click, n0 = sampler.sample(96,False,lock,1,101000101)
        pnr, n1 = sampler.sample(96,True,lock,4,101000101)
        last, _ = sampler.sample(59,True,lock,2,101000101,37)
        assert np.array_equal(click,pnr>0) and np.array_equal(n0,n1)
        assert np.array_equal(pnr.sum(1),n1) and np.array_equal(last,pnr[37:])
    save(out,'plan.json',dict(build=build,records_per_route=N,block=BLOCK,workers=16,
        comparisons='Four fixed optical pair events selected from analytic laws before sampling; four hardware and four optical CDFs.',
        hashes={p.relative_to(ROOT).as_posix():H(p) for p in (src,Path(__file__),FIT/'fit.npz',HIST,BASE/'prepared.npz')}))
    rows=[]
    for lock in (False,True):
        for pnr in (False,True):
            seed=101010000+2*int(lock)+int(pnr)
            times=[]
            for rep in range(8):
                start=perf_counter()
                sampler.sample(8192,pnr,lock,16,101020000+2*int(lock)+int(pnr),rep*8192)
                if rep: times.append((perf_counter()-start)*1e6/8192)
            clicks=np.zeros(len(original),np.int64)
            photons=np.zeros_like(clicks)
            hits=np.zeros(len(masks),int)
            count_hits=np.zeros_like(hits)
            for off in range(0,N,BLOCK):
                x,n=sampler.sample(BLOCK,pnr,lock,16,seed,off)
                if pnr: assert np.array_equal(x.sum(1),n)
                binary=x>0
                clicks+=np.bincount(binary.sum(1),minlength=len(clicks))
                photons+=np.bincount(n.astype(int),minlength=len(photons))
                for j,row in enumerate(fit['rows']):
                    g,h=row['groups']
                    word=(binary[:,8*g:8*g+8]@powers)+256*(binary[:,8*h:8*h+8]@powers)
                    hits[j]+=masks[j][word].sum()
                    if pnr:
                        ng=x[:,8*g:8*g+8].sum(1)
                        nh=x[:,8*h:8*h+8].sum(1)
                        count_hits[j]+=count_events[j][ng,nh].sum()
            # Simultaneous 99% over 24 two-sided fixed-event intervals, eight sampled CDFs
            # and the one independent experimental CDF (33 tests in total).
            alpha=.01/33
            def gaps(found,truth):
                lo=beta.ppf(alpha/2,found,N-found+1)
                hi=beta.ppf(1-alpha/2,found+1,N-found)
                return [np.maximum(np.maximum(lo-truth,truth-hi),0).tolist(),(found/N-truth).tolist()]
            click_lower, cg = gaps(hits,np.array(optical))
            count_lower, ngaps = gaps(count_hits,np.array(count_optical)) if pnr else ([],[])
            rows.append(dict(locked=lock,pnr=pnr,seed=seed,records=N,us=times,median_us=float(np.median(times)),
                click_event_difference=cg,count_event_difference=ngaps,click_tv_lower99=click_lower,count_tv_lower99=count_lower,
                hw_ks=float(abs(clicks.cumsum()/N-hardware_cdf).max()),optical_photon_ks=float(abs(photons.cumsum()/N-optical_cdf).max())))
            np.savez_compressed(out/f'hist_{int(lock)}_{int(pnr)}.npz',click=clicks,photon=photons,click_hits=hits,count_hits=count_hits)
            print(rows[-1],flush=True)
    base=[]
    g,h=fit['rows'][0]['groups']
    reduced=Sampler(dll,[g,h])
    ports=np.r_[8*g+np.arange(8),8*h+np.arange(8)]
    q=physical(np.load(ORDER),ports)
    truth=np.load(FIT/'pair_0.npz')['truth']
    flib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
    for k in (2,4):
        law,times=frugal(flib,q,k,101030000+k*100)
        base.append(dict(k=k,tv=float(abs(law-truth).sum()/2),median_us=float(np.median(times)),us=times))
    times=[]
    for j in range(8):
        start=perf_counter()
        reduced.sample(BLOCK,False,False,1,101040000,j*BLOCK)
        if j:times.append((perf_counter()-start)*1e6/BLOCK)
    eps=np.sqrt(np.log(2*33/.01)/(2*N))
    eh=np.sqrt(np.log(2*33/.01)/(2*int(original.sum())))
    save(out,'summary.json',dict(rows=rows,frugal=base,reduced_us=times,reduced_median_us=float(np.median(times)),
        reduced_tv=fit['rows'][0]['click_tv'],reduced_groups=[g,h],hw_records=int(original.sum()),
        optical_ks_radius99=eps,hardware_ks_radius99=eps+eh,validation='chunk, threads, click/PNR equivalence and recordwise photon totals passed'))
    manifest(out)


if __name__=='__main__':
    arg=argparse.ArgumentParser()
    sub=arg.add_subparsers(dest='action',required=True)
    bench=sub.add_parser('benchmark')
    bench.add_argument('output',type=Path)
    draw=sub.add_parser('draw')
    draw.add_argument('output',type=Path)
    draw.add_argument('--shots',type=int,default=1024)
    draw.add_argument('--seed',type=int,required=True)
    draw.add_argument('--workers',type=int,default=1)
    draw.add_argument('--pnr',action='store_true')
    draw.add_argument('--lock-total',action='store_true')
    args=arg.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.action=='benchmark':
            run(args.output.resolve())
        else:
            s=Sampler(RUN/'copula.dll')
            x,n=s.sample(args.shots,args.pnr,args.lock_total,args.workers,args.seed)
            with args.output.open('xb') as stream:np.savez_compressed(stream,records=x,total_photons=n)
