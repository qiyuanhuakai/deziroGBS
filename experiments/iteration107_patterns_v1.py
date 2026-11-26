"""Correct resolved photon allocations at fixed spatial totals using the optical Fock probabilities."""
import argparse
import ctypes as ct
import itertools
import math
from pathlib import Path
import shutil
import subprocess
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,Native,I,D,W,bind,save,manifest,occupancy_prior
from iteration100_projection_v1 import physical
from iteration105_jiuzhang_v1 import inputs
from iteration106_sample_v1 import DATA,read
from iteration107_tables_v1 import OUT as TABLES,BASE,LIMIT,kernel

OUT=DATA/'iteration107_patterns_v1'
ORDER=5


def layouts():
    states=[];offset=np.full((ORDER+1,ORDER+1),-1,np.int32);site=[]
    for n in range(ORDER+1):
        combos=[]
        for seq in itertools.combinations_with_replacement(range(8),n):
            rank=sum(math.comb(j+k-1,k) for k,j in enumerate(seq,1))
            combos.append((rank,seq))
        site.append([v for k,v in sorted(combos)])
    for a in range(ORDER+1):
        for b in range(ORDER+1-a):
            offset[a,b]=len(states)
            for right in site[b]:
                for left in site[a]:
                    seq=(*left,*(j+8 for j in right))
                    states.append(sum(1<<(4*j) for j in seq))
    lookup={v:i for i,v in enumerate(states)}
    left=[];right=[];starts=[]
    for v in states:
        starts.append(len(left));digits=[(v>>(4*j))&15 for j in range(16)]
        choices=[0]
        for j,d in enumerate(digits):
            if d:choices=[x+(k<<(4*j)) for x in choices for k in range(d+1)]
        left.extend(lookup[x] for x in choices);right.extend(lookup[v-x] for x in choices)
    return np.array(states,np.uint64),offset,site,np.array(starts),np.array(left),np.array(right)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
    dll=out/'fock.dll'
    subprocess.run([shutil.which('g++'),'-std=c++20','-O3','-march=x86-64-v3','-ffp-contract=off','-fopenmp','-shared','-I',str(eigen),str(ROOT/'native/photon_patterns107_v1.cpp'),'-o',str(dll)],check=True)
    runtime=Native(BASE/'spatial.dll');fn=bind(ct.CDLL(str(dll)),'patterns107',[I,I,I,D,W,D])
    words,offset,site,starts,left,right=layouts()
    occ=((words[:,None]>>(4*np.arange(16,dtype=np.uint64)))&15).astype(int)
    mask=(occ>0)@(1<<np.arange(16));z=inputs()[0]
    with np.load(TABLES/'tables.npz') as source:pairs=source['pairs'];weights=source['weights']
    counts=np.load(TABLES/'counts.npz')['target_counts']
    oracle=bind(ct.CDLL(str(DATA/'iteration105_source_groups_v1/extended.dll')),'oracle96',[I,I,D,D,D,D])
    pos=[];tabs=[];cdf=[];probs=[];rows=[]
    for g,(i,j) in enumerate(pairs):
        ids=np.r_[8*i+np.arange(8),8*j+np.arange(8)]
        q=physical(z,ids);components=[]
        for spec in q:
            p=np.empty(len(words));assert fn(16,len(words),16,spec,words,p)==0
            components.append(p)
        prob=np.add.reduceat(components[0][left]*components[1][right],starts)
        table=np.empty(len(words),np.float32);cuts=np.empty(len(words),np.uint64);mult=np.empty_like(prob)
        ref=counts[g].reshape(LIMIT,LIMIT);target=np.empty(65536);assert oracle(16,2,q,np.zeros((2,32)),target,np.empty(2))==0
        prior=[occupancy_prior(weights[k],LIMIT-1) for k in (i,j)]
        previous=(prior[1].T@ref@prior[0]).ravel()
        err=rounding=0.
        for a in range(ORDER+1):
            for b in range(ORDER+1-a):
                size=len(site[a])*len(site[b]);ix=slice(offset[a,b],offset[a,b]+size)
                o=occ[ix];w=np.r_[weights[i],weights[j]]
                comb=math.factorial(a)*math.factorial(b)/np.prod(np.array([math.factorial(k) for k in range(ORDER+1)])[o],axis=1)
                pm=comb*np.prod(w**o,axis=1);assert abs(pm.sum()-1)<1e-12
                mass=ref[b,a]
                err=max(err,abs(mass-prob[ix].sum()))
                table[ix],cuts[ix],_,r=kernel(pm,prob[ix])
                rounding+=mass*r;mult[ix]=mass*pm
        updated=previous+np.bincount(mask,weights=prob-mult,minlength=65536)
        assert err<1e-11 and updated.min()>-1e-11
        probs.append(prob);tabs.append(table);cdf.append(cuts)
        rows.append(dict(pair=[int(i),int(j)],sector_mass=float(prob.sum()),max_sector_reference_discrepancy=float(err),weighted_conditional_rounding=rounding,old_click_tv=float(abs(previous-target).sum()/2),new_click_tv=float(abs(updated-target).sum()/2)))
        if (i,j)==(34,35):np.savez_compressed(out/'panel34_35.npz',model=updated,previous=previous,optical=target)
        if g%20==0:print(g,rows[-1],flush=True)
    choose=np.array([[math.comb(i,k) if k<=i else 0 for k in range(ORDER+1)] for i in range(8+ORDER)],np.int32)
    sizes=np.array([len(x) for x in site],np.int32)
    np.savez_compressed(out/'tables.npz',codes=words,offset=offset,sizes=sizes,choose=choose,move=tabs,cdf=cdf)
    np.savez_compressed(out/'optical.npz',codes=words,prob=probs,offset=offset,sizes=sizes)
    save(out,'summary.json',dict(rows=rows,max_total=ORDER,patterns=len(words),scope='Two independently squeezed spectra. Exact numerical joint PNR sectors up to five detected photons across sixteen detector bins.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
