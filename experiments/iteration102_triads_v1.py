"""Invert the physical three-group photon PGF and sample a child given two ancestor counts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from iteration100_spatial_v1 import Native, compact, local_kernel, realify, cdf_rows, realized, ORDER, save, manifest, H
from iteration102_forest_v1 import FIT, BASE, DATA, Tables

TRI=DATA/'iteration102_triads_v1'


def triple(lib,matrices,spectra,caps):
    # Coordinates are child, parent, grandparent. Schur-eliminate grandparent
    # at a fixed phase; eliminate parent for a batch of phases; expand the
    # remaining determinant in the child photon count via Newton's recurrence.
    h=np.concatenate(matrices)
    sizes=[len(m) for m in matrices]
    da,db,dc=sizes
    d=da+db
    ca,cb,cc=map(int,caps)
    ny=1<<cb.bit_length()
    nz=1<<cc.bit_length()
    if min(ny,nz)<2:raise ValueError('Fourier grid requires positive caps')
    ys=np.exp(-2j*np.pi*np.arange(ny)/ny)
    zs=np.exp(-2j*np.pi*np.arange(nz//2+1)/nz)
    ys[ny//2]=zs[-1]=-1
    kernels=[]
    for x in spectra:
        k,logp,_=local_kernel(h@x@h.T)
        e,v=np.linalg.eigh(k[d:,d:])
        kernels.append((k[:d,:d],k[:d,d:]@v,e,logp))
    fourier=np.empty((nz//2+1,ny,ca+1),complex)
    identity=np.eye(db)
    for iz,z in enumerate(zs):
        eigs=[]
        pref=np.ones(ny,complex)
        for initial,cross,e,logp in kernels:
            k=initial+(cross*(z/(1-z*e)))@cross.T
            ab=k[:da,da:]
            bb=k[da:,da:]
            ev=np.linalg.eigvals(bb)
            pref*=np.exp(logp-.5*np.log1p(-z*e).sum()-.5*np.log1p(-ys[:,None]*ev).sum(1))
            rhs=np.broadcast_to(ab.T,(ny,db,da))
            solved=np.linalg.solve(identity-ys[:,None,None]*bb,rhs)
            effective=k[:da,:da]+ys[:,None,None]*(ab@solved)
            eigs.append(np.linalg.eigvals(effective))
        values=np.ascontiguousarray(np.column_stack(eigs))
        assert lib.complex(ny,values.shape[1],ca,values.view(float),pref.view(float),fourier[iz].view(float))==0
    # The endpoints become real after inverting the parent axis.
    along=np.fft.ifft(fourier,axis=1)
    assert abs(along[[0,-1]].imag).max()<2e-12
    return np.fft.irfft(along,n=nz,axis=0)[:cc+1,:cb+1,:].copy()


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    t=Tables(FIT/'prepared.npz')
    lib=Native(BASE/'spatial.dll')
    z=np.load(ORDER)
    v=z['v'][np.argsort(z['order'])]
    maps=[realify(v[8*g:8*g+8]) for g in range(len(t.order))]
    grams=[m.T@m for m in maps]
    mats=[compact(g) for g in grams]
    targets=[int(g) for g in t.order if t.parent[g]>=0 and t.parent[t.parent[g]]>=0]
    def run(g):
        p=int(t.parent[g])
        a=int(t.parent[p])
        nodes=[g,p,a]
        law=triple(lib,[mats[i] for i in nodes],z['x'],t.caps[nodes])
        # Check real PGFs away from the inversion contour, and all three
        # pair marginals against separate bivariate optical evaluations.
        from iteration100_spatial_v1 import split_joint
        err=0.
        for axis,(left,right) in enumerate(((p,g),(a,g),(a,p))):
            ref=split_joint(lib,grams[left],grams[right],z['x'],int(t.caps[left]),int(t.caps[right]))
            err=max(err,float(abs(law.sum(axis)-ref).sum()/2))
        grid=[np.arange(s) for s in law.shape]
        e=0.
        for values in [(0,0,0),(.31,.82,.57),(.94,.09,.75),(.99,.98,.97)]:
            mat=sum((1-u)*grams[i] for u,i in zip(values,[a,p,g]))
            expected=np.exp(-sum(np.linalg.slogdet(np.eye(len(x))+x@mat)[1] for x in z['x'])/2)
            result=np.einsum('i,j,k,ijk->',*[u**n for u,n in zip(values,grid)],law,optimize=True)
            e=max(e,abs(result-expected))
        negative=float(-law[law<0].sum())
        clean=law.clip(0)
        mass=clean.sum(2)
        clean[mass==0]=t.marg[g]
        cuts=cdf_rows(clean)
        pc=t.pair(a,p)
        actual=pc[:,:,None]*realized(cuts)
        tv=float(abs(law-actual).sum()/2)
        # This bounds per-triad roundoff when the grandparent/parent marginal
        # is the prepared pair law. Induction contributes at most one sum per path.
        assert max(tv,negative,err,e)<3e-11,(nodes,tv,negative,err,e)
        row=dict(groups=[a,p,g],shape=list(law.shape),pair_marginal_tv=err,pgf_error=float(e),
                 conditional_tv=tv,negative_mass=negative)
        print('triad',a,p,g,tv,flush=True)
        return g,cuts,row
    # Reserve capacity once, retaining the original pair conditionals for the
    # first two levels. The same layout also permits samples of marginalized pairs.
    offsets=t.offset.copy()
    second=np.full(len(t.order),-1,np.int32)
    arrays=[t.cdf]
    pos=len(t.cdf)
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for g,cuts,row in pool.map(run,targets):
            offsets[g]=pos
            second[g]=row['groups'][0]
            arrays.append(cuts.ravel())
            pos+=cuts.size
            rows.append(row)
    np.savez_compressed(out/'prepared.npz',cdf=np.concatenate(arrays),offset=offsets,second=second)
    save(out,'summary.json',dict(triads=rows,additional_pairs=len(rows),total_spatial_pairs=len(rows)+sum(t.parent>=0).item(),
        conservative_path_tv=sum(r['conditional_tv']+r['pair_marginal_tv'] for r in rows)
        +len(t.order)*2e-14,prepare_s=perf_counter()-start,
        source_hashes={p.name:H(p) for p in [Path(__file__),FIT/'prepared.npz',ORDER]}))
    manifest(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):
        prepare(ap.parse_args().output.resolve())
