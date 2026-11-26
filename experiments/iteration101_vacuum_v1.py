"""Scan all two-group vacuum events, then expand the worst new event to complete laws."""
import argparse
import ctypes as ct
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

from iteration101_copula_v1 import DATA, BASE, OLD, K, features
from iteration100_projection_v1 import Projector, physical
from iteration100_spatial_v1 import ORDER, Tree, Native, I, D, bind, realify, split_joint, save, manifest


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    lib=Native(BASE/'spatial.dll')
    model=Tree(lib,dict(np.load(BASE/'prepared.npz')))
    proj=Projector(model)
    fit=DATA/'iteration101_fit_v1'
    corr=np.load(fit/'fit.npz')['corr']
    rows=json.loads((fit/'summary.json').read_text())['rows']
    z=np.load(ORDER)
    v=z['v'][np.argsort(z['order'])]
    t=np.array([realify(v[8*g:8*g+8]) for g in range(len(corr))])
    a,b=np.triu_indices(len(corr),1)
    log_prob=np.zeros(len(a))
    eye=np.eye(16)
    for x in z['x']:
        tx=t@x
        local=eye+tx@t.transpose(0,2,1)
        chol=np.linalg.cholesky(local)
        log_local=2*np.log(np.diagonal(chol,axis1=1,axis2=2)).sum(1)
        white=np.array([solve_triangular(l,h,lower=True) for l,h in zip(chol,t)])
        wx=white@x
        for off in range(0,len(a),256):
            i=a[off:off+256]
            j=b[off:off+256]
            cross=wx[i]@white[j].transpose(0,2,1)
            sign,delta=np.linalg.slogdet(eye-cross@cross.transpose(0,2,1))
            assert np.all(sign==1)
            log_prob[off:off+len(i)]-=.5*(log_local[i]+log_local[j]+delta)
    optical=np.exp(log_prob)
    coeff=np.array([features(proj.totals[g])[1][0] for g in range(len(corr))])
    approx=np.sum(coeff[a]*coeff[b]*corr[a,b,None]**np.arange(K+1),axis=1)
    error=approx-optical
    idx=int(abs(error).argmax())
    g,h=int(a[idx]),int(b[idx])
    ports=np.r_[8*g+np.arange(8),8*h+np.arange(8)]
    truth=np.empty(65536)
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    assert oracle(16,2,physical(z,ports),np.zeros((2,32)),truth,np.empty(2))==0
    assert abs(optical[idx]-truth[0])<2e-12
    gram=t.transpose(0,2,1)@t
    ref=split_joint(lib,gram[g],gram[h],z['x'],int(model.caps[g]),int(model.caps[h]))
    left=features(proj.totals[g])[1]
    right=features(proj.totals[h])[1]
    joint=(left*corr[g,h]**np.arange(K+1))@right.T
    law=(proj.mask[h].T@joint.T@proj.mask[g]).ravel()
    tree=proj.clicks(ports)
    assert abs(law[0]-approx[idx])<1e-13 and abs(ref[0,0]-optical[idx])<2e-12
    for row in rows:
        i,j=sorted(row['groups'])
        k=np.where((a==i)&(b==j))[0].item()
        prev=np.load(fit/f'pair_{rows.index(row)}.npz')
        assert abs(optical[k]-prev['truth'][0])<2e-12 and abs(approx[k]-prev['bits'][0])<2e-12
    np.savez_compressed(out/'scan.npz',a=a,b=b,optical=optical,model=approx)
    np.savez_compressed(out/'worst.npz',counts=joint,reference=ref,bits=law,truth=truth,tree=tree)
    save(out,'summary.json',dict(pairs=len(a),worst_pair=[g,h],vacuum_model=approx[idx],vacuum_optical=optical[idx],
         absolute_error_percentiles=np.percentile(abs(error),[50,90,99,100]).tolist(),above_one_percent=int((abs(error)>.01).sum()),
         count_tv=abs(joint-ref).sum()/2,click_tv=abs(law-truth).sum()/2,tree_click_tv=abs(tree-truth).sum()/2))
    manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):
        main(ap.parse_args().output.resolve())
