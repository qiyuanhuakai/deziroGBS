"""All 10,296 high-power bank pairs: certified optical compression, exact integer capture histograms."""
import argparse
import ctypes as ct
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration106_field_v1 import actual_input
from iteration107_cliques_v1 import optical_x
from iteration107_fast_v1 import gaussian_click,oracle,ONE
from iteration115_visible_v1 import ROOT,DATA,I,P
from iteration100_baselines_v1 import frugal

SITES=np.ascontiguousarray(list(itertools.combinations(range(144),2)),np.int32)


def empirical(raw):
    pop=np.array([i.bit_count() for i in range(256)],np.uint8)
    cats=torch.arange(1,9,dtype=torch.uint8,device='cuda')
    cross=torch.zeros((1152,1152),device='cuda',dtype=torch.int32)
    for off in range(0,len(raw),4096):
        board=pop[raw[off:off+4096]]
        if len(board)%8:board=np.pad(board,((0,(-len(board))%8),(0,0)))
        z=torch.as_tensor(board,device='cuda')
        one=(z[:,:,None]==cats).to(torch.int8).reshape(len(board),1152)
        cross+=torch._int_mm(one.T.contiguous(),one)
    h=cross.cpu().numpy()
    margin=h.diagonal().reshape(144,8);i,j=SITES.T
    out=np.zeros((len(i),9,9),np.int32)
    out[:,1:,1:]=h.reshape(144,8,144,8).transpose(0,2,1,3)[i,j]
    out[:,1:,0]=margin[i]-out[:,1:,1:].sum(2)
    out[:,0,1:]=margin[j]-out[:,1:,1:].sum(1)
    out[:,0,0]=len(raw)-out.sum((1,2))
    assert out.min()>=0 and np.all(out.sum((1,2))==len(raw))
    return out


def captures():
    meta=json.loads((DATA/'iteration112_hardware_v1/sources.json').read_text())
    directory=ROOT.parent/'external_sources/jiuzhang3_iteration112_independent'
    saved=np.load(DATA/'iteration114_saturation_v1/covariances.npz');result=[]
    for item in meta:
        path=directory/item['member']
        with path.open('rb') as fh:assert hashlib.file_digest(fh,'sha256').hexdigest()==item['sha256']
        raw=np.memmap(path,np.uint8,'r').reshape(-1,144)
        v=empirical(raw);result.append(v);q=v/len(raw);h=np.arange(9)
        cross=np.einsum('kij,i,j->k',q,h,h)-(q.sum(2)@h)*(q.sum(1)@h)
        assert abs(cross-saved[f'cov{item["file"]}'][tuple(SITES.T)]).max()<1e-12
    return result


def occupancies(w,cap):
    mask=np.arange(256);size=np.array([int(v).bit_count() for v in mask])
    bits=(mask[:,None]>>np.arange(8))&1
    weights=w.astype(np.longdouble)@bits.T
    powers=weights[:,:,None]**np.arange(cap+1)
    out=np.zeros((144,9,cap+1),np.longdouble)
    for h in range(9):
        for k in range(h+1):
            out[:,h]+=(-1)**(h-k)*math.comb(8-k,h-k)*powers[:,size==k].sum(1)
    assert out.min()>-1e-12 and abs(out.sum(1)-1).max()<1e-12
    return np.ascontiguousarray(out.clip(0),float)


def bright_conditional(raw,r):
    groups=raw.reshape(144,8,50)
    u=np.linalg.svd(groups,full_matrices=False)[0]
    rotated=(u.conj().transpose(0,2,1)@groups).reshape(-1,50)
    x=optical_x(rotated,r)
    g=len(SITES);small=np.empty((g,2,4,4));logvac=np.zeros(g)
    for off in range(0,g,64):
        sites=SITES[off:off+64]*8
        h=np.stack([sites[:,0],sites[:,0]+1152,sites[:,1],sites[:,1]+1152],axis=1)
        lo=(sites[:,:,None]+np.arange(1,8)).reshape(len(sites),-1)
        dark=np.column_stack((lo,lo+1152))
        for s,mat in enumerate(x):
            a=mat[h[:,:,None],h[:,None,:]]
            ab=mat[h[:,:,None],dark[:,None,:]]
            dd=np.eye(28)+mat[dark[:,:,None],dark[:,None,:]]
            small[off:off+len(sites),s]=a-ab@np.linalg.solve(dd,ab.transpose(0,2,1))
            logvac[off:off+len(sites)]-=.5*np.linalg.slogdet(dd)[1]
    return small,abs(u[:,:,0])**2,np.sqrt(np.maximum(0,-np.expm1(logvac)))


def joint(lib,x,occ):
    p0=np.exp(-.5*np.linalg.slogdet(np.eye(4)+x)[1].sum(1))
    k=np.eye(4)-np.linalg.inv(np.eye(4)+x)
    count=len(k);d=np.zeros((count,2,3,3));d[:,:,0,0]=1
    for mask in range(1,16):
        ids=[j for j in range(4) if mask>>j&1]
        a=sum(j<2 for j in ids);b=len(ids)-a
        d[:,:,a,b]+=(-1)**len(ids)*np.linalg.det(k[:,:,ids][:,:,:,ids])
    poly=np.zeros((count,5,5))
    for a in range(3):
        for b in range(3):poly[:,a:a+3,b:b+3]+=d[:,0,a,b,None,None]*d[:,1]
    h=np.empty((count,9,9));tail=np.empty(count)
    assert lib.joint115(count,occ.shape[-1]-1,8,*[a.ctypes.data for a in (poly,p0,SITES,occ,h,tail)])==0
    assert abs(h.sum((1,2))-1).max()<1e-10 and tail.min()>-1e-10
    return h,tail


def summarise(law,hw):
    return [dict(avg=float(v.mean()),max=float(v.max()),p95=float(np.quantile(v,.95)))
            for v in (abs(law-x/x.sum((1,2),keepdims=True)).sum((1,2))/2 for x in hw)]


def run(out):
    out.mkdir(exist_ok=True,parents=True)
    dll=out/('joint.dll' if os.name=='nt' else 'joint.so')
    subprocess.run(['g++','-O3','-fopenmp','-shared','-fPIC',str(ROOT/'native/bank_joint115_v1.cpp'),'-o',str(dll)],check=True)
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent)) if os.name=='nt' else None
    lib=ct.CDLL(str(dll.resolve()));lib.joint115.argtypes=[I,I,I]+[P]*6
    hw=captures();raw,r=actual_input()
    x,w,bound=bright_conditional(raw,r)
    w/=w.sum(1,keepdims=True);occ=occupancies(w,48)
    ref,ref_tail=joint(lib,x,occ)
    model=np.load(DATA/'iteration115_tilt_v1/model.npz')
    assert abs(np.diff(model['split'],axis=1,prepend=0)-w).max()<1e-11
    tilts=np.zeros_like(ref);tail=np.zeros(len(ref))
    ids=(2*SITES[:,:,None]+np.arange(2)).reshape(-1,4)
    for weight,mats in zip(model['weights'],model['marginal']):
        part=np.stack([mat[ids[:,:,None],ids[:,None,:]] for mat in mats],axis=1)
        hh,tt=joint(lib,part,occ)
        tilts+=weight*hh;tail+=weight*tt
    delta=abs(tilts-ref).sum((1,2))/2
    summary=dict(pairs=len(SITES),reference_photon_cap=48,reference_projection_bound_mean=float(bound.mean()),
        reference_projection_bound_max=float(bound.max()),reference_count_tail_max=float(ref_tail.max()),
        model_count_tail_max=float(tail.max()),model_tv_vs_compressed_reference_mean=float(delta.mean()),
        model_tv_vs_compressed_reference_max=float(delta.max()),model_vs_hardware=summarise(tilts,hw),
        compressed_reference_vs_hardware=summarise(ref,hw),capture_vs_capture=summarise(hw[0]/hw[0].sum((1,2),keepdims=True),hw[1:])[0])
    arrays=dict(sites=SITES,optical_compressed=ref,model=tilts,compression=bound,reference_tail=ref_tail,
                model_tail=tail,hardware10=hw[0],hardware11=hw[1])
    pop=np.array([i.bit_count() for i in range(256)]);mask=np.arange(65536)
    code=9*pop[mask&255]+pop[mask>>8]
    checks=[]
    from iteration100_spatial_v1 import ROOT as _root
    fl=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
    tests=[(34,35),(1,143),(50,70),tuple(map(int,SITES[bound.argmax()])),tuple(map(int,SITES[delta.argmax()]))]
    for i,(a,b) in enumerate(dict.fromkeys(tests)):
        j=np.flatnonzero((SITES[:,0]==a)&(SITES[:,1]==b))[0]
        sites=(8*np.array([a,b])[:,None]+np.arange(8)).ravel();xx=optical_x(raw[sites],r)
        truth=gaussian_click(oracle(True),xx,ONE)
        count=np.bincount(code,weights=truth,minlength=81).reshape(9,9)
        err=float(abs(ref[j]-count).sum()/2)
        assert err<bound[j]+ref_tail[j]+1e-9
        entry=dict(banks=[a,b],reference_tv=err,model_tv=float(abs(tilts[j]-count).sum()/2),
                   measured_hardware_tv=[float(abs(count-h[j]/h[j].sum()).sum()/2) for h in hw])
        if i==0:
            for k in (2,4):
                prob,times=frugal(fl,np.eye(32)[None]+xx,k,115610000+100*k)
                count_f=np.bincount(code,weights=prob,minlength=81)
                entry[f'frugal{k}']=dict(complete_word_tv=float(abs(prob-truth).sum()/2),
                    count_pair_tv=float(abs(count_f-count.ravel()).sum()/2),us=float(np.median(times)),times=times)
        arrays[f'exact_word_{a}_{b}']=truth;checks.append(entry)
    summary['independent_65536_word_checks']=checks
    np.savez_compressed(out/'counts.npz',**arrays)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=4):run(p.parse_args().output)
