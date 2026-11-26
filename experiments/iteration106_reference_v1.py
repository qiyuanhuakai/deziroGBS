"""Exact numerical panel probabilities of the adaptive grouping rule from marked Gaussian determinants."""
import argparse
import ctypes as ct
import json
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT, I, J, D, Native, bind, save, manifest, realify
from iteration103_sample_v1 import DATA, BASE
from iteration105_jiuzhang_v1 import xpair
from iteration106_groups_v1 import actual,synthetic

OUT=DATA/'iteration106_reference_v1'
CAPS={2:4,4:2,8:1}


class Marked:
    def __init__(self,dll,t,r,nodes):
        self.runtime=Native(BASE/'spatial.dll')
        self.fn=bind(ct.CDLL(str(dll)),'group106_law',[I,I,I,I,I,D,D,J,J,D,D,J,D])
        self.parts=[]
        # Separate top-level trees. Different roots are distinguishable.
        children={v[j] for v in nodes if v['source']<0 for j in ('left','right')}
        for root in sorted(set(range(len(nodes)))-children):
            members=[]
            def visit(i):
                v=nodes[i]
                if v['source']<0:visit(v['left']);visit(v['right'])
                members.append(i)
            visit(root)
            mapping={k:i for i,k in enumerate(members)}
            sources=list(nodes[root]['ids'])
            order={k:nodes[k]['cut'] for k in members}
            for k in members[::-1]:
                v=nodes[k]
                if v['source']<0:
                    for j in ('left','right'):order[v[j]]=max(order[v[j]],v['cut'])
            ix=[];meta=[];vac=[]
            for k in members:
                v=nodes[k];cols=[2*sources.index(i)+j for i in v['ids'] for j in (0,1)]
                meta.append([mapping.get(v['left'],-1),mapping.get(v['right'],-1),len(ix),len(cols),v['cut'],order[k]])
                ix.extend(cols)
                vac.append(np.prod(1-np.tanh(r[list(v['ids'])])**2))
            roots=np.zeros(len(members),np.int32);roots[-1]=1
            transfer=np.ascontiguousarray(np.concatenate([t[j] for j in sources],axis=1))
            self.parts.append((transfer,np.array(meta,np.int32),np.array(ix,np.int32),np.repeat(np.tanh(r[sources]),2),np.array(vac),roots))
    def evaluate(self,ids,points,workers=16,total=False):
        result=np.ones(len(points),complex)
        for t,meta,ix,lam,vac,roots in self.parts:
            if total:
                t=np.linalg.qr(t,mode='reduced')[1]
                z=np.broadcast_to(points.reshape(-1,1),(len(points),len(t)))
            else:t=t[ids];z=points
            a=np.empty(len(points),complex)
            args=[np.ascontiguousarray(t).view(float),np.ascontiguousarray(z,dtype=complex).view(float),meta,ix,lam,vac,roots,a.view(float)]
            assert self.fn(len(z),len(t),t.shape[1],len(meta),workers,*args)==0
            result*=a
        return result
    def clicks(self,ids):
        words=np.arange(1<<len(ids),dtype=np.uint32)
        # A set bit marks an allowed detector; every other port in the panel is vacuum.
        points=((words[:,None]>>np.arange(len(ids),dtype=np.uint32))&1)
        cdf=self.evaluate(ids,points)
        assert max(abs(cdf.imag))<1e-10 and abs(cdf[-1]-1)<1e-9
        p=cdf.real.astype(np.longdouble)
        for j in range(len(ids)):
            x=p.reshape(-1,2,1<<j);x[:,1]-=x[:,0]
        assert p.min()>-3e-10
        return p.astype(float)


def truth_click(t,r,ids):
    m=len(ids);cov=np.eye(2*m)
    for u,a in zip(t,r):
        b=realify(u[ids]);cov+=b@xpair(a)@b.T
    lib=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    p=np.empty(1<<m)
    assert lib(m,1,cov[None,:,:],np.zeros((1,2*m)),p,np.empty(2))==0
    return p


def sectors(r,cut):
    p=np.array([1.])
    q=np.tanh(r)**2
    for a in q:p=np.convolve(p,(1-a)*a**np.arange(cut+1))[:cut+1]
    return p


def probability_tail(r,n=1024):
    q=np.tanh(r)**2
    z=np.linspace(1.00001,.99999/np.sqrt(q.max()),500)
    return float(np.exp((np.log1p(-q)-np.log1p(-z[:,None]**2*q)).sum(1)-n*np.log(z)).min())


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    dll=out/'laws.dll'
    eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
    subprocess.run([shutil.which('g++'),'-O3','-std=c++20','-march=x86-64-v3','-ffp-contract=off','-fopenmp','-shared','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(ROOT/'native/source_laws106_v1.cpp'),'-o',str(dll)],check=True)
    rows=[]
    for label,case,panels in [('synthetic',synthetic(16),[np.arange(16)]),('jiuzhang',actual(CAPS),[np.arange(16),np.r_[288:296,856:864],np.arange(0,128,8),np.array([0,1,7,8,15,30,50,51])])]:
        t,r,nodes=case
        model=Marked(dll,t,r,nodes)
        for j,ids in enumerate(panels):
            start=perf_counter();p=model.clicks(ids);elapsed=perf_counter()-start
            if label=='synthetic':
                q=truth_click(t,r,ids);base=Marked(dll,t,r,nodes[:len(r)]).clicks(ids)
            else:
                if j<2:
                    z=np.load(DATA/f'iteration105_jiuzhang_v1/optical_panel_{1-j}.npz');q=z['truth'];base=z['coherent']
                else:
                    from iteration100_projection_v1 import physical
                    from iteration105_jiuzhang_v1 import inputs
                    v=physical(inputs()[0],ids);q=np.empty(1<<len(ids))
                    lib=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
                    assert lib(len(ids),2,v,np.zeros((2,2*len(ids))),q,np.empty(2))==0
                    base=Marked(dll,t,r,nodes[:len(r)]).clicks(ids)
            neg=float(-p[p<0].sum());assert neg<1e-7
            tv=float(abs(p-q).sum()/2)
            row=dict(label=label,panel=j,ports=ids.tolist(),base_tv=float(abs(base-q).sum()/2),group_tv=tv,timing_s=elapsed,negative_mass=neg)
            if label=='synthetic':
                row['full_output_sector_tail']=float(1-sectors(r,5).sum())
                assert tv<=row['full_output_sector_tail']+1e-8
            rows.append(row)
            print(row,flush=True)
            np.savez_compressed(out/f'{label}_{j}.npz',target=q,model=p,source_labelled=base,ports=ids)
        w=np.exp(-2j*np.pi*np.arange(513)/1024)
        f=model.evaluate([],w,total=True)
        p=np.fft.irfft(f)
        assert abs(p.sum()-1)<1e-10 and p.min()>-1e-12
        if label=='jiuzhang':
            q=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf'][:len(p)]
            total=dict(tv=float(abs(p-q).sum()/2),ks=float(abs(p.cumsum()-q.cumsum()).max()),mean=float(p@np.arange(len(p))),fft_tail=probability_tail(r))
            np.savez_compressed(out/'total.npz',model=p,target=q)
    save(out,'summary.json',dict(rows=rows,total=total,caps=CAPS))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
