"""How much interference is lost when the actual 25 sources are partitioned into coherent groups?"""
import argparse
import tempfile
import shutil
import subprocess
import ctypes as ct
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,Native,realify,I,D,bind,save,manifest
from iteration103_sample_v1 import BASE,DATA
from iteration105_jiuzhang_v1 import inputs,xpair,JZ


def group_by_overlap(weight,cap):
    sets=[(k,) for k in range(len(weight))]
    while True:
        pairs=[(float(weight[np.ix_(a,b)].sum()),a,b) for i,a in enumerate(sets) for b in sets[:i] if len(a)+len(b)<=cap]
        if not pairs:break
        _,a,b=max(pairs)
        sets.remove(a);sets.remove(b);sets.append(tuple(sorted(a+b)))
    return sorted(sets)


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    z,modes,rr=inputs()
    native=Native(BASE/'spatial.dll')
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    # Replay the same recursive determinant algorithm in the platform's
    # extended precision, retaining all sectors through a single transform.
    origin=(ROOT/'native/click_oracle96_v1.cpp').read_text()
    fast=origin.replace('using Q=__float128;','using Q=long double; static_assert(sizeof(Q)>8);').replace('sqrtq(','std::sqrt(').replace('expq(','std::exp(').replace('sectors>8','sectors>128')
    assert fast!=origin
    with tempfile.TemporaryDirectory() as temp:
        src=Path(temp)/'oracle.cpp';src.write_text(fast)
        subprocess.run([shutil.which('g++'),'-O3','-std=c++20','-ffp-contract=off','-shared',str(src),'-o',str(out/'extended.dll')],check=True)
    extended=bind(ct.CDLL(str(out/'extended.dll')),'oracle96',[I,I,D,D,D,D])
    max_check=0.
    # The compact QR retains physical losses and relative phases of all columns.
    t=z['T'][np.argsort(z['order'])]
    _,compact=np.linalg.qr(t,mode='reduced')
    features=np.array([[realify(compact[:,2*i:2*i+2])@xpair(rr[s*25+i])@realify(compact[:,2*i:2*i+2]).T for i in range(25)] for s in range(2)])
    w=sum(f.reshape(25,-1)@f.reshape(25,-1).T for f in features)
    np.fill_diagonal(w,0)
    all_pairs=float(w.sum()/2)
    total=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf'][:1024]
    coord=np.arange(len(total))
    true_var=float(total@(coord-total@coord)**2)
    panel_ids=[np.arange(0,128,8),np.array([0,1,7,8,15,30,50,51])]
    panel_ids += [np.r_[np.arange(8)+288,np.arange(8)+856],np.arange(16)]
    exact=[]
    for ids in panel_ids:
        from iteration100_projection_v1 import physical
        truth=np.empty(2**len(ids))
        assert oracle(len(ids),2,physical(z,ids),np.zeros((2,2*len(ids))),truth,np.empty(2))==0
        exact.append(truth)
    configurations=[('one_source',1,[(i,) for i in range(25)])]
    for cap in (2,4,8,16):
        configurations.append(('overlap',cap,group_by_overlap(w,cap)))
    configurations+=[('index_order',4,[tuple(range(i,min(25,i+4))) for i in range(0,25,4)]),('all',25,[tuple(range(25))])]
    rows=[]
    omega=np.exp(-2j*np.pi*np.arange(513)/1024)
    for name,cap,sets in configurations:
        log=np.zeros_like(omega)
        for f in features:
            for group in sets:
                ev=np.linalg.eigvalsh(f[list(group)].sum(0))
                log-=.5*np.log1p((1-omega[:,None])*ev).sum(1)
        p=np.fft.irfft(np.exp(log))
        variance=float(p@(coord-p@coord)**2)
        gained=float(sum(w[np.ix_(g,g)].sum()/2 for g in sets))
        assert abs((true_var-variance)-(all_pairs-gained))<3e-8
        panel_tvs=[]
        for ids,tru in zip(panel_ids,exact):
            m=len(ids);covs=[]
            for s in range(2):
                for group in sets:
                    q=np.eye(2*m)
                    for k in group:
                        b=realify(modes[s*25+k][ids])
                        q+=b@xpair(rr[s*25+k])@b.T
                    covs.append(q)
            # Different groups carry distinct spectral labels and their click
            # words combine by a bitwise OR.
            law=np.empty(1<<m)
            assert extended(m,len(covs),np.array(covs),np.zeros((len(covs),2*m)),law,np.empty(2))==0
            if name=='all':max_check=max(max_check,float(abs(law-tru).sum()/2))
            if name=='one_source' and len(panel_tvs)>=2:
                jp=JZ/f'optical_panel_{len(panel_tvs)-2}.npz'
                if jp.exists():max_check=max(max_check,float(abs(law-np.load(jp)['coherent']).sum()/2))
            panel_tvs.append(float(abs(law-tru).sum()/2))
        rows.append(dict(method=name,cap=cap,groups=[list(g) for g in sets],variance=variance,variance_missing=all_pairs-gained,photon_tv=float(abs(p-total).sum()/2),photon_ks=float(abs(p.cumsum()-total.cumsum()).max()),click_tvs=panel_tvs))
        np.savez_compressed(out/f'total_{name}_{cap}.npz',law=p)
        print(name,cap,rows[-1]['photon_tv'],panel_tvs,flush=True)
    assert rows[-1]['photon_tv']<3e-12 and max(rows[-1]['click_tvs'])<1e-9 and max_check<1e-9
    np.savez_compressed(out/'interference.npz',pair_terms=w)
    save(out,'summary.json',dict(rows=rows,extended_precision_max_tv_vs_binary128=max_check,reference_variance=true_var,variance_cross_source=all_pairs,panels=[p.tolist() for p in panel_ids],scope='Exact numerical optical marginals under source labels. The variance guide uses the full transfer; no benchmark panel or hardware record enters the partition.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
