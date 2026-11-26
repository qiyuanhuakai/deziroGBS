"""Exact-interference regression, synthetic baselines, sector tails and measured width scaling."""
import argparse
import ctypes as ct
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_baselines_v1 import frugal,pair_reference,purification,pe_sample
from iteration100_spatial_v1 import realify,save,manifest
from iteration104_validate_v1 import interval
from iteration105_jiuzhang_v1 import xpair
from iteration106_groups_v1 import Groups,forest,synthetic,DATA
from iteration106_reference_v1 import Marked,truth_click,sectors,OUT as REF
from iteration106_sample_v1 import benchmark,panel_hist


def covariance(t,r,ids):
    q=np.eye(2*len(ids))
    for u,a in zip(t,r):
        b=realify(u[ids]);q+=b@xpair(a)@b.T
    return q


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    t,r,nodes=synthetic(16)
    g=Groups(t,r,nodes)
    ids=np.arange(16);q=covariance(t,r,ids);truth=truth_click(t,r,ids)
    ref=Marked(REF/'laws.dll',t,r,nodes).clicks(ids)
    h=np.zeros(65536,np.int64);ph=np.zeros(169,np.int64)
    pair_ids=[0,15];pair_true=pair_reference(covariance(t,r,pair_ids)[None,:,:],12)
    for i in range(128):
        h+=panel_hist(g.draw(8192,False,106050000,i*8192,16),ids)
        x=g.draw(8192,True,106050001,i*8192,16)
        ph+=np.bincount(np.minimum(x[:,pair_ids],12)@np.array([1,13]),minlength=169)
    trial=panel_hist(g.draw(32768,False,106050002,workers=16),ids)
    event=ref>truth
    assert abs(h[event].sum()/h.sum()-ref[event].sum())<.003
    audit=interval(truth,h,trial,int(h.sum()),.0005)
    peer={}
    lib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
    for k in (2,4):
        law,times=frugal(lib,q[None,:,:],k,106051000+20*k)
        peer[k]=dict(tv=float(abs(law-truth).sum()/2),us=float(np.median(times)))
    st,u,res=purification(covariance(t,r,pair_ids))
    def pd(n,seed,offset):
        x,stats=pe_sample(st,u,n,seed=seed+offset,device='cpu',return_stats=True,impl='v2',window=4,max_photons=64)
        assert not np.any(stats.censored)
        return x[:,:2]
    pe=benchmark(pd,2048,106052000,7)
    peh=np.zeros(169,int);ourh=np.zeros(169,int)
    for j in range(16):
        y=pd(2048,106060000,j);x=g.draw(2048,True,106060050,j*2048,1)
        for arr,count in [(y,peh),(x[:,pair_ids],ourh)]:
            count+=np.bincount(np.minimum(arr,12)@np.array([1,13]),minlength=169)
    rows=[]
    for m in [4,16,144,1152,4097,16384]:
        system=Groups(*synthetic(m))
        for pnr in (False,True):
            for workers in ([1,16] if m==16 else [16]):
                n=2048 if m<8192 else 256
                result=benchmark(lambda n,s,o:system.draw(n,pnr,s,o,workers),n,106070000+m+int(pnr),7)
                rows.append(dict(width=m,pnr=pnr,workers=workers,**result))
        raw=system.draw(93,True,106080000+m,11,1)
        assert np.array_equal(raw[7:]>0,system.draw(86,False,106080000+m,18,16))
    # Eight differently pumped twin beams enter the same sixteen-output unitary.
    # At one emitted pair the closed-form interference amplitudes are independent
    # of the Fock recursion and the determinant reference.
    rr=np.linspace(.11,.17,8)*(-1.)**np.arange(8)
    from scipy.linalg import dft
    u0=dft(16,scale='sqrtn')*np.exp(1j*.17*np.arange(16)**2)
    tt=np.stack([u0[:,2*i:2*i+2] for i in range(8)])
    tree=forest(8,[list(range(8))],np.ones((8,8)),{2:3,4:2,8:1})
    many=Groups(tt,rr,tree)
    amp=np.zeros((16,16),complex)
    for u,a in zip(tt,rr):amp+=np.tanh(a)*(np.outer(u[:,0],u[:,1])+np.outer(u[:,1],u[:,0]))
    pi=abs(amp)**2;pi*=1/pi.sum()
    expected=np.triu(pi+pi.T,1)+np.diag(np.diag(pi))
    obs=np.zeros((16,16),int)
    for j in range(64):
        x=many.draw(8192,True,106099000,8192*j,16)
        hits=x[x.sum(1)==2]
        rridx,col=np.where(hits);_,paircols=np.unique(rridx,return_index=True)
        first=col[paircols];last=col[np.r_[paircols[1:]-1,len(col)-1]]
        np.add.at(obs,(first,last),1)
    gap=float(abs(obs/obs.sum()-expected).sum()/2)
    assert gap<.023 and obs.sum()>50000
    # Source-labelled fallback is exactly the previous native optical path,
    # including its per-record output at nonzero offsets and across threads.
    old=Groups(t,r,forest(len(r),[],np.eye(len(r)),{}))
    assert np.array_equal(old.draw(2048,True,106099005,7,16),old.base.draw(2048,True,106099005,7,1))
    np.savez_compressed(out/'heldout.npz',click=h,trial=trial,model=ref,truth=truth,pnr=ph,pnr_target=pair_true,many=obs,many_target=expected)
    save(out,'summary.json',dict(widths=rows,click_tv=float(abs(ref-truth).sum()/2),click_sample=audit,full_output_fallback_mass=float(1-sectors(r,5).sum()),
         click_records=int(h.sum()),pnr_records=int(ph.sum()),frugal=peer,pe_us=pe['us'],equal_pnr_samples=int(peh.sum()),pe_empirical_tv=float(abs(peh/peh.sum()-pair_true).sum()/2),our_empirical_tv=float(abs(ourh/ourh.sum()-pair_true).sum()/2),eight_source_two_photon_records=int(obs.sum()),eight_source_conditional_empirical_tv=gap))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
