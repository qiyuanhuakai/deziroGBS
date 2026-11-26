"""Exact optical references, two-photon interference identities and equal-input CPU baselines."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.linalg import null_space
from threadpoolctl import threadpool_limits
from iteration100_baselines_v1 import frugal,pair_reference,pe_sample,GaussianInput
from iteration100_spatial_v1 import Native,I,D,bind,save,manifest
from iteration103_sample_v1 import BASE,DATA
from iteration104_validate_v1 import interval
from iteration104_transfer_v1 import make_cov
from iteration105_two_mode_v1 import OUT,TwoMode,example


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    dll=OUT/'two.dll'
    native=Native(BASE/'spatial.dll')
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    def clicks(q):
        m=q.shape[1]//2
        z=np.empty(2**m)
        assert oracle(m,len(q),q,np.zeros((len(q),2*m)),z,np.empty(2))==0
        return z
    bs=np.array([[1,1],[1,-1]])/np.sqrt(2)
    twin=TwoMode(dll,bs,[.91,.91],[True])
    split=TwoMode(dll,bs,[.91,.91],[True],independent_arms=True)
    paired=TwoMode(dll,bs,[.91,-.91])
    hom={}
    for j,(model,name) in enumerate(((twin,'indistinguishable'),(split,'labelled'),(paired,'oppositely_squeezed'))):
        x=model.draw(262144,True,105010000+j,workers=16)
        sector=x.sum(1)==2
        coincidence=int((x[sector,0]==1).sum())
        hom[name]=dict(records=len(x),two_photon_records=int(sector.sum()),coincidences=coincidence,prob=coincidence/sector.sum(),odd_records=int(np.any(x%2,axis=1).sum()),unequal=int(np.sum(x[:,0]!=x[:,1])))
        if j==0:assert not (x%2).any() and not coincidence
        elif j==1:assert abs(coincidence/sector.sum()-.5)<.009
        else:assert np.array_equal(x[:,0],x[:,1])
    t,r=example(16);model=TwoMode(dll,t,r);q=make_cov(t,r)
    truth=clicks(q)
    single=r.copy();single[:,1]=0
    previous=clicks(make_cov(t,single))
    first=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
    peer=[]
    for k in (2,4):
        pr,times=frugal(first,q,k,105012000+100*k)
        peer.append(dict(K=k,us=float(np.median(times)),tv=float(abs(pr-truth).sum()/2)))
    word=np.uint32(1)<<np.arange(16,dtype=np.uint32)
    def simulate(n,seed):
        h=np.zeros(65536,np.int64)
        for off in range(0,n,8192):
            x=model.draw(8192,False,seed,off,16)
            h+=np.bincount(x@word,minlength=len(h))
        return h
    trial=simulate(131072,105013000)
    hold=simulate(2097152,105013001)
    click=interval(truth,hold,trial,int(hold.sum()),.0005)
    assert abs(click['event_gap'])<.003
    np.savez_compressed(out/'click16.npz',truth=truth,rank1=previous,trial=trial,hold=hold)
    # Observe the first and last port of the same correlated 16-output input.
    ids=[0,15,16,31];small=np.ascontiguousarray(q[:,ids][:,:,ids])
    fock=pair_reference(small,12)
    states=[]
    for u,row in zip(t,r):
        sub=u[[0,15]]
        e,v=np.linalg.eigh(np.eye(2)-sub.conj().T@sub)
        iso=np.vstack([sub,np.sqrt(e)[:,None]*v.conj().T])
        iso[:,row<0]*=1j
        unitary=np.column_stack((iso,null_space(iso.conj().T)))
        assert abs(unitary.conj().T@unitary-np.eye(4)).max()<1e-12
        states.append((GaussianInput(abs(row),(0,1),4),unitary))
    def pe(n,seed):
        total=np.zeros((n,2),np.uint32)
        for s,(state,u) in enumerate(states):
            x,st=pe_sample(state,u,n,seed=seed+s,device='cpu',return_stats=True,impl='v2',window=4,max_photons=96)
            assert not np.any(st.censored)
            total+=np.asarray(x[:,:2],np.uint32)
        return total
    def codes(x):return np.bincount(x.clip(0,12)@np.array([1,13]),minlength=169)
    pn=[];mine=[];ours=np.zeros(169,np.int64);other=ours.copy()
    for rep in range(17):
        tic=perf_counter();x=pe(2048,105014000+2*rep);elapsed=(perf_counter()-tic)*1e6/2048
        tic=perf_counter();y=model.draw(2048,True,105015000,rep*2048)[:,[0,15]].copy();local=(perf_counter()-tic)*1e6/2048
        if rep:pn.append(elapsed);mine.append(local);ours+=codes(y);other+=codes(x)
    blind=np.zeros(169,np.int64)
    for i in range(128):blind+=codes(model.draw(8192,True,105015001,i*8192,16)[:,[0,15]])
    pnr=interval(fock,blind,ours,int(blind.sum()),.0005)
    pnr['full_pnr_upper']=float(pnr['upper']+fock.reshape(13,13)[-1].sum()+fock.reshape(13,13)[:-1,-1].sum())
    assert abs(pnr['event_gap'])<.004
    np.savez_compressed(out/'pnr.npz',truth=fock,ours=ours,peer=other,hold=blind)
    save(out,'summary.json',dict(hom=hom,rank1_complete_click_tv=float(abs(previous-truth).sum()/2),native_click=click,click_records=int(hold.sum()),frugal=peer,local_pnr=pnr,pnr_records=int(blind.sum()),pnr_comparison=dict(shots=int(ours.sum()),our_us=float(np.median(mine)),pe_us=float(np.median(pn)),our_empirical_tv=float(abs(ours/ours.sum()-fock).sum()/2),pe_empirical_tv=float(abs(other/other.sum()-fock).sum()/2))))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
