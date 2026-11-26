"""Reduce a general passive transfer with independently squeezed inputs to its leading supermode per spectrum."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_baselines_v1 import pair_reference
from iteration100_spatial_v1 import ORDER,Native,realify,I,D,bind,save,manifest
from iteration103_sample_v1 import DATA,BASE
from iteration104_spectral_v1 import Allocation,OUT


def reduce_input(transfer,squeezing):
    r=np.asarray(squeezing,float)
    t=np.asarray(transfer,complex)
    if r.ndim!=2:raise ValueError('Expected spectra by squeezed input modes')
    if t.ndim==2:t=np.broadcast_to(t,(len(r),)+t.shape)
    if t.ndim!=3 or (len(t),t.shape[2])!=r.shape or min(t.shape)<1 or not (np.isfinite(t).all() and np.isfinite(r).all()):
        raise ValueError('Incompatible transfer matrix or squeezing')
    weights=[];eta=[];hot=[];modes=[];discarded=[]
    for u,row in zip(t,r):
        if np.linalg.svd(u,compute_uv=False)[0]>1+1e-12:raise ValueError('The transfer must be passive')
        k=int(np.argmax(abs(row)))
        p=abs(u[:,k])**2
        efficiency=min(1.,p.sum())
        weights.append(p if efficiency>0 else np.ones(len(u)))
        eta.append(efficiency);modes.append(k);hot.append(row[k])
        discarded.extend(np.delete(row,k))
    with mp.workdps(65):
        log_overlap=-mp.fsum(mp.log(mp.cosh(float(x))) for x in discarded)
        upper=mp.sqrt(-mp.expm1(log_overlap))
        overlap=mp.exp(log_overlap)
    model=Allocation(OUT/'allocation.dll',weights,hot,eta)
    return model,dict(retained=modes,squared_vacuum_overlap=float(overlap),discarded_trace_distance=float(upper),allocation_tv=model.bound,complete_click_and_pnr_upper=min(1.,float(upper)+model.bound))


def make_cov(t,r):
    out=[]
    for u,row in zip(t,r):
        b=realify(u)
        out.append(np.eye(len(b))+(b*(np.expm1(np.r_[-2*row,2*row])/2))@b.T)
    return np.array(out)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    native=Native(BASE/'spatial.dll')
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    rows=[]
    for m in (16,1152,4097):
        rng=np.random.default_rng(104060000+m)
        rotations=[]
        for j in range(2):
            a=rng.normal(size=(m,8))+1j*rng.normal(size=(m,8))
            u=np.linalg.qr(a,mode='reduced')[0]
            rotations.append(u*np.linspace(.89,.62,8))
        t=np.array(rotations)
        for scale in ((.003,.025,.12) if m==16 else (.025,)):
            r=np.array([[1.35]+[scale]*7,[1.1]+[scale/2]*7])
            model,diag=reduce_input(t,r)
            row=dict(modes=m,discarded_squeezing=scale,**diag)
            if m==16:
                q=make_cov(t,r)
                kept=r.copy();kept[:,1:]=0
                qk=make_cov(t,kept)
                def clicks(cov):
                    v=np.empty(65536)
                    assert oracle(16,2,cov,np.zeros((2,32)),v,np.empty(2))==0
                    return v
                full,approx=clicks(q),clicks(qk)
                row['complete_16bit_tv']=float(abs(full-approx).sum()/2)
                assert row['complete_16bit_tv']<=diag['complete_click_and_pnr_upper']
                ids=[0,15,16,31]
                fc=pair_reference(np.ascontiguousarray(q[:,ids][:,:,ids]),12)
                ac=pair_reference(np.ascontiguousarray(qk[:,ids][:,:,ids]),12)
                tv=float(abs(fc-ac).sum()/2)
                tail=fc.reshape(13,13)[-1].sum()+fc.reshape(13,13)[:-1,-1].sum()
                row['two_detector_full_pnr_tv_interval']=[max(0.,tv-model.bound),float(tv+tail+model.bound)]
                if scale==.025:
                    np.savez_compressed(out/'input16.npz',transfer=t,squeezing=r)
                    x=model.draw(1048576,False,104061000,workers=16)
                    observed=np.bincount(x@(np.uint32(1)<<np.arange(16,dtype=np.uint32)),minlength=65536)/len(x)
                    row['native_empirical_click_tv']=float(abs(observed-full).sum()/2)
                    event=approx>full
                    row['event_error_vs_implemented']=float(observed[event].sum()-approx[event].sum())
            times=[]
            for j in range(9):
                start=perf_counter();model.draw(2048,True,104062000+m,j*2048,16)
                if j>1:times.append((perf_counter()-start)*1e6/2048)
            row['pnr_us']=float(np.median(times))
            rows.append(row)
    z=np.load(ORDER)
    raw=z['T'][np.argsort(z['order'])]
    # Each two-mode squeezed source gives two single-mode squeezers on the
    # sum and difference ports. The column phases choose the squeeze quadrature.
    tr=np.empty_like(raw)
    tr[:,::2]=1j*(raw[:,::2]+raw[:,1::2])/np.sqrt(2)
    tr[:,1::2]=(raw[:,::2]-raw[:,1::2])/np.sqrt(2)
    rr=np.repeat(z['squeezing'].T,2,axis=1)
    model,jz=reduce_input(tr,rr)
    ids=[289,861,0,8,123,402,568,535]
    local=make_cov(np.broadcast_to(tr[ids],(2,len(ids),tr.shape[1])),rr)
    from iteration100_projection_v1 import physical
    assert np.max(abs(local-physical(z,ids)))<1e-11
    kept=np.zeros_like(rr)
    for s,k in enumerate(jz['retained']):kept[s,k]=rr[s,k]
    mean=sum(np.sum(abs(tr)**2*np.sinh(row)**2) for row in kept)
    jz['projected_mean_photons']=float(mean)
    jz['optical_mean_photons']=float(sum(np.sum(abs(tr)**2*np.sinh(row)**2) for row in rr))
    qk=make_cov(np.broadcast_to(tr[ids],(2,len(ids),tr.shape[1])),kept)
    # Only the local photon-number pair is needed for a full-machine TV witness.
    ix=[0,1,len(ids),len(ids)+1]
    p=pair_reference(np.ascontiguousarray(local[:,ix][:,:,ix]),12)
    q=pair_reference(np.ascontiguousarray(qk[:,ix][:,:,ix]),12)
    jz['actual_two_port_pnr_tv_lower']=max(0.,float(abs(p-q).sum()/2)-model.bound)
    save(out,'summary.json',dict(rows=rows,jiuzhang_rank_reduction=jz,physical_pair=[289,861],scope='Arbitrary passive transfer, pure independent squeezed inputs per unresolved spectrum. All outputs, including thresholding and loss, contract trace distance.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('benchmark','draw'))
    p.add_argument('output',type=Path)
    p.add_argument('--input',type=Path)
    p.add_argument('--seed',type=int)
    p.add_argument('--shots',type=int,default=2048)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--pnr',action='store_true')
    args=p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.action=='benchmark':main(args.output.resolve())
        else:
            if args.seed is None or args.input is None:p.error('--input and --seed are required')
            with np.load(args.input) as z:model,diag=reduce_input(z['transfer'],z['squeezing'])
            x=model.draw(args.shots,args.pnr,args.seed,workers=args.workers)
            with args.output.open('xb') as f:np.savez_compressed(f,records=x,tv_upper=diag['complete_click_and_pnr_upper'])
