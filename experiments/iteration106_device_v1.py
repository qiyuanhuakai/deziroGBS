"""Frozen optical model, file-order hardware calibration and matched external GPU conditions."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import I,D,bind,Native,save,manifest
from iteration100_baselines_v1 import pair_reference
from iteration100_projection_v1 import physical
from iteration105_jiuzhang_v1 import inputs
from iteration106_field_v1 import actual_input,factors,DATA,BASE
from iteration106_packed_v1 import Packed,model_case,compress,field_panel,OUT as PACKED
from iteration106_sample_v1 import read,benchmark,panel_hist


def fit_hardware(fs,profile,train):
    x=np.stack([2*f.T@f for f in fs])
    fun=bind(ct.CDLL(str(PACKED/'pairs.dll')),'census106',[I,I,I,I,D,D,D,D,D])
    m=fs.shape[-1]//2;idx=np.tril_indices(m,-1);single=np.empty(m);joint=np.empty((m,m))
    p=train/train.sum();coords=np.arange(len(p))
    mu=p@coords;var=p@(coords-mu)**2
    weights=np.array(profile['weights']);values=np.array(profile['values'])
    def scale(params):
        a,b=params;g=values**b
        return a*g/(weights@g)
    def residual(params):
        assert fun(m,len(fs),len(weights),16,x,scale(params),weights,single,joint)==0
        cov=joint[idx]-single[idx[0]]*single[idx[1]]
        return np.array([(1-single).sum()-mu,((1-single)@single+2*cov.sum()-var)/20])
    fit=least_squares(residual,[1.,1.],bounds=([.85,0],[1.2,4]),xtol=1e-11,gtol=1e-11,ftol=1e-11)
    gap=residual(fit.x)
    return dict(values=scale(fit.x).tolist(),weights=weights.tolist()),dict(mean=float(mu),variance=float(var),mean_residual=float(gap[0]),variance_residual=float(20*gap[1]),scale=float(fit.x[0]),exponent=float(fit.x[1]),records=int(train.sum()))


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    native=Native(BASE/'spatial.dll')
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    t,r=actual_input();z=inputs()[0];weights=(np.sinh(r)**2).sum(0)
    names=['optical','hardware','rank2']
    first=perf_counter();op,eff,rr,prof=model_case(1);torch.cuda.synchronize();cold=perf_counter()-first
    m,w,_=compress(t,weights,1)
    hw=np.load(DATA/'iteration104_record_order_v1/windows.npz')['histograms']
    fit_start=perf_counter()
    cal,params=fit_hardware(factors(eff,rr),prof,hw[:32].sum(0))
    fit_seconds=perf_counter()-fit_start
    hold=hw[32:].sum(0);hardware=Packed(m,w,rr,cal)
    second,eff2,_,prof2=model_case(2)
    routes=[op,hardware,second]
    times=[];pairs=[];runs=[];qa=[]
    peer=read(DATA/'iteration103_baselines_v1/summary.json')['pnr'][0]
    ids=np.array(peer['ports']);truth=pair_reference(physical(z,ids),12);tail=truth.reshape(13,13)[-1].sum()+truth.reshape(13,13)[:-1,-1].sum()
    truth_n=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf'][:1024]
    for k,(name,model,pm,tr) in enumerate(zip(names,routes,[prof,cal,prof2],[eff,eff,eff2])):
        fs=factors(tr,rr);sub=np.r_[ids,len(t)+ids]
        qp=sum(a*pair_reference(np.stack([np.eye(4)+2*b*f[:,sub].T@f[:,sub] for f in fs]),12) for a,b in zip(pm['weights'],pm['values']))
        pairs.append(dict(method=name,complete_lower=float(abs(qp-truth).sum()/2),complete_upper=float(abs(qp-truth).sum()/2+tail)))
        eig=np.concatenate([np.linalg.eigvalsh(2*f@f.T) for f in fs]);omega=np.exp(-2j*np.pi*np.arange(513)/1024)
        total=np.fft.irfft(sum(a*np.exp(-.5*np.log1p((1-omega[:,None])*b*eig).sum(1)) for a,b in zip(pm['weights'],pm['values'])))
        ids0=[np.load(PACKED/f'panel_{j}.npz')['ports'] for j in (0,1,4)]
        laws=[field_panel(fs,pm,ix,oracle) if name=='hardware' else np.load(PACKED/f'panel_{j}.npz')[str(1 if k==0 else 2)] for j,ix in zip((0,1,4),ids0)]
        references=[np.load(PACKED/f'panel_{j}.npz')['target'] for j in (0,1,4)]
        qa.append(dict(method=name,total_tv=float(abs(total-truth_n).sum()/2),click_tvs=[float(abs(l-q).sum()/2) for l,q in zip(laws,references)]))
        for pnr in (False,True):
            for j,n in enumerate((4,64,2048,16384)):
                x=benchmark(lambda n,s,o:model.draw(n,pnr,s,o),n,106150000+k*100+10*pnr+j,13)
                times.append(dict(method=name,pnr=pnr,**x))
            if k==2:continue
            n=1048576;hist=np.zeros(1153,np.int64);ph=np.zeros(1024,np.int64);pair=np.zeros(169,np.int64)
            observed=[0]*len(laws)
            for off in range(0,n,4096):
                x=model.draw(4096,pnr,106155000+2*k+pnr,off)
                hist+=np.bincount(np.count_nonzero(x,axis=1),minlength=len(hist))
                if pnr:
                    ph+=np.bincount(x.sum(1).astype(np.int64),minlength=len(ph))
                    pair+=np.bincount(x[:,ids].astype(np.int64).clip(0,12)@np.array([1,13]),minlength=len(pair))
                for j,(ix,l,q) in enumerate(zip(ids0,laws,references)):observed[j]+=panel_hist(x,ix)[l>q].sum()
            cdf=hist.cumsum()/n
            gaps=[float(o/n-l[l>q].sum()) for o,l,q in zip(observed,laws,references)]
            assert max(map(abs,gaps))<.004
            row=dict(method=name,pnr=pnr,n=n,hardware_holdout_ks=float(abs(cdf-hold.cumsum()/hold.sum()).max()),panel_gaps=gaps)
            if pnr:
                row.update(total_sampling_ks=float(abs(ph.cumsum()/n-total.cumsum()).max()),pair_sampling_tv=float(abs(pair/n-qp).sum()/2))
                assert row['total_sampling_ks']<.004 and row['pair_sampling_tv']<.004
            runs.append(row)
            np.savez_compressed(out/f'{name}_{int(pnr)}.npz',click=hist,photons=ph,pair=pair,model_pair=qp,model_total=total)
            print(row,flush=True)
        # Stress pathwise agreement between full reconstructed and packed FP32 fields.
        from iteration106_field_v1 import Field
        direct=Field(tr,rr,pm)
        a=direct.draw(4096,True,106158000+k);b=model.draw(4096,True,106158000+k)
        assert np.count_nonzero(a!=b)<20
        qa[-1]['dense_packed_different_cells']=int(np.count_nonzero(a!=b));qa[-1]['compared_cells']=a.size
        assert np.array_equal(b[123:]>0,model.draw(4096-123,False,106158000+k,123))
    same=[];prior=read(DATA/'iteration104_device_release_v1/validation.json')
    for j,ports in enumerate(prior['observed_ports']):
        ix=np.array(ports);gs=np.unique(ix//8);n=64 if len(ix)==144 else 2048
        projected=Packed(m[gs],w[ix],rr,prof)
        tic=benchmark(lambda n,s,o:projected.draw(n,False,s,o).astype(np.uint8),n,106159000+j,15)
        row=dict(panel=j,modes=len(ix),batch=n,us=tic['us'])
        if len(ix)==8:
            truth8=np.empty(256)
            assert oracle(8,2,physical(z,ix),np.zeros((2,16)),truth8,np.empty(2))==0
            l=field_panel(factors(eff,rr),prof,ix,oracle)
            row['complete_tv']=float(abs(l-truth8).sum()/2)
            h=sum((panel_hist(projected.draw(2048,False,106160000+j,k*2048),np.arange(8)) for k in range(128)))
            row['empirical_tv']=float(abs(h/h.sum()-truth8).sum()/2)
        same.append(row)
    radius=float(np.sqrt(np.log(2/.0005)/(2*1048576))+np.sqrt(np.log(2/.0005)/(2*hold.sum())))
    save(out,'summary.json',dict(times=times,runs=runs,optical=qa,pnr=pairs,external_gpu=same,
        fit=params,hardware_profile=cal,fit_seconds=fit_seconds,setup_s=cold,hardware_holdout_records=int(hold.sum()),
        two_sample_ks_radius=radius,windows='First 32 file-order windows for two fitted moments; last 32 for evaluation.',
        hw_interval_assumption='iid hardware records within the evaluated portion.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
