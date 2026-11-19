"""Measure all optical/window pairs, then sample both PNR and binary Gaussian-copula models."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[k]='1'
import argparse
import ctypes as C
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import scipy
from scipy.special import ndtr
from iteration92_common_v2 import ROOT,H,save,manifest,read
from iteration92_quality_v1 import confidence

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
src=ROOT/'native/gaussian_pair_copula95_v1.cpp'
ref=ROOT/'results/iteration93_reference_v2'
hardware=ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz'
layout=ROOT/'results/iteration83_loss_pilot_v1/input.npz'
prior=ROOT/'results/iteration95_third_click_v1/quality_3.npz'
totals=ROOT/'results/iteration84_photon_cdf_v1/reference.npz'
inputs={i:[ref/f'input_{i}.npz'] for i in range(4)}
inputs[4]=[ROOT/f'results/iteration94_actual_quality_v1/input_{s}.npz' for s in (0,1)]
blas_path=next((Path(scipy.__file__).parent.parent/'scipy.libs').glob('*openblas*.dll'))
files=[Path(__file__),src,src.with_name('gaussian_click_window95_v2.cpp'),src.with_name('gaussian_click_window95_v1.cpp'),
       src.with_name('gaussian_rng_domain84_v1.hpp'),hardware,layout,prior,totals,*[f for paths in inputs.values() for f in paths],
       *[ref/f'law_{j}.npz' for j in range(4)]]
save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},blas_sha256=H(blas_path),
    calibration='single-port photon PGF; invert all two-port threshold probabilities, eigendecompose; full positive spectrum or top 16/64 with diagonal residual',
    baseline='exact all-pair DP of physical-order 4+4 window mixtures and unweighted optical window',
    samples='65536 small or 65536 full-size per configuration, counts in batches; cap64 overflow symbol 65 retained',
    seeds='quality 0x95d00000+16*case+method; timing 0x95d10000+512*case+32*(rep+1)+2*method+observable',
    timing='three warm serial repetitions, n=2048, full output allocation included, one native BLAS worker',
    family='0.01/32 per bound: 8 small click TV bounds, 6 small full PNR bounds, 2 pure parity lower bounds, 6 actual model CDF bands and 1 shared hardware band, spare unspent',
    precision='pair joint/covariance errors deterministic vs fixed optical input. PNR/click distributions estimated from complete records, iid nominal bounds.'))
cmd=[shutil.which('g++'),'-shared','-O3','-std=c++20','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math',
     '-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-o',str(out/'pair.dll')]
b=subprocess.run(cmd,capture_output=True,text=True)
save(out,'build.json',dict(command=cmd,code=b.returncode,stderr=b.stderr));b.check_returncode()
runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
blasdir=os.add_dll_directory(str(blas_path.parent));blas=C.CDLL(str(blas_path));lib=C.CDLL(str(out/'pair.dll'))
blas.scipy_openblas_set_num_threads(1);assert blas.scipy_openblas_get_num_threads()==1
I,P,U=C.c_int,C.c_void_p,C.c_uint64
D,B=[np.ctypeslib.ndpointer(dtype=t,flags='C_CONTIGUOUS') for t in (np.float64,np.uint8)]
def bind(name,args,result=I):
    fn=getattr(lib,name);fn.argtypes=args;fn.restype=result;return fn
optical=bind('window95_moments',[I,I,D,D,D,D,D,D])
targets=bind('pair95_optical',[I,I,I,D,D,D,D,D,D,D])
invert=bind('pair95_invert',[I,D,D,D,D])
forward=bind('pair95_forward',[I,D,D,D,D])
eigen=bind('pair95_eigen',[I,D,D,D]);prepare=bind('window95_prepare',[I,I,I,I,D,D,D],P)
tilt=bind('window95_tilt',[P,I,C.c_double],P);release=bind('window95_destroy',[P],None)
window_pair=bind('pair95_window',[P,D,D])
sample=bind('pair95_sample',[I,I,I,D,D,D,I,U,U,I,B])
assert bind('pair95_gemm',[P])(C.cast(blas.scipy_cblas_dgemm,P))==0
cap=64;delta=.01/32;N=65536
hw=np.load(hardware);hist=hw['total_click_histogram']
observed_cdf=hist.cumsum()/hist.sum();pnr_cdf=np.load(totals)['photon_cdf']
photon_tail=read(totals.with_name('reference.json'))['tail_upper']

def errors(single,joint,model_single,model_joint):
    a,b=np.triu_indices(len(single),1)
    target=joint[a,b]-single[a]*single[b]
    approx=model_joint[a,b]-model_single[a]*model_single[b]
    diff=model_joint[a,b]-joint[a,b]
    di=model_single[a]-single[a];dj=model_single[b]-single[b]
    tv=(np.abs(diff)+np.abs(di-diff)+np.abs(dj-diff)+np.abs(diff-di-dj))/2
    worst=int(tv.argmax())
    return dict(max_pair_tv=float(tv[worst]),mean_pair_tv=float(tv.mean()),
        pair_rms_cov=float(np.sqrt(np.mean((approx-target)**2))),relative_cov_l2=float(np.linalg.norm(approx-target)/np.linalg.norm(target)),
        worst=[int(a[worst]),int(b[worst])],marginal_max=float(max(abs(model_single-single))))

rows=[];timings=[];setups=[]
for case,paths in inputs.items():
    start=time.perf_counter();m=len(np.load(paths[0])['u']);s=len(paths)
    q=np.empty((s,2*m,2*m));mean=np.empty((s,2*m))
    for sec,file in enumerate(paths):
        z=np.load(file);u=np.ascontiguousarray(z['u'])
        assert optical(m,len(z['t']),u.view(float),z['t'],z['alpha'].view(float),z['noise'],q[sec],mean[sec])==0
    if case==4:
        perm=np.argsort(np.load(layout)['order']);ix=np.r_[perm,perm+m]
        q=np.ascontiguousarray(q[:,ix][:,:,ix]);mean=np.ascontiguousarray(mean[:,ix])
    single=np.empty(m);joint=np.empty((m,m));photon_cov=np.empty_like(joint)
    local=np.empty((m,cap+1));cuts=np.empty_like(local)
    assert targets(m,s,cap,q,mean,single,joint,local,cuts,photon_cov)==0
    assert max(abs(local[:,0]-(1-single)))<1e-12 and np.all(cuts[:,1:]>=cuts[:,:-1])
    threshold=np.ascontiguousarray(cuts[:,0]);corr=np.empty_like(joint)
    assert invert(m,threshold,single,joint,corr)==0
    values=np.empty(m);vectors=np.empty_like(joint)
    assert eigen(m,corr,values,vectors)==0
    calibration_seconds=time.perf_counter()-start
    check=np.empty_like(joint);assert forward(m,threshold,single,corr,check)==0
    setup=dict(case=case,m=m,calibration_seconds=calibration_seconds,min_eigen=values[0],max_eigen=values[-1],
        negative_eigenvalues=int(sum(values< -1e-10)),inversion_max_error=float(np.max(abs(check-joint))),
        max_port_overflow=max(0.,float((1-local.sum(1)).max())))
    assert setup['inversion_max_error']<2e-9
    if case<4:
        refdata=np.load(ref/f'law_{case}.npz');law=refdata['click']
        bits=((np.arange(2**m)[:,None]>>np.arange(m))&1)
        assert np.max(abs((bits.T*law)@bits-joint))<1e-10
    else:
        plan=prepare(m,4,4,s,q,mean,np.zeros(4));assert plan
        calib=np.load(prior);win_s=np.zeros(m);win_j=np.zeros_like(joint)
        first=np.empty(m);both=np.empty_like(joint)
        assert window_pair(plan,first,both)==0
        setup['window_base']=errors(single,joint,first,both)
        for slope,weight in zip(calib['slopes'],calib['weights']):
            h=tilt(plan,4,slope);assert h and window_pair(h,first,both)==0;release(h)
            win_s+=weight*first;win_j+=weight*both
        release(plan);setup['window_third']=errors(single,joint,win_s,win_j)
        assert abs(win_s.sum()-single.sum())<1e-8
        assert abs((win_j-np.outer(win_s,win_s)).sum()-(joint-np.outer(single,single)).sum())<1e-7
        np.savez_compressed(out/'window_pairs.npz',single=win_s,joint=win_j)
    np.savez_compressed(out/f'optical_{case}.npz',single=single,joint=joint,photon_cov=photon_cov,local=local,
        thresholds=cuts,correlation=corr,eigenvalues=values)
    setups.append(setup);save(out,'setups.json',setups)
    for method,rank in enumerate(([m,16,64] if case==4 else [m])):
        start=time.perf_counter()
        base=max(0.,values[-rank-1]) if rank<m else 0.
        factor=np.ascontiguousarray(vectors[:,-rank:]*np.sqrt(np.maximum(0.,values[-rank:]-base)))
        norms=np.einsum('ij,ij->i',factor,factor)
        factor/=np.sqrt(np.maximum(1.,norms))[:,None]
        residual=np.sqrt(np.maximum(0.,1-norms))
        effective=factor@factor.T;effective.flat[::m+1]+=residual**2
        assert np.max(abs(effective.diagonal()-1))<1e-11
        factor_seconds=time.perf_counter()-start
        pair=np.empty_like(joint);assert forward(m,threshold,single,effective,pair)==0
        row=dict(case=case,method=method,rank=rank,factor_seconds=factor_seconds,**errors(single,joint,single,pair))
        exp=0x95d00000+16*case+method
        def draw(n,experiment,off,pnr):
            data=np.empty((n,m),np.uint8)
            assert sample(m,rank,cap,factor,residual,cuts,n,experiment,off,pnr,data)==0
            return data
        allcounts=[];summary_h=np.zeros(m+1,int);sum_photons=[];overflows=0
        for off in range(0,N,2048):
            cnt=draw(2048,exp,off,1)
            if off in (0,N-2048):
                assert np.array_equal(draw(2048,exp,off,0),cnt>0)
                assert np.array_equal(draw(5,exp,off+11,1),cnt[11:16])
            overflows+=int((cnt>cap).any(1).sum())
            total=cnt.sum(1).astype(int);total[(cnt>cap).any(1)]=2**24
            summary_h+=np.bincount((cnt>0).sum(1).astype(int),minlength=m+1)
            sum_photons.extend(total)
            if case==4:
                np.savez_compressed(out/f'full_{method}_{off}.npz',counts=cnt)
            else:allcounts.append(cnt)
        sum_photons=np.asarray(sum_photons)
        row['overflows']=overflows
        if case<4:
            c=np.concatenate(allcounts);labels=(c>0)@(1<<np.arange(m))
            row['click_tv'],_=confidence(labels,law,N,8192,delta)
            if case in (0,2):row['odd_photon_mass']=float(np.mean(sum_photons%2==1))
            if case in (0,1,3):
                if case==3:
                    target=np.r_[refdata['pnr'].ravel(),float(read(ref/'row_3.json')['pnr_tail']),0.]
                    labels=np.where((c<=64).all(1),c[:,0].astype(int)*65+c[:,1],len(target)-1)
                else:
                    target=np.r_[refdata['probabilities'],float(read(ref/f'row_{case}.json')['total_tail']),0.]
                    keys=refdata['keys'];codes=sum(c[:,j].astype(np.uint64)<<(8*j) for j in range(m))
                    pos=np.searchsorted(keys,codes);inside=pos<len(keys);inside[inside]&=keys[pos[inside]]==codes[inside]
                    labels=np.where(inside,pos,len(keys));labels[(c>cap).any(1)]=len(target)-1
                row['pnr_tv'],_=confidence(labels,target,N,8192,delta,float(read(ref/f'row_{case}.json')['tail_chernoff']))
            np.savez_compressed(out/f'small_{case}.npz',counts=c)
        else:
            curve=np.searchsorted(np.sort(sum_photons),np.arange(len(pnr_cdf)),side='right')/N
            ks=float(max(abs(curve-pnr_cdf)));error=math.sqrt(math.log(2/delta)/(2*N))
            row['photon_cdf']=[ks,max(0.,ks-error-float(photon_tail)),min(1.,ks+error+float(photon_tail))]
            ks=float(max(abs(summary_h.cumsum()/N-observed_cdf)))
            hw_error=error+math.sqrt(math.log(2/delta)/(2*hist.sum()))
            row['hardware_click_count']=[ks,max(0.,ks-hw_error),min(1.,ks+hw_error)]
            row['photon_mean']=float(sum_photons.mean()) if not overflows else None
            row['photon_variance']=float(sum_photons.var()) if not overflows else None
        rows.append(row)
        for rep in (-1,0,1,2):
            for obs in ([0,1] if rep%2==0 else [1,0]):
                n=128 if rep<0 else 2048;now=time.perf_counter()
                c=draw(n,0x95d10000+512*case+32*(rep+1)+2*method+obs,0,obs)
                elapsed=time.perf_counter()-now
                if rep>=0:timings.append(dict(case=case,method=method,pnr=obs,n=n,rep=rep,seconds=elapsed,
                    hash=hashlib.sha256(c.tobytes()).hexdigest()))
        save(out,'summary.json',dict(rows=rows,timings=timings,setups=setups))
        print('completed',case,rank,row['max_pair_tv'],flush=True)
manifest(out)
