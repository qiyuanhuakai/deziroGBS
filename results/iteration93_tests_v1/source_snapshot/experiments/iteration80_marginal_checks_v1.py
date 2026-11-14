"""Complete conditional click marginals and unconditional vacuum diagnostics.
Pure fixed-N no-hit probabilities use coefficients of det(I-z G^T G)^(-1/2).
Latent-pair no-hit probabilities are averages of s_l^(N/2).
All distances here are numerical marginal distances, hence full-law lower bounds.
"""
from pathlib import Path
import os,json,hashlib,argparse,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
save('plan.json',dict(scope=__doc__,source_sha256=H(Path(__file__)),fixed_total=34,marginal_ports=8,unconditional_r=1.1,budget=8,negative_probability_policy='Raise; no clipping'))
def coefficients(eigen,exponent,degree):
    # Positive power-sum recurrence, independent of photon sampling core.
    traces=np.array([exponent*np.sum(eigen**k) for k in range(1,degree+1)],np.longdouble);h=np.zeros(degree+1,np.longdouble);h[0]=1
    for j in range(1,degree+1):h[j]=np.dot(traces[:j],h[j-1::-1])/j
    return h
def normalizer(shape,degree):
    x=np.ones(degree+1,np.longdouble)
    for j in range(1,degree+1):x[j]=x[j-1]*(np.longdouble(shape)+j-1)/j
    return x
def spectra(v,ids):
    # SVD factors preserve nonnegativity at rank-deficient no-hit events.
    # No eigenvalue clipping: R R^T has the surviving anomalous singular values.
    remaining=v[[i for i in range(len(v)) if i not in ids]];_,singular,Vh=np.linalg.svd(remaining,full_matrices=False);factor=singular[:,None]*Vh
    ep=np.linalg.svd(factor@factor.T,compute_uv=False)**2;eg=singular**2;er=np.linalg.svd(np.r_[remaining.real,remaining.imag],compute_uv=False)**2
    return ep,eg,er
def mobius(vacuum):
    size=len(vacuum);law=vacuum[np.arange(size)^(size-1)].copy()
    for bit in range(size.bit_length()-1):
        for mask in range(size):
            if mask>>bit&1:law[mask]-=law[mask^(1<<bit)]
    assert law.min()>=0 and abs(law.sum()-1)<1e-10
    return law
# Validate fixed-total vacuum coefficient formula against complete pure Fock laws.
d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');v=d['v'];base=int(d['cutoff'])+1;states=np.array([(d['codes']//base**j)%base for j in range(4)]).T;totals=states.sum(axis=1);validation=[]
for N in (2,4,8,14):
    active=totals==N;ground=d['target'][:-1][active];ground=ground/ground.sum();x=states[active]
    for mask in range(16):
        ids=[j for j in range(4) if mask>>j&1];actual=float(ground[np.all(x[:,ids]==0,axis=1)].sum()) if ids else 1.
        if not ids:predicted=1.
        elif len(ids)==4:predicted=0.
        else:ep,_,_=spectra(v,ids);predicted=float(coefficients(ep,.5,N//2)[-1]/normalizer(v.shape[1]/2,N//2)[-1])
        error=abs(actual-predicted);assert error<1e-12;validation.append(dict(total=N,mask=mask,reference=actual,formula=predicted,error=error))
save('fock_validation.json',dict(passed=True,rows=validation));rows=[]
for case in (0,1,2,3):
    base=ROOT/'results/iteration80_wide_pilot_v1';d=np.load(base/f'input_{case}.npz');v=d['v'];m,n=v.shape;weights=d['weights'].copy();weights/=weights.sum();table=np.concatenate([np.load(base/f'table_{case}_{first}.npz')['couplings'] for first in range(0,len(weights),64)]);table/=table.sum(axis=(1,2))[:,None,None];marginals=table.sum(axis=2);N=34;K=8;normal=normalizer(n/2,N//2)[-1];thermal_normal=normalizer(n,N)[-1];real_normal=normalizer(n/2,N)[-1];vp=np.ones(1<<K,np.longdouble);vt=vp.copy();vc=vp.copy();vr=vp.copy();subset_pair=np.ones((1<<K,len(weights)),np.longdouble)
    for mask in range(1,1<<K):
        ids=[j for j in range(K) if mask>>j&1];ep,eg,er=spectra(v,ids);vp[mask]=coefficients(ep,.5,N//2)[-1]/normal;vc[mask]=coefficients(eg,1,N)[-1]/thermal_normal;vr[mask]=coefficients(er,.5,N)[-1]/real_normal
        pair_nohit=1-2*marginals[:,ids].sum(axis=1)+table[:,ids][:,:,ids].sum(axis=(1,2));assert pair_nohit.min()>=0 and pair_nohit.max()<=1;subset_pair[mask]=pair_nohit;vt[mask]=np.dot(weights.astype(np.longdouble),pair_nohit.astype(np.longdouble)**(N//2))
    pure,transport,complex_sphere,real_sphere=map(mobius,(vp,vt,vc,vr));counts=np.load(base/f'counts_{case}.npz')['counts'];labels=((counts[:,:K]>0).astype(np.int64)@2**np.arange(K));hist=np.bincount(labels,minlength=1<<K)/len(counts);empirical=float(abs(hist-transport).sum()/2);assert empirical<.04
    row=dict(case=case,inputs=n,outputs=m,fixed_total=N,marginal_ports=K,transport_numerical_tv=float(abs(transport-pure).sum()/2),complex_sphere_numerical_tv=float(abs(complex_sphere-pure).sum()/2),real_sphere_numerical_tv=float(abs(real_sphere-pure).sum()/2),empirical_tv_to_transport=empirical,maximum_normalization_error=float(max(abs(v.sum()-1) for v in (pure,transport,complex_sphere,real_sphere))))
    np.savez_compressed(OUT/f'conditional_{case}.npz',pure=pure,transport=transport,complex_sphere=complex_sphere,real_sphere=real_sphere,vacuum_pure=vp,vacuum_transport=vt,histogram=hist,subset_pair_nohit=subset_pair);save(f'conditional_{case}.json',row);rows.append(row);print(row,flush=True)
    # Exact-budget hybrid vacuum events, at a fixed strong squeezing.
    z=np.longdouble(np.tanh(1.1))**2;a=np.longdouble(n)/2;nb=normalizer(a,256)*(1-z)**a*z**np.arange(257);unconditional=[]
    ratio=z*(a+256)/257;omitted=float(nb[-1]*ratio/(1-ratio));assert ratio<1
    for size in (1,4,8,16,32):
        if size>=m:continue
        ids=list(range(size));ep,eg,er=spectra(v,ids);hp=coefficients(ep,.5,256);pure_vac=float(np.exp(a*np.log1p(-z)-.5*np.log1p(-z*ep.astype(np.longdouble)).sum()));check_sum=float((1-z)**a*np.dot(z**np.arange(257),hp));assert abs(check_sum-pure_vac)<1e-11+omitted
        s=1-2*marginals[:,ids].sum(axis=1)+table[:,ids][:,:,ids].sum(axis=(1,2));assert s.min()>=0 and s.max()<=1
        full=np.exp(a*(np.log1p(-z)-np.log1p(-z*s.astype(np.longdouble))));low=np.sum(nb[:5,None]*s.astype(np.longdouble)[None,:]**np.arange(5)[:,None],axis=0);exact_low=(1-z)**a*np.dot(z**np.arange(5),hp[:5]);hybrid=float(exact_low+np.dot(weights,full-low))
        hc=coefficients(eg,1,512)/normalizer(n,512);hr=coefficients(er,.5,512)/normalizer(a,512);old=float(exact_low+np.dot(nb[5:],hc[10::2]));real=float(exact_low+np.dot(nb[5:],hr[10::2]));unconditional.append(dict(ports=size,target=pure_vac,transport=hybrid,complex_sphere=old,real_sphere=real,transport_numerical_tv_lower=abs(hybrid-pure_vac),complex_sphere_numerical_tv_lower=abs(old-pure_vac),positive_total_tail_upper=omitted))
    save(f'unconditional_{case}.json',dict(r=1.1,budget=8,rows=unconditional));print('unconditional',case,unconditional,flush=True)
save('summary.json',dict(passed=True,rows=rows,fock_checks=len(validation)));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
