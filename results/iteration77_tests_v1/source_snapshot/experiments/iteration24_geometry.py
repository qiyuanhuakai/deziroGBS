"""Post-hoc Gaussian-isometry and virtual Schmidt-basis checks."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian_kernel_chain import contract_gaussians,reverse_kernel_chain
from gbslab.gaussian_schmidt_mps import decompose_gaussian,_validate_basis
from gbslab.banded_gaussian_mps import pairing_covariance
from gbslab.gaussian_mps_feasibility import marginal_photon_tails
from iteration22_external import ARCHIVE_SHA,PREFIX,digest,load_entry,save


def inspect_geometry(kernels,bonds):
    grams=[]
    for i,K in enumerate(kernels):
        output=bonds[i]+1; right=bonds[i+1]; order=list(range(output,len(K)))+list(range(output))
        C,c,_=contract_gaussians(K,K.conj()[np.ix_(order,order)],output)
        target=np.block([[np.zeros((right,right)),np.eye(right)],[np.eye(right),np.zeros((right,right))]])
        grams.append(dict(site=i,gram_pairing_error=float(np.max(np.abs(C-target))) if right else 0.,gram_log_norm=c.real,gram_log_phase=c.imag))
    ks,bs,_=reverse_kernel_chain(kernels,bonds); prefix=np.zeros((0,0),complex); cuts=[]
    for i,K in enumerate(ks):
        l,r=bs[i:i+2]; po=list(range(i,i+l))+list(range(i)); ko=list(range(l,len(K)))+list(range(l))
        X,_,_=contract_gaussians(prefix[np.ix_(po,po)],K[np.ix_(ko,ko)],l)
        order=list(range(1+r,1+r+i))+list(range(1+r)); prefix=X[np.ix_(order,order)]
        if r:
            V=pairing_covariance(prefix); size=len(prefix); ids=list(range(i+1,size))+list(range(size+i+1,2*size)); virtual=V[np.ix_(ids,ids)]
            nu=.5*(np.diag(virtual)[:r]+np.diag(virtual)[r:]); target=np.diag(np.r_[nu,nu])
            q=np.maximum(0.,(nu-1)/(nu+1))
            cuts.append(dict(cut=len(kernels)-i-1,virtual_modes=r,thermal_covariance_residual=float(np.max(np.abs(virtual-target))),minimum_raw_nu=float(nu.min()),nu=nu.tolist(),parameters=q.tolist()))
    return grams,sorted(cuts,key=lambda c:c['cut'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=False)
    sources=sorted({'experiments/iteration24_geometry.py','experiments/iteration22_external.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))}); hashes={f:digest(ROOT/f) for f in sources}
    for f in sources:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'plan.json',dict(source_hashes=hashes,archive_sha256=ARCHIVE_SHA,stage='Post-hoc after observed low-sector distance lower bound',known_calibrations=list(range(12)),all_external_sites=144,all_external_cuts=143,tolerance=1e-8,rank=200,dimension=4))
    assert digest(args.archive)==ARCHIVE_SHA; checks=[]
    def check(name,value): checks.append(dict(name=name,passed=bool(value)))
    calibrations=[]
    for i in range(12):
        with np.load(ROOT/f'results/iteration24_chain/case_{i:02d}.npz',allow_pickle=False) as a: B=a['original']
        ks,rs,scalar,_=decompose_gaussian(B); grams,cuts=inspect_geometry(ks,list(range(4))+[0])
        ge=max(r['gram_pairing_error'] for r in grams); te=max(r['thermal_covariance_residual'] for r in cuts)
        qe=max(float(np.max(np.abs(np.asarray(c['parameters'])-r['parameters']))) for c,r in zip(cuts,rs))
        check(f'calibration {i}',max(ge,te,qe)<1e-8); calibrations.append(dict(case=i,gram_error=ge,thermal_error=te,parameter_error=qe))
    save(out/'calibrations.json',calibrations)
    entries=[]; kernels=[]; bases=[]; bonds=[0]
    with zipfile.ZipFile(args.archive) as z:
        for i in range(144):
            K=load_entry(z,PREFIX+f'Q/{i}.npy',entries).astype(complex); K[np.diag_indices_from(K)]*=2; kernels.append(K)
            if i<143:
                basis=load_entry(z,PREFIX+f'List_vst/{i}.npy',entries)[:200]; _validate_basis(basis); bases.append(basis); bonds.append(basis.shape[1])
    bonds.append(0); save(out/'external_inputs.json',entries)
    grams,cuts=inspect_geometry(kernels,bonds); save(out/'grams.json',grams)
    with np.load(ROOT/'results/iteration24_chain/case_56.npz',allow_pickle=False) as a: B=a['forward']; V=a['covariance']
    chain=json.loads((ROOT/'results/iteration24_chain/case_56.json').read_text())
    check('all local normalized isometries',max(r['gram_pairing_error'] for r in grams)<1e-8 and max(abs(r['gram_log_phase']) for r in grams)<1e-8)
    local_norm=math.fsum(r['gram_log_norm'] for r in grams); global_norm=chain['directions']['forward']['raw_log_norm_squared']
    check('product local scales matches global norm',abs(local_norm-global_norm)<1e-8)
    for cut,basis in zip(cuts,bases):
        q=np.asarray(cut['parameters']); nu=np.asarray(cut['nu']); k=cut['cut']
        check(f'cut {k} diagonal thermal',cut['thermal_covariance_residual']<1e-8 and cut['minimum_raw_nu']>=1-1e-8)
        weights=np.prod(1-q)*np.prod(q[None,:]**basis,axis=1); tail=max(0.,1-math.fsum(weights))
        cut.update(kept_weights=weights.tolist(),tail_estimate=tail)
        ids=list(range(k))+list(range(144,144+k)); block=V[np.ix_(ids,ids)]; omega=np.block([[np.zeros((k,k)),np.eye(k)],[-np.eye(k),np.zeros((k,k))]])
        values=np.linalg.eigvals(1j*omega@block); positives=np.sort(values.real)[k:]
        target=np.sort(np.r_[nu,np.ones(k-len(nu))])
        error=float(np.max(np.abs(positives-target))); cut['physical_prefix_spectrum_error']=error
        check(f'cut {k} independent physical prefix spectrum',error<1e-8 and np.max(np.abs(values.imag))<1e-8)
    save(out/'cuts.json',cuts)
    inverse=np.linalg.solve(np.eye(144)-B@B.conj().T,np.eye(144)); N=inverse-np.eye(144); M=inverse@B
    tails=[float(marginal_photon_tails(float(N[i,i].real),abs(M[i,i]),[4])['tail_upper_estimates'][0]) for i in range(144)]
    check('sources unchanged',hashes=={f:digest(ROOT/f) for f in sources})
    eligible=all(c['passed'] for c in checks); loss=math.fsum(c['tail_estimate'] for c in cuts)+math.fsum(tails)
    summary=dict(checks=checks,count=len(checks),passed=sum(c['passed'] for c in checks),budget_conditions_passed=eligible,cut_tail_sum=math.fsum(c['tail_estimate'] for c in cuts),physical_tail_sum=math.fsum(tails),physical_tails=tails,conditional_pure_trace_upper_estimate=math.sqrt(min(1.,loss)) if eligible else None,local_scale_log_norm=local_norm,global_scale_log_norm=global_norm,scale_log_error=abs(local_norm-global_norm),scope='Internal Gaussian state defined by archived kernels only. Projector bound applies only if isometry and original thermal virtual-basis conditions pass; float64 estimates, not experimental or finite-precision TV certification.')
    save(out/'summary.json',summary); print(json.dumps({k:v for k,v in summary.items() if k not in ('checks','physical_tails')},indent=2))


if __name__=='__main__': main()
