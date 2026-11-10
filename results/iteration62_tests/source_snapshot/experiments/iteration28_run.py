"""Frozen guarded resource panel and independent realified SDP controls."""
import argparse
import json
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter
import traceback
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian_resource import guarded_resource,passive_active_covariance,pure_pairing,omega,resource_diagnostics
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from gbslab.banded_gaussian_mps import pairing_covariance,pairing_distance
from gbslab.low_clicks import gaussian_low_clicks,low_click_tv
from iteration23_support import save,digest


def independent_sdp(V,cp):
    n=len(V); O=omega(n//2); X=cp.Variable((n,n),symmetric=True)
    physical=cp.bmat([[X,-O],[O,X]]); constraints=[physical >> 0,V-X >> 0]
    problem=cp.Problem(cp.Minimize(cp.trace(X)),constraints)
    value=problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=250)
    if X.value is None: raise ArithmeticError('SDP returned no covariance')
    P=np.asarray(X.value); S=np.asarray(constraints[1].dual_value)
    diagnostics=dict(status=problem.status,objective=float(value),minimum_physical_eigenvalue=float(np.linalg.eigvalsh(P+1j*O)[0]),minimum_noise_eigenvalue=float(np.linalg.eigvalsh(V-P)[0]),minimum_dual_eigenvalue=float(np.linalg.eigvalsh(S)[0]),iterations=problem.solver_stats.num_iters,solve_time=problem.solver_stats.solve_time)
    return P,S,diagnostics


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--external',type=Path,required=True); p.add_argument('--sdp-packages',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    # Append so the already imported main NumPy/SciPy runtime remains authoritative.
    sys.path.append(str(args.sdp_packages.resolve())); import cvxpy as cp; import clarabel
    sources=['experiments/iteration28_run.py','experiments/iteration23_support.py',*[f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py')]]
    hashes={f:digest(ROOT/f) for f in sources}
    for name in sources:
        dst=out/'source_snapshot'/name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dst)
    shutil.copy2(ROOT/'research/iteration28_panel_plan.md',out/'frozen_plan.md')
    selected='results/iteration27_source/selected.json'; target='results/iteration24_chain/case_56.npz'; old_mps='results/iteration26_panel/case_29.npz'
    dependencies={f:digest(ROOT/f) for f in (selected,target,old_mps)}
    assert dependencies[target]==json.loads((ROOT/'results/iteration24_manifest.json').read_text())['artifacts'][target]
    assert dependencies[old_mps]==json.loads((ROOT/'results/iteration26_manifest.json').read_text())['artifacts'][old_mps]
    external=args.external.resolve()/'jiuzhang2_mixed_cov.npy'; expected=next(r['sha256'] for r in json.loads((ROOT/selected).read_text()) if r['local_name']==external.name); assert digest(external)==expected
    M=np.load(external,allow_pickle=False); compressed=passive_active_covariance(M); O=compressed['basis']; W=compressed['active_covariance']; m=len(M)//2; k=compressed['active_modes']
    np.savez_compressed(out/'active_support.npz',basis=O,active_covariance=W,normal_spectrum=compressed['normal_spectrum'])
    save(out/'compression.json',{key:value for key,value in compressed.items() if not isinstance(value,np.ndarray)})
    jobs=[dict(panel='small',modes=m,seed=s,loss_kind=loss,margin=1e-6) for m in (2,4,6) for s in (281301,281302) for loss in ('uniform','heterogeneous')]
    jobs += [dict(panel='source',modes=144,active_modes=k,margin=margin,primary=margin==1e-6) for margin in (1e-4,1e-5,1e-6,1e-7)]
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=dependencies,external_sha256=expected,environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,cvxpy=cp.__version__,clarabel=clarabel.__version__,numpy_path=np.__file__,scipy_path=scipy.__file__,cvxpy_path=cp.__file__)))
    B_old=np.load(ROOT/target,allow_pickle=False)['forward']; old_cov=pairing_covariance(B_old)
    old=np.load(ROOT/old_mps,allow_pickle=False); mps=dict(patterns=old['patterns'],probabilities=old['mps_probabilities'],selected_mass=float(sum(old['mps_probabilities'])))
    rows=[]
    for i,job in enumerate(jobs):
        row=dict(sequence=i,**job); start=perf_counter(); print('starting',i,job,flush=True)
        try:
            if job['panel']=='small':
                n=job['modes']; r=np.linspace(.1,.8,n); eta=.55 if job['loss_kind']=='uniform' else np.linspace(.2,.8,n)
                V=covariance_from_interferometer(haar_unitary(n,np.random.default_rng(job['seed'])),r,eta)
            else: V=W
            result=guarded_resource(V,margin=job['margin']); row.update(diagnostics=result['diagnostics'],tightened_diagnostics=result['tightened_diagnostics'],optimizer_success=result['optimizer_success'],optimizer_message=result['optimizer_message'],iterations=result['iterations'],evaluations=result['evaluations'],accepted_before_encoding=result['numerically_accepted'])
            arrays=dict(pure_raw=result['pure_covariance'],dual_factor=result['dual_factor'],dual=result['dual_multiplier'],remainder_raw=result['remainder'])
            save(out/f'case_{i:02d}.history.json',result['history'])
            # Always retain the candidate, including rejected ones.
            if result['numerically_accepted']:
                B,P,roundtrip=pure_pairing(result['pure_covariance']); R=(V-P+V.T-P.T)/2
                eig=float(np.linalg.eigvalsh(R)[0]); row.update(pairing_roundtrip=roundtrip,minimum_encoded_remainder=eig)
                row['accepted']=eig>=job['margin']/4
                if row['accepted']:
                    noise=np.linalg.cholesky(R); arrays.update(pairing=B,pure=P,noise_factor=noise)
            else: row['accepted']=False
            if job['panel']=='small':
                reference,dual,info=independent_sdp(V,cp); arrays.update(input=V,sdp_pure=reference,sdp_dual=dual); row['independent_sdp']=info
                row['sdp_trace_minus_dual']=info['objective']-row['diagnostics']['dual_trace']; row['primal_trace_minus_sdp']=row['diagnostics']['primal_trace']-info['objective']
                if job['loss_kind']=='uniform':
                    squeezed=eta*np.exp(-2*r)+1-eta; exact=float(np.sum(squeezed+1/squeezed)); row['analytic_optimal_trace']=exact; row['sdp_analytic_error']=info['objective']-exact
                assert info['minimum_physical_eigenvalue']>-2e-8 and info['minimum_noise_eigenvalue']>-2e-8
                assert row['sdp_trace_minus_dual']>-2e-7 and -2e-7<row['primal_trace_minus_sdp']<2e-4
                if 'sdp_analytic_error' in row: assert abs(row['sdp_analytic_error'])<2e-7
            elif row['accepted']:
                Ua=O[:m,:k]+1j*O[m:,:k]; fullB=Ua@B@Ua.T; fullB=(fullB+fullB.T)/2; fullP=pairing_covariance(fullB); full_noise=O@noise
                constructed=fullP+full_noise@full_noise.T; full_dual_factor=O@result['dual_factor']; full_dual=full_dual_factor@full_dual_factor.T
                row.update(full_pure_mean_photons=float((np.trace(fullP)-2*m)/4),constructed_input_error_operator=float(np.linalg.norm(constructed-M,ord=2)),constructed_input_error_max=float(np.max(abs(constructed-M))),original_input_minimum_remainder=float(np.linalg.eigvalsh(M-fullP)[0]),candidate_pure_distance=pairing_distance(fullB,B_old),candidate_covariance_max_difference=float(np.max(abs(fullP-old_cov))))
                row['full_original_diagnostics']=resource_diagnostics(M,fullP,full_dual)
                row['full_constructed_diagnostics']=resource_diagnostics(constructed,fullP,full_dual)
                low=gaussian_low_clicks(fullB); row['pure_core_threshold_comparison']=low_click_tv(low,mps)
                arrays.update(full_pairing=fullB,full_noise_factor=full_noise,full_dual_factor=full_dual_factor,low_click_patterns=low['patterns'],low_click_probabilities=low['probabilities'])
            np.savez_compressed(out/f'case_{i:02d}.npz',**arrays); row['status']='completed'
        except Exception: row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-start; rows.append(row); save(out/f'case_{i:02d}.json',row); save(out/'rows.json',rows); print('finished',i,row['status'],'accepted',row.get('accepted'),flush=True)
    unchanged=hashes=={f:digest(ROOT/f) for f in sources}; assert unchanged
    save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),accepted=sum(r.get('accepted',False) for r in rows),primary_accepted=next(r.get('accepted',False) for r in rows if r.get('primary')),unchanged_sources=unchanged,dependencies=dependencies))


if __name__=='__main__': main()
