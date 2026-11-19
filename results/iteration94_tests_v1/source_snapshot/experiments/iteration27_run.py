"""Mixed low-click calibration and explicit external-covariance provenance audit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import mpmath as mp
import numpy as np
import scipy.linalg as la
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from gbslab.mixed_low_clicks import mixed_gaussian_low_clicks
from gbslab.gaussian import covariance_from_interferometer,haar_unitary
from gbslab.banded_gaussian_mps import pairing_covariance
from gbslab.exact import ThresholdModel
from test_mixed_low_clicks import direct_complement_reference
from iteration23_support import digest,save


def complement_reference(V,patterns):
    m=len(V)//2; Q=(V+np.eye(2*m))/2; cache={():1.}
    def vacuum(S):
        S=tuple(sorted(S))
        if S not in cache:
            ix=list(S)+[i+m for i in S]; sign,logdet=np.linalg.slogdet(Q[np.ix_(ix,ix)]); assert sign==1
            cache[S]=float(np.exp(-.5*logdet))
        return cache[S]
    result=[]
    for row in patterns:
        clicks=[int(i) for i in row if i>=0]; outside=[i for i in range(m) if i not in clicks]
        value=vacuum(outside)
        if len(clicks)>=1: value-=vacuum(outside+[clicks[0]])
        if len(clicks)==2: value+=vacuum(range(m))-vacuum(outside+[clicks[1]])
        result.append(value)
    return np.array(result)


def covariance_audit(mixed,pure,candidate,out):
    m=len(mixed)//2; O=np.block([[np.zeros((m,m)),np.eye(m)],[-np.eye(m),np.zeros((m,m))]])
    info={}
    for name,V in [('source_mixed',mixed),('source_pure_times_two',pure),('kernel_chain_candidate',candidate)]:
        info[name]=dict(shape=list(V.shape),symmetry_residual=float(np.max(abs(V-V.T))),minimum_eigenvalue=float(np.linalg.eigvalsh(V)[0]),minimum_uncertainty_eigenvalue=float(np.linalg.eigvalsh(V+1j*O)[0]),purity_symplectic_residual=float(np.max(abs(V@O@V-O))),mean_photons=float((np.trace(V)-2*m)/4))
    info['candidate_vs_author_pure']=dict(max_absolute=float(np.max(abs(candidate-pure))),frobenius=float(np.linalg.norm(candidate-pure)))
    witnesses={}; checks=[]
    for name,P in [('author_pure',pure),('candidate_pure',candidate)]:
        D=mixed-P; vals,vecs=np.linalg.eigh(D); w=vecs[:,0]; value=float(w@D@w)
        independent=float(la.eigh(D,subset_by_index=(0,0),driver='evr')[0][0]); residual=float(np.linalg.norm(D@w-value*w))
        with mp.workdps(80):
            wm=[mp.mpf(float(x)) for x in w]
            exact_dot=mp.fsum(wm[i]*(mp.mpf(float(mixed[i,j]))-mp.mpf(float(P[i,j])))*wm[j] for i in range(2*m) for j in range(2*m))
            high=float(exact_dot); high_text=mp.nstr(exact_dot,75)
        witnesses[name]=w
        info['mixed_minus_'+name]=dict(minimum_eigenvalue=float(vals[0]),maximum_eigenvalue=float(vals[-1]),scipy_minimum_eigenvalue=independent,rayleigh_float64=value,rayleigh_80_digit=high_text,witness_norm=float(np.linalg.norm(w)),eigenvector_residual=residual,positive_semidefinite_at_supplied_precision=high>=0,displacement_covariance_convention='hbar=2; source stored pure covariance multiplied by two')
        checks.extend([abs(vals[0]-independent)<2e-11,abs(value-high)<2e-11,residual<2e-11,abs(np.linalg.norm(w)-1)<1e-12])
    np.savez_compressed(out/'negative_witnesses.npz',**witnesses)
    info['checks']=len(checks); info['checks_passed']=sum(checks); save(out/'covariance_audit.json',info); assert all(checks)
    return info


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--external',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False); external=args.external.resolve()
    sources=sorted({'experiments/iteration27_run.py','experiments/iteration23_support.py','tests/test_mixed_low_clicks.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))})
    hashes={f:digest(ROOT/f) for f in sources}
    for f in sources:
        target=out/'source_snapshot'/f; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,target)
    shutil.copy2(ROOT/'research/iteration27_panel_plan.md',out/'frozen_plan.md')
    selected=json.loads((ROOT/'results/iteration27_source/selected.json').read_text())
    for row in selected: assert digest(external/row['local_name'])==row['sha256']
    target='results/iteration24_chain/case_56.npz'; target_sha=digest(ROOT/target)
    assert target_sha==json.loads((ROOT/'results/iteration24_manifest.json').read_text())['artifacts'][target]
    with np.load(ROOT/target,allow_pickle=False) as data: candidate=pairing_covariance(data['forward'])
    mixed=np.load(external/'jiuzhang2_mixed_cov.npy',allow_pickle=False); pure=2*np.load(external/'best_opt_cov_jiuzhang2_mixed_cov.npy',allow_pickle=False)
    jobs=[dict(panel='small',modes=m,seed=s,eta=e,noise=n) for m in (2,4,6) for s in (271101,271102) for e in (.15,.55,.9) for n in (0.,.02)]
    jobs += [dict(panel='candidate',modes=144,eta=e,noise=n) for e in (.2,.6,1.) for n in (0.,.02)]
    jobs += [dict(panel=name,modes=144) for name in ('source_mixed','source_pure_times_two')]
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,external_members=selected,dependencies={target:target_sha},convention='Zero displacement, hbar=2, xxpp; noise is added quadrature covariance, not mean photon number'))
    audit=covariance_audit(mixed,pure,candidate,out)
    rows=[]
    for i,job in enumerate(jobs):
        start=perf_counter(); row=dict(sequence=i,**job); print('starting',i,job,flush=True)
        try:
            if job['panel']=='small':
                m=job['modes']; U=haar_unitary(m,np.random.default_rng(job['seed'])); V=covariance_from_interferometer(U,np.linspace(.05,.8,m),job['eta'])+job['noise']*np.eye(2*m)
            elif job['panel']=='candidate': V=job['eta']*candidate+(1-job['eta']+job['noise'])*np.eye(288)
            else: V=mixed if job['panel']=='source_mixed' else pure
            result=mixed_gaussian_low_clicks(V); probs=result['probabilities']; patterns=result['patterns']
            if job['panel']=='small':
                high=mixed_gaussian_low_clicks(V,backend='mpmath'); independent=direct_complement_reference(V,patterns)
                reference=ThresholdModel(V); full=reference.full_distribution(); indices=[sum(1<<int(k) for k in pair if k>=0) for pair in patterns]
                row.update(float_vs_high_max_absolute=float(np.max(abs(probs-high['probabilities']))),high_vs_independent_max_relative=float(np.max(abs(high['probabilities']-independent)/np.maximum(abs(independent),1e-70))),float_vs_full_max_absolute=float(np.max(abs(probs-full[indices]))),full_reference_diagnostics=reference.diagnostics)
                assert row['float_vs_high_max_absolute']<2e-12 and row['float_vs_full_max_absolute']<2e-12 and row['high_vs_independent_max_relative']<2e-12
                np.savez_compressed(out/f'case_{i:02d}.reference.npz',covariance=V,full_threshold=full,high_precision=high['probabilities'],independent_high_precision=independent)
            else:
                rng=np.random.default_rng(271144); ids=np.r_[0,np.sort(rng.choice(np.arange(1,145),12,replace=False)),np.sort(rng.choice(np.arange(145,len(probs)),36,replace=False))]
                independent=complement_reference(V,patterns[ids]); error=float(np.max(abs(probs[ids]-independent))/probs[0]); row['independent_complement_checks']=len(ids); row['complement_max_error_divided_by_vacuum']=error
                assert error<2e-8
                np.savez_compressed(out/f'case_{i:02d}.reference.npz',indices=ids,independent_complement=independent)
            np.savez_compressed(out/f'case_{i:02d}.npz',patterns=patterns,probabilities=probs)
            row.update({k:v for k,v in result.items() if k not in ('patterns','probabilities')}); m=job['modes']
            row.update(vacuum=float(probs[0]),one_click_mass=float(probs[1:1+m].sum()),two_click_mass=float(probs[1+m:].sum()),status='passed')
        except Exception: row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-start; rows.append(row); save(out/f'case_{i:02d}.json',row); save(out/'rows.json',rows); print('finished',i,row['status'],flush=True)
    unchanged=hashes=={f:digest(ROOT/f) for f in sources}
    save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),passed=sum(r['status']=='passed' for r in rows),unchanged_sources=unchanged,covariance_checks=audit['checks'],covariance_checks_passed=audit['checks_passed'],dependencies={target:target_sha}))
    assert unchanged


if __name__=='__main__': main()
