"""Audit frozen frontier states, determinant distances, and the full test suite."""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import unittest
import numpy as np
from iteration14_run import save
from iteration10_reference import core_parameters
from verify_iteration11 import RecordingResult
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'tests'))
from gbslab.banded_gaussian_mps import pairing_distance, pairing_covariance, truncate_band, frontier_element_bound
from gbslab.local_mps import local_circuit
from gbslab.gaussian import haar_unitary
from gbslab.mps_probe import mps_to_dense
from test_banded_gaussian_mps import reference_box


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); checks=[]
    check=lambda name,value:checks.append(dict(check=name,passed=bool(value)))
    read=lambda p:json.loads(p.read_text())
    panel=ROOT/'results/iteration20_panel'; scan=ROOT/'results/iteration20_band_scan'
    plan=read(panel/'plan.json'); rows=read(panel/'rows.json'); scanplan=read(scan/'plan.json'); scanrows=read(scan/'rows.json')
    check('complete planned panel',len(plan['jobs'])==len(rows)==60 and read(panel/'completion.json')['finished']==60)
    check('complete band scan',len(scanrows)==250 and all(r['status']=='complete' for r in scanrows))
    for folder,hashes in ((panel,read(panel/'metadata.json')['source_sha256']),(scan,read(scan/'source_hashes.json'))):
        for f,digest in hashes.items():
            check(str(folder.name)+' frozen '+f,hashlib.sha256((folder/'source_snapshot'/f).read_bytes()).hexdigest()==digest)
            check(str(folder.name)+' current '+f,hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest)
    pars=core_parameters(.9,.65)
    check('scan row unique',len({(r['index'],r['width']) for r in scanrows})==250)
    for i,job in enumerate(scanplan['jobs']):
        with np.load(scan/f'input_{i:02d}.npz') as data: u=data['U']; B=data['pairing']
        expected=local_circuit(job['modes'],job['depth'],np.random.default_rng(job['seed']))[1]
        check(f'scan {i} source input',np.array_equal(u,expected) and np.allclose(B,-math.tanh(pars['core_squeezing'])*(u@u.T),atol=1e-15,rtol=0))
        for r in (x for x in scanrows if x['index']==i):
            w=r['width']; C=truncate_band(B,w); label=f'scan {i}/{w}'
            check(label+' identity',all(r[k]==v for k,v in job.items()))
            check(label+' distance',abs(pairing_distance(B,C)-r['band_distance'])<1e-13)
            check(label+' normalizable norm',abs(np.linalg.norm(C,2)-r['pairing_norm'])<1e-13 and r['pairing_norm']<1)
            check(label+' allocation',all(int(r['allocation_upper_elements'][str(d)])==frontier_element_bound(job['modes'],d,w) for d in scanplan['dimensions']))
            if 'independent_covariance_fidelity_error' in r:
                V=pairing_covariance(B); W=pairing_covariance(C); _,ld=np.linalg.slogdet((V+W)/2)
                error=abs(1-r['band_distance']**2-math.exp(-.5*ld))
                check(label+' covariance fidelity',error<1e-12 and abs(error-r['independent_covariance_fidelity_error'])<1e-14)
                m=job['modes']; omega=np.block([[np.zeros((m,m)),np.eye(m)],[-np.eye(m),np.zeros((m,m))]])
                check(label+' covariance purity',np.max(np.abs(W@omega@W-omega))<1e-12)
    completed=quality=failed=samples=0
    for i,(job,r) in enumerate(zip(plan['jobs'],rows)):
        label=f'job {i}'; m=job['modes']; d=job['dimension']; w=job['width']; rank=job['rank']; stem=panel/f'run_{i:02d}'
        check(label+' identity',r['sequence']==i and all(r[k]==v for k,v in job.items()))
        raw=read(stem.with_suffix('.json')); check(label+' raw record',dict(raw,sequence=i)==r)
        bound=frontier_element_bound(m,d,w)
        if r['status']=='resource_rejected':
            failed+=1; check(label+' preflight rejection',bound>job['max_elements'] and 'Conservative frontier allocation cap' in r['error'] and not stem.with_suffix('.npz').exists()); continue
        completed+=1
        check(label+' completed status',r['status'] in ('accepted','budget_missed'))
        with np.load(stem.with_suffix('.npz')) as data: tensors=[data[f'A{k}'] for k in range(m)]
        info=r['info']; rng=np.random.default_rng(job['seed']); u=haar_unitary(m,rng) if job['ensemble']=='haar' else local_circuit(m,job['depth'],rng)[1]
        B=-math.tanh(pars['core_squeezing'])*(u@u.T); C=truncate_band(B,w); delta=pairing_distance(B,C)
        check(label+' finite shapes',all(np.all(np.isfinite(a)) and a.shape[1]==d for a in tensors) and tensors[0].shape[0]==tensors[-1].shape[2]==1 and all(tensors[k].shape[2]==tensors[k+1].shape[0] for k in range(m-1)))
        check(label+' center norm',abs(np.linalg.norm(tensors[0])-1)<1e-11)
        for k,a in enumerate(tensors[1:],1):
            matrix=a.reshape(a.shape[0],-1); check(label+f' right canonical {k}',np.max(np.abs(matrix@matrix.conj().T-np.eye(a.shape[0])))<1e-11)
        check(label+' storage and rank',sum(a.size for a in tensors)==info['stored_complex_elements'] and max(a.shape[2] for a in tensors)==info['max_bond_actual']<=rank)
        check(label+' frontier accounting',len(info['frontier_bonds'])==m-1 and len(info['transition_counts'])==m and info['frontier_elements']<=info['conservative_element_bound']==bound<=job['max_elements'])
        losses=info['rounding_losses']; joint=-math.expm1(math.fsum(math.log1p(-q) for q in losses)); mass=info['box_mass']
        check(label+' nested survival',len(losses)==m-1 and all(0<=q<1 for q in losses) and abs(joint-info['rounding_joint_tail'])<1e-13)
        e=math.sqrt(max(0.,1-mass*(1-joint))); upper=min(1.,delta+e); lower=max(0.,delta-e)
        check(label+' error accounting',0<mass<=1+1e-12 and abs(delta-info['band_distance'])<1e-13 and abs(e-info['distance_to_band_gaussian'])<1e-13 and abs(upper-r['construction_trace_budget'])<1e-13 and abs(delta+e-info['unclipped_construction_trace_budget'])<1e-13)
        check(label+' target status',r['target_met']==(upper<=job['target'])==(r['status']=='accepted'))
        if job['panel']=='original_unresolved': check(label+' candidate ruled out in pure-state distance',lower>.1)
        if job['panel']=='quality':
            quality+=1; state=mps_to_dense(tensors)
            with np.load(stem.with_suffix('.reference.npz')) as data: original=data['gaussian_box']
            band=reference_box(C,d); recurrence_original=reference_box(B,d)
            check(label+' independent original recurrence',np.max(np.abs(original-recurrence_original))<1e-12)
            check(label+' exact box mass',abs(np.linalg.norm(band)**2-mass)<1e-12)
            overlap=abs(np.vdot(band,state))**2
            check(label+' nested overlap identity',abs(overlap-mass*(1-joint))<1e-12)
            actual=math.sqrt(max(0.,1-abs(np.vdot(original,state))**2))
            check(label+' original full-state distance',abs(actual-r['actual_state_distance'])<1e-12 and lower-1e-10<=actual<=upper+1e-10)
        with np.load(stem.with_suffix('.samples.npz')) as data: alpha=data['alpha']; observed=data['samples']
        z=np.random.default_rng(job['seed']+10000).standard_normal((32,m)); expected_alpha=.5j*math.sqrt(pars['residual_variance'])*(z@u.T)
        check(label+' original correlated displacement',np.array_equal(alpha,expected_alpha))
        check(label+' observed samples only',observed.shape==(32,m) and np.all((observed==0)|(observed==1)) and r['completed_samples']==32 and r['sampling_status']=='complete')
        samples+=len(observed)
    check('all states small references resource rejections samples',completed==56 and quality==20 and failed==4 and samples==1792)
    save(out/'artifact_checks.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        r=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    result=dict(tests_run=r.testsRun,passed=len(r.successes),failures=len(r.failures),errors=len(r.errors),
        skipped_cases=sum('setUpClass' not in t.id() for t,_ in r.skipped),skipped_classes=sum('setUpClass' in t.id() for t,_ in r.skipped),
        new_tests_passed=sum(s.startswith('test_banded_gaussian_mps.') for s in r.successes),
        failure_details=[dict(test=t.id(),traceback=msg) for t,msg in r.failures],error_details=[dict(test=t.id(),traceback=msg) for t,msg in r.errors],
        skip_details=[dict(test=t.id(),reason=msg) for t,msg in r.skipped],full_gate_passed=r.wasSuccessful() and not r.skipped)
    old=read(ROOT/'results/iteration19_validation/validation.json')
    result['unchanged_failure_ids']={x['test'] for k in ('failure_details','error_details') for x in old[k]}=={x['test'] for k in ('failure_details','error_details') for x in result[k]}
    save(out/'validation.json',result)
    for f in ('experiments/verify_iteration20.py','src/gbslab/banded_gaussian_mps.py','tests/test_banded_gaussian_mps.py'):
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2)); print('Artifact checks',sum(c['passed'] for c in checks),'/',len(checks))
    if not all(c['passed'] for c in checks): raise AssertionError('Artifact checks failed; preserve this output directory')


if __name__=='__main__': main()
