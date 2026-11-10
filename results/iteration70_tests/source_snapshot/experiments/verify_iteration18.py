"""Audit frozen plans, full degree sequences, reference paths and regression suite."""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import unittest
import mpmath as mp
import numpy as np
from iteration14_run import save
from verify_iteration11 import RecordingResult
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.meixner_vacuum import vacuum_to_click


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); checks=[]
    check=lambda name,value:checks.append(dict(check=name,passed=bool(value)))
    read=lambda p:json.loads(p.read_text())
    panel=ROOT/'results/iteration18_panel'; ref=ROOT/'results/iteration18_reference'
    plan=read(panel/'plan.json'); inputs=read(panel/'inputs.json'); rows=read(panel/'rows.json'); controls=read(panel/'controls.json')
    check('all planned input/method/degree/control rows',len(inputs)==len(plan['jobs'])==60 and len(rows)==7800 and len(controls)==120)
    check('unique full method degree inventory',len({(r['index'],r['policy'],r['degree']) for r in rows})==7800 and all(r['status']=='complete' for r in rows))
    check('all control identities unique and completed',len({(r['index'],r['k']) for r in controls})==120 and all(r['status']=='complete' for r in controls))
    lookup={(r['index'],r['policy'],r['degree']):r for r in rows}
    for directory,hashes in ((panel,read(panel/'metadata.json')['source_sha256']),(ref,read(ref/'source_hashes.json'))):
        for f,digest in hashes.items():
            check(directory.name+' frozen source '+f,hashlib.sha256((directory/'source_snapshot'/f).read_bytes()).hexdigest()==digest)
            check(directory.name+' current source '+f,hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest)
    max_reference_error=mp.mpf(0); max_precision_delta=mp.mpf(0)
    for i,(job,item) in enumerate(zip(plan['jobs'],inputs)):
        check(f'{i} planned identity',item['index']==i and all(item[k]==v for k,v in job.items()))
        with np.load(panel/f'input_{i:02d}.npz') as data: target=data['target']; u=data['U']
        check(f'{i} target normalization',np.all(target>=0) and abs(math.fsum(target)-1)<1e-12)
        if 'old_index' in job:
            old=ROOT/f"results/iteration17_panel/input_{job['old_index']:02d}.npz"
            check(f'{i} reused input hash',hashlib.sha256(old.read_bytes()).hexdigest()==item['old_input_sha256'])
            with np.load(old) as previous: check(f'{i} reused U',np.array_equal(u,previous['U']))
        m=job['modes']; cardinality=np.array([s.bit_count() for s in range(1<<m)])
        a=job['eta']*math.sinh(job['r'])*math.cosh(job['r']); n=job['eta']*math.sinh(job['r'])**2
        for policy in plan['reference_policies']:
            label=f'{i}/{policy}'
            with np.load(panel/f'input_{i:02d}_{policy}.npz') as data:
                coeff=data['vacuum_coefficients']; exact=data['exact_vacuum']; radii=data['subset_radii']; degrees=data['degrees']; savedraw=data['raw']; savedrepair=data['repaired']
            prefix=coeff.cumsum(axis=0); raw=vacuum_to_click(prefix); repair=np.maximum(raw,0); repair/=repair.sum(axis=1)[:,None]
            check(label+' saved degrees',list(degrees)==plan['saved_degrees'])
            check(label+' coefficients and raw finite',coeff.shape==(65,1<<m) and np.all(np.isfinite(coeff)) and np.all(np.isfinite(raw)))
            check(label+' saved laws reproduce',np.array_equal(raw[degrees],savedraw) and np.array_equal(repair[degrees],savedrepair))
            check(label+' spectral contraction',np.all(radii>=0) and radii.max()<1)
            b=lookup[i,policy,0]['reference_mean']; theoretical=max(abs(n-b+a),abs(n-b-a))/(1+b)
            check(label+' analytical full radius',abs(radii.max()-theoretical)<1e-13)
            # Exact subset values invert to the same target via a distinct SVD path.
            check(label+' SVD exact-vacuum inversion',.5*np.abs(vacuum_to_click(exact)-target).sum()<1e-9)
            if policy=='physical_mean':
                check(label+' positive even coefficients',np.all(coeff>=0) and np.all(coeff[1::2]==0))
                check(label+' globally minimal reference radius',radii.max()<=lookup[i,'paper',0]['spectral_radius']+1e-13)
            for d in range(65):
                row=lookup[i,policy,d]; sub=label+f'/D{d}'; mass=math.fsum(raw[d]); negative=-raw[d,raw[d]<0].sum()
                check(sub+' normalized signed/positive mass',abs(mass-1)<1e-11 and abs(math.fsum(repair[d])-1)<1e-12 and np.all(repair[d]>=0))
                quantities=dict(raw_mass=mass,negative_mass=negative,repair_l1=np.abs(repair[d]-raw[d]).sum(),tv_repaired=.5*np.abs(repair[d]-target).sum(),half_l1_signed=.5*np.abs(raw[d]-target).sum())
                check(sub+' saved metrics',all(abs(value-row[key])<1e-13 for key,value in quantities.items()))
                if 'old_wick_regression_error' in row: check(sub+' old Wick regression',row['old_wick_regression_error']<1e-11)
                if policy=='physical_mean':
                    tail=.5*np.dot(2.**cardinality,np.maximum(exact-prefix[d],0))
                    check(sub+' tail estimate',abs(tail-row['signed_half_l1_bound_estimate'])<1e-12)
                    check(sub+' diagnostic bound covers signed and repaired within floating tolerance',quantities['half_l1_signed']<=tail+1e-9 and quantities['tv_repaired']<=row['repaired_tv_bound_estimate']+1e-9)
            if i<12:
                references=read(ref/f'input_{i:02d}_{policy}.json'); low=references['reference80']; high=references['reference120']
                with mp.workdps(140):
                    for d in range(65):
                        delta=max(abs(mp.mpf(a)-mp.mpf(b)) for a,b in zip(low['raw_laws_decimal'][d],high['raw_laws_decimal'][d]))
                        error=max(abs(mp.mpf(float(a))-mp.mpf(b)) for a,b in zip(raw[d],high['raw_laws_decimal'][d]))
                        max_reference_error=max(max_reference_error,error); max_precision_delta=max(max_precision_delta,delta)
                        check(label+f'/D{d} high precision convergence',delta<mp.mpf('1e-65'))
                        check(label+f'/D{d} independent matrix-polynomial reference',error<mp.mpf('1e-13'))
                    check(label+' high precision target',max(abs(mp.mpf(float(a))-mp.mpf(b)) for a,b in zip(target,high['target_decimal']))<mp.mpf('1e-13'))
    for row in controls:
        i=row['index']; k=row['k']; f=row['failure_mass']
        with np.load(panel/f'input_{i:02d}_retained{k}.npz') as data: q=data['distribution']
        with np.load(panel/f'input_{i:02d}.npz') as data: target=data['target']
        check(f'control {i}/{k} extended probability law',f>=0 and np.all(q>=0) and abs(math.fsum(q)+f-1)<1e-12)
        check(f'control {i}/{k} extended TV',abs(.5*(np.abs(q-target).sum()+f)-row['tv_extended'])<1e-13)
    save(out/'artifact_checks.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),max_reference_raw_error=str(max_reference_error),max_reference_precision_delta=str(max_precision_delta),checks=checks))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        r=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    result=dict(tests_run=r.testsRun,passed=len(r.successes),failures=len(r.failures),errors=len(r.errors),
        skipped_cases=sum('setUpClass' not in t.id() for t,_ in r.skipped),skipped_classes=sum('setUpClass' in t.id() for t,_ in r.skipped),
        new_tests_passed=sum(s.startswith('test_meixner_vacuum.') for s in r.successes),
        failure_details=[dict(test=t.id(),traceback=msg) for t,msg in r.failures],error_details=[dict(test=t.id(),traceback=msg) for t,msg in r.errors],
        skip_details=[dict(test=t.id(),reason=msg) for t,msg in r.skipped],full_gate_passed=r.wasSuccessful() and not r.skipped)
    old=read(ROOT/'results/iteration17_validation/validation.json')
    result['unchanged_failure_ids']={x['test'] for k in ('failure_details','error_details') for x in old[k]}=={x['test'] for k in ('failure_details','error_details') for x in result[k]}
    save(out/'validation.json',result)
    for f in ('experiments/verify_iteration18.py','src/gbslab/meixner_vacuum.py','tests/test_meixner_vacuum.py'):
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2)); print('Artifact checks',sum(c['passed'] for c in checks),'/',len(checks))

if __name__=='__main__': main()
