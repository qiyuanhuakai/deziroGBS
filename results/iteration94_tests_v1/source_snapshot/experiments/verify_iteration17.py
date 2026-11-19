"""Audit all saved signed/repaired laws, controls, high-precision references and tests."""
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
from gbslab.meixner_high_order import threshold_transform
from gbslab.meixner_probe import uniform_parameters

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); checks=[]
    check=lambda name,value:checks.append(dict(check=name,passed=bool(value)))
    read=lambda path:json.loads(path.read_text())
    panel=ROOT/'results/iteration17_panel'; ext=ROOT/'results/iteration17_extension'; ref=ROOT/'results/iteration17_reference'
    inputs=read(panel/'inputs.json'); jobs=read(panel/'plan.json')['jobs']; rows=read(panel/'rows.json'); extra=read(ext/'rows.json'); controls=read(panel/'controls.json')
    check('36 planned inputs, 276 initial laws, 216 extension laws, 72 controls',len(inputs)==len(jobs)==36 and len(rows)==276 and len(extra)==216 and len(controls)==72)
    check('extension covers every m8/m12 input',read(ext/'plan.json')['input_indices']==list(range(12,36)))
    check('all row keys complete and unique',len({(r['index'],r['degree']) for r in rows})==276 and len({(r['index'],r['degree']) for r in extra})==216 and all(r['status']=='complete' for r in rows+extra+controls))
    check('controls complete and unique',len({(r['index'],r['k']) for r in controls})==72)
    for i,(job,item) in enumerate(zip(jobs,inputs)):
        check(f'input {i} planned identity',item['index']==i and all(item[k]==v for k,v in job.items()))
        with np.load(panel/f'input_{i:02d}.npz') as data:
            target=data['target']; U=data['U']; thermal=data['thermal']
        check(f'input {i} valid target',np.all(target>=0) and abs(math.fsum(target)-1)<1e-12)
        check(f'input {i} unitary',np.allclose(U.conj().T@U,np.eye(len(U)),atol=1e-12,rtol=0))
        check(f'input {i} thermal metric',abs(.5*np.abs(thermal-target).sum()-item['thermal_tv'])<1e-14)
    for directory,hashes in ((panel,read(panel/'metadata.json')['source_sha256']),(ext,read(ext/'source_hashes.json')),(ref,read(ref/'source_hashes.json'))):
        for f,digest in hashes.items():
            check(f'{directory.name} frozen source {f}',hashlib.sha256((directory/'source_snapshot'/f).read_bytes()).hexdigest()==digest)
            check(f'{directory.name} current source {f}',hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest)
    for directory,table in ((panel,rows),(ext,extra)):
        for row in table:
            i=row['index']; d=row['degree']; m=inputs[i]['modes']; label=f'{directory.name}/{i}/D{d}'
            with np.load(panel/f'input_{i:02d}.npz') as data: target=data['target']
            with np.load(directory/f'input_{i:02d}_D{d}.npz') as data: raw=data['raw']; repaired=data['repaired']; support=data['subset_coefficients']
            expected=np.maximum(raw,0); expected/=expected.sum()
            check(label+' finite and mass',np.all(np.isfinite(raw)) and abs(math.fsum(raw)-1)<1e-12)
            check(label+' positive repair',np.all(repaired>=0) and np.array_equal(expected,repaired))
            quantities={'raw_mass':math.fsum(raw),'negative_mass':-raw[raw<0].sum(),'repair_l1':np.abs(repaired-raw).sum(),'tv_repaired':.5*np.abs(repaired-target).sum(),'half_l1_signed':.5*np.abs(raw-target).sum()}
            for key,value in quantities.items(): check(label+' '+key,abs(value-row[key])<1e-13)
            check(label+' coefficient count',row['coefficient_count']==math.comb(m+d,d))
            pars=uniform_parameters(m,inputs[i]['r'],inputs[i]['eta']); b=pars['b']
            bits=(np.arange(1<<m)[:,None]>>np.arange(m))&1
            base=np.prod(np.where(bits,b/(1+b),1/(1+b)),axis=1)
            check(label+' saved support yields law',np.allclose(base*threshold_transform(support,b),raw,rtol=0,atol=1e-14))
            if directory==ext and d<=6:
                with np.load(panel/f'input_{i:02d}_D{d}.npz') as old: check(label+' unchanged prefix',np.array_equal(raw,old['raw']) and row['prefix_regression_error']==0)
            if 'degree_two_regression_error' in row: check(label+' low-degree regression',row['degree_two_regression_error']<1e-13)
        for i in sorted({r['index'] for r in table}):
            with np.load(directory/f'input_{i:02d}_coefficients.npz') as data: indices=data['indices']; coefficients=data['coefficients']
            D=max(r['degree'] for r in table if r['index']==i); m=inputs[i]['modes']
            check(f'{directory.name}/{i} coefficient inventory',indices.shape==(math.comb(m+D,D),m) and len(set(map(tuple,indices)))==len(indices) and np.all(indices>=0) and np.all(indices.sum(axis=1)<=D) and np.all(np.isfinite(coefficients)))
    for row in controls:
        i=row['index']; k=row['k']; f=row['failure_mass']
        with np.load(panel/f'input_{i:02d}_retained{k}.npz') as data: q=data['distribution']
        with np.load(panel/f'input_{i:02d}.npz') as data: target=data['target']
        check(f'control {i}/{k} extended mass',f>=0 and np.all(q>=0) and abs(math.fsum(q)+f-1)<1e-12)
        check(f'control {i}/{k} extended TV',abs(.5*(np.abs(q-target).sum()+f)-row['tv_extended'])<1e-13)
    check('high precision covers every m4 input',read(ref/'plan.json')['input_indices']==list(range(12)) and len(read(ref/'rows.json'))==12)
    for i in range(12):
        references=read(ref/f'input_{i:02d}.json'); low=references['reference80']; high=references['reference120']
        with mp.workdps(140):
            with np.load(panel/f'input_{i:02d}_coefficients.npz') as data: coefficients=data['coefficients']
            c=list(map(mp.mpf,high['coefficients_decimal']))
            check(f'reference {i} coefficient count',len(c)==len(coefficients)==495)
            check(f'reference {i} coefficient convergence',max(abs(mp.mpf(a)-b) for a,b in zip(low['coefficients_decimal'],c))<mp.mpf('1e-65'))
            check(f'reference {i} shifted versus unshifted coefficients',max(abs(mp.mpf(float(a))-b)/max(1,abs(b)) for a,b in zip(coefficients,c))<mp.mpf('1e-13'))
            for d,law in enumerate(high['raw_laws_decimal']):
                with np.load(panel/f'input_{i:02d}_D{d}.npz') as data: raw=data['raw']
                check(f'reference {i}/{d} raw convergence',max(abs(mp.mpf(a)-mp.mpf(b)) for a,b in zip(low['raw_laws_decimal'][d],law))<mp.mpf('1e-65'))
                check(f'reference {i}/{d} independent raw law',max(abs(mp.mpf(float(a))-mp.mpf(b)) for a,b in zip(raw,law))<mp.mpf('1e-13'))
            with np.load(panel/f'input_{i:02d}.npz') as data: target=data['target']
            check(f'reference {i} independent target',max(abs(mp.mpf(float(a))-mp.mpf(b)) for a,b in zip(target,high['target_decimal']))<mp.mpf('1e-13'))
            check(f'reference {i} target convergence',max(abs(mp.mpf(a)-mp.mpf(b)) for a,b in zip(low['target_decimal'],high['target_decimal']))<mp.mpf('1e-65'))
    save(out/'artifact_checks.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        r=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    result=dict(tests_run=r.testsRun,passed=len(r.successes),failures=len(r.failures),errors=len(r.errors),
        skipped_cases=sum('setUpClass' not in t.id() for t,_ in r.skipped),skipped_classes=sum('setUpClass' in t.id() for t,_ in r.skipped),
        new_tests_passed=sum(s.startswith('test_meixner_high_order.') for s in r.successes),
        failure_details=[dict(test=t.id(),traceback=msg) for t,msg in r.failures],error_details=[dict(test=t.id(),traceback=msg) for t,msg in r.errors],
        skip_details=[dict(test=t.id(),reason=msg) for t,msg in r.skipped],full_gate_passed=r.wasSuccessful() and not r.skipped)
    old=read(ROOT/'results/iteration16_validation/validation.json')
    result['unchanged_failure_ids']={x['test'] for k in ('failure_details','error_details') for x in old[k]}=={x['test'] for k in ('failure_details','error_details') for x in result[k]}
    save(out/'validation.json',result)
    for f in ('experiments/verify_iteration17.py','tests/test_meixner_high_order.py','src/gbslab/meixner_high_order.py'):
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2)); print('Artifact checks',sum(c['passed'] for c in checks),'/',len(checks))

if __name__=='__main__': main()
